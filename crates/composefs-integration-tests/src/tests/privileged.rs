//! Privileged integration tests requiring root and fs-verity support.
//!
//! These tests run `cfsctl` without `--insecure` on a real ext4 filesystem
//! with the verity feature enabled. They need root to create loop mounts.
//!
//! When run on the host (not as root), each test automatically re-executes
//! itself inside a bcvk ephemeral VM where it has real root and kernel
//! fs-verity support. The `COMPOSEFS_IN_VM` env var prevents infinite
//! recursion — see [`require_privileged`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use xshell::{Shell, cmd};

use composefs_oci::composefs::fsverity::{FsVerityHashValue, Sha256HashValue, Sha512HashValue};
use composefs_oci::composefs::repository::{Repository, RepositoryConfig};

use crate::{cfsctl, create_test_rootfs, integration_test};

/// Ensure we're running in a privileged environment, or re-exec this test inside a VM.
///
/// If already root (e.g. inside a bcvk VM), returns `Ok(None)` and the
/// test proceeds normally.
///
/// If not root and `COMPOSEFS_TEST_IMAGE` is set, spawns
/// `bcvk ephemeral run-ssh <image> -- cfsctl-integration-tests --exact <test>`
/// and returns `Ok(Some(()))` — the caller should return immediately since
/// the test already ran in the VM.
///
/// If not root and no test image is configured, returns an error.
///
/// This is also used by cstor tests which need user namespace support
/// (via `podman unshare`) that may not be available on GHA runners.
pub fn require_privileged(test_name: &str) -> Result<Option<()>> {
    require_privileged_with_memory(test_name, "4G")
}

/// Like [`require_privileged`], but allows specifying VM memory size.
///
/// Tests that pull large container images inside the VM may need more
/// memory since `/var` is backed by tmpfs.
pub fn require_privileged_with_memory(test_name: &str, memory: &str) -> Result<Option<()>> {
    if rustix::process::getuid().is_root() {
        return Ok(None);
    }

    // We're on the host without root — delegate to a VM.
    if std::env::var_os("COMPOSEFS_IN_VM").is_some() {
        bail!("COMPOSEFS_IN_VM is set but we're not root — VM setup is broken");
    }

    let image = std::env::var("COMPOSEFS_TEST_IMAGE").map_err(|_| {
        anyhow::anyhow!(
            "not root and COMPOSEFS_TEST_IMAGE not set; \
             run `just test-integration-vm` to build the image and run all tests"
        )
    })?;

    run_in_vm(test_name, &image, Some(memory))?;
    Ok(Some(()))
}

/// Directory in which to keep the journal and console of each VM whose
/// test fails, as `<dir>/<test>-<attempt>/`, e.g. for CI to upload.
const VM_LOG_DIR_ENV: &str = "COMPOSEFS_VM_LOG_DIR";

/// Run `test_name` inside a bcvk ephemeral VM booted from `image`.
///
/// When the VM dies or stops answering, ssh exits 255 with no other
/// output, so if [`VM_LOG_DIR_ENV`] is set the VM's journal and console
/// are captured and kept when the test fails.
fn run_in_vm(test_name: &str, image: &str, memory: Option<&str>) -> Result<()> {
    let sh = Shell::new()?;
    let bcvk = std::env::var("BCVK_PATH").unwrap_or_else(|_| "bcvk".into());
    let memory_args = memory.map(|m| ["--memory", m]).into_iter().flatten();

    let log_dir = std::env::var_os(VM_LOG_DIR_ENV).map(|base| {
        // Set by nextest; keeps a retry from overwriting the failed attempt's logs.
        let attempt = std::env::var("NEXTEST_ATTEMPT").unwrap_or_else(|_| "1".into());
        PathBuf::from(base).join(format!("{test_name}-{attempt}"))
    });
    let mut log_args = Vec::new();
    if let Some(dir) = &log_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating VM log directory {}", dir.display()))?;
        log_args.push("--log-dir".to_owned());
        log_args.push(format!("journal,console={}", dir.display()));
    }

    cmd!(
        sh,
        "{bcvk} ephemeral run-ssh {memory_args...} {log_args...} {image} -- cfsctl-integration-tests --exact {test_name}"
    )
    .run()?;

    if let Some(dir) = &log_dir {
        // Best effort: only failures' logs are interesting.
        let _ = std::fs::remove_dir_all(dir);
    }
    Ok(())
}

/// Check if user namespaces work (needed for podman unshare).
fn userns_works() -> bool {
    std::process::Command::new("podman")
        .args(["unshare", "true"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Ensure user namespace support is available, or re-exec this test inside a VM.
///
/// Unlike `require_privileged`, this doesn't require root — it just needs
/// working user namespaces (for `podman unshare`). If user namespaces work,
/// the test proceeds normally. Otherwise, it dispatches to a VM.
///
/// Returns `Ok(None)` if the test should proceed, `Ok(Some(()))` if it was
/// dispatched to a VM and the caller should return immediately.
pub fn require_userns(test_name: &str) -> Result<Option<()>> {
    // If we're root (e.g. in VM), userns works
    if rustix::process::getuid().is_root() {
        return Ok(None);
    }

    // Check if userns works on this host
    if userns_works() {
        return Ok(None);
    }

    // userns doesn't work — delegate to a VM
    if std::env::var_os("COMPOSEFS_IN_VM").is_some() {
        bail!("COMPOSEFS_IN_VM is set but userns doesn't work — VM setup is broken");
    }

    let image = std::env::var("COMPOSEFS_TEST_IMAGE").map_err(|_| {
        anyhow::anyhow!(
            "user namespaces not available and COMPOSEFS_TEST_IMAGE not set; \
             run `just build-test-image` or use `just test-integration-vm`"
        )
    })?;

    run_in_vm(test_name, &image, None)?;
    Ok(Some(()))
}

/// A temporary directory backed by a loopback ext4 filesystem with verity support.
///
/// tmpfs doesn't support fs-verity, so privileged tests that need verity
/// (i.e. running cfsctl without `--insecure`) must use a real filesystem.
/// This creates a sparse file, formats it as ext4 with the verity feature,
/// and loop-mounts it to a temp directory.
pub(crate) struct VerityTempDir {
    mountpoint: PathBuf,
    _backing: tempfile::TempDir,
}

impl VerityTempDir {
    pub(crate) fn new() -> Result<Self> {
        let backing = tempfile::tempdir()?;
        let img = backing.path().join("fs.img");
        let mountpoint = backing.path().join("mnt");
        std::fs::create_dir(&mountpoint)?;

        let sh = Shell::new()?;
        cmd!(sh, "truncate -s 256M {img}").run()?;
        cmd!(sh, "mkfs.ext4 -q -O verity -b 4096 {img}").run()?;
        cmd!(sh, "mount -o loop {img} {mountpoint}").run()?;

        // Create a repo subdirectory (cfsctl needs it to exist)
        std::fs::create_dir(mountpoint.join("repo"))?;

        Ok(Self {
            mountpoint,
            _backing: backing,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.mountpoint
    }
}

impl Drop for VerityTempDir {
    fn drop(&mut self) {
        let _ = std::process::Command::new("umount")
            .arg(&self.mountpoint)
            .status();
    }
}

/// A temporary directory backed by `fuse-overlayfs`.
///
/// Unlike the kernel's overlayfs, `fuse-overlayfs` doesn't support
/// `O_TMPFILE` (see bootc-dev/bootc#2340), which is exactly the
/// configuration `bootc container ukify` runs into in CI environments
/// (e.g. GitLab CI, or nested-overlay container build contexts) that use
/// it as their storage driver. This is used to reproduce that failure
/// mode and confirm the repository's named-tmpfile fallback handles it.
struct FuseOverlayTempDir {
    mountpoint: PathBuf,
    _backing: tempfile::TempDir,
}

impl FuseOverlayTempDir {
    fn new() -> Result<Self> {
        let backing = tempfile::tempdir()?;
        let lower = backing.path().join("lower");
        let upper = backing.path().join("upper");
        let work = backing.path().join("work");
        let mountpoint = backing.path().join("merged");
        for dir in [&lower, &upper, &work, &mountpoint] {
            std::fs::create_dir(dir)?;
        }

        let sh = Shell::new()?;
        cmd!(
            sh,
            "fuse-overlayfs -o lowerdir={lower},upperdir={upper},workdir={work} {mountpoint}"
        )
        .run()
        .context("mounting fuse-overlayfs")?;

        // Sanity check: confirm this mount really doesn't support
        // O_TMPFILE, so the test actually exercises the fallback path
        // rather than passing vacuously if fuse-overlayfs behavior changes.
        match rustix::fs::openat(
            rustix::fs::CWD,
            &mountpoint,
            rustix::fs::OFlags::RDWR | rustix::fs::OFlags::TMPFILE,
            rustix::fs::Mode::from_raw_mode(0o644),
        ) {
            Err(rustix::io::Errno::OPNOTSUPP) => {}
            Ok(_) => bail!(
                "fuse-overlayfs at {} unexpectedly supports O_TMPFILE; \
                 this test no longer exercises the fallback path",
                mountpoint.display()
            ),
            Err(e) => {
                return Err(e).with_context(|| {
                    format!("unexpected error probing O_TMPFILE support on {mountpoint:?}")
                });
            }
        }

        std::fs::create_dir(mountpoint.join("repo"))?;

        Ok(Self {
            mountpoint,
            _backing: backing,
        })
    }

    fn path(&self) -> &Path {
        &self.mountpoint
    }
}

impl Drop for FuseOverlayTempDir {
    fn drop(&mut self) {
        let _ = std::process::Command::new("umount")
            .arg(&self.mountpoint)
            .status();
    }
}

/// Reproduces bootc-dev/bootc#2340: on `fuse-overlayfs`, which doesn't
/// support `O_TMPFILE`, ingesting object data used to fail outright with
/// `EOPNOTSUPP` ("Opening temp file in objects directory"). This is
/// exactly the path bootc's composefs digest computation exercises
/// (`ensure_object_from_reader`, via `ensure_object_from_fd`) when reading
/// a container rootfs through a storage backend without O_TMPFILE support.
///
/// Verifies the repository's named-tmpfile fallback lets object ingestion
/// (including the dedup case) succeed and produce correct, retrievable
/// objects on such filesystems.
fn privileged_repo_on_fuse_overlayfs() -> Result<()> {
    if require_privileged("privileged_repo_on_fuse_overlayfs")?.is_some() {
        return Ok(());
    }

    let overlay = FuseOverlayTempDir::new()?;
    let repo_path = overlay.path().join("repo");
    let repo = init_insecure_repo_at::<Sha256HashValue>(
        &repo_path,
        composefs_oci::composefs::fsverity::Algorithm::Sha256 { lg_blocksize: 12 },
    )?;

    let data = b"bootc-dev/bootc#2340: O_TMPFILE fallback on fuse-overlayfs".repeat(1024);

    // First call exercises the named-tmpfile fallback's rename-into-place path.
    let id = repo.ensure_object_from_reader(data.as_slice(), data.len() as u64)?;
    ensure!(
        repo.read_object(&id)? == data,
        "object content mismatch after storing via fuse-overlayfs fallback"
    );

    // Second call with identical data exercises the dedup (unlink) path.
    let id2 = repo.ensure_object_from_reader(data.as_slice(), data.len() as u64)?;
    ensure!(id == id2, "expected identical digest for identical content");

    Ok(())
}
integration_test!(privileged_repo_on_fuse_overlayfs);

fn privileged_check_root() -> Result<()> {
    if require_privileged("privileged_check_root")?.is_some() {
        return Ok(());
    }
    Ok(())
}
integration_test!(privileged_check_root);

fn privileged_repo_without_insecure() -> Result<()> {
    if require_privileged("privileged_repo_without_insecure")?.is_some() {
        return Ok(());
    }

    let sh = Shell::new()?;
    let cfsctl = cfsctl()?;
    let verity_dir = VerityTempDir::new()?;
    let repo = verity_dir.path().join("repo");

    // Init on ext4+verity: meta.json gets verity enabled → secure repo
    cmd!(sh, "{cfsctl} --repo {repo} init").run()?;

    let output = cmd!(sh, "{cfsctl} --repo {repo} gc").read()?;
    ensure!(
        output.contains("Objects: 0 removed"),
        "gc on fresh repo failed: {output}"
    );
    Ok(())
}
integration_test!(privileged_repo_without_insecure);

/// Build a bootable test OCI image, mount it via `cfsctl oci mount` (plain
/// and `--bootable`), and verify the filesystem content differs correctly.
/// The plain mount should contain /boot/EFI/Linux/test-6.1.0.efi (the UKI),
/// while the bootable mount should have an empty /boot (transform_for_boot
/// clears it) but still have /usr content intact.
fn privileged_oci_bootable_mount() -> Result<()> {
    if require_privileged("privileged_oci_bootable_mount")?.is_some() {
        return Ok(());
    }

    let sh = Shell::new()?;
    let cfsctl = cfsctl()?;
    let verity_dir = VerityTempDir::new()?;
    let repo_path = verity_dir.path().join("repo");
    let repo_arg = repo_path.to_str().unwrap();
    let hash = "sha256";

    composefs_oci::test_util::create_test_bootable_oci_image(&repo_path, "boot-test:v1")?;

    let inspect_output = cmd!(
        sh,
        "{cfsctl} --insecure --hash {hash} --repo {repo_arg} oci inspect boot-test:v1"
    )
    .read()?;
    let inspect: serde_json::Value = serde_json::from_str(&inspect_output)?;
    ensure!(
        inspect.get("composefs_erofs").is_some(),
        "inspect should show composefs_erofs field"
    );
    ensure!(
        inspect.get("composefs_boot_erofs").is_some(),
        "inspect should show composefs_boot_erofs field"
    );

    // Plain mount: full filesystem including /boot
    let mountpoint1 = tempfile::tempdir()?;
    let mp1 = mountpoint1.path().to_str().unwrap();
    cmd!(
        sh,
        "{cfsctl} --insecure --hash {hash} --repo {repo_arg} oci mount boot-test:v1 {mp1}"
    )
    .run()?;

    ensure!(
        mountpoint1
            .path()
            .join("boot/EFI/Linux/test-6.1.0.efi")
            .exists(),
        "plain mount should contain UKI at /boot/EFI/Linux/test-6.1.0.efi"
    );

    cmd!(sh, "umount {mp1}").run()?;

    // Bootable mount: /boot empty, /usr intact
    let mountpoint2 = tempfile::tempdir()?;
    let mp2 = mountpoint2.path().to_str().unwrap();
    cmd!(
        sh,
        "{cfsctl} --insecure --hash {hash} --repo {repo_arg} oci mount --bootable boot-test:v1 {mp2}"
    )
    .run()?;

    let boot_dir = mountpoint2.path().join("boot");
    ensure!(
        boot_dir.is_dir(),
        "bootable mount should have /boot directory"
    );
    let boot_entries: Vec<_> = std::fs::read_dir(&boot_dir)?.collect();
    ensure!(
        boot_entries.is_empty(),
        "bootable mount /boot should be empty, found {} entries",
        boot_entries.len()
    );

    ensure!(
        !mountpoint2
            .path()
            .join("boot/EFI/Linux/test-6.1.0.efi")
            .exists(),
        "bootable mount should NOT contain UKI"
    );

    ensure!(
        mountpoint2
            .path()
            .join("usr/lib/modules/6.1.0/vmlinuz")
            .exists(),
        "bootable mount should still have kernel at /usr/lib/modules/6.1.0/vmlinuz"
    );

    let os_release = std::fs::read_to_string(mountpoint2.path().join("etc/os-release"))?;
    ensure!(
        os_release.contains("ID=test"),
        "bootable mount os-release missing ID=test: {os_release:?}"
    );

    cmd!(sh, "umount {mp2}").run()?;

    Ok(())
}
integration_test!(privileged_oci_bootable_mount);

/// Build a test OCI image, mount it via `cfsctl oci mount`, and verify
/// the filesystem content. Uses the library only for image creation (test
/// setup); all verification goes through the CLI.
fn privileged_oci_pull_mount() -> Result<()> {
    if require_privileged("privileged_oci_pull_mount")?.is_some() {
        return Ok(());
    }

    let sh = Shell::new()?;
    let cfsctl = cfsctl()?;
    let verity_dir = VerityTempDir::new()?;
    let repo_path = verity_dir.path().join("repo");
    let repo_arg = repo_path.to_str().unwrap();

    // Create a test OCI image with EROFS linked (library used only for setup)
    composefs_oci::test_util::create_test_oci_image(&repo_path, "mount-test:v1")?;

    // test_util creates SHA-256 repos; tell cfsctl to match
    let hash = "sha256";

    // Verify inspect shows the EROFS ref
    let inspect_output = cmd!(
        sh,
        "{cfsctl} --insecure --hash {hash} --repo {repo_arg} oci inspect mount-test:v1"
    )
    .read()?;
    let inspect: serde_json::Value = serde_json::from_str(&inspect_output)?;
    ensure!(
        inspect.get("composefs_erofs").is_some(),
        "inspect should show composefs_erofs field"
    );

    // Mount via cfsctl oci mount
    let mountpoint = tempfile::tempdir()?;
    let mp = mountpoint.path().to_str().unwrap();
    cmd!(
        sh,
        "{cfsctl} --insecure --hash {hash} --repo {repo_arg} oci mount mount-test:v1 {mp}"
    )
    .run()?;

    // Verify file content at the mountpoint
    let hostname = std::fs::read_to_string(mountpoint.path().join("etc/hostname"))?;
    ensure!(hostname == "test-host", "hostname mismatch: {hostname:?}");

    let os_release = std::fs::read_to_string(mountpoint.path().join("etc/os-release"))?;
    ensure!(
        os_release.contains("ID=test"),
        "os-release missing ID: {os_release:?}"
    );

    // busybox is a 4096-byte external file (random data seeded from size)
    let busybox = std::fs::read(mountpoint.path().join("usr/bin/busybox"))?;
    ensure!(
        busybox.len() == 4096,
        "busybox size mismatch: expected 4096, got {}",
        busybox.len()
    );

    let sh_target = std::fs::read_link(mountpoint.path().join("usr/bin/sh"))?;
    ensure!(
        sh_target.to_str() == Some("busybox"),
        "sh symlink target mismatch: {sh_target:?}"
    );

    // App layer has a 512-byte README (external, random data)
    let readme = std::fs::read(mountpoint.path().join("usr/share/doc/README"))?;
    ensure!(
        readme.len() == 512,
        "README size mismatch: expected 512, got {}",
        readme.len()
    );

    ensure!(mountpoint.path().join("tmp").is_dir(), "/tmp missing");
    ensure!(mountpoint.path().join("var").is_dir(), "/var missing");
    ensure!(
        mountpoint.path().join("usr/lib").is_dir(),
        "/usr/lib missing"
    );

    Ok(())
}
integration_test!(privileged_oci_pull_mount);

/// Test `cfsctl mount` with `--upperdir`/`--workdir`/`--read-write` flags.
///
/// Exercises three scenarios:
/// 1. Upperdir present but no `--read-write` → mount stays read-only.
/// 2. Upperdir + `--read-write` → writes land in the upper directory.
/// 3. Plain mount (no upperdir) → baseline sanity check.
fn privileged_mount_upper_layer() -> Result<()> {
    if require_privileged("privileged_mount_upper_layer")?.is_some() {
        return Ok(());
    }

    let sh = Shell::new()?;
    let cfsctl = cfsctl()?;
    let verity_dir = VerityTempDir::new()?;
    let repo_path = verity_dir.path().join("repo");
    let repo_arg = repo_path.to_str().unwrap();

    cmd!(sh, "{cfsctl} --insecure --repo {repo_arg} init").run()?;

    let fixture_dir = tempfile::tempdir()?;
    let rootfs = create_test_rootfs(fixture_dir.path())?;
    let rootfs = rootfs.to_str().unwrap();

    // Use a named ref so we can mount via "refs/upper-test" — the bare output
    // of create-image is "algo:hex" but mount expects just the hex or a ref path.
    cmd!(
        sh,
        "{cfsctl} --insecure --repo {repo_arg} create-image {rootfs} upper-test"
    )
    .run()?;
    let image_name = "refs/upper-test";

    // --- Scenario 1: upperdir without --read-write keeps the mount read-only ---
    {
        let upper1 = tempfile::tempdir()?;
        let work1 = tempfile::tempdir()?;
        let mp1 = tempfile::tempdir()?;
        let upper1_path = upper1.path().to_str().unwrap();
        let work1_path = work1.path().to_str().unwrap();
        let mp1_path = mp1.path().to_str().unwrap();

        cmd!(
            sh,
            "{cfsctl} --insecure --repo {repo_arg} mount {image_name} {mp1_path}
             --upperdir {upper1_path} --workdir {work1_path}"
        )
        .run()?;

        // The rootfs fixture has etc/hostname — verify the mount is populated.
        let hostname = std::fs::read_to_string(mp1.path().join("etc/hostname"))?;
        ensure!(
            hostname == "integration-test\n",
            "hostname mismatch in scenario 1: {hostname:?}"
        );

        // Without --read-write the mount must be read-only.
        let write_result = std::fs::write(mp1.path().join("should_fail"), b"data");
        ensure!(
            write_result.is_err(),
            "write to read-only overlayfs mount should fail"
        );

        cmd!(sh, "umount {mp1_path}").run()?;
    }

    // --- Scenario 2: upperdir + --read-write allows writes that persist in upperdir ---
    {
        let upper2 = tempfile::tempdir()?;
        let work2 = tempfile::tempdir()?;
        let mp2 = tempfile::tempdir()?;
        let upper2_path = upper2.path().to_str().unwrap();
        let work2_path = work2.path().to_str().unwrap();
        let mp2_path = mp2.path().to_str().unwrap();

        cmd!(
            sh,
            "{cfsctl} --insecure --repo {repo_arg} mount {image_name} {mp2_path}
             --upperdir {upper2_path} --workdir {work2_path} --read-write"
        )
        .run()?;

        // Write a new file through the mount.
        let new_file = mp2.path().join("upper_test_file");
        std::fs::write(&new_file, b"hello from upper")?;

        // Read it back through the mount.
        let content = std::fs::read(&new_file)?;
        ensure!(
            content == b"hello from upper",
            "newly written file content mismatch"
        );

        cmd!(sh, "umount {mp2_path}").run()?;

        // After unmount the file must be visible in the upperdir itself.
        let upper_file = upper2.path().join("upper_test_file");
        ensure!(
            upper_file.exists(),
            "written file should persist in upperdir after unmount"
        );
        let upper_content = std::fs::read(&upper_file)?;
        ensure!(
            upper_content == b"hello from upper",
            "upperdir file content mismatch after unmount"
        );
    }

    // --- Scenario 3: plain mount (no upperdir) still works as a baseline ---
    {
        let mp3 = tempfile::tempdir()?;
        let mp3_path = mp3.path().to_str().unwrap();

        cmd!(
            sh,
            "{cfsctl} --insecure --repo {repo_arg} mount {image_name} {mp3_path}"
        )
        .run()?;

        // Check a known file rather than read_dir to avoid keeping a dirfd open.
        ensure!(
            mp3.path().join("etc/hostname").exists(),
            "plain mount should expose etc/hostname from rootfs"
        );

        cmd!(sh, "umount {mp3_path}").run()?;
    }

    Ok(())
}
integration_test!(privileged_mount_upper_layer);

/// A repository in `dir` that requires fs-verity.
fn strict_sha512_repo(dir: &Path) -> Result<Repository<Sha512HashValue>> {
    let (repo, _) = Repository::<Sha512HashValue>::init_path(
        rustix::fs::CWD,
        dir.join("repo"),
        RepositoryConfig::new(composefs_oci::composefs::fsverity::Algorithm::SHA512),
    )?;
    ensure!(!repo.is_insecure(), "repository should require fs-verity");
    Ok(repo)
}

/// Reads the directory tree at `rootfs` like cfsctl does, with large files
/// stored in `repo`.
fn read_rootfs(
    repo: &Arc<Repository<Sha512HashValue>>,
    rootfs: &Path,
) -> Result<composefs_oci::composefs::tree::FileSystem<Sha512HashValue>> {
    // Like cfsctl: the path is absolute, but the API needs a directory fd.
    let dirfd = rustix::fs::open(
        ".",
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::DIRECTORY | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )?;
    tokio::runtime::Runtime::new()?.block_on(composefs_oci::composefs::fs::read_filesystem(
        dirfd,
        rootfs.to_path_buf(),
        Some(repo.clone()),
    ))
}

/// Mounts the image set `id` of `repo`, runs `check` on the mount point,
/// and unmounts it.
fn check_image_set_mount(
    repo: &Repository<Sha512HashValue>,
    id: &str,
    check: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    let mountpoint = tempfile::tempdir()?;
    let mnt = mountpoint.path();
    composefs_oci::composefs::mount::mount_at(repo.mount_image_set(id)?, rustix::fs::CWD, mnt)?;
    let result = check(mnt);
    let sh = Shell::new()?;
    cmd!(sh, "umount {mnt}").run()?;
    result
}

/// The names in the directory `path`, sorted.
fn dir_names(path: &Path) -> Result<Vec<String>> {
    let mut names = std::fs::read_dir(path)?
        .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
        .collect::<Result<Vec<_>>>()?;
    names.sort();
    Ok(names)
}

/// Mount an image set of two images, the root filesystem and its
/// `/usr/lib/modules` split off into an image of its own, from a repository
/// that requires fs-verity, and check that the mount shows the whole tree.
///
/// cfsctl has no image set commands, so this goes through the library.
fn privileged_image_set_mount() -> Result<()> {
    use std::ffi::OsStr;

    use composefs_oci::composefs::image_set::{ImageSet, ImageSetComponent};

    if require_privileged("privileged_image_set_mount")?.is_some() {
        return Ok(());
    }

    let verity_dir = VerityTempDir::new()?;
    let repo = Arc::new(strict_sha512_repo(verity_dir.path())?);

    let fixture_dir = tempfile::tempdir()?;
    let rootfs = create_test_rootfs(fixture_dir.path())?;
    let kver = rootfs.join("usr/lib/modules/6.1.0");
    std::fs::create_dir_all(kver.join("kernel"))?;
    let module = "m".repeat(128 * 1024);
    std::fs::write(kver.join("kernel/test.ko"), &module)?;
    std::fs::write(kver.join("modules.dep"), "kernel/test.ko:\n")?;

    let mut fs = read_rootfs(&repo, &rootfs)?;
    let modules = fs.split_off(OsStr::new("/usr/lib/modules"))?;
    let format = repo.erofs_version();
    let component = |digest, target: &str| ImageSetComponent {
        digest,
        format,
        target: target.into(),
    };
    let set = ImageSet::new(vec![
        component(fs.commit_image(&repo, None)?, "/"),
        component(modules.commit_image(&repo, None)?, "/usr/lib/modules"),
    ])?;
    let id = repo.write_image_set(Some("test"), &set)?.to_hex();

    check_image_set_mount(&repo, &id, |mnt| {
        let read = |path: &str| std::fs::read_to_string(mnt.join(path));
        let names = |path: &str| dir_names(&mnt.join(path));
        // From the root image
        ensure!(read("etc/hostname")? == "integration-test\n");
        ensure!(read("usr/bin/hello")? == "x".repeat(128 * 1024));
        // From the modules image, merged into the root image's /usr/lib
        ensure!(read("usr/lib/modules/6.1.0/kernel/test.ko")? == module);
        ensure!(read("usr/lib/modules/6.1.0/modules.dep")? == "kernel/test.ko:\n");
        ensure!(
            names("usr/lib")? == ["modules", "readme.txt"],
            "{:?}",
            names("usr/lib")
        );
        ensure!(names("usr/lib/modules")? == ["6.1.0"]);
        let mountinfo = std::fs::read_to_string("/proc/self/mountinfo")?;
        let line = mountinfo
            .lines()
            .find(|line| line.contains(&format!("composefs:{id}")))
            .context("image set mount not in mountinfo")?;
        ensure!(line.contains("verity=require"), "{line}");
        Ok(())
    })
}
integration_test!(privileged_image_set_mount);

/// The writer escapes `trusted.overlay.*` xattrs, so the image set tests
/// write unescaped ones under this decoy name and patch it.
const OVERLAY_DECOY: &[u8] = b"ovxrlay";

/// Writes the image of `dumpfile`, under a root directory entry, into `repo`,
/// with every [`OVERLAY_DECOY`] patched to `overlay`.  The superblock's xattr
/// name filter is turned off in a patched image, so that the kernel doesn't
/// skip the patched names.
fn write_patched_image(
    repo: &Repository<Sha512HashValue>,
    dumpfile: &str,
) -> Result<Sha512HashValue> {
    use composefs_oci::composefs::{
        dumpfile::dumpfile_to_validated_filesystem,
        erofs::{
            format::{FEATURE_COMPAT_XATTR_FILTER, Superblock},
            writer::mkfs_erofs_versioned,
        },
    };

    /// Where the EROFS superblock starts in a composefs image.
    const SUPERBLOCK_OFFSET: usize = 1024;
    const FEATURE_COMPAT: usize =
        SUPERBLOCK_OFFSET + std::mem::offset_of!(Superblock, feature_compat);

    let dumpfile = format!("/ 0 40755 2 0 0 0 1000.0 - - -\n{dumpfile}");
    let fs = dumpfile_to_validated_filesystem::<Sha512HashValue>(&dumpfile)?;
    let mut data = mkfs_erofs_versioned(&fs, repo.erofs_version()).to_vec();
    let mut patched = false;
    while let Some(at) = data
        .windows(OVERLAY_DECOY.len())
        .position(|w| w == OVERLAY_DECOY)
    {
        data[at..at + OVERLAY_DECOY.len()].copy_from_slice(b"overlay");
        patched = true;
    }
    if patched {
        let compat = &mut data[FEATURE_COMPAT..FEATURE_COMPAT + 4];
        let flags = u32::from_le_bytes(compat.try_into()?) & !FEATURE_COMPAT_XATTR_FILTER;
        compat.copy_from_slice(&flags.to_le_bytes());
    }
    repo.write_image(None, &data)
}

/// Overlayfs xattrs on a component's ancestor directories are refused.
/// overlayfs (with `redirect_dir=on`) acts on them in every lower layer but
/// the last, so `trusted.overlay.redirect` could show the files of a
/// component below outside its target, and `trusted.overlay.opaque` could
/// hide them.
///
/// The set is the root, a `/usr/lib/modules` component whose `/usr` carries
/// the xattr, and a `/usr/lib/firmware` component below it.
fn privileged_image_set_overlay_xattrs() -> Result<()> {
    use composefs_oci::composefs::image_set::{ImageSet, ImageSetComponent};

    const ROOT: &str = "\
/usr 0 40755 2 0 0 0 1000.0 - - -
/usr/bin 0 40755 2 0 0 0 1000.0 - - -
/usr/bin/true 4 100755 1 0 0 0 1000.0 - root -
/usr/lib 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/firmware 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules 0 40755 2 0 0 0 1000.0 - - -
";
    const FIRMWARE: &str = "\
/usr 0 40755 2 0 0 0 1000.0 - - -
/usr/lib 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/firmware 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/firmware/fw.bin 2 100644 1 0 0 0 1000.0 - fw -
/usr/lib/firmware/inject 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/firmware/inject/bin 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/firmware/inject/bin/evil 4 100755 1 0 0 0 1000.0 - evil -
";
    let modules = |usr_xattr: &str| {
        format!(
            "\
/usr 0 40755 2 0 0 0 1000.0 - - - {usr_xattr}
/usr/lib 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules/a.ko 1 100644 1 0 0 0 1000.0 - a -
"
        )
    };

    if require_privileged("privileged_image_set_overlay_xattrs")?.is_some() {
        return Ok(());
    }
    let verity_dir = VerityTempDir::new()?;
    let repo = strict_sha512_repo(verity_dir.path())?;
    let format = repo.erofs_version();
    let image = |dumpfile: &str| write_patched_image(&repo, dumpfile);
    let component = |digest, target: &str| ImageSetComponent {
        digest,
        format,
        target: target.into(),
    };
    let root = image(ROOT)?;
    let firmware = image(FIRMWARE)?;
    for (xattr, refused) in [
        ("user.foo=bar", false),
        // Escaped by the writer, or not used by the mount
        ("trusted.overlay.opaque=y", false),
        ("user.overlay.opaque=y", false),
        ("user.overlay.redirect=/usr/lib/firmware/inject", false),
        ("trusted.ovxrlay.redirect=/usr/lib/firmware/inject", true),
        ("trusted.ovxrlay.opaque=y", true),
    ] {
        let set = ImageSet::new(vec![
            component(root.clone(), "/"),
            component(image(&modules(xattr))?, "/usr/lib/modules"),
            component(firmware.clone(), "/usr/lib/firmware"),
        ])?;
        match repo.write_image_set(None, &set) {
            Err(err) => {
                let err = format!("{err:#}");
                ensure!(refused, "{xattr}: write_image_set refused: {err}");
                ensure!(
                    err.contains("/usr has the overlayfs xattr trusted.overlay."),
                    "{xattr}: {err}"
                );
            }
            Ok(id) => {
                check_image_set_mount(&repo, &id.to_hex(), |mnt| {
                    for (dir, expected) in [
                        ("usr/bin", &["true"][..]),
                        ("usr/lib/firmware", &["fw.bin", "inject"]),
                        ("usr/lib/modules", &["a.ko"]),
                    ] {
                        let names = dir_names(&mnt.join(dir))?;
                        ensure!(names == expected, "/{dir} has {names:?}, not {expected:?}");
                    }
                    Ok(())
                })
                .with_context(|| format!("{xattr} on /usr of the modules image"))?;
                ensure!(!refused, "{xattr}: write_image_set accepted it");
            }
        }
    }
    Ok(())
}
integration_test!(privileged_image_set_overlay_xattrs);

/// Overlayfs xattrs on the directories inside a component's target are
/// refused too.  An absolute `trusted.overlay.redirect` restarts the lookup
/// at the root of the layers below, so a `/usr/share` component whose
/// `/usr/share/inject` is redirected to `/usr/lib` would show the files of a
/// `/usr/lib/modules` component below it under `/usr/share/inject/modules`.
/// The writer's own `trusted.overlay.metacopy` and `trusted.overlay.redirect`
/// on a regular file, which name its object, are fine.
fn privileged_image_set_descendant_xattrs() -> Result<()> {
    use composefs_oci::composefs::image_set::{ImageSet, ImageSetComponent};

    const ROOT: &str = "\
/usr 0 40755 2 0 0 0 1000.0 - - -
/usr/lib 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules 0 40755 2 0 0 0 1000.0 - - -
/usr/share 0 40755 2 0 0 0 1000.0 - - -
";
    const MODULES: &str = "\
/usr 0 40755 2 0 0 0 1000.0 - - -
/usr/lib 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules/payload.ko 8 100644 1 0 0 0 1000.0 - payload! -
";
    /// An external file of the `/usr/share` component, larger than any inline one.
    const DATA: &[u8] = &[b'd'; 64 * 1024];

    if require_privileged("privileged_image_set_descendant_xattrs")?.is_some() {
        return Ok(());
    }
    let verity_dir = VerityTempDir::new()?;
    let repo = strict_sha512_repo(verity_dir.path())?;
    let format = repo.erofs_version();
    let data = repo.ensure_object(DATA)?;
    let share = |xattr: &str| {
        format!(
            "\
/usr 0 40755 2 0 0 0 1000.0 - - -
/usr/share 0 40755 2 0 0 0 1000.0 - - -
/usr/share/data {} 100644 1 0 0 0 1000.0 {} - {}
/usr/share/inject 0 40755 2 0 0 0 1000.0 - - - {xattr}
",
            DATA.len(),
            data.to_object_pathname(),
            data.to_hex()
        )
    };
    let component = |digest, target: &str| ImageSetComponent {
        digest,
        format,
        target: target.into(),
    };
    let root = write_patched_image(&repo, ROOT)?;
    let modules = write_patched_image(&repo, MODULES)?;
    for (xattr, refused) in [
        ("user.foo=bar", false),
        // Escaped by the writer
        ("trusted.overlay.redirect=/usr/lib", false),
        ("trusted.ovxrlay.redirect=/usr/lib", true),
        ("trusted.ovxrlay.opaque=y", true),
    ] {
        let set = ImageSet::new(vec![
            component(root.clone(), "/"),
            component(write_patched_image(&repo, &share(xattr))?, "/usr/share"),
            component(modules.clone(), "/usr/lib/modules"),
        ])?;
        match repo.write_image_set(None, &set) {
            Err(err) => {
                let err = format!("{err:#}");
                ensure!(refused, "{xattr}: write_image_set refused: {err}");
                ensure!(
                    err.contains("/usr/share/inject has the overlayfs xattr trusted.overlay."),
                    "{xattr}: {err}"
                );
            }
            Ok(id) => {
                check_image_set_mount(&repo, &id.to_hex(), |mnt| {
                    let inject = dir_names(&mnt.join("usr/share/inject"))?;
                    let payload =
                        std::fs::read_to_string(mnt.join("usr/share/inject/modules/payload.ko"));
                    ensure!(
                        inject.is_empty(),
                        "/usr/share/inject has {inject:?}, \
                         and /usr/share/inject/modules/payload.ko reads {payload:?}"
                    );
                    ensure!(std::fs::read(mnt.join("usr/share/data"))? == DATA);
                    let modules = dir_names(&mnt.join("usr/lib/modules"))?;
                    ensure!(
                        modules == ["payload.ko"],
                        "/usr/lib/modules has {modules:?}"
                    );
                    Ok(())
                })
                .with_context(|| format!("{xattr} on /usr/share/inject"))?;
                ensure!(!refused, "{xattr}: write_image_set accepted it");
            }
        }
    }
    Ok(())
}
integration_test!(privileged_image_set_descendant_xattrs);

/// A target whose first directory is named like a whiteout stub is refused.
/// Every V0/V1 image has whiteout stubs `00` to `ff` at its root for the names
/// it lacks, and they hide those names in the layers below, so with the
/// targets `/usr/lib/modules` then `/ab/x`, the modules image's `/ab` stub
/// would hide all of the `/ab/x` image.  A name that only looks similar works.
fn privileged_image_set_stub_named_target() -> Result<()> {
    use std::ffi::OsStr;

    use composefs_oci::composefs::{
        erofs::format::FormatVersion,
        image_set::{ImageSet, ImageSetComponent},
    };

    if require_privileged("privileged_image_set_stub_named_target")?.is_some() {
        return Ok(());
    }
    let verity_dir = VerityTempDir::new()?;
    let mut repo = strict_sha512_repo(verity_dir.path())?;
    repo.set_erofs_version(FormatVersion::V1);
    let repo = Arc::new(repo);
    let component = |digest, target: &str| ImageSetComponent {
        digest,
        format: FormatVersion::V1,
        target: target.into(),
    };

    for (top, refused) in [("abc", false), ("ab", true)] {
        let fixture_dir = tempfile::tempdir()?;
        let rootfs = create_test_rootfs(fixture_dir.path())?;
        std::fs::create_dir_all(rootfs.join("usr/lib/modules/6.1.0"))?;
        std::fs::write(rootfs.join("usr/lib/modules/6.1.0/modules.dep"), "\n")?;
        std::fs::create_dir_all(rootfs.join(top).join("x"))?;
        std::fs::write(rootfs.join(top).join("keep"), "keep\n")?;
        std::fs::write(rootfs.join(top).join("x/payload"), "payload\n")?;

        let target = format!("/{top}/x");
        let mut fs = read_rootfs(&repo, &rootfs)?;
        let modules = fs.split_off(OsStr::new("/usr/lib/modules"))?;
        let x = fs.split_off(OsStr::new(&target))?;
        let set = match ImageSet::new(vec![
            component(fs.commit_image(&repo, None)?, "/"),
            component(modules.commit_image(&repo, None)?, "/usr/lib/modules"),
            component(x.commit_image(&repo, None)?, &target),
        ]) {
            Err(err) => {
                ensure!(refused, "{target}: {err}");
                ensure!(
                    err.to_string().contains("named like the whiteout stubs"),
                    "{target}: {err}"
                );
                continue;
            }
            Ok(set) => set,
        };
        let id = repo.write_image_set(None, &set)?.to_hex();
        check_image_set_mount(&repo, &id, |mnt| {
            let read = |path: &str| std::fs::read_to_string(mnt.join(path));
            ensure!(read(&format!("{top}/keep"))? == "keep\n");
            ensure!(read("usr/lib/modules/6.1.0/modules.dep")? == "\n");
            let payload = read(&format!("{top}/x/payload"));
            ensure!(
                payload.as_deref().ok() == Some("payload\n"),
                "{target}/payload: {payload:?}"
            );
            Ok(())
        })
        .with_context(|| format!("image set with target {target}"))?;
        ensure!(!refused, "{target}: accepted");
    }
    Ok(())
}
integration_test!(privileged_image_set_stub_named_target);

/// Verify that `init` on a verity-capable filesystem enables verity on
/// meta.json, and that `--require-verity` succeeds on such a repo.
fn privileged_init_enables_verity() -> Result<()> {
    if require_privileged("privileged_init_enables_verity")?.is_some() {
        return Ok(());
    }

    let sh = Shell::new()?;
    let cfsctl = cfsctl()?;
    let verity_dir = VerityTempDir::new()?;
    let repo = verity_dir.path().join("repo");

    let output = cmd!(sh, "{cfsctl} --repo {repo} init").read()?;
    ensure!(
        output.contains("verity") && output.contains("required"),
        "init should report verity as required, got: {output}"
    );

    // --require-verity should succeed on this repo
    let output = cmd!(sh, "{cfsctl} --require-verity --repo {repo} gc").read()?;
    ensure!(
        output.contains("Objects: 0 removed"),
        "--require-verity gc should work on secure repo, got: {output}"
    );

    Ok(())
}
integration_test!(privileged_init_enables_verity);

/// Verify that `init --insecure` on a verity-capable filesystem does NOT
/// enable verity on meta.json, and `--require-verity` fails.
fn privileged_init_insecure_skips_verity() -> Result<()> {
    if require_privileged("privileged_init_insecure_skips_verity")?.is_some() {
        return Ok(());
    }

    let sh = Shell::new()?;
    let cfsctl = cfsctl()?;
    let verity_dir = VerityTempDir::new()?;
    let repo = verity_dir.path().join("repo");

    let output = cmd!(sh, "{cfsctl} --repo {repo} init --insecure").read()?;
    ensure!(
        output.contains("insecure"),
        "init --insecure should say insecure, got: {output}"
    );

    // --require-verity should fail even though the filesystem supports verity,
    // because init --insecure skipped enabling it on meta.json
    let result = cmd!(sh, "{cfsctl} --require-verity --repo {repo} gc").read();
    ensure!(
        result.is_err(),
        "--require-verity should fail on insecure-initialized repo"
    );

    // But operations without --require-verity should work fine
    let output = cmd!(sh, "{cfsctl} --repo {repo} gc").read()?;
    ensure!(
        output.contains("Objects: 0 removed"),
        "gc should work on insecure repo, got: {output}"
    );

    Ok(())
}
integration_test!(privileged_init_insecure_skips_verity);

/// Verify that repositories initialized while another thread forks still
/// get fs-verity on meta.json.  A child that lingers before exec holds a
/// copy of the writable meta.json tmpfile, so fs-verity has to be enabled
/// on a copy of it, and that copy is what must be linked.
///
/// Whether a child holds the tmpfile at the right moment is up to the
/// scheduler, so this keeps initializing repositories until composefs logs
/// that it made such a copy, and fails if it made none before the deadline.
/// Without the fix, it failed in 3 of 3 runs.
fn privileged_init_verity_while_forking() -> Result<()> {
    use composefs_oci::composefs::fsverity::{Algorithm, measure_verity};
    use std::cell::Cell;
    use std::os::unix::process::CommandExt;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Counts the log records of fs-verity copies made on the threads that
    /// set `COUNTING`, so that other tests running meanwhile don't count.
    struct CopyCounter;
    thread_local! {
        static COUNTING: Cell<bool> = const { Cell::new(false) };
    }
    static COPIES: AtomicUsize = AtomicUsize::new(0);
    impl log::Log for CopyCounter {
        fn enabled(&self, metadata: &log::Metadata) -> bool {
            metadata.level() <= log::Level::Debug && COUNTING.get()
        }
        fn log(&self, record: &log::Record) {
            if self.enabled(record.metadata())
                && record.args().to_string().contains("fs-verity on a copy")
            {
                COPIES.fetch_add(1, Ordering::Relaxed);
            }
        }
        fn flush(&self) {}
    }

    /// Children to fork, enough for many to land while meta.json is written.
    const FORKS: usize = 100;
    /// How long each child keeps its copies of the parent's file descriptors.
    const LINGER: Duration = Duration::from_millis(20);
    /// When to stop anyway, should forking be slow.
    const MAX_RUN: Duration = Duration::from_secs(60);

    if require_privileged("privileged_init_verity_while_forking")?.is_some() {
        return Ok(());
    }

    log::set_logger(&CopyCounter).map_err(|_| anyhow::anyhow!("another logger is set"))?;
    log::set_max_level(log::LevelFilter::Debug);
    let verity_dir = VerityTempDir::new()?;
    let path = verity_dir.path().join("forking");
    let stop = AtomicBool::new(false);
    let forks = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let forker = scope.spawn(|| -> Result<()> {
            while !stop.load(Ordering::Relaxed) {
                let mut child = std::process::Command::new("true");
                // SAFETY: the closure runs in the child between fork and
                // exec, where it only sleeps, which allocates nothing and
                // takes no locks.
                unsafe {
                    child.pre_exec(|| {
                        std::thread::sleep(LINGER);
                        Ok(())
                    });
                }
                child.status().context("running a lingering child")?;
                forks.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        });
        let init = || -> Result<(usize, usize)> {
            let (mut iterations, mut failures) = (0, 0);
            let start = Instant::now();
            COUNTING.set(true);
            while (forks.load(Ordering::Relaxed) < FORKS || COPIES.load(Ordering::Relaxed) == 0)
                && start.elapsed() < MAX_RUN
                && !forker.is_finished()
            {
                iterations += 1;
                let (repo, _) = Repository::<Sha256HashValue>::init_path(
                    rustix::fs::CWD,
                    &path,
                    RepositoryConfig::new(Algorithm::SHA256),
                )?;
                let meta = std::fs::File::open(path.join("meta.json"))?;
                if repo.is_insecure() || measure_verity::<Sha256HashValue>(&meta).is_err() {
                    failures += 1;
                }
                drop(repo);
                std::fs::remove_dir_all(&path)?;
            }
            Ok((iterations, failures))
        };
        let result = init();
        stop.store(true, Ordering::Relaxed);
        forker.join().expect("forking thread panicked")?;
        let (iterations, failures) = result?;
        let forks = forks.load(Ordering::Relaxed);
        let copies = COPIES.load(Ordering::Relaxed);
        println!("{iterations} repositories initialized during {forks} forks, {copies} copies");
        ensure!(
            failures == 0,
            "{failures} of {iterations} repositories initialized during {forks} forks \
             have no fs-verity on meta.json"
        );
        ensure!(
            copies > 0,
            "no child held meta.json open for writing in {iterations} repositories \
             initialized during {forks} forks, so the copy was never tested"
        );
        Ok(())
    })
}
integration_test!(privileged_init_verity_while_forking);

/// Verify that `oci pull` into a read-only bind-mounted repository fails
/// immediately with a clear "not writable" error instead of a confusing
/// tar header error.
fn privileged_pull_readonly_repo() -> Result<()> {
    if require_privileged("privileged_pull_readonly_repo")?.is_some() {
        return Ok(());
    }

    let sh = Shell::new()?;
    let cfsctl = cfsctl()?;
    let verity_dir = VerityTempDir::new()?;
    let repo = verity_dir.path().join("repo");

    cmd!(sh, "{cfsctl} --repo {repo} init").run()?;

    // Bind-mount the repo read-only over itself
    cmd!(sh, "mount --bind {repo} {repo}").run()?;
    cmd!(sh, "mount -o remount,ro,bind {repo}").run()?;

    // Use a bogus oci: reference — the writable check fires before any
    // image processing so the source doesn't matter.
    let output = cmd!(
        sh,
        "{cfsctl} --repo {repo} oci pull oci:/nonexistent ignored"
    )
    .ignore_status()
    .output()?;

    // Clean up the bind mount before asserting
    cmd!(sh, "umount {repo}").run()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}{stderr}");

    ensure!(
        !output.status.success(),
        "pull into read-only repo should fail"
    );
    ensure!(
        combined.contains("not writable") || combined.contains("Read-only file system"),
        "expected writable or EROFS error, got: {combined}"
    );
    ensure!(
        !combined.contains("header error") && !combined.contains("invalid octal"),
        "should NOT produce misleading tar header errors, got: {combined}"
    );

    Ok(())
}
integration_test!(privileged_pull_readonly_repo);

// ============================================================================
// Filesystem-specific reflink / hardlink tests
// ============================================================================

/// A temporary directory backed by a loop-mounted filesystem.
///
/// Supports ext4 (with verity) and XFS (with reflinks).  The backing sparse
/// file is 512 MB — large enough for a synthetic OCI image in
/// containers-storage plus a composefs repo.
pub(crate) struct LoopTempDir {
    mountpoint: PathBuf,
    _backing: tempfile::TempDir,
}

impl LoopTempDir {
    /// Create a loop-mounted ext4 filesystem with verity support.
    pub(crate) fn ext4_verity() -> Result<Self> {
        Self::create("mkfs.ext4", &["-q", "-O", "verity", "-b", "4096"])
    }

    /// Create a loop-mounted XFS filesystem with reflink support.
    pub(crate) fn xfs_reflink() -> Result<Self> {
        Self::create("mkfs.xfs", &["-q", "-m", "reflink=1"])
    }

    fn create(mkfs: &str, args: &[&str]) -> Result<Self> {
        let backing = tempfile::tempdir()?;
        let img = backing.path().join("fs.img");
        let mountpoint = backing.path().join("mnt");
        std::fs::create_dir(&mountpoint)?;

        let sh = Shell::new()?;
        cmd!(sh, "truncate -s 512M {img}").run()?;
        cmd!(sh, "{mkfs} {args...} {img}").run()?;
        cmd!(sh, "mount -o loop {img} {mountpoint}").run()?;

        Ok(Self {
            mountpoint,
            _backing: backing,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.mountpoint
    }
}

impl Drop for LoopTempDir {
    fn drop(&mut self) {
        let _ = std::process::Command::new("umount")
            .arg(&self.mountpoint)
            .status();
    }
}

/// Create a minimal OCI directory image with files large enough to exercise
/// the `ensure_object_from_file` path (> 64 bytes, the inline threshold),
/// including one non-world-readable file (see bootc-dev/bootc#2324).
///
/// Returns the path to the OCI directory.
pub(super) fn create_oci_layout_with_large_files(parent: &Path) -> Result<PathBuf> {
    use cap_std_ext::cap_std;
    use ocidir::oci_spec::image::{
        ConfigBuilder, ImageConfigurationBuilder, Platform, PlatformBuilder, RootFsBuilder,
    };

    let oci_dir = parent.join("oci-image");
    std::fs::create_dir_all(&oci_dir)?;

    let dir = cap_std::fs::Dir::open_ambient_dir(&oci_dir, cap_std::ambient_authority())?;
    let ocidir = ocidir::OciDir::ensure(dir)?;

    let mut manifest = ocidir.new_empty_manifest()?.build()?;

    let runtime_config = ConfigBuilder::default().build()?;
    let rootfs = RootFsBuilder::default()
        .typ("layers")
        .diff_ids(Vec::<String>::new())
        .build()?;
    let mut config = ImageConfigurationBuilder::default()
        .architecture("amd64")
        .os("linux")
        .rootfs(rootfs)
        .config(runtime_config)
        .build()?;

    // Create a layer with several files > INLINE_CONTENT_MAX_V0 (64 bytes)
    // so they go through the ensure_object_from_file path during cstor import.
    // The image must have /usr (required by transform_for_oci).
    let mut layer_builder = ocidir.create_layer(None)?;

    // Add /usr directory (required by composefs OCI transformations)
    let mut usr_hdr = tar::Header::new_gnu();
    usr_hdr.set_entry_type(tar::EntryType::Directory);
    usr_hdr.set_size(0);
    usr_hdr.set_mode(0o755);
    usr_hdr.set_uid(0);
    usr_hdr.set_gid(0);
    usr_hdr.set_mtime(1234567890);
    usr_hdr.set_cksum();
    layer_builder.append_data(&mut usr_hdr, "usr/", &[] as &[u8])?;

    for i in 0..5u8 {
        let data = vec![i.wrapping_mul(0x37); 4096];
        let name = format!("usr/file_{i}.bin");
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(1234567890);
        header.set_cksum();
        layer_builder.append_data(&mut header, &name, &data[..])?;
    }

    // Also add a non-world-readable file (mode 0o600).  This regression-tests
    // bootc-dev/bootc#2324: without `consumer_has_cap_dac_override` wired up,
    // cstor import would inline this file's content into the varlink stream,
    // materialize it into a memfd on the consumer side, and then crash with
    // EXDEV when zerocopy mode tried to reflink/hardlink the memfd (memfds
    // can never be zero-copied to a real filesystem).  With the fix, the
    // privileged (direct, in-process) import path recognizes that it can
    // bypass DAC checks and emits a real dirfd+filename reference instead,
    // so this file is hardlinked/reflinked just like the world-readable ones.
    {
        let data = vec![0xffu8; 4096];
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o600);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(1234567890);
        header.set_cksum();
        layer_builder.append_data(&mut header, "usr/restricted_file.bin", &data[..])?;
    }
    let layer = layer_builder.into_inner()?.complete()?;

    ocidir.push_layer(&mut manifest, &mut config, layer, "test layer", None);

    let platform: Platform = PlatformBuilder::default()
        .architecture("amd64")
        .os("linux")
        .build()?;
    ocidir.insert_manifest_and_config(manifest, config, None, platform)?;

    Ok(oci_dir)
}

/// Copy an OCI directory image into containers-storage on a specific filesystem.
///
/// Uses `skopeo copy` with the `[overlay@root+runroot]` syntax to target
/// a containers-storage instance at the given mount point.
///
/// Returns the storage root path.
pub(super) fn copy_oci_to_cstor(sh: &Shell, oci_dir: &Path, mount: &Path) -> Result<PathBuf> {
    let storage_root = mount.join("storage");
    let run_root = mount.join("run");
    std::fs::create_dir_all(&storage_root)?;
    std::fs::create_dir_all(&run_root)?;

    let oci_ref = format!("oci:{}", oci_dir.display());
    let cstor_ref = format!(
        "containers-storage:[overlay@{}+{}]test:latest",
        storage_root.display(),
        run_root.display()
    );

    // Run in a private mount namespace so that any bind mounts the overlay
    // driver creates (e.g. on storage/overlay) don't leak into our namespace.
    // This ensures the diff files we later open are on the raw filesystem,
    // not behind a bind mount that would cause EXDEV on hardlinks.
    cmd!(sh, "unshare -m skopeo copy {oci_ref} {cstor_ref}").run()?;

    Ok(storage_root)
}

/// Open a repository at `path`, initializing it first.  Uses insecure mode
/// so the tests work on filesystems without verity (XFS).
fn init_insecure_repo_at<ObjectID: FsVerityHashValue>(
    path: &Path,
    algorithm: composefs_oci::composefs::fsverity::Algorithm,
) -> Result<Arc<Repository<ObjectID>>> {
    std::fs::create_dir_all(path)?;
    let fd = rustix::fs::open(
        path,
        rustix::fs::OFlags::CLOEXEC | rustix::fs::OFlags::RDONLY,
        0.into(),
    )?;
    let (repo, _created) = Repository::<ObjectID>::init_path(
        &fd,
        ".",
        RepositoryConfig::new(algorithm).set_insecure(),
    )?;
    Ok(Arc::new(repo))
}

/// Pull a containers-storage image into a composefs repo with an explicit
/// storage root and local-fetch mode, and return the import stats.
fn cstor_pull<ObjectID: FsVerityHashValue>(
    repo: &Arc<Repository<ObjectID>>,
    storage_root: &Path,
    local_fetch: composefs_oci::LocalFetchOpt,
) -> Result<composefs_oci::ImportStats> {
    let opts = composefs_oci::PullOptions {
        local_fetch,
        storage_root: Some(storage_root),
        ..Default::default()
    };

    let rt = tokio::runtime::Runtime::new()?;
    let pull_result = rt
        .block_on(async {
            composefs_oci::pull(repo, "containers-storage:test:latest", None, opts).await
        })
        .context("containers-storage pull failed")?;
    Ok(pull_result.stats)
}

/// Dispatch a cstor pull with the given algorithm and local-fetch mode.
///
/// This creates a fresh repo at `repo_path` with the appropriate hash type
/// and pulls the `test:latest` image from containers-storage.
fn cstor_pull_with_algorithm(
    storage_root: &Path,
    repo_path: &Path,
    algorithm: composefs_oci::composefs::fsverity::Algorithm,
    local_fetch: composefs_oci::LocalFetchOpt,
) -> Result<composefs_oci::ImportStats> {
    match algorithm {
        composefs_oci::composefs::fsverity::Algorithm::Sha256 { .. } => {
            let repo = init_insecure_repo_at::<Sha256HashValue>(repo_path, algorithm)?;
            cstor_pull(&repo, storage_root, local_fetch)
        }
        composefs_oci::composefs::fsverity::Algorithm::Sha512 { .. } => {
            let repo = init_insecure_repo_at::<Sha512HashValue>(repo_path, algorithm)?;
            cstor_pull(&repo, storage_root, local_fetch)
        }
    }
}

/// On ext4 (no reflink support), the import should skip FICLONE after the
/// first probe fails and use hardlinks instead (zero-copy).  The `skopeo
/// copy` step runs in `unshare -m` to prevent the overlay driver's bind
/// mount from interfering with hardlinks.
///
/// Covers all combinations of `{sha256, sha512} × {auto, zerocopy}`.
fn privileged_cstor_import_ext4_hardlink() -> Result<()> {
    use composefs_oci::LocalFetchOpt;
    use composefs_oci::composefs::fsverity::Algorithm;

    if require_privileged("privileged_cstor_import_ext4_hardlink")?.is_some() {
        return Ok(());
    }

    let algorithms = [(Algorithm::SHA256, "sha256"), (Algorithm::SHA512, "sha512")];
    let modes = [
        (LocalFetchOpt::IfPossible, "auto"),
        (LocalFetchOpt::ZeroCopy, "zerocopy"),
    ];

    let sh = Shell::new()?;

    for (algorithm, alg_name) in &algorithms {
        // One LoopTempDir per algorithm: enabling verity is irreversible and
        // algorithm-specific, so we cannot reuse across algorithms.
        let fs = LoopTempDir::ext4_verity()?;

        let oci_dir = create_oci_layout_with_large_files(fs.path())?;
        let storage_root = copy_oci_to_cstor(&sh, &oci_dir, fs.path())?;

        for (mode, mode_name) in &modes {
            let repo_dir = fs.path().join(format!("repo-{alg_name}-{mode_name}"));
            let stats = cstor_pull_with_algorithm(&storage_root, &repo_dir, *algorithm, *mode)?;

            println!("ext4 [{alg_name}/{mode_name}] import stats: {stats:?}");
            ensure!(
                stats.objects_reflinked == 0,
                "ext4 [{alg_name}/{mode_name}] should not reflink any objects, got {} reflinked",
                stats.objects_reflinked,
            );
            ensure!(
                stats.objects_hardlinked > 0,
                "ext4 [{alg_name}/{mode_name}] should hardlink objects, got 0 hardlinked (copied={})",
                stats.objects_copied,
            );
            ensure!(
                stats.objects_copied == 0,
                "ext4 [{alg_name}/{mode_name}] should not need copies, got {} copied",
                stats.objects_copied,
            );
            println!(
                "ext4 [{alg_name}/{mode_name}]: {} hardlinked, {} already present",
                stats.objects_hardlinked, stats.objects_already_present,
            );
        }
    }

    Ok(())
}
integration_test!(privileged_cstor_import_ext4_hardlink);

/// On XFS with reflink support, importing from containers-storage on the same
/// filesystem should use reflinks (zero-copy), and the import stats should
/// show `objects_reflinked > 0`.
///
/// Covers all combinations of `{sha256, sha512} × {auto, zerocopy}`.
fn privileged_cstor_import_xfs_reflink() -> Result<()> {
    use composefs_oci::LocalFetchOpt;
    use composefs_oci::composefs::fsverity::Algorithm;

    if require_privileged("privileged_cstor_import_xfs_reflink")?.is_some() {
        return Ok(());
    }

    // Skip if mkfs.xfs is not available (e.g. Debian bootc images).
    if !Path::new("/usr/sbin/mkfs.xfs").exists() && !Path::new("/sbin/mkfs.xfs").exists() {
        println!("SKIP: mkfs.xfs not available");
        return Ok(());
    }

    let algorithms = [(Algorithm::SHA256, "sha256"), (Algorithm::SHA512, "sha512")];
    let modes = [
        (LocalFetchOpt::IfPossible, "auto"),
        (LocalFetchOpt::ZeroCopy, "zerocopy"),
    ];

    let sh = Shell::new()?;

    for (algorithm, alg_name) in &algorithms {
        let fs = LoopTempDir::xfs_reflink()?;

        let oci_dir = create_oci_layout_with_large_files(fs.path())?;
        let storage_root = copy_oci_to_cstor(&sh, &oci_dir, fs.path())?;

        for (mode, mode_name) in &modes {
            let repo_dir = fs.path().join(format!("repo-{alg_name}-{mode_name}"));
            let stats = cstor_pull_with_algorithm(&storage_root, &repo_dir, *algorithm, *mode)?;

            println!("XFS [{alg_name}/{mode_name}] import stats: {stats:?}");
            ensure!(
                stats.objects_reflinked > 0,
                "XFS [{alg_name}/{mode_name}] should reflink objects, got 0 reflinked (hardlinked={}, copied={})",
                stats.objects_hardlinked,
                stats.objects_copied,
            );
            ensure!(
                stats.objects_copied == 0,
                "XFS [{alg_name}/{mode_name}] should not need copies, got {} copied",
                stats.objects_copied,
            );
            println!(
                "XFS [{alg_name}/{mode_name}]: {} reflinked, {} already present",
                stats.objects_reflinked, stats.objects_already_present,
            );
        }
    }

    Ok(())
}
integration_test!(privileged_cstor_import_xfs_reflink);

// ============================================================================
// FUSE integration test
// ============================================================================

struct MountGuard {
    mountpoint: PathBuf,
    child: Option<std::process::Child>,
}

impl Drop for MountGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = rustix::mount::unmount(&self.mountpoint, rustix::mount::UnmountFlags::DETACH);
    }
}

fn bigfile_content() -> Vec<u8> {
    vec![b'A'; 600]
}

fn biglib_content() -> Vec<u8> {
    (0u8..=255).cycle().take(800).collect()
}

fn build_test_filesystem(
    repo: &Repository<Sha256HashValue>,
) -> Result<composefs_oci::composefs::tree::FileSystem<Sha256HashValue>> {
    use std::collections::BTreeMap;
    use std::ffi::OsStr;

    use composefs_oci::composefs::generic_tree::{LeafId, Stat};
    use composefs_oci::composefs::tree::{
        Directory, FileSystem, Inode, Leaf, LeafContent, RegularFile,
    };

    fn mkstat(mode: u32, uid: u32, gid: u32, mtime: i64) -> Stat {
        Stat {
            st_mode: mode,
            st_uid: uid,
            st_gid: gid,
            st_mtim_sec: mtime,
            st_mtim_nsec: 0,
            xattrs: BTreeMap::new(),
        }
    }

    let root_stat = mkstat(0o755, 0, 0, 1_700_000_000);

    let mut fs = FileSystem::<Sha256HashValue>::new(root_stat);

    let hello_id = LeafId(fs.leaves.len());
    {
        let mut xattrs = BTreeMap::new();
        xattrs.insert(
            OsStr::new("user.test").into(),
            Box::from(b"hello-value".as_ref()),
        );
        fs.leaves.push(Leaf {
            stat: Stat {
                st_mode: 0o755,
                st_uid: 0,
                st_gid: 0,
                st_mtim_sec: 1_700_000_001,
                st_mtim_nsec: 0,
                xattrs,
            },
            content: LeafContent::Regular(RegularFile::Inline(
                b"hello world binary stub".as_ref().into(),
            )),
        });
    }

    let readme_id = LeafId(fs.leaves.len());
    fs.leaves.push(Leaf {
        stat: mkstat(0o644, 0, 0, 1_700_000_002),
        content: LeafContent::Regular(RegularFile::Inline(
            b"readme text content\n".as_ref().into(),
        )),
    });

    let hostname_id = LeafId(fs.leaves.len());
    fs.leaves.push(Leaf {
        stat: mkstat(0o644, 0, 0, 1_700_000_003),
        content: LeafContent::Regular(RegularFile::Inline(b"integration-test\n".as_ref().into())),
    });

    let os_release_id = LeafId(fs.leaves.len());
    fs.leaves.push(Leaf {
        stat: mkstat(0o644, 0, 0, 1_700_000_004),
        content: LeafContent::Regular(RegularFile::Inline(b"ID=test\nNAME=Test\n".as_ref().into())),
    });

    let symlink_id = LeafId(fs.leaves.len());
    fs.leaves.push(Leaf {
        stat: mkstat(0o777, 0, 0, 1_700_000_005),
        content: LeafContent::Symlink(OsStr::new("../usr/lib/os-release").into()),
    });

    let devnull_id = LeafId(fs.leaves.len());
    fs.leaves.push(Leaf {
        stat: mkstat(0o666, 0, 0, 0),
        content: LeafContent::CharacterDevice(rustix::fs::makedev(1, 3)),
    });

    let fifo_id = LeafId(fs.leaves.len());
    fs.leaves.push(Leaf {
        stat: mkstat(0o644, 0, 0, 1_700_000_006),
        content: LeafContent::Fifo,
    });

    let bigfile_data = bigfile_content();
    let bigfile_hash = repo.ensure_object(&bigfile_data)?;
    let bigfile_id = LeafId(fs.leaves.len());
    fs.leaves.push(Leaf {
        stat: mkstat(0o755, 0, 0, 1_700_000_007),
        content: LeafContent::Regular(RegularFile::External(
            bigfile_hash,
            bigfile_data.len() as u64,
        )),
    });

    let biglib_data = biglib_content();
    let biglib_hash = repo.ensure_object(&biglib_data)?;
    let biglib_id = LeafId(fs.leaves.len());
    fs.leaves.push(Leaf {
        stat: mkstat(0o755, 0, 0, 1_700_000_008),
        content: LeafContent::Regular(RegularFile::External(biglib_hash, biglib_data.len() as u64)),
    });

    let mut usr_bin = Directory::<Sha256HashValue>::new(mkstat(0o755, 0, 0, 1_700_000_010));
    usr_bin.insert(OsStr::new("hello"), Inode::leaf(hello_id));
    usr_bin.insert(OsStr::new("hello2"), Inode::leaf(hello_id));
    usr_bin.insert(OsStr::new("bigfile"), Inode::leaf(bigfile_id));

    let mut usr_lib = Directory::<Sha256HashValue>::new(mkstat(0o755, 0, 0, 1_700_000_011));
    usr_lib.insert(OsStr::new("readme.txt"), Inode::leaf(readme_id));
    usr_lib.insert(OsStr::new("os-release"), Inode::leaf(os_release_id));
    usr_lib.insert(OsStr::new("biglib.so"), Inode::leaf(biglib_id));

    let mut usr = Directory::<Sha256HashValue>::new(mkstat(0o755, 0, 0, 1_700_000_012));
    usr.insert(OsStr::new("bin"), Inode::Directory(Box::new(usr_bin)));
    usr.insert(OsStr::new("lib"), Inode::Directory(Box::new(usr_lib)));

    let mut etc = Directory::<Sha256HashValue>::new(mkstat(0o755, 0, 0, 1_700_000_013));
    etc.insert(OsStr::new("hostname"), Inode::leaf(hostname_id));
    etc.insert(OsStr::new("os-release"), Inode::leaf(symlink_id));

    let mut dev = Directory::<Sha256HashValue>::new(mkstat(0o755, 0, 0, 1_700_000_014));
    dev.insert(OsStr::new("null"), Inode::leaf(devnull_id));

    let mut tmp_dir = Directory::<Sha256HashValue>::new(mkstat(0o1777, 0, 0, 1_700_000_015));
    tmp_dir.insert(OsStr::new("fifo"), Inode::leaf(fifo_id));

    fs.root
        .insert(OsStr::new("usr"), Inode::Directory(Box::new(usr)));
    fs.root
        .insert(OsStr::new("etc"), Inode::Directory(Box::new(etc)));
    fs.root
        .insert(OsStr::new("dev"), Inode::Directory(Box::new(dev)));
    fs.root
        .insert(OsStr::new("tmp"), Inode::Directory(Box::new(tmp_dir)));

    Ok(fs)
}

fn privileged_fuse_dumpfile_roundtrip() -> Result<()> {
    use std::os::unix::fs::MetadataExt as _;
    use std::time::{Duration, Instant};

    use composefs_oci::composefs::{
        dumpfile::write_dumpfile,
        erofs::{
            reader::erofs_to_filesystem,
            writer::{ValidatedFileSystem, mkfs_erofs},
        },
        repository::{Repository, RepositoryConfig},
    };

    if require_privileged("privileged_fuse_dumpfile_roundtrip")?.is_some() {
        return Ok(());
    }

    let work_dir = tempfile::tempdir()?;
    let mountpoint = work_dir.path().join("mnt");
    let repo_path = work_dir.path().join("repo");
    std::fs::create_dir(&mountpoint)?;
    std::fs::create_dir(&repo_path)?;

    let repo_fd = rustix::fs::open(
        &repo_path,
        rustix::fs::OFlags::CLOEXEC | rustix::fs::OFlags::RDONLY,
        rustix::fs::Mode::empty(),
    )?;
    let (mut repo, _created) = Repository::<Sha256HashValue>::init_path(
        &repo_fd,
        ".",
        RepositoryConfig::default().set_insecure(),
    )?;
    repo.set_insecure();

    let synthetic = build_test_filesystem(&repo)?;
    let erofs_bytes = mkfs_erofs(&ValidatedFileSystem::new(synthetic)?);
    let canonical_fs = erofs_to_filesystem::<Sha256HashValue>(&erofs_bytes)?;

    let image_id = repo.write_image(None, &erofs_bytes)?;
    let image_name = image_id.to_hex();

    let mut expected_buf = Vec::new();
    write_dumpfile(&mut expected_buf, &canonical_fs)?;
    let expected_dump = String::from_utf8(expected_buf)?;

    let pre_mount_dev = std::fs::metadata(&mountpoint)?.dev();

    let cfsctl_bin = cfsctl()?;
    let child = std::process::Command::new(&cfsctl_bin)
        .arg("--repo")
        .arg(&repo_path)
        .arg("mount")
        .arg("--fuse=yes")
        .arg("--foreground")
        .arg(&image_name)
        .arg(&mountpoint)
        .spawn()
        .context("spawning cfsctl mount --fuse")?;

    let mut guard = MountGuard {
        mountpoint: mountpoint.clone(),
        child: Some(child),
    };

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(child) = guard.child.as_mut()
            && let Some(status) = child.try_wait()?
        {
            bail!("cfsctl mount --fuse exited before mount was ready: {status}");
        }
        if std::fs::metadata(&mountpoint)
            .map(|m| m.dev())
            .unwrap_or(pre_mount_dev)
            != pre_mount_dev
        {
            break;
        }
        if Instant::now() >= deadline {
            bail!("timed out waiting for FUSE mount");
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let bigfile_actual = std::fs::read(mountpoint.join("usr/bin/bigfile"))
        .context("reading bigfile from FUSE mount")?;
    ensure!(
        bigfile_actual == bigfile_content(),
        "bigfile content mismatch: got {} bytes, expected {}",
        bigfile_actual.len(),
        bigfile_content().len(),
    );
    let biglib_actual = std::fs::read(mountpoint.join("usr/lib/biglib.so"))
        .context("reading biglib.so from FUSE mount")?;
    ensure!(
        biglib_actual == biglib_content(),
        "biglib.so content mismatch: got {} bytes, expected {}",
        biglib_actual.len(),
        biglib_content().len(),
    );

    let sh = Shell::new()?;
    let mp = mountpoint.to_str().context("non-UTF-8 mountpoint")?;
    let repo_arg = repo_path.to_str().context("non-UTF-8 repo path")?;
    let actual_dump = cmd!(
        sh,
        "{cfsctl_bin} --repo {repo_arg} create-dumpfile --no-propagate-usr-to-root {mp}"
    )
    .read()?;

    drop(guard);

    similar_asserts::assert_eq!(
        expected_dump.trim_end_matches('\n'),
        actual_dump.trim_end_matches('\n')
    );

    Ok(())
}
integration_test!(privileged_fuse_dumpfile_roundtrip);
