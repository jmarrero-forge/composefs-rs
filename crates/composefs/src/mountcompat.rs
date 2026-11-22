//! Compatibility helpers for older Linux kernel mount APIs.
//!
//! This module provides fallback implementations for mount operations
//! on kernels that don't support the modern mount API, including
//! loopback device setup and temporary mount handling.

use std::{
    io::Result,
    os::fd::{AsFd, BorrowedFd, OwnedFd},
};

// This file contains a bunch of helpers that deal with the pre-6.15 mount API

// First: the simple pass-through versions of all of our helpers, for 6.15 or later, along with
// documentation about why they're required.

/// Sets one of the "dir" mount options on an overlayfs to the given file descriptor.  This can
/// either be a freshly-created mount or a O_PATH file descriptor.  On 6.15 kernels this can be
/// done by directly calling `fsconfig_set_fd()`.  On pre-6.15 kernels, it needs to be done by
/// reopening the file descriptor `O_RDONLY` and calling `fsconfig_set_fd()` because `O_PATH` fds
/// are rejected.  On very old kernels this needs to be done by way of `fsconfig_set_string()` and
/// `/proc/self/fd/`.
#[cfg(not(feature = "pre-6.15"))]
pub fn overlayfs_set_fd(fs_fd: BorrowedFd, key: &str, fd: BorrowedFd) -> rustix::io::Result<()> {
    rustix::mount::fsconfig_set_fd(fs_fd, key, fd)
}

/// Sets the "lowerdir+" and "datadir+" mount options of an overlayfs mount to the provided file
/// descriptors.  On 6.15 kernels this can be done by directly calling `fsconfig_set_fd()`.  On
/// pre-6.15 kernels, it needs to be done by reopening the file descriptor `O_RDONLY` and calling
/// `fsconfig_set_fd()` because `O_PATH` fds are rejected.  On very old kernels this needs to be
/// done by calculating a `"lowerdir=lower::data"` string using `/proc/self/fd/` filenames and
/// setting it via `fsconfig_set_string()`.
pub fn overlayfs_set_lower_and_data_fds(
    fs_fd: impl AsFd,
    lower: impl AsFd,
    data_fds: &[BorrowedFd<'_>],
) -> rustix::io::Result<()> {
    overlayfs_set_lowers_and_data_fds(fs_fd, &[lower.as_fd()], data_fds)
}

/// The longest `lowerdir` option value that `fsconfig_set_string()` accepts: the kernel copies
/// string parameters with `strndup_user(value, 256)`, which leaves 255 bytes and the NUL.
///
/// This only limits kernels before 6.15 (the `pre-6.15` feature), where all layers go into one
/// `lowerdir` string; see [`overlayfs_check_lowers()`].
pub const LOWERDIR_OPTION_MAX: usize = 255;

/// Checks that `lowers` and `data_fds` can be passed to
/// [`overlayfs_set_lowers_and_data_fds()`], with an error that names the limit if not.  On 6.15
/// kernels each layer is its own option, so this always succeeds.
#[cfg(not(feature = "pre-6.15"))]
pub fn overlayfs_check_lowers(
    _lowers: &[BorrowedFd<'_>],
    _data_fds: &[BorrowedFd<'_>],
) -> Result<()> {
    Ok(())
}

/// Like [`overlayfs_set_lower_and_data_fds()`], with several lower layers, the first one
/// topmost.
#[cfg(not(feature = "pre-6.15"))]
pub fn overlayfs_set_lowers_and_data_fds(
    fs_fd: impl AsFd,
    lowers: &[BorrowedFd<'_>],
    data_fds: &[BorrowedFd<'_>],
) -> rustix::io::Result<()> {
    for lower in lowers {
        overlayfs_set_fd(fs_fd.as_fd(), "lowerdir+", *lower)?;
    }
    for data_fd in data_fds {
        overlayfs_set_fd(fs_fd.as_fd(), "datadir+", *data_fd)?;
    }
    Ok(())
}

/// Prepares an open erofs image file for mounting.  On kernels versions after 6.12 this is a
/// simple passthrough.  On older kernels (like on RHEL 9) we need to create a loopback device.
#[cfg(not(feature = "rhel9"))]
pub fn make_erofs_mountable(image: OwnedFd) -> Result<OwnedFd> {
    Ok(image)
}

/// Prepares a mounted filesystem for further use.  On 6.15 kernels this is a no-op, due to the
/// expanded number of operations which can be performed on "detached" mounts.  On earlier kernels
/// we need to create a temporary directory and mount the filesystem there to avoid failures,
/// making sure to detach the mount and remove the directory later.  This function returns an `impl
/// AsFd` which also implements the `Drop` trait in order to facilitate this cleanup.
#[cfg(not(feature = "pre-6.15"))]
pub fn prepare_mount(mnt_fd: OwnedFd) -> Result<impl AsFd> {
    Ok(mnt_fd)
}

// Now: support for pre-6.15 kernels

/// Sets one of the "dir" mount options on an overlayfs to the given file descriptor.
///
/// Uses `fsconfig_set_string()` with a `/proc/self/fd/` path.  The previous
/// implementation tried `fsconfig_set_fd()` (with O_PATH fds reopened as O_RDONLY),
/// but `fsconfig_set_fd()` for overlayfs layer options was only added in 6.13 so
/// that broke on 6.12 kernels (e.g. CentOS Stream 10).  The string-based approach
/// works on all kernels with the new mount API.
#[cfg(feature = "pre-6.15")]
pub fn overlayfs_set_fd(fs_fd: BorrowedFd, key: &str, fd: BorrowedFd) -> rustix::io::Result<()> {
    rustix::mount::fsconfig_set_string(fs_fd, key, crate::util::proc_self_fd(fd))
}

/// Sets the "lowerdir+" and "datadir+" mount options of an overlayfs mount to the provided file
/// descriptors, the first lower layer topmost.
///
/// Constructs a `lowerdir` string using `/proc/self/fd/` paths, with `:` between the lower
/// layers and `::` as the separator for the data layers.  The kernel takes at most
/// [`LOWERDIR_OPTION_MAX`] bytes for it, which leaves room for about a dozen layers: a longer
/// string fails with `ENAMETOOLONG` here rather than a bare `EINVAL` from the kernel, and
/// [`overlayfs_check_lowers()`] says why.
#[cfg(feature = "pre-6.15")]
pub fn overlayfs_set_lowers_and_data_fds(
    fs_fd: impl AsFd,
    lowers: &[BorrowedFd<'_>],
    data_fds: &[BorrowedFd<'_>],
) -> rustix::io::Result<()> {
    let arg = lowerdir_arg(&raw_fds(lowers), &raw_fds(data_fds));
    if arg.len() > LOWERDIR_OPTION_MAX {
        return Err(rustix::io::Errno::NAMETOOLONG);
    }
    rustix::mount::fsconfig_set_string(fs_fd.as_fd(), "lowerdir", arg)
}

/// Checks that `lowers` and `data_fds` fit in the single `lowerdir` option that
/// [`overlayfs_set_lowers_and_data_fds()`] passes to kernels before 6.15, with an error that
/// names the limit if not.
#[cfg(feature = "pre-6.15")]
pub fn overlayfs_check_lowers(
    lowers: &[BorrowedFd<'_>],
    data_fds: &[BorrowedFd<'_>],
) -> Result<()> {
    check_lowerdir_arg(
        &lowerdir_arg(&raw_fds(lowers), &raw_fds(data_fds)),
        lowers.len(),
    )
}

#[cfg(feature = "pre-6.15")]
fn raw_fds(fds: &[BorrowedFd<'_>]) -> Vec<std::os::fd::RawFd> {
    use std::os::fd::AsRawFd;

    fds.iter().map(|fd| fd.as_raw_fd()).collect()
}

#[cfg(any(test, feature = "pre-6.15"))]
fn lowerdir_arg(lowers: &[std::os::fd::RawFd], data_fds: &[std::os::fd::RawFd]) -> String {
    let path = |fd: &std::os::fd::RawFd| format!("/proc/self/fd/{fd}");
    let mut arg = lowers.iter().map(path).collect::<Vec<_>>().join(":");
    for data_fd in data_fds {
        arg.push_str("::");
        arg.push_str(&path(data_fd));
    }
    arg
}

#[cfg(any(test, feature = "pre-6.15"))]
fn check_lowerdir_arg(arg: &str, lowers: usize) -> Result<()> {
    if arg.len() <= LOWERDIR_OPTION_MAX {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "{lowers} lower layers need a {}-byte overlayfs lowerdir option, but kernels \
             before 6.15 (this build's pre-6.15 feature) take at most {LOWERDIR_OPTION_MAX} bytes",
            arg.len()
        ),
    ))
}

/// Prepares a mounted filesystem for further use.
///
/// On pre-6.15 kernels, this mounts the filesystem to a temporary directory and returns
/// an O_PATH file descriptor to it. The temporary mount is automatically cleaned up when
/// the returned value is dropped.
#[cfg(feature = "pre-6.15")]
pub fn prepare_mount(mnt_fd: OwnedFd) -> Result<impl AsFd> {
    tmpmount::TmpMount::mount(mnt_fd)
}

/// Prepares an open erofs image file for mounting.
///
/// On RHEL9 kernels (before 6.12), this creates a loopback device because erofs
/// cannot directly mount files on these older kernels.
#[cfg(feature = "rhel9")]
pub fn make_erofs_mountable(image: OwnedFd) -> Result<OwnedFd> {
    loopify(image)
}

// Finally, we have two submodules which do the heavy lifting for loopback devices and temporary
// mountpoints.

// Required before Linux 6.15: it's not possible to use detached mounts with OPEN_TREE_CLONE or
// overlayfs.  Convert them into a non-floating form by mounting them on a temporary directory and
// reopening them as an O_PATH fd.
#[cfg(feature = "pre-6.15")]
mod tmpmount {
    use std::{
        io::Result,
        os::fd::{AsFd, BorrowedFd, OwnedFd},
    };

    use rustix::fs::{Mode, OFlags, open};
    use rustix::mount::{MoveMountFlags, UnmountFlags, move_mount, unmount};

    pub(super) struct TmpMount {
        dir: tempfile::TempDir,
        fd: OwnedFd,
    }

    impl TmpMount {
        pub(super) fn mount(mnt_fd: OwnedFd) -> Result<impl AsFd> {
            let tmp = tempfile::TempDir::new()?;
            move_mount(
                mnt_fd.as_fd(),
                "",
                rustix::fs::CWD,
                tmp.path(),
                MoveMountFlags::MOVE_MOUNT_F_EMPTY_PATH,
            )?;
            let fd = open(
                tmp.path(),
                OFlags::PATH | OFlags::DIRECTORY | OFlags::CLOEXEC,
                Mode::empty(),
            )?;
            Ok(TmpMount { dir: tmp, fd })
        }
    }

    impl AsFd for TmpMount {
        fn as_fd(&self) -> BorrowedFd<'_> {
            self.fd.as_fd()
        }
    }

    impl Drop for TmpMount {
        fn drop(&mut self) {
            let _ = unmount(self.dir.path(), UnmountFlags::DETACH);
        }
    }
}

/// Required before 6.12: erofs can't directly mount files.
/// Uses the composefs-ioctls crate for the loop device ioctl.
#[cfg(feature = "rhel9")]
fn loopify(image: OwnedFd) -> Result<OwnedFd> {
    composefs_ioctls::loop_device::loopify(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowerdir_arg_orders_layers() {
        assert_eq!(lowerdir_arg(&[5], &[]), "/proc/self/fd/5");
        assert_eq!(
            lowerdir_arg(&[5, 6], &[7, 8]),
            "/proc/self/fd/5:/proc/self/fd/6::/proc/self/fd/7::/proc/self/fd/8"
        );
    }

    #[test]
    fn lowerdir_arg_length_limit() {
        // Each "/proc/self/fd/NN" is 16 bytes: n lowers and one data layer take
        // 16n + (n - 1) + 2 + 16 bytes, exactly the limit for n = 14.
        let cases: [(Vec<std::os::fd::RawFd>, usize, bool); 4] = [
            (vec![10; 13], 238, true),
            (vec![10; 14], 255, true),
            ([vec![100], vec![10; 13]].concat(), 256, false),
            (vec![10; 15], 272, false),
        ];
        for (lowers, len, fits) in cases {
            let arg = lowerdir_arg(&lowers, &[11]);
            assert_eq!(arg.len(), len, "{arg}");
            let result = check_lowerdir_arg(&arg, lowers.len());
            assert_eq!(result.is_ok(), fits, "{len} bytes: {result:?}");
            if let Err(err) = result {
                let msg = err.to_string();
                assert!(
                    msg.contains(&format!("{} lower layers need a {len}-byte", lowers.len()))
                        && msg.contains("at most 255 bytes"),
                    "{msg}"
                );
            }
        }
    }
}
