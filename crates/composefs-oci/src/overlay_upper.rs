//! Turn an overlayfs upper directory into an OCI layer.
//!
//! Running a tool (a package manager, `tar -x`, ...) inside a writable
//! overlayfs mounted over an image leaves exactly its changes in the overlay's
//! upper directory. [`write_upper_tar`] writes that directory as an OCI tar
//! layer, and [`import_upper`] imports it into a repository, so that it can be
//! stacked on the base image's layers (see [`crate::derive_image`]).
//!
//! The upper must come from an overlay mounted with `metacopy=off`,
//! `redirect_dir=off` and `index=off`, which keeps it self-contained: every
//! changed file is a full copy, and the only overlay-specific entries are
//! whiteouts and opaque directories. Those are translated to their OCI
//! spelling:
//!
//! - a whiteout (a character device 0/0) becomes an empty `.wh.<name>`;
//! - an opaque directory (`trusted.overlay.opaque=y`, or `user.overlay.opaque=y`
//!   with [`UpperOptions::userxattr`]) gets a `.wh..wh..opq` entry right after its own entry,
//!   before any of its children, since the opaque marker clears whatever the
//!   directory held at that point.
//!
//! An upper with `metacopy` or `redirect` xattrs (from an overlay mounted with
//! those features on) is refused rather than converted, because its files
//! would depend on the lower layers. Other overlay-private xattrs (`origin`,
//! `impure`, ...) and `security.selinux` are dropped. An escaped overlay
//! xattr (`trusted.overlay.overlay.X`, which the overlay shows as
//! `trusted.overlay.X`) is kept under the name the overlay shows. That name
//! doesn't reach a composefs image as overlay metadata: an image keeps only
//! allowlisted xattrs (with `user.*` at most), and the EROFS writer escapes
//! `trusted.overlay.*` anyway. Everything else, such as `security.capability`
//! and `user.*`, is kept as `SCHILY.xattr.*` PAX records. An xattr name
//! containing `=` is refused, since a PAX record ends its key at the first
//! `=`. Files hardlinked within the upper become tar hardlinks. A file whose
//! name starts with `.wh.` is refused, since a layer would read it as a
//! whiteout. Sockets are left out, as container builds do.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{self, Seek, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use composefs::digest::{Digest, Sha256};
use composefs::fsverity::FsVerityHashValue;
use composefs::repository::Repository;
use tar_core::EntryType;
use tar_core::builder::EntryBuilder;

use crate::{ImportStats, OciDigest, import_layer, sha256_output_to_digest};

const BLOCK: usize = 512;
const WHITEOUT_PREFIX: &[u8] = b".wh.";
const OPAQUE_MARKER: &[u8] = b".wh..wh..opq";
const PAX_XATTR_PREFIX: &str = "SCHILY.xattr.";
/// The overlay's own xattrs, for normal mounts and with `userxattr`.
const TRUSTED_OVERLAY_PREFIX: &[u8] = b"trusted.overlay.";
const USER_OVERLAY_PREFIX: &[u8] = b"user.overlay.";
/// What follows the overlay prefix in an escaped xattr: the overlay shows
/// `<prefix>overlay.X` as `<prefix>X`.
const ESCAPED_OVERLAY_INFIX: &[u8] = b"overlay.";
/// Overlay features whose xattrs make an upper depend on its lower layers.
const REFUSED_OVERLAY_XATTRS: [&[u8]; 2] = [b"metacopy", b"redirect"];
/// Dropped because the boot image is relabeled from the image's policy anyway.
const DROPPED_XATTRS: [&[u8]; 1] = [b"security.selinux"];

/// What [`write_upper_tar`] wrote.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UpperStats {
    /// Regular files written with their content.
    pub files: u64,
    /// Directories.
    pub directories: u64,
    /// Symbolic links.
    pub symlinks: u64,
    /// Hardlinks to an entry written earlier in the layer.
    pub hardlinks: u64,
    /// Devices and FIFOs.
    pub special: u64,
    /// Sockets, which are left out.
    pub sockets: u64,
    /// Whiteouts, written as `.wh.<name>`.
    pub whiteouts: u64,
    /// Opaque directories, written with `.wh..wh..opq`.
    pub opaque_directories: u64,
    /// Directories left out because the lower has them with the same
    /// metadata (see [`UpperOptions::lower`]).
    pub unchanged_directories: u64,
    /// Top-level entries that were left out because they were excluded.
    pub excluded: Vec<OsString>,
}

/// Options for [`write_upper_tar`] and [`import_upper`].
#[derive(Debug, Default, Clone, Copy)]
pub struct UpperOptions<'a> {
    /// Top-level entries of the upper to leave out (e.g. `boot` or `var`);
    /// they are reported in [`UpperStats::excluded`].
    pub excluded: &'a [&'a OsStr],
    /// A mount of the overlay's lower (the base image). Directories that
    /// only hold changes are copied up with new mtimes; with the lower,
    /// those whose mode, owner and xattrs are unchanged are left out of the
    /// layer, so the derived tree keeps the base's directory metadata.
    /// That keeps unchanged parts of the tree byte for byte the same, which
    /// is what lets an image split into several EROFS images keep the
    /// images of the parts a layer doesn't touch.
    pub lower: Option<&'a Path>,
    /// The overlay was mounted with `userxattr`, so its own xattrs are
    /// `user.overlay.*`; otherwise they are `trusted.overlay.*`, and
    /// `user.overlay.*` xattrs are ordinary file content.
    pub userxattr: bool,
}

/// A layer imported by [`import_upper`].
#[derive(Debug)]
pub struct ImportedUpper<ObjectID> {
    /// The layer's diff_id (and, being uncompressed, its digest).
    pub diff_id: OciDigest,
    /// The fs-verity digest of the layer's splitstream.
    pub verity: ObjectID,
    /// The size of the tar stream in bytes.
    pub size: u64,
    /// What the layer contains.
    pub upper: UpperStats,
    /// Import statistics.
    pub import: ImportStats,
}

struct Xattrs {
    kept: Vec<(Vec<u8>, Vec<u8>)>,
    opaque: bool,
}

/// Reads the xattrs of `path`, leaving out the overlay's own (with the
/// `userxattr` namespace if set) and `security.selinux`, and unescaping
/// escaped overlay xattrs to the names the overlay shows. With `refuse`, an
/// overlay xattr that makes the upper depend on the lower is an error.
fn read_xattrs(path: &Path, userxattr: bool, refuse: bool) -> Result<Xattrs> {
    let overlay_prefix = if userxattr {
        USER_OVERLAY_PREFIX
    } else {
        TRUSTED_OVERLAY_PREFIX
    };
    let mut kept = Vec::new();
    let mut opaque = false;
    for name in list_xattrs(path)? {
        if let Some(key) = name.strip_prefix(overlay_prefix) {
            if let Some(escaped) = key.strip_prefix(ESCAPED_OVERLAY_INFIX) {
                let value = get_xattr(path, &name)?;
                kept.push(([overlay_prefix, escaped].concat(), value));
                continue;
            }
            if refuse && REFUSED_OVERLAY_XATTRS.contains(&key) {
                bail!(
                    "{} has overlayfs xattr {}: the upper must come from an overlay \
                     mounted with metacopy=off and redirect_dir=off",
                    path.display(),
                    String::from_utf8_lossy(&name)
                );
            }
            if key == b"opaque" {
                opaque = get_xattr(path, &name)? == b"y";
            }
            continue;
        }
        if DROPPED_XATTRS.contains(&name.as_slice()) {
            continue;
        }
        let value = get_xattr(path, &name)?;
        kept.push((name, value));
    }
    kept.sort();
    Ok(Xattrs { kept, opaque })
}

fn list_xattrs(path: &Path) -> Result<Vec<Vec<u8>>> {
    let mut buf = vec![0u8; 1024];
    loop {
        match rustix::fs::llistxattr(path, &mut buf) {
            Ok(n) => {
                return Ok(buf[..n]
                    .split(|&b| b == 0)
                    .filter(|n| !n.is_empty())
                    .map(<[u8]>::to_vec)
                    .collect());
            }
            Err(rustix::io::Errno::RANGE) => buf.resize(buf.len() * 4, 0),
            Err(e) => {
                return Err(e).with_context(|| format!("Listing xattrs of {}", path.display()));
            }
        }
    }
}

fn get_xattr(path: &Path, name: &[u8]) -> Result<Vec<u8>> {
    let name_os = OsStr::from_bytes(name);
    let mut buf = vec![0u8; 256];
    loop {
        match rustix::fs::lgetxattr(path, name_os, &mut buf) {
            Ok(n) => {
                buf.truncate(n);
                return Ok(buf);
            }
            Err(rustix::io::Errno::RANGE) => buf.resize(buf.len() * 4, 0),
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "Reading xattr {} of {}",
                        String::from_utf8_lossy(name),
                        path.display()
                    )
                });
            }
        }
    }
}

fn pad(out: &mut impl Write, len: u64) -> io::Result<()> {
    let rem = (len % BLOCK as u64) as usize;
    if rem != 0 {
        out.write_all(&[0u8; BLOCK][..BLOCK - rem])?;
    }
    Ok(())
}

fn entry(
    fs_path: &Path,
    path: &[u8],
    meta: &std::fs::Metadata,
    xattrs: &[(Vec<u8>, Vec<u8>)],
) -> Result<EntryBuilder> {
    let mut b = EntryBuilder::new_ustar();
    b.path(path);
    b.mode(meta.mode() & 0o7777)?;
    b.uid(meta.uid().into())?;
    b.gid(meta.gid().into())?;
    b.mtime(meta.mtime().max(0) as u64)?;
    b.size(0)?;
    for (name, value) in xattrs {
        let name = std::str::from_utf8(name).with_context(|| {
            format!(
                "{}: xattr name {} is not UTF-8",
                fs_path.display(),
                String::from_utf8_lossy(name)
            )
        })?;
        if name.contains('=') {
            bail!(
                "{}: xattr name {name} contains '=', which a PAX xattr record can't represent",
                fs_path.display()
            );
        }
        b.add_pax(&format!("{PAX_XATTR_PREFIX}{name}"), value);
    }
    Ok(b)
}

fn empty_file(path: &[u8]) -> Result<EntryBuilder> {
    let mut b = EntryBuilder::new_ustar();
    b.path(path);
    b.mode(0o644)?;
    b.size(0)?;
    b.entry_type(EntryType::Regular);
    Ok(b)
}

/// A directory of the lower: its metadata and kept xattrs.
type LowerDir = (std::fs::Metadata, Vec<(Vec<u8>, Vec<u8>)>);

struct Writer<'a, W: Write> {
    out: &'a mut W,
    opts: UpperOptions<'a>,
    links: HashMap<(u64, u64), Vec<u8>>,
    stats: UpperStats,
}

impl<W: Write> Writer<'_, W> {
    fn header(&mut self, mut b: EntryBuilder) -> Result<()> {
        self.out.write_all(&b.finish_bytes())?;
        Ok(())
    }

    /// The lower's directory at `layer_path`, if it has one: its metadata
    /// and kept xattrs.
    fn lower_dir(&self, layer_path: &[u8]) -> Result<Option<LowerDir>> {
        let Some(lower) = self.opts.lower else {
            return Ok(None);
        };
        let lower_path = lower.join(OsStr::from_bytes(layer_path));
        match std::fs::symlink_metadata(&lower_path) {
            Ok(m) if m.is_dir() => {
                let xattrs = read_xattrs(&lower_path, self.opts.userxattr, false)?.kept;
                Ok(Some((m, xattrs)))
            }
            Ok(_) => Ok(None),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => {
                Err(e).with_context(|| format!("Reading metadata of {}", lower_path.display()))
            }
        }
    }

    /// Writes the children of `dir` (whose path in the layer is `rel`).
    /// `lower_visible` says whether the lower's tree at `rel` still shows in
    /// the merged result: not below an opaque directory, nor below one the
    /// lower doesn't have, so that no directory there can be left out as
    /// unchanged.
    fn walk(&mut self, dir: &Path, rel: &[u8], lower_visible: bool) -> Result<()> {
        let mut names: Vec<OsString> = std::fs::read_dir(dir)
            .with_context(|| format!("Reading {}", dir.display()))?
            .map(|e| e.map(|e| e.file_name()))
            .collect::<io::Result<_>>()
            .with_context(|| format!("Reading {}", dir.display()))?;
        names.sort();
        for name in names {
            if rel.is_empty() && self.opts.excluded.contains(&name.as_os_str()) {
                self.stats.excluded.push(name);
                continue;
            }
            if name.as_bytes().starts_with(WHITEOUT_PREFIX) {
                bail!(
                    "{}: a name starting with .wh. would be read as a whiteout in an OCI layer",
                    dir.join(&name).display()
                );
            }
            let path = dir.join(&name);
            let mut layer_path = rel.to_vec();
            if !layer_path.is_empty() {
                layer_path.push(b'/');
            }
            layer_path.extend_from_slice(name.as_bytes());
            self.write_one(&path, rel, name.as_bytes(), layer_path, lower_visible)?;
        }
        Ok(())
    }

    fn write_one(
        &mut self,
        path: &Path,
        parent: &[u8],
        name: &[u8],
        layer_path: Vec<u8>,
        lower_visible: bool,
    ) -> Result<()> {
        let meta = std::fs::symlink_metadata(path)
            .with_context(|| format!("Reading metadata of {}", path.display()))?;
        let ft = meta.file_type();
        if ft.is_char_device() && meta.rdev() == 0 {
            let mut wh = parent.to_vec();
            if !wh.is_empty() {
                wh.push(b'/');
            }
            wh.extend_from_slice(WHITEOUT_PREFIX);
            wh.extend_from_slice(name);
            self.header(empty_file(&wh)?)?;
            self.stats.whiteouts += 1;
            return Ok(());
        }
        if ft.is_socket() {
            self.stats.sockets += 1;
            return Ok(());
        }
        let xattrs = read_xattrs(path, self.opts.userxattr, true)?;
        let mut b = entry(path, &layer_path, &meta, &xattrs.kept)?;
        if ft.is_dir() {
            let lower = match lower_visible {
                true => self.lower_dir(&layer_path)?,
                false => None,
            };
            let children_see_lower = lower.is_some() && !xattrs.opaque;
            let unchanged = lower.is_some_and(|(m, x)| {
                !xattrs.opaque
                    && m.mode() == meta.mode()
                    && m.uid() == meta.uid()
                    && m.gid() == meta.gid()
                    && x == xattrs.kept
            });
            if unchanged {
                self.stats.unchanged_directories += 1;
            } else {
                b.entry_type(EntryType::Directory);
                self.header(b)?;
                self.stats.directories += 1;
            }
            if xattrs.opaque {
                let mut marker = layer_path.clone();
                marker.push(b'/');
                marker.extend_from_slice(OPAQUE_MARKER);
                self.header(empty_file(&marker)?)?;
                self.stats.opaque_directories += 1;
            }
            return self.walk(path, &layer_path, children_see_lower);
        }
        // Any non-directory can be hardlinked: the first name is written,
        // the others link to it.
        if meta.nlink() > 1 {
            let key = (meta.dev(), meta.ino());
            if let Some(target) = self.links.get(&key) {
                b.entry_type(EntryType::Link);
                b.link_name(target.clone());
                self.header(b)?;
                self.stats.hardlinks += 1;
                return Ok(());
            }
            self.links.insert(key, layer_path.clone());
        }
        if ft.is_symlink() {
            let target = std::fs::read_link(path)
                .with_context(|| format!("Reading symlink {}", path.display()))?;
            b.entry_type(EntryType::Symlink);
            b.link_name(target.as_os_str().as_bytes());
            self.header(b)?;
            self.stats.symlinks += 1;
        } else if ft.is_file() {
            b.entry_type(EntryType::Regular);
            b.size(meta.len())?;
            self.header(b)?;
            let mut f = File::open(path).with_context(|| format!("Opening {}", path.display()))?;
            let copied = io::copy(&mut f, self.out)?;
            if copied != meta.len() {
                bail!("{} changed size while being read", path.display());
            }
            pad(self.out, copied)?;
            self.stats.files += 1;
        } else {
            let ty = if ft.is_fifo() {
                EntryType::Fifo
            } else if ft.is_char_device() {
                EntryType::Char
            } else if ft.is_block_device() {
                EntryType::Block
            } else {
                bail!("{}: unsupported file type", path.display());
            };
            b.entry_type(ty);
            if ty != EntryType::Fifo {
                let rdev = meta.rdev();
                b.device(rustix::fs::major(rdev), rustix::fs::minor(rdev))?;
            }
            self.header(b)?;
            self.stats.special += 1;
        }
        Ok(())
    }
}

/// Writes the overlayfs upper directory `upper` as an uncompressed OCI tar
/// layer to `out`; see the [module documentation](self) for the translation.
///
/// The upper's own root directory is not an entry: its metadata stays the
/// base image's. See [`UpperOptions`] for what else is left out.
pub fn write_upper_tar(
    upper: &Path,
    out: &mut impl Write,
    opts: &UpperOptions<'_>,
) -> Result<UpperStats> {
    let mut w = Writer {
        out,
        opts: *opts,
        links: HashMap::new(),
        stats: UpperStats::default(),
    };
    w.walk(upper, b"", true)?;
    w.out.write_all(&[0u8; 2 * BLOCK])?;
    w.out.flush()?;
    Ok(w.stats)
}

/// A writer that hashes what goes through it.
struct HashingWriter<W> {
    inner: W,
    hasher: Sha256,
    len: u64,
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.len += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Writes `upper` as a tar layer (see [`write_upper_tar`]) and imports it into
/// `repo` under its diff_id.
///
/// The tar stream goes through an unlinked temporary file in `tmpdir`, since
/// a layer is stored under its diff_id, which is only known once the whole
/// stream has been written.
pub async fn import_upper<ObjectID: FsVerityHashValue>(
    repo: &Arc<Repository<ObjectID>>,
    upper: &Path,
    opts: &UpperOptions<'_>,
    tmpdir: &Path,
) -> Result<ImportedUpper<ObjectID>> {
    let file = tempfile::tempfile_in(tmpdir)
        .with_context(|| format!("Creating a temporary file in {}", tmpdir.display()))?;
    let mut w = HashingWriter {
        inner: io::BufWriter::new(file),
        hasher: Sha256::new(),
        len: 0,
    };
    let upper_stats = write_upper_tar(upper, &mut w, opts)
        .with_context(|| format!("Writing {} as a tar layer", upper.display()))?;
    let HashingWriter { inner, hasher, len } = w;
    let mut file = inner.into_inner().map_err(|e| e.into_error())?;
    file.rewind()?;
    let diff_id = sha256_output_to_digest(hasher.finalize());
    let (verity, import) = import_layer(repo, &diff_id, None, tokio::fs::File::from_std(file))
        .await
        .with_context(|| format!("Importing layer {diff_id}"))?;
    Ok(ImportedUpper {
        diff_id,
        verity,
        size: len,
        upper: upper_stats,
        import,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};

    fn layer_entries(tar: &[u8]) -> Vec<(String, EntryType, u64, Vec<(String, Vec<u8>)>, String)> {
        let mut out = Vec::new();
        let mut archive = ::tar::Archive::new(tar);
        for e in archive.entries().unwrap() {
            let mut e = e.unwrap();
            let path = e.path().unwrap().to_string_lossy().into_owned();
            let ty = EntryType::from_byte(e.header().entry_type().as_byte());
            let size = e.header().size().unwrap();
            let link = e
                .link_name()
                .unwrap()
                .map(|l| l.to_string_lossy().into_owned())
                .unwrap_or_default();
            let mut xattrs = Vec::new();
            if let Some(exts) = e.pax_extensions().unwrap() {
                for ext in exts {
                    let ext = ext.unwrap();
                    if let Some(k) = ext.key().unwrap().strip_prefix(PAX_XATTR_PREFIX) {
                        xattrs.push((k.to_owned(), ext.value_bytes().to_vec()));
                    }
                }
            }
            out.push((path, ty, size, xattrs, link));
        }
        out
    }

    /// What an unprivileged process can create: everything but whiteouts and
    /// trusted.* xattrs, with a userxattr-style opaque directory. The
    /// privileged integration test covers a real overlay upper.
    #[test]
    fn upper_to_tar() {
        let dir = tempfile::tempdir().unwrap();
        let up = dir.path();
        std::fs::create_dir_all(up.join("usr/bin")).unwrap();
        std::fs::create_dir_all(up.join("usr/share/opaque")).unwrap();
        std::fs::create_dir_all(up.join("boot")).unwrap();
        std::fs::write(up.join("usr/bin/tool"), b"#!/bin/sh\n").unwrap();
        std::fs::set_permissions(up.join("usr/bin/tool"), PermissionsExt::from_mode(0o755))
            .unwrap();
        std::fs::hard_link(up.join("usr/bin/tool"), up.join("usr/bin/tool2")).unwrap();
        symlink("tool", up.join("usr/bin/tool-link")).unwrap();
        std::fs::write(up.join("usr/share/opaque/new"), b"new").unwrap();
        std::fs::write(up.join("boot/vmlinuz"), b"k").unwrap();
        let xattrs_ok = rustix::fs::setxattr(
            up.join("usr/share/opaque"),
            "user.overlay.opaque",
            b"y",
            rustix::fs::XattrFlags::empty(),
        )
        .is_ok()
            && rustix::fs::setxattr(
                up.join("usr/bin/tool"),
                "user.test",
                b"\x00\xffbinary",
                rustix::fs::XattrFlags::empty(),
            )
            .is_ok();

        let mut tar = Vec::new();
        let excluded = [OsStr::new("boot")];
        let opts = UpperOptions {
            excluded: &excluded,
            userxattr: true,
            ..Default::default()
        };
        let stats = write_upper_tar(up, &mut tar, &opts).unwrap();
        let entries = layer_entries(&tar);
        let summary: Vec<(&str, EntryType)> = entries.iter().map(|e| (e.0.as_str(), e.1)).collect();
        let mut expected = vec![
            ("usr", EntryType::Directory),
            ("usr/bin", EntryType::Directory),
            ("usr/bin/tool", EntryType::Regular),
            ("usr/bin/tool-link", EntryType::Symlink),
            ("usr/bin/tool2", EntryType::Link),
            ("usr/share", EntryType::Directory),
            ("usr/share/opaque", EntryType::Directory),
        ];
        if xattrs_ok {
            expected.push(("usr/share/opaque/.wh..wh..opq", EntryType::Regular));
        }
        expected.push(("usr/share/opaque/new", EntryType::Regular));
        similar_asserts::assert_eq!(summary, expected);
        assert_eq!(stats.excluded, vec![OsString::from("boot")]);
        assert_eq!(stats.hardlinks, 1);
        let tool = entries.iter().find(|e| e.0 == "usr/bin/tool").unwrap();
        assert_eq!(tool.2, 10);
        assert_eq!(
            entries.iter().find(|e| e.0 == "usr/bin/tool2").unwrap().4,
            "usr/bin/tool"
        );
        assert_eq!(
            entries
                .iter()
                .find(|e| e.0 == "usr/bin/tool-link")
                .unwrap()
                .4,
            "tool"
        );
        if xattrs_ok {
            assert_eq!(stats.opaque_directories, 1);
            assert_eq!(
                tool.3,
                vec![("user.test".to_owned(), b"\x00\xffbinary".to_vec())]
            );
            // The opaque marker itself is not carried over as an xattr.
            let opaque = entries.iter().find(|e| e.0 == "usr/share/opaque").unwrap();
            assert!(opaque.3.is_empty(), "{:?}", opaque.3);
        }
    }

    #[test]
    fn unchanged_lower_directories_are_left_out() {
        let lower = tempfile::tempdir().unwrap();
        let upper = tempfile::tempdir().unwrap();
        for d in ["usr/lib", "usr/share"] {
            std::fs::create_dir_all(lower.path().join(d)).unwrap();
            std::fs::create_dir_all(upper.path().join(d)).unwrap();
        }
        // A mode change is a metadata change; a new directory is new.
        std::fs::set_permissions(
            upper.path().join("usr/share"),
            PermissionsExt::from_mode(0o700),
        )
        .unwrap();
        std::fs::create_dir(upper.path().join("usr/lib/new")).unwrap();
        std::fs::write(upper.path().join("usr/lib/new/file"), b"x").unwrap();

        let mut tar = Vec::new();
        let opts = UpperOptions {
            lower: Some(lower.path()),
            ..Default::default()
        };
        let stats = write_upper_tar(upper.path(), &mut tar, &opts).unwrap();
        let paths: Vec<String> = layer_entries(&tar).into_iter().map(|e| e.0).collect();
        similar_asserts::assert_eq!(paths, ["usr/lib/new", "usr/lib/new/file", "usr/share"]);
        assert_eq!(stats.unchanged_directories, 2);
    }

    /// Below an opaque directory nothing of the lower is left, so a
    /// recreated directory there must be written even if the lower has one
    /// with the same metadata at that path.
    #[test]
    fn directories_below_an_opaque_one_are_written() {
        let lower = tempfile::tempdir().unwrap();
        let upper = tempfile::tempdir().unwrap();
        for root in [lower.path(), upper.path()] {
            std::fs::create_dir_all(root.join("a/b")).unwrap();
            std::fs::create_dir_all(root.join("a/empty")).unwrap();
        }
        std::fs::write(upper.path().join("a/b/f"), b"x").unwrap();
        if rustix::fs::setxattr(
            upper.path().join("a"),
            "user.overlay.opaque",
            b"y",
            rustix::fs::XattrFlags::empty(),
        )
        .is_err()
        {
            return; // no user xattrs on this filesystem
        }
        let mut tar = Vec::new();
        let opts = UpperOptions {
            lower: Some(lower.path()),
            userxattr: true,
            ..Default::default()
        };
        write_upper_tar(upper.path(), &mut tar, &opts).unwrap();
        let paths: Vec<String> = layer_entries(&tar).into_iter().map(|e| e.0).collect();
        similar_asserts::assert_eq!(paths, ["a", "a/.wh..wh..opq", "a/b", "a/b/f", "a/empty"]);

        // Without userxattr, user.overlay.* is content: no opaque marker, and
        // the xattr is kept.
        let mut tar = Vec::new();
        let opts = UpperOptions {
            lower: Some(lower.path()),
            ..Default::default()
        };
        let stats = write_upper_tar(upper.path(), &mut tar, &opts).unwrap();
        assert_eq!(stats.opaque_directories, 0);
        let entries = layer_entries(&tar);
        let a = entries.iter().find(|e| e.0 == "a").unwrap();
        assert_eq!(a.3, vec![("user.overlay.opaque".to_owned(), b"y".to_vec())]);
    }

    /// Sets `xattrs` on `path`, or returns false if the filesystem has no
    /// user xattrs.
    fn set_user_xattrs(path: &Path, xattrs: &[(&str, &[u8])]) -> bool {
        xattrs.iter().all(|(name, value)| {
            rustix::fs::setxattr(path, *name, value, rustix::fs::XattrFlags::empty()).is_ok()
        })
    }

    /// With `userxattr`, `user.overlay.*` is the overlay's: its bookkeeping
    /// is dropped, and escaped xattrs are kept as the overlay shows them.
    #[test]
    fn escaped_overlay_xattrs_are_unescaped() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path().join("d");
        std::fs::create_dir(&d).unwrap();
        let set: [(&str, &[u8]); 4] = [
            ("user.overlay.origin", b"o"),
            ("user.overlay.overlay.opaque", b"y"),
            ("user.overlay.overlay.overlay.redirect", b"/r"),
            ("user.other", b"x"),
        ];
        if !set_user_xattrs(&d, &set) {
            return;
        }
        let opts = UpperOptions {
            userxattr: true,
            ..Default::default()
        };
        let mut tar = Vec::new();
        let stats = write_upper_tar(dir.path(), &mut tar, &opts).unwrap();
        // An escaped opaque xattr is content, not this overlay's marker.
        assert_eq!(stats.opaque_directories, 0);
        let entries = layer_entries(&tar);
        let paths: Vec<&str> = entries.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(paths, ["d"]);
        let expected: Vec<(String, Vec<u8>)> = [
            ("user.other", b"x".as_slice()),
            ("user.overlay.opaque", b"y"),
            ("user.overlay.overlay.redirect", b"/r"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_vec()))
        .collect();
        assert_eq!(entries[0].3, expected);
    }

    /// A PAX record's key ends at its first `=`, so such a name would be
    /// read back as a different xattr.
    #[test]
    fn xattr_names_with_equals_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("f");
        std::fs::write(&f, b"").unwrap();
        if !set_user_xattrs(&f, &[("user.a=b", b"v")]) {
            return;
        }
        let err = write_upper_tar(dir.path(), &mut Vec::new(), &Default::default()).unwrap_err();
        let err = format!("{err:#}");
        assert!(
            err.contains("user.a=b") && err.contains("contains '='"),
            "{err}"
        );
    }

    #[test]
    fn hardlinked_fifos_stay_linked() {
        let dir = tempfile::tempdir().unwrap();
        rustix::fs::mknodat(
            rustix::fs::CWD,
            dir.path().join("a"),
            rustix::fs::FileType::Fifo,
            rustix::fs::Mode::from_raw_mode(0o644),
            0,
        )
        .unwrap();
        std::fs::hard_link(dir.path().join("a"), dir.path().join("b")).unwrap();
        let mut tar = Vec::new();
        let stats = write_upper_tar(dir.path(), &mut tar, &Default::default()).unwrap();
        assert_eq!((stats.special, stats.hardlinks), (1, 1));
        let entries = layer_entries(&tar);
        assert_eq!(entries[1].1, EntryType::Link);
        assert_eq!(entries[1].4, "a");
    }

    #[test]
    fn sockets_are_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let _listener = std::os::unix::net::UnixListener::bind(dir.path().join("sock")).unwrap();
        std::fs::write(dir.path().join("file"), b"x").unwrap();
        let mut tar = Vec::new();
        let stats = write_upper_tar(dir.path(), &mut tar, &Default::default()).unwrap();
        assert_eq!(stats.sockets, 1);
        let paths: Vec<String> = layer_entries(&tar).into_iter().map(|e| e.0).collect();
        assert_eq!(paths, ["file"]);
    }

    #[test]
    fn whiteout_named_files_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".wh.file"), b"").unwrap();
        let err = write_upper_tar(dir.path(), &mut Vec::new(), &Default::default()).unwrap_err();
        assert!(format!("{err:#}").contains("read as a whiteout"), "{err:#}");
    }

    #[test]
    fn upper_with_redirect_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("moved")).unwrap();
        if rustix::fs::setxattr(
            dir.path().join("moved"),
            "user.overlay.redirect",
            b"/orig",
            rustix::fs::XattrFlags::empty(),
        )
        .is_err()
        {
            return; // no user xattrs on this filesystem
        }
        let opts = UpperOptions {
            userxattr: true,
            ..Default::default()
        };
        let err = write_upper_tar(dir.path(), &mut Vec::new(), &opts).unwrap_err();
        assert!(format!("{err:#}").contains("redirect_dir=off"), "{err:#}");
    }
}
