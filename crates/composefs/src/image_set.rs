//! Image sets: several composefs images mounted as one tree.
//!
//! An image set is a small manifest listing composefs EROFS images that are
//! stacked, in order, as the lower layers of a single overlayfs mount over
//! the repository's object store.  The manifest is stored in the repository
//! as an fs-verity-enabled object, so its own digest pins every component
//! image, and through them every file: a single digest (for example on the
//! kernel command line) names the whole tree, just as the digest of a single
//! image does.
//!
//! This lets a tree be split into parts that change independently, such as
//! the kernel modules under `/usr/lib/modules` and the rest of the root
//! filesystem, while every part keeps its own digest and is shared between
//! the sets that use it.
//!
//! # Format
//!
//! The manifest is UTF-8 text, one record per line, each line ending in
//! `\n`, fields separated by a single space:
//!
//! ```text
//! composefs-image-set 1
//! algorithm fsverity-sha512-12
//! component 0 v1 <hex digest> /
//! component 1 v1 <hex digest> /usr/lib/modules
//! ```
//!
//! - The first line names the format and its version ([`IMAGE_SET_VERSION`]).
//! - `algorithm` is the fs-verity algorithm of every digest in the set, and
//!   must match the repository's.
//! - Each `component` line gives the component's position in the stack
//!   (`0` is the topmost overlayfs lower layer, and positions count up from
//!   there without gaps), the EROFS format version the image was written
//!   with, the fs-verity digest of the image, and its target: the directory
//!   the component provides.
//!
//! The encoding is canonical: parsing rejects any manifest that would not
//! serialize back to the same bytes, so a set has exactly one digest.
//!
//! # Targets and stacking order
//!
//! The first component is the root filesystem, with target `/`, and no other
//! component has that target.  Every other component contains its target
//! directory and, outside it, only the chain of the target's ancestor
//! directories (see [`crate::generic_tree::FileSystem::split_off`]), plus the
//! whiteout stubs at the root of every V0/V1 image.  Several components may
//! share a target.
//!
//! No target starts with a directory named like one of those stubs, `00` to
//! `ff`, whatever the formats in the set: a stub in any V0/V1 component, the
//! root included, would hide that directory in the components below it.
//!
//! No directory of a component other than the root, be it an ancestor of the
//! target, the target or one below it, may carry an overlayfs xattr
//! (`trusted.overlay.*`, other than the escaped `trusted.overlay.overlay.*`
//! that the writer turns image xattrs into): with `redirect_dir=on`, a
//! `trusted.overlay.redirect` or `trusted.overlay.opaque` there could move
//! files of the components below to another path, or hide them.  Since an
//! absolute redirect restarts the lookup at the root of the layers below,
//! even a directory deep inside the target could show files from outside
//! it.  The root directory of each image is exempt, since overlayfs ignores
//! these xattrs on a layer's root (the V1 writer puts
//! `trusted.overlay.opaque` there).  Files aren't checked: on a regular file
//! overlayfs only acts on the `trusted.overlay.metacopy` and
//! `trusted.overlay.redirect` the writer uses to name its object, which can
//! give the file other content, but under `verity=require` only content
//! matching the digest in its own metacopy xattr.  So file content stays
//! inside the target only under `verity=require`: with fs-verity off, as for
//! an insecure repository, a file's redirect can show the content of another
//! component's file.  `user.overlay.*` xattrs aren't checked either: the
//! mount doesn't use `userxattr`, so overlayfs ignores them.
//!
//! [`ImageSet::new`] and [`ImageSet::parse`] enforce the rules on targets, and
//! [`ImageSetComponent::check_format`] and
//! [`ImageSetComponent::check_contents`] check an image against its
//! component; [`crate::repository::Repository`] runs both when writing a set
//! and when mounting one.  Together they mean a component can only provide
//! files below its target (though with fs-verity off, as above, their
//! content can come from outside it): overlayfs gives a path present in
//! several components the content of the topmost one, and since the root
//! comes first, the metadata of a directory present in several (such as
//! `/usr` in both of the components above) is always the root's, while the
//! entries of the directories merge.

use std::{collections::HashSet, fmt};

use thiserror::Error;

use zerocopy::FromBytes;

use crate::{
    erofs::{
        format::{
            COMPOSEFS_MAGIC, COMPOSEFS_VERSION, COMPOSEFS_VERSION_V0, COMPOSEFS_VERSION_V1,
            ComposefsHeader, FormatVersion, XATTR_OVERLAY_ESCAPED_PREFIX, XATTR_OVERLAY_PREFIX,
        },
        reader::{ErofsReaderError, Image, InodeHeader, MAX_DIRECTORY_DEPTH},
    },
    fsverity::{FsVerityHashValue, compute_verity},
};

/// The first word of every image set manifest.
pub const IMAGE_SET_MAGIC: &str = "composefs-image-set";

/// The manifest format version written and accepted by this code.
pub const IMAGE_SET_VERSION: u32 = 1;

/// The maximum number of components in one set.
///
/// overlayfs stacks at most 500 layers (`OVL_MAX_STACK`), including the
/// object store's data-only layer, and on 6.15 and later kernels each layer
/// is passed on its own, so this is a format limit rather than a mount one.
/// Before 6.15 all layers share one `lowerdir` option of at most
/// [`LOWERDIR_OPTION_MAX`](crate::mountcompat::LOWERDIR_OPTION_MAX) bytes,
/// which fits only about a dozen; how many exactly depends on the file
/// descriptor numbers at mount time, so mounting checks that, with an error
/// naming the limit.
pub const IMAGE_SET_MAX_COMPONENTS: usize = 256;

/// The target of the root component.
pub const ROOT_TARGET: &str = "/";

/// The largest manifest that will be parsed.
pub const IMAGE_SET_MAX_SIZE: usize = 1024 * 1024;

/// An error in an image set manifest.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ImageSetError {
    /// The manifest is larger than [`IMAGE_SET_MAX_SIZE`].
    #[error("image set manifest is {0} bytes, more than the limit of {IMAGE_SET_MAX_SIZE}")]
    TooLarge(usize),
    /// The manifest is not UTF-8.
    #[error("image set manifest is not UTF-8")]
    NotUtf8,
    /// A line of the manifest is malformed.
    #[error("image set manifest line {line}: {reason}")]
    Line {
        /// The 1-based line number.
        line: usize,
        /// What is wrong with it.
        reason: String,
    },
    /// The manifest uses a different fs-verity algorithm than expected.
    #[error("image set uses {found}, but this repository uses {expected}")]
    AlgorithmMismatch {
        /// The algorithm named in the manifest.
        found: String,
        /// The algorithm of the repository's object IDs.
        expected: String,
    },
    /// The set has no components.
    #[error("image set has no components")]
    Empty,
    /// The set has more than [`IMAGE_SET_MAX_COMPONENTS`] components.
    #[error("image set has {0} components, more than the limit of {IMAGE_SET_MAX_COMPONENTS}")]
    TooManyComponents(usize),
    /// A component's target is not a normalized absolute path.
    #[error("invalid image set target {0:?}: expected a normalized absolute path")]
    InvalidTarget(String),
    /// The same image is listed twice.
    #[error("image {0} appears more than once in the image set")]
    DuplicateImage(String),
    /// The first component isn't the root filesystem.
    #[error("the first component of an image set must have target {ROOT_TARGET}, not {0:?}")]
    RootNotFirst(String),
    /// A component other than the first has target `/`.
    #[error(
        "only the first component of an image set may have target {ROOT_TARGET}, not component {0}"
    )]
    ExtraRoot(usize),
    /// A component's image doesn't match the component.
    #[error("image {digest} of image set component {target}: {reason}")]
    InvalidComponent {
        /// The image's digest.
        digest: String,
        /// The component's target.
        target: String,
        /// What is wrong with the image.
        reason: String,
    },
    /// A target starts with a directory named like a V0/V1 whiteout stub.
    #[error(
        "invalid image set target {0:?}: its first directory is named like the whiteout \
         stubs (00 to ff) at the root of V0/V1 images, which would hide it"
    )]
    StubNameTarget(String),
    /// The manifest parses, but isn't in canonical form.
    #[error("image set manifest is not in canonical form")]
    NotCanonical,
}

/// One image in an [`ImageSet`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSetComponent<ObjectID> {
    /// The fs-verity digest of the EROFS image.
    pub digest: ObjectID,
    /// The EROFS format version the image was written with.
    pub format: FormatVersion,
    /// The directory this component provides, as an absolute path.
    pub target: String,
}

/// Whether an image whose header has `composefs_version` was written in `format`.
fn format_matches(format: FormatVersion, composefs_version: u32) -> bool {
    match format {
        // V0 images get version 1 when they contain whiteouts, like C mkcomposefs
        FormatVersion::V0 => [COMPOSEFS_VERSION_V0, COMPOSEFS_VERSION_V1]
            .iter()
            .any(|v| v.get() == composefs_version),
        FormatVersion::V1 => COMPOSEFS_VERSION_V1.get() == composefs_version,
        FormatVersion::V2 => COMPOSEFS_VERSION.get() == composefs_version,
    }
}

/// Whether `name` is the name of one of the whiteout stubs `00` to `ff` that
/// every V0/V1 image has at its root.
fn is_whiteout_stub_name(name: &[u8]) -> bool {
    name.len() == 2
        && name
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}

/// Whether `name` is an xattr that overlayfs acts on in a lower layer: a
/// `trusted.overlay.*` one that isn't escaped as `trusted.overlay.overlay.*`.
fn is_overlay_xattr(name: &[u8]) -> bool {
    name.starts_with(XATTR_OVERLAY_PREFIX) && !name.starts_with(XATTR_OVERLAY_ESCAPED_PREFIX)
}

/// How many bytes from the start of an image [`ImageSetComponent::check_format`] needs.
pub const COMPONENT_HEADER_SIZE: usize = size_of::<ComposefsHeader>();

impl<ObjectID: FsVerityHashValue> ImageSetComponent<ObjectID> {
    fn invalid(&self, reason: impl Into<String>) -> ImageSetError {
        ImageSetError::InvalidComponent {
            digest: self.digest.to_hex(),
            target: self.target.clone(),
            reason: reason.into(),
        }
    }

    /// Checks that `header`, the first [`COMPONENT_HEADER_SIZE`] bytes of this
    /// component's image, is the header of a composefs image in the format
    /// the component declares.
    pub fn check_format(&self, header: &[u8]) -> Result<(), ImageSetError> {
        let (header, _) = ComposefsHeader::read_from_prefix(header)
            .map_err(|_| self.invalid("too short for a composefs header"))?;
        if header.magic != COMPOSEFS_MAGIC {
            return Err(self.invalid("not a composefs image"));
        }
        let version = header.composefs_version.get();
        if !format_matches(self.format, version) {
            return Err(self.invalid(format!(
                "listed as {}, but its header has composefs version {version}",
                format_name(self.format)
            )));
        }
        Ok(())
    }

    /// Checks that `image`, all of this component's image, contains nothing
    /// outside the component's target but the target's ancestor directories,
    /// and at the root the whiteout stubs of V0/V1 images, and that none of
    /// its directories but the image's root carries an overlayfs xattr: not
    /// the ancestors, not the target, and not the directories below it (see
    /// the [module documentation](self)).  Any image passes for the root
    /// component, whose target is `/`.
    ///
    /// A directory on the way to the target with two entries of the same name
    /// is refused: this check and the kernel could each pick a different one
    /// (the kernel's lookup is a binary search over sorted entries), and the
    /// one the kernel picks could hold anything.
    ///
    /// Like the image reader, it refuses directories nested 2048 or more
    /// levels deep, and a directory linked from more than one place (or
    /// from inside itself), which no real image has: otherwise a link to a
    /// directory seen before could hide how deep its subtree really goes.
    ///
    /// This looks at every directory, and at the inode of every entry in one,
    /// but at no file data.
    pub fn check_contents(&self, image: &[u8]) -> Result<(), ImageSetError> {
        if self.target == ROOT_TARGET {
            return Ok(());
        }
        let too_deep = || {
            self.invalid(format!(
                "nests directories {MAX_DIRECTORY_DEPTH} or more levels deep"
            ))
        };
        let target_depth = self.target[1..].split('/').count();
        if target_depth >= MAX_DIRECTORY_DEPTH {
            return Err(too_deep());
        }
        let erofs = |e: ErofsReaderError| self.invalid(format!("reading image: {e}"));
        let image = Image::open(image)
            .and_then(Image::restrict_to_composefs)
            .map_err(erofs)?;
        let check_xattrs = |nid: u64, path: &dyn Fn() -> String| match image
            .xattr_names(nid)
            .map_err(erofs)?
            .into_iter()
            .find(|xattr| is_overlay_xattr(xattr))
        {
            Some(xattr) => Err(self.invalid(format!(
                "{} has the overlayfs xattr {}",
                path(),
                String::from_utf8_lossy(&xattr)
            ))),
            None => Ok(()),
        };
        let linked_twice =
            |path: &str| self.invalid(format!("links {path} to a directory seen before"));
        let mut nid = image.sb.root_nid.get() as u64;
        let mut seen = HashSet::from([nid]);
        // The path so far is always a prefix of the target.
        let mut path = "";
        for name in self.target[1..].split('/') {
            let mut next = None;
            for (entry, entry_nid) in image.directory_entries(nid).map_err(erofs)? {
                if entry == name.as_bytes() {
                    if next.replace(entry_nid).is_some() {
                        return Err(self.invalid(format!("has two entries {path}/{name}")));
                    }
                } else if !(path.is_empty()
                    && is_whiteout_stub_name(&entry)
                    && image.inode(entry_nid).map_err(erofs)?.is_whiteout())
                {
                    return Err(self.invalid(format!(
                        "contains {path}/{} outside its target",
                        String::from_utf8_lossy(&entry)
                    )));
                }
            }
            path = &self.target[..path.len() + 1 + name.len()];
            let Some(next) = next else {
                return Err(self.invalid(format!("has no {path}")));
            };
            if !image.inode(next).map_err(erofs)?.mode().is_dir() {
                return Err(self.invalid(format!("{path} is not a directory")));
            }
            if !seen.insert(next) {
                return Err(linked_twice(path));
            }
            check_xattrs(next, &|| path.to_string())?;
            nid = next;
        }

        // The target's subtree, depth first.  A crafted image can link a
        // directory from several places, or from inside itself, so `seen`
        // refuses every directory reached twice, the ancestors included.
        // Directories are kept as their inode, parent and name, and a path is
        // built only for an error: queuing whole paths would take memory
        // quadratic in the depth of the tree.
        struct SubDir {
            nid: u64,
            parent: usize,
            name: Vec<u8>,
            depth: usize,
        }
        let path_of = |dirs: &[SubDir], mut at: usize| {
            let mut names = Vec::new();
            while at != 0 {
                names.push(String::from_utf8_lossy(&dirs[at].name).into_owned());
                at = dirs[at].parent;
            }
            names.reverse();
            format!("{}/{}", self.target, names.join("/"))
        };
        let mut dirs = vec![SubDir {
            nid,
            parent: 0,
            name: Vec::new(),
            depth: target_depth,
        }];
        let mut pending = vec![0];
        while let Some(at) = pending.pop() {
            for (entry, entry_nid) in image.directory_entries(dirs[at].nid).map_err(erofs)? {
                if image.inode(entry_nid).map_err(erofs)?.mode().is_dir() {
                    if !seen.insert(entry_nid) {
                        let path = format!(
                            "{}/{}",
                            path_of(&dirs, at).trim_end_matches('/'),
                            String::from_utf8_lossy(&entry)
                        );
                        return Err(linked_twice(&path));
                    }
                    let depth = dirs[at].depth + 1;
                    if depth >= MAX_DIRECTORY_DEPTH {
                        return Err(too_deep());
                    }
                    dirs.push(SubDir {
                        nid: entry_nid,
                        parent: at,
                        name: entry,
                        depth,
                    });
                    let child = dirs.len() - 1;
                    check_xattrs(entry_nid, &|| path_of(&dirs, child))?;
                    pending.push(child);
                }
            }
        }
        Ok(())
    }
}

/// A validated list of composefs images to mount as one tree.
///
/// See the [module documentation](self) for the format and semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSet<ObjectID> {
    components: Vec<ImageSetComponent<ObjectID>>,
}

fn format_name(format: FormatVersion) -> &'static str {
    match format {
        FormatVersion::V0 => "v0",
        FormatVersion::V1 => "v1",
        FormatVersion::V2 => "v2",
    }
}

fn parse_format(s: &str) -> Option<FormatVersion> {
    match s {
        "v0" => Some(FormatVersion::V0),
        "v1" => Some(FormatVersion::V1),
        "v2" => Some(FormatVersion::V2),
        _ => None,
    }
}

/// Checks that `target` can be the target of an image set component: `/`, or
/// an absolute path without empty, `.` or `..` components, trailing slash,
/// whitespace or control characters, whose first component isn't named like
/// a whiteout stub (see the [module documentation](self)).
pub fn validate_target(target: &str) -> Result<(), ImageSetError> {
    if !valid_target(target) {
        return Err(ImageSetError::InvalidTarget(target.to_string()));
    }
    if target[1..]
        .split('/')
        .next()
        .is_some_and(|first| is_whiteout_stub_name(first.as_bytes()))
    {
        return Err(ImageSetError::StubNameTarget(target.to_string()));
    }
    Ok(())
}

fn valid_target(target: &str) -> bool {
    if target == ROOT_TARGET {
        return true;
    }
    let Some(rest) = target.strip_prefix('/') else {
        return false;
    };
    !target.chars().any(|c| c.is_whitespace() || c.is_control())
        && rest
            .split('/')
            .all(|name| !name.is_empty() && name != "." && name != "..")
}

impl<ObjectID: FsVerityHashValue> ImageSet<ObjectID> {
    /// Creates a set from its components, topmost first: the root filesystem,
    /// then the others (see the [module documentation](self)).
    pub fn new(components: Vec<ImageSetComponent<ObjectID>>) -> Result<Self, ImageSetError> {
        if components.is_empty() {
            return Err(ImageSetError::Empty);
        }
        if components.len() > IMAGE_SET_MAX_COMPONENTS {
            return Err(ImageSetError::TooManyComponents(components.len()));
        }
        let mut seen = HashSet::new();
        for (order, component) in components.iter().enumerate() {
            validate_target(&component.target)?;
            match (order, component.target == ROOT_TARGET) {
                (0, false) => return Err(ImageSetError::RootNotFirst(component.target.clone())),
                (1.., true) => return Err(ImageSetError::ExtraRoot(order)),
                _ => {}
            }
            if !seen.insert(&component.digest) {
                return Err(ImageSetError::DuplicateImage(component.digest.to_hex()));
            }
        }
        Ok(Self { components })
    }

    /// The components, topmost first.
    pub fn components(&self) -> &[ImageSetComponent<ObjectID>] {
        &self.components
    }

    /// Serializes the set as a manifest.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_string().into_bytes()
    }

    /// The fs-verity digest of the manifest, which identifies the set.
    pub fn compute_id(&self) -> ObjectID {
        compute_verity(&self.to_bytes())
    }

    /// Parses and validates a manifest.
    pub fn parse(data: &[u8]) -> Result<Self, ImageSetError> {
        if data.len() > IMAGE_SET_MAX_SIZE {
            return Err(ImageSetError::TooLarge(data.len()));
        }
        let text = std::str::from_utf8(data).map_err(|_| ImageSetError::NotUtf8)?;
        let Some(body) = text.strip_suffix('\n') else {
            return Err(ImageSetError::Line {
                line: text.lines().count().max(1),
                reason: "missing final newline".into(),
            });
        };

        let mut components = Vec::new();
        for (index, line) in body.split('\n').enumerate() {
            let line_number = index + 1;
            let err = |reason: String| ImageSetError::Line {
                line: line_number,
                reason,
            };
            let fields: Vec<&str> = line.split(' ').collect();
            match (index, fields.as_slice()) {
                (0, [IMAGE_SET_MAGIC, version]) => {
                    if *version != IMAGE_SET_VERSION.to_string() {
                        return Err(err(format!("unsupported image set version {version:?}")));
                    }
                }
                (0, _) => {
                    return Err(err(format!(
                        "expected '{IMAGE_SET_MAGIC} {IMAGE_SET_VERSION}'"
                    )));
                }
                (1, ["algorithm", algorithm]) => {
                    let expected = ObjectID::ALGORITHM.to_string();
                    if *algorithm != expected {
                        return Err(ImageSetError::AlgorithmMismatch {
                            found: algorithm.to_string(),
                            expected,
                        });
                    }
                }
                (1, _) => return Err(err("expected 'algorithm <algorithm>'".into())),
                (_, ["component", order, format, digest, target]) => {
                    if *order != components.len().to_string() {
                        return Err(err(format!(
                            "component order {order:?}, expected {}",
                            components.len()
                        )));
                    }
                    let format = parse_format(format)
                        .ok_or_else(|| err(format!("unknown EROFS format {format:?}")))?;
                    let digest = ObjectID::from_hex(digest)
                        .map_err(|e| err(format!("invalid digest: {e}")))?;
                    components.push(ImageSetComponent {
                        digest,
                        format,
                        target: target.to_string(),
                    });
                }
                _ => {
                    return Err(err(
                        "expected 'component <order> <format> <digest> <target>'".into(),
                    ));
                }
            }
        }

        let set = Self::new(components)?;
        // Rejects everything the strict parse above lets through but that
        // would serialize differently (such as upper-case hex digits), so
        // that a set has exactly one encoding and thus one digest.
        if set.to_bytes() != data {
            return Err(ImageSetError::NotCanonical);
        }
        Ok(set)
    }
}

impl<ObjectID: FsVerityHashValue> fmt::Display for ImageSet<ObjectID> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{IMAGE_SET_MAGIC} {IMAGE_SET_VERSION}")?;
        writeln!(f, "algorithm {}", ObjectID::ALGORITHM)?;
        for (order, component) in self.components.iter().enumerate() {
            writeln!(
                f,
                "component {order} {} {} {}",
                format_name(component.format),
                component.digest.to_hex(),
                component.target
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsverity::{Sha256HashValue, Sha512HashValue};

    fn digest(byte: u8) -> Sha512HashValue {
        Sha512HashValue::from_hex(format!("{byte:02x}").repeat(64)).unwrap()
    }

    fn component(byte: u8, target: &str) -> ImageSetComponent<Sha512HashValue> {
        ImageSetComponent {
            digest: digest(byte),
            format: FormatVersion::V1,
            target: target.into(),
        }
    }

    fn two_components() -> ImageSet<Sha512HashValue> {
        ImageSet::new(vec![
            component(0xaa, "/"),
            component(0xbb, "/usr/lib/modules"),
        ])
        .unwrap()
    }

    #[test]
    fn serialize_and_parse_round_trip() {
        let set = two_components();
        let text = set.to_string();
        let expected = format!(
            "composefs-image-set 1\n\
             algorithm fsverity-sha512-12\n\
             component 0 v1 {} /\n\
             component 1 v1 {} /usr/lib/modules\n",
            "aa".repeat(64),
            "bb".repeat(64)
        );
        assert_eq!(text, expected);
        assert_eq!(ImageSet::parse(text.as_bytes()).unwrap(), set);
        assert_eq!(set.compute_id(), compute_verity(text.as_bytes()));
    }

    #[test]
    fn identity_depends_on_order_and_content() {
        let three = |first, second| {
            ImageSet::new(vec![
                component(0xaa, "/"),
                component(first, "/usr/lib/modules"),
                component(second, "/usr/lib/modules"),
            ])
            .unwrap()
        };
        let other_target =
            ImageSet::new(vec![component(0xaa, "/"), component(0xbb, "/usr/lib")]).unwrap();
        assert_ne!(
            three(0xbb, 0xcc).compute_id(),
            three(0xcc, 0xbb).compute_id()
        );
        assert_ne!(two_components().compute_id(), other_target.compute_id());
    }

    #[test]
    fn new_rejects_invalid_sets() {
        let cases: [(Vec<ImageSetComponent<Sha512HashValue>>, ImageSetError); 9] = [
            (vec![], ImageSetError::Empty),
            (
                vec![component(0xbb, "/usr/lib/modules"), component(0xaa, "/")],
                ImageSetError::RootNotFirst("/usr/lib/modules".into()),
            ),
            (
                vec![component(0xbb, "/usr/lib/modules")],
                ImageSetError::RootNotFirst("/usr/lib/modules".into()),
            ),
            (
                vec![component(0xaa, "/"), component(0xbb, "/")],
                ImageSetError::ExtraRoot(1),
            ),
            (
                vec![component(0xaa, "/"), component(0xaa, "/usr")],
                ImageSetError::DuplicateImage("aa".repeat(64)),
            ),
            (
                vec![component(0xaa, "usr")],
                ImageSetError::InvalidTarget("usr".into()),
            ),
            (
                vec![component(0xaa, "/"), component(0xbb, "/ab/x")],
                ImageSetError::StubNameTarget("/ab/x".into()),
            ),
            (
                vec![
                    component(0xaa, "/"),
                    component(0xbb, "/usr/lib/modules"),
                    component(0xcc, "/0f"),
                ],
                ImageSetError::StubNameTarget("/0f".into()),
            ),
            (
                (0..=IMAGE_SET_MAX_COMPONENTS)
                    .map(|i| ImageSetComponent {
                        digest: compute_verity(&i.to_le_bytes()),
                        format: FormatVersion::V1,
                        target: "/".into(),
                    })
                    .collect(),
                ImageSetError::TooManyComponents(IMAGE_SET_MAX_COMPONENTS + 1),
            ),
        ];
        for (components, expected) in cases {
            assert_eq!(ImageSet::new(components).unwrap_err(), expected);
        }
        // Names that only look a bit like whiteout stubs are fine, and so are
        // stub names further down
        for target in ["/abc", "/a", "/AB", "/0g/x", "/usr/ab", "/usr/00/x"] {
            let set = ImageSet::new(vec![component(0xaa, "/"), component(0xbb, target)]);
            assert!(set.is_ok(), "{target}: {set:?}");
        }
    }

    #[test]
    fn targets() {
        for target in ["/", "/usr", "/usr/lib/modules", "/a.b/c-d"] {
            assert!(valid_target(target), "{target:?} should be valid");
            assert_eq!(validate_target(target), Ok(()));
        }
        assert_eq!(
            validate_target("usr"),
            Err(ImageSetError::InvalidTarget("usr".into()))
        );
        assert_eq!(
            validate_target("/ab/x"),
            Err(ImageSetError::StubNameTarget("/ab/x".into()))
        );
        for target in [
            "",
            "usr",
            "/usr/",
            "//usr",
            "/usr//lib",
            "/./usr",
            "/usr/..",
            "/a b",
            "/a\tb",
        ] {
            assert!(!valid_target(target), "{target:?} should be invalid");
        }
    }

    #[test]
    fn parse_rejects_malformed_manifests() {
        let good = two_components().to_string();
        let aa = "aa".repeat(64);
        let header = "composefs-image-set 1\nalgorithm fsverity-sha512-12\n";
        let cases: Vec<(String, &str)> = vec![
            (good.trim_end().to_string(), "missing final newline"),
            (
                good.replace("image-set 1", "image-set 2"),
                "unsupported image set version",
            ),
            (
                good.replace("composefs-image-set", "composefs-imageset"),
                "expected 'composefs-image-set 1'",
            ),
            (good.replace("sha512", "sha256"), "uses fsverity-sha256-12"),
            (
                good.replace("component 1", "component 2"),
                "component order \"2\", expected 1",
            ),
            (good.replace(" v1 ", " v9 "), "unknown EROFS format"),
            (good.replace(&aa, &aa[..64]), "invalid digest"),
            (
                good.replace(" /usr/lib/modules", " /usr/lib/modules extra"),
                "expected 'component",
            ),
            (
                good.replace(" /usr/lib/modules", "  /usr/lib/modules"),
                "expected 'component",
            ),
            (
                good.replace(" /usr/lib/modules", " usr/lib/modules"),
                "invalid image set target",
            ),
            (
                good.replace(" /\n", " /usr\n"),
                "first component of an image set must have target /",
            ),
            (format!("{header}\n"), "expected 'component"),
            (header.to_string(), "no components"),
            (
                good.replace(&aa, &aa.to_uppercase()),
                "not in canonical form",
            ),
            (good.replace('\n', "\r\n"), "unsupported image set version"),
            (
                good.replace(" /usr/lib/modules", " /ff/lib"),
                "named like the whiteout stubs",
            ),
        ];
        for (text, expected) in cases {
            let err = ImageSet::<Sha512HashValue>::parse(text.as_bytes()).unwrap_err();
            assert!(
                err.to_string().contains(expected),
                "{text:?}: expected {expected:?} in {err}"
            );
        }
        assert_eq!(
            ImageSet::<Sha512HashValue>::parse(&[0xff, b'\n']).unwrap_err(),
            ImageSetError::NotUtf8
        );
        assert_eq!(
            ImageSet::<Sha512HashValue>::parse(&vec![b'a'; IMAGE_SET_MAX_SIZE + 1]).unwrap_err(),
            ImageSetError::TooLarge(IMAGE_SET_MAX_SIZE + 1)
        );
    }

    #[test]
    fn algorithm_follows_object_id() {
        let digest = Sha256HashValue::from_hex("cc".repeat(32)).unwrap();
        let set = ImageSet::new(vec![ImageSetComponent {
            digest,
            format: FormatVersion::V2,
            target: "/".into(),
        }])
        .unwrap();
        let text = set.to_string();
        assert!(text.contains("algorithm fsverity-sha256-12\n"));
        assert!(text.contains(" v2 "));
        assert_eq!(ImageSet::parse(text.as_bytes()).unwrap(), set);
        assert!(matches!(
            ImageSet::<Sha512HashValue>::parse(text.as_bytes()).unwrap_err(),
            ImageSetError::AlgorithmMismatch { .. }
        ));
    }

    /// A V1 or V2 image of the filesystem `dumpfile` describes, under a root
    /// directory entry.
    fn image_of(dumpfile: &str, version: FormatVersion) -> Box<[u8]> {
        let dumpfile = format!("/ 0 40755 2 0 0 0 1000.0 - - -\n{dumpfile}");
        let fs = crate::dumpfile::dumpfile_to_validated_filesystem::<Sha512HashValue>(&dumpfile)
            .unwrap();
        crate::erofs::writer::mkfs_erofs_versioned(&fs, version)
    }

    const MODULES: &str = "\
/usr 0 40755 2 0 0 0 1000.0 - - -
/usr/lib 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules/6.12 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules/6.12/a.ko 3 100644 1 0 0 0 1000.0 - abc -
";

    #[test]
    fn component_contents() {
        let extra = |line: &str| format!("{MODULES}{line}\n");
        let cases = [
            (MODULES.to_string(), "/usr/lib/modules", None),
            (MODULES.to_string(), "/usr/lib", None),
            (MODULES.to_string(), "/", None),
            (extra("/etc 0 40755 2 0 0 0 1000.0 - - -"), "/", None),
            (
                extra("/etc 0 40755 2 0 0 0 1000.0 - - -"),
                "/usr/lib/modules",
                Some("contains /etc outside its target"),
            ),
            (
                extra("/usr/bin 0 40755 2 0 0 0 1000.0 - - -"),
                "/usr/lib/modules",
                Some("contains /usr/bin outside its target"),
            ),
            (
                extra("/usr/lib/os-release 1 100644 1 0 0 0 1000.0 - x -"),
                "/usr/lib/modules",
                Some("contains /usr/lib/os-release outside its target"),
            ),
            // A file whose name looks like a whiteout stub, but isn't one
            (
                extra("/ab 1 100644 1 0 0 0 1000.0 - x -"),
                "/usr/lib/modules",
                Some("contains /ab outside its target"),
            ),
            (
                MODULES.to_string(),
                "/usr/lib/firmware",
                Some("contains /usr/lib/modules outside its target"),
            ),
            (
                "/usr 0 40755 2 0 0 0 1000.0 - - -\n/usr/lib 0 40755 2 0 0 0 1000.0 - - -\n"
                    .to_string(),
                "/usr/lib/modules",
                Some("has no /usr/lib/modules"),
            ),
            (
                MODULES.to_string(),
                "/usr/lib/modules/6.12/a.ko",
                Some("/usr/lib/modules/6.12/a.ko is not a directory"),
            ),
        ];
        for version in [FormatVersion::V1, FormatVersion::V2] {
            for (dumpfile, target, expected) in &cases {
                let image = image_of(dumpfile, version);
                let component = ImageSetComponent {
                    digest: compute_verity::<Sha512HashValue>(&image),
                    format: version,
                    target: target.to_string(),
                };
                let result = component.check_contents(&image);
                match expected {
                    None => assert_eq!(result, Ok(()), "{version:?} {target}"),
                    Some(expected) => {
                        let err = result.unwrap_err().to_string();
                        assert!(err.contains(expected), "{version:?} {target}: {err}");
                    }
                }
            }
        }

        let component = component(0xbb, "/usr/lib/modules");
        let err = component.check_contents(b"not an image").unwrap_err();
        assert!(err.to_string().contains("reading image"), "{err}");
    }

    /// A component whose image names an ancestor of its target twice, once
    /// for a directory with files outside the target.  The kernel's lookup
    /// may well resolve that one, so the check must not pick the other.
    #[test]
    fn component_contents_duplicate_entry() {
        // Each image is written with a decoy name next to the real one, then
        // patched to repeat the real name: the writer itself never writes
        // duplicates.
        let cases = [
            (
                "\
/usr 0 40755 2 0 0 0 1000.0 - - -
/usr/bin 0 40755 2 0 0 0 1000.0 - - -
/usr/bin/evil 4 100644 1 0 0 0 1000.0 - evil -
/usr/lib 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules 0 40755 2 0 0 0 1000.0 - - -
/usz 0 40755 2 0 0 0 1000.0 - - -
/usz/lib 0 40755 2 0 0 0 1000.0 - - -
/usz/lib/modules 0 40755 2 0 0 0 1000.0 - - -
",
                "usz",
                "usr",
                "has two entries /usr",
            ),
            (
                "\
/usr 0 40755 2 0 0 0 1000.0 - - -
/usr/lib 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/evil 4 100644 1 0 0 0 1000.0 - evil -
/usr/lib/modules 0 40755 2 0 0 0 1000.0 - - -
/usr/lic 0 40755 2 0 0 0 1000.0 - - -
/usr/lic/modules 0 40755 2 0 0 0 1000.0 - - -
",
                "lic",
                "lib",
                "has two entries /usr/lib",
            ),
        ];
        for version in [FormatVersion::V1, FormatVersion::V2] {
            for (dumpfile, decoy, real, expected) in cases {
                let image = image_of(dumpfile, version);
                let component = ImageSetComponent {
                    digest: digest(0xcc),
                    format: version,
                    target: "/usr/lib/modules".into(),
                };
                assert!(component.check_contents(&image).is_err(), "{version:?}");
                let mut patched = image.to_vec();
                let at = patched
                    .windows(decoy.len())
                    .position(|w| w == decoy.as_bytes())
                    .expect("decoy name in image");
                assert!(
                    !patched[at + 1..]
                        .windows(decoy.len())
                        .any(|w| w == decoy.as_bytes()),
                    "decoy name appears once"
                );
                patched[at..at + decoy.len()].copy_from_slice(real.as_bytes());
                let err = component.check_contents(&patched).unwrap_err().to_string();
                assert!(err.contains(expected), "{version:?}: {err}");
            }
        }
    }

    /// A component with an overlayfs xattr on its target, one of its ancestors
    /// or a directory below it, which with `redirect_dir=on` could move or
    /// hide files of the components below it.
    #[test]
    fn component_contents_overlay_xattrs() {
        // The writer escapes trusted.overlay.* xattrs, so unescaped ones are
        // written under a decoy name, then patched.
        const DECOY: &[u8] = b"ovxrlay";
        let modules =
            format!("{MODULES}/usr/lib/modules/6.12/kernel 0 40755 2 0 0 0 1000.0 - - -\n");
        let with_xattr = |path: &str, xattr: &str| -> String {
            modules
                .lines()
                .map(|line| match line.split_once(' ') {
                    Some((p, _)) if p == path => format!("{line} {xattr}\n"),
                    _ => format!("{line}\n"),
                })
                .collect()
        };
        let refused =
            |path: &str, name: &str| Some(format!("{path} has the overlayfs xattr {name}"));
        let cases = [
            (
                "/usr",
                "trusted.ovxrlay.redirect=/usr/lib/firmware",
                refused("/usr", "trusted.overlay.redirect"),
            ),
            (
                "/usr/lib",
                "trusted.ovxrlay.opaque=y",
                refused("/usr/lib", "trusted.overlay.opaque"),
            ),
            (
                "/usr/lib/modules",
                "trusted.ovxrlay.redirect=/usr/lib/firmware",
                refused("/usr/lib/modules", "trusted.overlay.redirect"),
            ),
            (
                "/usr/lib/modules",
                "trusted.ovxrlay.metacopy=",
                refused("/usr/lib/modules", "trusted.overlay.metacopy"),
            ),
            // Inside the target: an absolute redirect restarts the lookup at
            // the root of the layers below
            (
                "/usr/lib/modules/6.12",
                "trusted.ovxrlay.redirect=/usr/lib/firmware",
                refused("/usr/lib/modules/6.12", "trusted.overlay.redirect"),
            ),
            (
                "/usr/lib/modules/6.12/kernel",
                "trusted.ovxrlay.opaque=y",
                refused("/usr/lib/modules/6.12/kernel", "trusted.overlay.opaque"),
            ),
            // What the writer puts on a regular file that names an object
            (
                "/usr/lib/modules/6.12/a.ko",
                "trusted.ovxrlay.metacopy= trusted.ovxrlay.redirect=/aa/bb",
                None,
            ),
            // Image xattrs as the writer escapes them, and user ones, which the
            // mount doesn't act on
            ("/usr", "trusted.overlay.opaque=y", None),
            ("/usr/lib", "trusted.overlay.redirect=/etc", None),
            ("/usr", "user.overlay.opaque=y", None),
            ("/usr", "user.foo=bar", None),
        ];
        for version in [FormatVersion::V1, FormatVersion::V2] {
            for (path, xattr, expected) in &cases {
                let mut image = image_of(&with_xattr(path, xattr), version).to_vec();
                let mut patched = 0;
                while let Some(at) = image.windows(DECOY.len()).position(|w| w == DECOY) {
                    image[at..at + DECOY.len()].copy_from_slice(b"overlay");
                    patched += 1;
                }
                assert_eq!(patched > 0, xattr.contains("ovxrlay"), "{xattr}");
                let component = ImageSetComponent {
                    digest: digest(0xcc),
                    format: version,
                    target: "/usr/lib/modules".into(),
                };
                let result = component.check_contents(&image);
                match expected {
                    None => assert_eq!(result, Ok(()), "{version:?} {path} {xattr}"),
                    Some(expected) => {
                        let err = result.unwrap_err().to_string();
                        assert!(err.contains(expected), "{version:?} {path} {xattr}: {err}");
                    }
                }
            }
        }
    }

    /// A component whose target holds a chain of directories as deep as the
    /// image reader allows, or one deeper, and one with an overlayfs xattr at
    /// the end of the chain.
    #[test]
    fn component_contents_deep_tree() {
        use crate::{erofs::writer::ValidatedFileSystem, generic_tree::Stat, tree};
        use std::ffi::OsStr;

        const DECOY: &[u8] = b"ovxrlay";
        // A chain /usr/d/d/..., its last directory `depth` levels deep
        fn chain(depth: usize, version: FormatVersion, xattr_at_end: bool) -> Vec<u8> {
            let stat = |xattr: bool| {
                let mut stat = Stat {
                    st_mode: 0o755,
                    st_uid: 0,
                    st_gid: 0,
                    st_mtim_sec: 1000,
                    st_mtim_nsec: 0,
                    xattrs: Default::default(),
                };
                if xattr {
                    let name = OsStr::new("trusted.ovxrlay.opaque");
                    stat.xattrs.insert(name.into(), Box::from(&b"y"[..]));
                }
                stat
            };
            let mut dir = tree::Directory::new(stat(xattr_at_end));
            for _ in 1..depth {
                let mut parent = tree::Directory::new(stat(false));
                parent.insert(OsStr::new("d"), tree::Inode::Directory(Box::new(dir)));
                dir = parent;
            }
            let mut fs = tree::FileSystem::<Sha512HashValue>::new(stat(false));
            fs.root
                .insert(OsStr::new("usr"), tree::Inode::Directory(Box::new(dir)));
            let fs = ValidatedFileSystem::new(fs).unwrap();
            let mut image = crate::erofs::writer::mkfs_erofs_versioned(&fs, version).to_vec();
            if let Some(at) = image.windows(DECOY.len()).position(|w| w == DECOY) {
                image[at..at + DECOY.len()].copy_from_slice(b"overlay");
            }
            image
        }

        // Building and dropping such a tree recurses once per level
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                let too_deep =
                    format!("nests directories {MAX_DIRECTORY_DEPTH} or more levels deep");
                let deepest = format!("/usr{}", "/d".repeat(MAX_DIRECTORY_DEPTH - 2));
                let cases = [
                    (MAX_DIRECTORY_DEPTH - 1, false, None),
                    (MAX_DIRECTORY_DEPTH, false, Some(too_deep.clone())),
                    (
                        MAX_DIRECTORY_DEPTH - 1,
                        true,
                        Some(format!(
                            "{deepest} has the overlayfs xattr trusted.overlay.opaque"
                        )),
                    ),
                ];
                let component = |version, target: &str| ImageSetComponent {
                    digest: digest(0xcc),
                    format: version,
                    target: target.into(),
                };
                for version in [FormatVersion::V1, FormatVersion::V2] {
                    for (depth, xattr, expected) in &cases {
                        let image = chain(*depth, version, *xattr);
                        let result = component(version, "/usr").check_contents(&image);
                        match expected {
                            None => assert_eq!(result, Ok(()), "{version:?} {depth}"),
                            Some(expected) => {
                                let err = result.unwrap_err().to_string();
                                assert!(err.ends_with(expected), "{version:?} {depth}: {err}");
                            }
                        }
                    }
                    // A target that deep is refused without looking further
                    let target = "/d".repeat(MAX_DIRECTORY_DEPTH);
                    let image = chain(1, version, false);
                    let err = component(version, &target)
                        .check_contents(&image)
                        .unwrap_err();
                    assert!(err.to_string().ends_with(&too_deep), "{version:?}: {err}");
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// A component that links a directory seen before, deeper down or back to
    /// an ancestor: since its subtree would then be walked at only one of its
    /// depths, any directory reached twice is refused.
    #[test]
    fn component_contents_linked_directory() {
        use crate::erofs::{
            format::{DirectoryEntryHeader, FileType, FileTypeField},
            reader::Image,
        };

        let dumpfile = format!(
            "{MODULES}\
/usr/lib/modules/6.12/kernel 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules/6.12/kernel/drivers 0 40755 2 0 0 0 1000.0 - - -
/usr/lib/modules/6.12/kernel/drivers/link 1 100644 1 0 0 0 1000.0 - x -
/usr/lib/modules/6.13 0 40755 2 0 0 0 1000.0 - - -
"
        );
        const LINK: &str = "/usr/lib/modules/6.12/kernel/drivers/link";
        let nid_of = |image: &[u8], path: &str| {
            let image = Image::open(image).unwrap();
            let mut nid = image.sb.root_nid.get() as u64;
            for name in path.split('/').filter(|name| !name.is_empty()) {
                nid = image
                    .directory_entries(nid)
                    .unwrap()
                    .into_iter()
                    .find_map(|(entry, entry_nid)| (entry == name.as_bytes()).then_some(entry_nid))
                    .unwrap_or_else(|| panic!("no {path}"));
            }
            nid
        };
        for version in [FormatVersion::V1, FormatVersion::V2] {
            let image = image_of(&dumpfile, version);
            let component = ImageSetComponent {
                digest: digest(0xcc),
                format: version,
                target: "/usr/lib/modules".into(),
            };
            assert_eq!(component.check_contents(&image), Ok(()), "{version:?}");
            // The directory entry of the file, the only place its inode
            // number is written
            let link_nid = nid_of(&image, LINK);
            let regular_file = FileTypeField::from(FileType::RegularFile);
            let entries: Vec<_> = (0..image.len() - size_of::<DirectoryEntryHeader>())
                .filter(|&at| {
                    DirectoryEntryHeader::ref_from_prefix(&image[at..]).is_ok_and(|(header, _)| {
                        header.inode_offset.get() == link_nid
                            && header.file_type == regular_file
                            && header.reserved == 0
                    })
                })
                .collect();
            assert_eq!(entries.len(), 1, "{version:?}: entries of {LINK}");
            // An earlier sibling, ancestors inside and outside the target, the
            // target itself and the root
            for linked in [
                "/usr/lib/modules/6.13",
                "/usr/lib/modules/6.12/kernel",
                "/usr/lib/modules/6.12",
                "/usr/lib/modules",
                "/usr",
                "/",
            ] {
                let mut patched = image.to_vec();
                let (header, _) =
                    DirectoryEntryHeader::mut_from_prefix(&mut patched[entries[0]..]).unwrap();
                header.inode_offset.set(nid_of(&image, linked));
                header.file_type = FileType::Directory.into();
                let err = component.check_contents(&patched).unwrap_err().to_string();
                assert!(
                    err.ends_with(&format!("links {LINK} to a directory seen before")),
                    "{version:?} {linked}: {err}"
                );
            }
        }
    }

    #[test]
    fn component_format() {
        for (written, listed, ok) in [
            (FormatVersion::V1, FormatVersion::V1, true),
            (FormatVersion::V2, FormatVersion::V2, true),
            (FormatVersion::V1, FormatVersion::V2, false),
            (FormatVersion::V2, FormatVersion::V1, false),
            (FormatVersion::V0, FormatVersion::V0, true),
            (FormatVersion::V2, FormatVersion::V0, false),
        ] {
            let image = image_of(MODULES, written);
            let component = ImageSetComponent {
                digest: compute_verity::<Sha512HashValue>(&image),
                format: listed,
                target: "/usr/lib/modules".into(),
            };
            let header = &image[..COMPONENT_HEADER_SIZE];
            let result = component.check_format(header);
            assert_eq!(result.is_ok(), ok, "{written:?} as {listed:?}: {result:?}");
            if !ok {
                let err = result.unwrap_err().to_string();
                assert!(
                    err.contains("but its header has composefs version"),
                    "{err}"
                );
            }
        }
        let component = component(0xbb, "/");
        for (header, expected) in [
            (&b"short"[..], "too short"),
            (&[0u8; 64][..], "not a composefs image"),
        ] {
            let err = component.check_format(header).unwrap_err().to_string();
            assert!(err.contains(expected), "{err}");
        }
    }
}
