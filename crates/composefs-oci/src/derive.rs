//! Derive a new image from one in the repository by adding layers to it.
//!
//! This is the repository-local equivalent of a `FROM base` + `RUN ...`
//! container build: the derived image's config is the base's with the new
//! layers' diff_ids and history entries appended, and its manifest lists the
//! base's layer descriptors followed by the new ones. Nothing of the base is
//! copied; the derived manifest and config reference the base's layer
//! splitstreams, which therefore stay alive as long as the derived image does.
//! The base's own manifest and config do not: keep a ref to the base too if
//! it is needed later (say, to derive from it again).
//!
//! Layers come from [`crate::overlay_upper::import_upper`] or
//! [`crate::import_layer`], and are recorded uncompressed
//! (`application/vnd.oci.image.layer.v1.tar`), so that their digest is their
//! diff_id.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use containers_image_proxy::oci_spec::image::{
    Descriptor, DescriptorBuilder, HistoryBuilder, ImageManifest, MediaType,
};
use fn_error_context::context;

use composefs::fsverity::FsVerityHashValue;
use composefs::repository::Repository;

use crate::oci_image::{OciImage, has_manifest, rewrite_manifest, tag_image};
use crate::{
    ImportStats, OciDigest, PullResult, ensure_oci_composefs_erofs, hash_sha256, write_config_raw,
};

/// Serializes `value` with every object's keys sorted, so that the same
/// image always gives the same bytes and digest: oci-spec keeps labels and
/// annotations in `HashMap`s, whose order changes from run to run.
fn canonical_json(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    use serde_json::Value;
    fn sorted(value: Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut entries: Vec<_> = map.into_iter().collect();
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                Value::Object(entries.into_iter().map(|(k, v)| (k, sorted(v))).collect())
            }
            Value::Array(items) => Value::Array(items.into_iter().map(sorted).collect()),
            other => other,
        }
    }
    Ok(serde_json::to_vec(&sorted(serde_json::to_value(value)?))?)
}

/// A layer to append to a base image.
#[derive(Debug, Clone)]
pub struct DerivedLayer<ObjectID> {
    /// The layer's diff_id, also used as its (uncompressed) blob digest.
    pub diff_id: OciDigest,
    /// The fs-verity digest of the layer's splitstream in the repository.
    pub verity: ObjectID,
    /// The size of the uncompressed tar stream.
    pub size: u64,
    /// The history entry's `created_by`, e.g. the command that made it.
    pub created_by: String,
}

/// Writes an image consisting of the image `base_manifest_digest` plus
/// `layers`, with `labels` added to its config, links its EROFS image as a
/// pull does, and tags it `reference` if given.
///
/// `created`, an RFC 3339 timestamp, is recorded as the config's `created`
/// and in the new history entries; without it the base's `created` stays.
/// The base manifest's annotations are not carried over, since they describe
/// the base (and may claim a seal that the derived image doesn't have).
#[context("Deriving an image from {base_manifest_digest}")]
pub fn derive_image<ObjectID: FsVerityHashValue>(
    repo: &Arc<Repository<ObjectID>>,
    base_manifest_digest: &OciDigest,
    layers: &[DerivedLayer<ObjectID>],
    labels: &[(&str, &str)],
    created: Option<&str>,
    reference: Option<&str>,
) -> Result<PullResult<ObjectID>> {
    ensure!(!layers.is_empty(), "No layers to add");
    let base = OciImage::open(repo, base_manifest_digest, None)?;
    let mut config = base
        .config()
        .context("The base is an artifact, not a container image")?
        .clone();

    let mut refs: HashMap<Box<str>, ObjectID> = base.layer_refs().clone();
    let mut diff_ids = config.rootfs().diff_ids().clone();
    let mut history = config.history().clone().unwrap_or_default();
    let mut descriptors: Vec<Descriptor> = base.manifest().layers().clone();
    for layer in layers {
        let diff_id = layer.diff_id.to_string();
        ensure!(
            !diff_ids.contains(&diff_id),
            "Layer {diff_id} is already part of the image"
        );
        diff_ids.push(diff_id.clone());
        refs.insert(diff_id.into(), layer.verity.clone());
        let mut entry = HistoryBuilder::default().created_by(&layer.created_by);
        if let Some(created) = created {
            entry = entry.created(created);
        }
        history.push(entry.build()?);
        descriptors.push(
            DescriptorBuilder::default()
                .media_type(MediaType::ImageLayer)
                .digest(layer.diff_id.clone())
                .size(layer.size)
                .build()?,
        );
    }
    config.rootfs_mut().set_diff_ids(diff_ids);
    config.set_history(Some(history));
    if let Some(created) = created {
        config.set_created(Some(created.to_owned()));
    }
    if !labels.is_empty() {
        let mut inner = config.config().clone().unwrap_or_default();
        let mut all = inner.labels().clone().unwrap_or_default();
        all.extend(
            labels
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned())),
        );
        inner.set_labels(Some(all));
        config.set_config(Some(inner));
    }

    let config_json = canonical_json(&config)?;
    let config_digest = hash_sha256(&config_json);
    let manifest: ImageManifest = {
        let mut m = base.manifest().clone();
        m.set_media_type(Some(MediaType::ImageManifest));
        m.set_config(
            DescriptorBuilder::default()
                .media_type(MediaType::ImageConfig)
                .digest(config_digest.clone())
                .size(config_json.len() as u64)
                .build()?,
        );
        m.set_layers(descriptors);
        m.set_annotations(None);
        m.set_subject(None);
        m.set_artifact_type(None);
        m
    };
    let manifest_json = canonical_json(&manifest)?;
    let manifest_digest = hash_sha256(&manifest_json);

    // Deriving the same image again changes nothing: rewriting its config
    // would drop the boot images linked to it since.
    if has_manifest(repo, &manifest_digest)?.is_none() {
        let (_, config_verity) = write_config_raw(
            repo,
            &config_json,
            refs.clone(),
            None,
            None,
            &HashMap::new(),
        )?;
        let layer_verities: Vec<(String, ObjectID)> = config
            .rootfs()
            .diff_ids()
            .iter()
            .map(|d| (d.clone(), refs[d.as_str()].clone()))
            .collect();
        let (_, manifest_verity) = rewrite_manifest(
            repo,
            &manifest_json,
            &manifest_digest,
            &config_verity,
            &layer_verities,
            None,
        )?;
        // Like a pull, link the image's EROFS (which rewrites the config and
        // manifest splitstreams).
        ensure_oci_composefs_erofs(repo, &manifest_digest, Some(&manifest_verity), None, None)?
            .context("The derived image produced no EROFS image")?;
    }
    if let Some(reference) = reference {
        tag_image(repo, &manifest_digest, reference)?;
    }
    let img = OciImage::open(repo, &manifest_digest, None)?;
    Ok(PullResult {
        manifest_digest,
        manifest_verity: img.manifest_verity().clone(),
        config_digest,
        config_verity: img.config_verity().clone(),
        stats: ImportStats::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;
    use composefs::fsverity::Sha256HashValue;
    use composefs::test::TestRepo;

    #[tokio::test]
    async fn derive_adds_a_layer() {
        let test_repo = TestRepo::<Sha256HashValue>::new();
        let repo = &test_repo.repo;
        let base = test_util::create_base_image(repo, Some("base:v1")).await;

        let upper = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(upper.path().join("usr/bin")).unwrap();
        std::fs::write(upper.path().join("usr/bin/added"), vec![7u8; 8192]).unwrap();
        // Paths and link targets over tar's 100 bytes go into PAX records.
        let long = format!("usr/share/{}", "d".repeat(120));
        std::fs::create_dir_all(upper.path().join(&long)).unwrap();
        std::fs::write(upper.path().join(&long).join("f"), b"long").unwrap();
        std::os::unix::fs::symlink(format!("/{long}/f"), upper.path().join("usr/bin/long-link"))
            .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let imported =
            crate::overlay_upper::import_upper(repo, upper.path(), &Default::default(), tmp.path())
                .await
                .unwrap();
        assert_eq!(imported.upper.files, 2);

        let layer = DerivedLayer {
            diff_id: imported.diff_id.clone(),
            verity: imported.verity.clone(),
            size: imported.size,
            created_by: "test: add /usr/bin/added".into(),
        };
        let labels: Vec<(String, String)> = (0..16)
            .map(|i| (format!("org.example.l{i}"), format!("v{i}")))
            .chain([("org.example.base".into(), base.manifest_digest.to_string())])
            .collect();
        let labels: Vec<(&str, &str)> = labels
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let derive = |reference| {
            derive_image(
                repo,
                &base.manifest_digest,
                std::slice::from_ref(&layer),
                &labels,
                Some("2026-10-30T00:00:00Z"),
                reference,
            )
            .unwrap()
        };
        let derived = derive(Some("derived:v1"));
        let base_img = OciImage::open(repo, &base.manifest_digest, None).unwrap();
        let img = OciImage::open(repo, &derived.manifest_digest, None).unwrap();
        assert!(img.image_ref(repo.erofs_version()).is_some());
        assert_eq!(img.manifest_verity(), &derived.manifest_verity);
        let mut expected = base_img.layer_diff_ids();
        expected.push(imported.diff_id.as_ref());
        assert_eq!(img.layer_diff_ids(), expected);
        assert_eq!(img.layer_descriptors().len(), expected.len());
        assert_eq!(
            img.label("org.example.base"),
            Some(base.manifest_digest.as_ref())
        );
        let history = img.config().unwrap().history().clone().unwrap();
        assert_eq!(
            history.last().unwrap().created_by().as_deref(),
            Some("test: add /usr/bin/added")
        );
        assert!(img.manifest().annotations().is_none());

        // The derived tree has the base's files plus the new one.
        let fs = crate::image::create_filesystem(
            &**repo,
            img.config_digest(),
            None,
            &Default::default(),
        )
        .unwrap();
        let mut dump = Vec::new();
        composefs::dumpfile::write_dumpfile(&mut dump, &fs).unwrap();
        let dump = String::from_utf8(dump).unwrap();
        assert!(dump.contains("/usr/bin/added 8192 "), "{dump}");
        assert!(dump.contains("/usr/bin/busybox "), "{dump}");
        assert!(dump.contains(&format!("/{long}/f 4 ")), "{dump}");
        assert!(
            dump.contains(&format!(" /{long}/f ")),
            "long symlink target: {dump}"
        );

        // The same inputs give the same image, whatever the HashMap order,
        // and deriving it again leaves the stored image (and whatever is
        // linked to it since) alone.
        let stored = OciImage::open(repo, &derived.manifest_digest, None)
            .unwrap()
            .manifest_verity()
            .clone();
        for _ in 0..4 {
            let again = derive(None);
            assert_eq!(again.manifest_digest, derived.manifest_digest);
            assert_eq!(again.manifest_verity, stored);
        }
        assert!(
            derive_image(
                repo,
                &base.manifest_digest,
                std::slice::from_ref(&layer),
                &labels,
                Some("2026-10-30T00:00:00Z"),
                Some("@bad"),
            )
            .is_err()
        );

        // Adding the same layer twice is refused.
        let err =
            derive_image(repo, &derived.manifest_digest, &[layer], &[], None, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("already part of the image"),
            "{err:#}"
        );
    }
}
