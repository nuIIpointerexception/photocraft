//! Optional real-file corpus: `corpus/psd/**/*.{psd,psb}` at the workspace
//! root (gitignored; see `corpus/psd/SOURCES.md`). Skips silently if absent.
//!
//! For each file: parse → document → flatten, compared with the file's own
//! merged composite (Photoshop's rendering) as the oracle; then document →
//! PSD → document → flatten, compared with the first flatten (our own export
//! must not change what the document looks like). Prints a per-file table.
//!
//! Without `PHOTOCRAFT_CORPUS` the comparisons are only reported: differences
//! are expected where features are not yet rendered (effects, text engine,
//! smart filters). With `PHOTOCRAFT_CORPUS` set (to the corpus directory, or
//! to `1` for the default location) the run asserts that the oracle pass
//! count and the export round-trip count do not fall below their floors.
//! Raise the floors when they improve; never lower them.
//! Set `PHOTOCRAFT_CORPUS_STRICT=1` to fail on import/export errors.

mod common;

use std::path::{Path, PathBuf};

use photocraft_io::*;
use photocraft_psd::PsdFile;

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("psd") || e.eq_ignore_ascii_case("psb")) {
            out.push(p);
        }
    }
}

const PASS_TOL: f32 = 2.0 / 255.0;
/// Export → re-import must render within one 8-bit step of the imported document.
const ROUNDTRIP_TOL: f32 = 1.0 / 255.0 + 1e-5;
/// Files whose flatten matches Photoshop's merged image (`corpus/psd`, 170 files).
const PASS_FLOOR: usize = 120;
/// Files whose export → re-import renders the same as the import.
const ROUNDTRIP_FLOOR: usize = 169;

/// Dissolve block size and tolerance (see [`dissolve_matches`]).
const DISSOLVE_BLOCK: i32 = 16;
const DISSOLVE_TOL: f32 = 0.1;

/// Oracle for documents with Dissolve layers. Photoshop's dissolve decides each pixel with a
/// position-only pseudo-random threshold (observed on psd-tools dissolve.psd: where dissolved
/// layers of equal opacity overlap, a pixel shows the top layer or nothing, never a lower one),
/// but its generator is not public and can't be recovered from one binary pattern; ours uses
/// another hash with the same rule. So outside the dissolve layers' bounds pixels must match
/// as usual (≤ 2/255), and inside them the premultiplied colour averaged over 16 × 16 blocks
/// (density and colour) must agree within 0.1 (binomial noise of a 50 % dissolve is about
/// 0.044 per block difference).
fn dissolve_matches(doc: &photocraft_doc::Document, ours: &[[f32; 4]], ps: &[[f32; 4]]) -> bool {
    let canvas = doc.bounds();
    let regions: Vec<_> = doc
        .walk()
        .into_iter()
        .filter(|(_, _, l)| l.visible && l.blend == photocraft_color::BlendMode::Dissolve)
        .map(|(_, _, l)| photocraft_compose::layer_bounds(l, canvas).intersect(&canvas))
        .filter(|r| !r.is_empty())
        .collect();
    let (w, h) = (canvas.width() as i32, canvas.height() as i32);
    if regions.is_empty() || ours.len() != ps.len() || ours.len() != (w as usize) * (h as usize) {
        return false;
    }
    let pm = |p: &[f32; 4], c: usize| if c < 3 { p[c] * p[3] } else { p[3] };
    let inside = |x: i32, y: i32| regions.iter().any(|r| r.contains(x + canvas.x0, y + canvas.y0));
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if !inside(x, y) && (0..4).any(|c| (pm(&ours[i], c) - pm(&ps[i], c)).abs() > PASS_TOL) {
                return false;
            }
        }
    }
    for by in (0..h).step_by(DISSOLVE_BLOCK as usize) {
        for bx in (0..w).step_by(DISSOLVE_BLOCK as usize) {
            let (x1, y1) = ((bx + DISSOLVE_BLOCK).min(w), (by + DISSOLVE_BLOCK).min(h));
            let n = ((x1 - bx) * (y1 - by)) as f32;
            for c in 0..4 {
                let (mut a, mut b) = (0.0, 0.0);
                for y in by..y1 {
                    for x in bx..x1 {
                        let i = (y * w + x) as usize;
                        a += pm(&ours[i], c);
                        b += pm(&ps[i], c);
                    }
                }
                if ((a - b) / n).abs() > DISSOLVE_TOL {
                    return false;
                }
            }
        }
    }
    true
}

#[test]
fn corpus_import_flatten_oracle() {
    let env = std::env::var_os("PHOTOCRAFT_CORPUS").filter(|v| !v.is_empty());
    let root = match &env {
        Some(v) if v != "1" => PathBuf::from(v),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/psd"),
    };
    if !root.is_dir() {
        assert!(env.is_none(), "PHOTOCRAFT_CORPUS is set but {} is not a directory", root.display());
        return;
    }
    let mut files = Vec::new();
    collect(&root, &mut files);
    files.sort();
    let (mut pass, mut diff, mut skipped, mut errors) = (0, 0, 0, 0);
    let (mut rt_same, mut rt_diff) = (0, Vec::new());
    eprintln!("{:<60} {:>6} {:>9} {:>8}  status", "file", "layers", "max_err", "bad_px%");
    for p in &files {
        let name = p.strip_prefix(&root).unwrap_or(p).display().to_string();
        let bytes = std::fs::read(p).unwrap_or_default();
        let file = match PsdFile::from_bytes(&bytes) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{name:<60} {:>6} {:>9} {:>8}  PARSE-ERROR {e}", "-", "-", "-");
                errors += 1;
                continue;
            }
        };
        let imp = match import(&name, &bytes) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("{name:<60} IMPORT-ERROR {e}");
                errors += 1;
                continue;
            }
        };
        let doc = &imp.document;
        // Export must always succeed, re-parse and re-import.
        let reimported = match export(doc, "x.psd", &ExportOptions::default()) {
            Ok(r) => {
                if let Err(e) = PsdFile::from_bytes(&r.bytes) {
                    eprintln!("{name:<60} REEXPORT-PARSE-ERROR {e}");
                    errors += 1;
                    continue;
                }
                match import(&name, &r.bytes) {
                    Ok(i) => i.document,
                    Err(e) => {
                        eprintln!("{name:<60} REIMPORT-ERROR {e}");
                        errors += 1;
                        continue;
                    }
                }
            }
            Err(e) => {
                eprintln!("{name:<60} EXPORT-ERROR {e}");
                errors += 1;
                continue;
            }
        };
        let ours = photocraft_compose::flatten(doc).px;
        // Our own export must not change how the document renders.
        let again = photocraft_compose::flatten(&reimported).px;
        let rt = if again.len() == ours.len() { common::max_diff(&ours, &again) } else { f32::INFINITY };
        if rt <= ROUNDTRIP_TOL {
            rt_same += 1;
        } else {
            rt_diff.push(format!("{name} ({rt:.4})"));
        }
        let layers = doc.layer_count();
        if file.has_real_merged_data() == Some(false) || file.layers().is_empty() {
            eprintln!("{name:<60} {layers:>6} {:>9} {:>8}  SKIP (no layers or no real composite)", "-", "-");
            skipped += 1;
            continue;
        }
        let Ok(merged) = merged_composite(&file) else {
            eprintln!("{name:<60} {layers:>6} {:>9} {:>8}  SKIP (merged not decodable)", "-", "-");
            skipped += 1;
            continue;
        };
        let m = common::max_diff(&ours, &merged);
        let bad = ours.iter().zip(&merged).filter(|(a, b)| (0..4).any(|c| (a[c] * a[3] - b[c] * b[3]).abs() > PASS_TOL)).count();
        let pct = 100.0 * bad as f32 / ours.len().max(1) as f32;
        let status = if m <= PASS_TOL {
            pass += 1;
            "PASS"
        } else if dissolve_matches(doc, &ours, &merged) {
            pass += 1;
            "PASS (dissolve metric)"
        } else {
            diff += 1;
            "DIFF"
        };
        let notes: Vec<&str> = imp.warnings.iter().map(String::as_str).take(2).collect();
        eprintln!("{name:<60} {layers:>6} {:>9.4} {:>7.2}%  {status} {}", m, pct, notes.join(" | "));
    }
    eprintln!("io corpus: {} files: {pass} pass (<= 2/255), {diff} differ, {skipped} skipped, {errors} errors", files.len());
    eprintln!("io corpus: export -> re-import renders the same for {rt_same} files; differs for {}: {}", rt_diff.len(), rt_diff.join(", "));
    if std::env::var_os("PHOTOCRAFT_CORPUS_STRICT").is_some() {
        assert_eq!(errors, 0);
    }
    if env.is_some() {
        assert!(pass >= PASS_FLOOR, "oracle pass count {pass} fell below the floor {PASS_FLOOR}");
        assert!(rt_same >= ROUNDTRIP_FLOOR, "export round trip {rt_same} fell below the floor {ROUNDTRIP_FLOOR}: {rt_diff:?}");
    }
}
