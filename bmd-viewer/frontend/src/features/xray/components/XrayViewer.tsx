// Study viewer: filename + overlay image + large L4 crop + DICOM metadata + note.
import { useStudy } from "@/features/patients/api/queries";
import { getDataProvider } from "@/lib/data";
import { XaiPanel } from "@/features/bmd/components/XaiPanel";
import { DensityBasis } from "@/features/bmd/components/DensityBasis";
import { StudyNote } from "./StudyNote";

const dp = getDataProvider();

// Friendly labels for common DICOM tags
const META_LABELS: Record<string, string> = {
  PatientID: "Patient ID",
  PatientSex: "Sex",
  PatientAge: "Age",
  StudyDate: "Study date",
  AcquisitionDate: "Acquisition date",
  Modality: "Modality",
  Manufacturer: "Manufacturer",
  ManufacturerModelName: "Device model",
  BodyPartExamined: "Body part",
  ViewPosition: "View position",
  KVP: "kVp",
  Exposure: "Exposure",
  Rows: "Rows",
  Columns: "Columns",
  BitsStored: "Bits stored",
  PhotometricInterpretation: "Photometric",
};

export function XrayViewer({ studyId }: { studyId: string }) {
  const { data: study } = useStudy(studyId);
  if (!study)
    return <div className="viewer empty">Select a study to view.</div>;

  if (study.status === "processing" || study.status === "uploaded") {
    return <div className="viewer loading">Analyzing… (auto-refresh)</div>;
  }
  if (study.status === "failed") {
    // 실패해도 원본 + 추출한 척추까지 보여주고, L4 분석 영역만 비워 실패 표시.
    const failOverlay = dp.assetUrl(study.overlay_path ?? null);
    const failPreview = dp.assetUrl(study.preview_path ?? null);
    const failImg = failOverlay ?? failPreview;
    return (
      <div className="viewer">
        <div className="image-pane">
          <div className="image-filename" title={study.original_filename ?? ""}>
            {study.original_filename ?? "(no filename)"}
          </div>
          <div className="image-stack">
            {failImg ? (
              <img
                key={study.id}
                src={`${failImg}?v=${study.id}`}
                alt="X-ray (original / detected vertebrae)"
              />
            ) : (
              <div className="placeholder">No preview available</div>
            )}
          </div>
          {failOverlay && (
            <p className="muted" style={{ marginTop: 8 }}>
              Original with detected vertebrae. L4 analysis could not be
              completed.
            </p>
          )}
        </div>
        <div className="result-pane">
          <div className="bmd-warning" role="alert">
            <span className="bmd-warning-icon" aria-hidden="true">
              ⚠
            </span>
            <div>
              <b>L4 analysis failed</b>
              <p>{study.error_message ?? "unknown cause"}</p>
            </div>
          </div>
          <div className="fail-placeholder">
            <span className="label">L4 bone density (BMD)</span>
            <span className="value">—</span>
            <p className="muted">
              BMD, density heatmap, and XAI are unavailable for this study.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const m = study.measurement;
  const overlay = dp.assetUrl(m?.gradcam_path ?? null);
  const preview = dp.assetUrl(study.preview_path);
  const imageSrc = overlay ?? preview;
  const l4Crop = dp.assetUrl(m?.l4_crop_path ?? null);
  const xaiDensity = dp.assetUrl(m?.xai_overlay_path ?? null);
  const xaiCam = dp.assetUrl(m?.xai_l4_cam_path ?? null);

  // Parse DICOM metadata JSON (stored as string)
  let meta: Record<string, unknown> = {};
  if (study.dicom_meta) {
    try {
      meta = JSON.parse(study.dicom_meta);
    } catch {
      meta = {};
    }
  }
  const metaEntries = Object.entries(meta).filter(
    ([, v]) => v !== null && v !== undefined && v !== ""
  );

  return (
    <div className="viewer">
      <div className="image-pane">
        {/* (3) Filename above the original image */}
        <div className="image-filename" title={study.original_filename ?? ""}>
          {study.original_filename ?? "(no filename)"}
        </div>

        <div className="image-stack">
          {imageSrc ? (
            <img
              key={study.id}
              src={imageSrc}
              alt="X-ray with L1–L5 segmentation"
            />
          ) : (
            <div className="placeholder">No preview</div>
          )}
        </div>

        {/* (5) Left column (L4 crop + DICOM metadata) beside the density basis,
             so the metadata sits right under the crop (no big gap). */}
        <div className="l4-density-row">
          <div className="l4-left-col">
            {l4Crop && (
              <div className="l4-crop-block">
                <h4>Extracted L4 vertebra</h4>
                <img src={`${l4Crop}?v=${study.id}`} alt="Extracted L4 crop" />
              </div>
            )}
            {/* (4) DICOM metadata (two-column), attached under the L4 crop */}
            <div className="dicom-meta">
              <h4>DICOM metadata</h4>
              {metaEntries.length > 0 && (
                <dl>
                  {metaEntries.map(([k, v]) => (
                    <div className="meta-row" key={k}>
                      <dt>{META_LABELS[k] ?? k}</dt>
                      <dd>{String(v)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          </div>
          {xaiDensity && (
            <DensityBasis
              densityUrl={`${xaiDensity}?v=${study.id}`}
              densityLow={m?.density_low}
              densityHigh={m?.density_high}
              roiMean={m?.roi_mean_intensity}
              bmdValue={m?.bmd_value}
            />
          )}
        </div>

        {/* Notes: full width below the row */}
        <StudyNote
          studyId={study.id}
          initialNote={study.note}
          initialUpdatedAt={study.note_updated_at}
        />
      </div>

      <div className="result-pane">
        {m && (
          <>
            {m.reliable === false && (
              <div className="bmd-warning" role="alert">
                <span className="bmd-warning-icon" aria-hidden="true">⚠</span>
                <div>
                  <b>Unreliable measurement — verify manually.</b>
                  <p>
                    {m.reliability_warning ??
                      "The target vertebra may be cut off or misidentified."}
                  </p>
                </div>
              </div>
            )}
            <div className={`bmd-headline${m.reliable === false ? " unreliable" : ""}`}>
              <span className="label">L4 bone density (BMD)</span>
              <span className="value">{m.bmd_value.toFixed(3)}</span>
              {m.t_score != null && (
                <span className="tscore">T-score {m.t_score.toFixed(2)}</span>
              )}
            </div>
            <div className="bmd-meta">
              <span>Confidence {((m.confidence ?? 0) * 100).toFixed(0)}%</span>
              <span>Model {m.model_version}</span>
              {m.exposure_corrected && <span>Exposure corrected</span>}
            </div>
            <XaiPanel
              factors={m.xai_factors}
              camUrl={xaiCam ? `${xaiCam}?v=${study.id}` : null}
            />
          </>
        )}
      </div>
    </div>
  );
}
