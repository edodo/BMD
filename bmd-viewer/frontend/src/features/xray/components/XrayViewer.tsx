// 스터디 뷰어: 원본 X-ray + 추출된 척추 오버레이 + L4 BMD + XAI.
import { useStudy } from "@/features/patients/api/queries";
import { getDataProvider } from "@/lib/data";
import { XaiPanel } from "@/features/bmd/components/XaiPanel";

const dp = getDataProvider();

export function XrayViewer({ studyId }: { studyId: string }) {
  const { data: study } = useStudy(studyId);
  if (!study) return <div className="viewer empty">스터디를 선택하세요.</div>;

  if (study.status === "processing" || study.status === "uploaded") {
    return <div className="viewer loading">분석 중입니다… (자동 갱신)</div>;
  }
  if (study.status === "failed") {
    return (
      <div className="viewer error">
        분석 실패: {study.error_message ?? "원인 미상"}
      </div>
    );
  }

  const m = study.measurement;
  const preview = dp.assetUrl(study.preview_path);

  return (
    <div className="viewer">
      <div className="image-pane">
        <div className="image-stack">
          {preview ? (
            <img src={preview} alt="X-ray" />
          ) : (
            <div className="placeholder">미리보기 없음</div>
          )}
          {/* 추출된 척추 오버레이 (정규화 bbox) */}
          {m?.segments.map((s) => (
            <div
              key={s.label}
              className={`seg-box ${s.is_target ? "target" : ""}`}
              style={{
                left: `${s.bbox_x * 100}%`,
                top: `${s.bbox_y * 100}%`,
                width: `${s.bbox_w * 100}%`,
                height: `${s.bbox_h * 100}%`,
              }}
            >
              <span>{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="result-pane">
        {m && (
          <>
            <div className="bmd-headline">
              <span className="label">L4 골밀도 (BMD)</span>
              <span className="value">{m.bmd_value.toFixed(3)}</span>
              {m.t_score != null && (
                <span className="tscore">T-score {m.t_score.toFixed(2)}</span>
              )}
            </div>
            <div className="bmd-meta">
              <span>신뢰도 {((m.confidence ?? 0) * 100).toFixed(0)}%</span>
              <span>모델 {m.model_version}</span>
              {m.exposure_corrected && <span>노출 보정됨</span>}
            </div>
            <XaiPanel factors={m.xai_factors} />
          </>
        )}
      </div>
    </div>
  );
}
