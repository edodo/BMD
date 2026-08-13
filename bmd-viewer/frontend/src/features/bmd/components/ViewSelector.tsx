// AP/LA 촬영 방향 표시 + 수동 재지정 (v18_spec.md Sec.4.5).
// 자동판별(DICOM ViewPosition 태그, 없으면 AP 기본)이 실제 촬영 방향과 다르면
// 의사가 여기서 바로잡아 재계산할 수 있다 — 측면(LA) 영상을 AP로 잘못 재면
// 연부조직 기준값이 후방 골구조물로 오염돼 BAR가 왜곡된다.
import { useEffect, useState } from "react";
import { useOverrideView } from "@/features/patients/api/queries";

export function ViewSelector({
  studyId,
  viewPosition,
  viewSource,
}: {
  studyId: string;
  viewPosition: "AP" | "LA" | null;
  viewSource: "auto" | "manual" | null;
}) {
  const override = useOverrideView(studyId);
  const current = viewPosition ?? "AP";
  const [pending, setPending] = useState<"AP" | "LA">(current);

  // Reset the pending pick whenever we switch studies or the server's
  // resolved view changes (e.g. after Apply succeeds).
  useEffect(() => {
    setPending(current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studyId, current]);

  const dirty = pending !== current;

  return (
    <div className="view-selector">
      <span className="view-label">
        View: <b>{current}</b>{" "}
        <span className="view-source">
          ({viewSource === "manual" ? "manually set" : "auto-detected"})
        </span>
      </span>
      <div className="view-toggle">
        {(["AP", "LA"] as const).map((v) => (
          <button
            key={v}
            type="button"
            className={`view-btn${pending === v ? " active" : ""}`}
            onClick={() => setPending(v)}
          >
            {v}
          </button>
        ))}
        {dirty && (
          <button
            type="button"
            className="view-apply-btn"
            onClick={() => override.mutate(pending)}
            disabled={override.isPending}
          >
            {override.isPending ? "Recalculating…" : "Apply"}
          </button>
        )}
      </div>
      {override.isError && (
        <span className="view-error">
          {(override.error as any)?.response?.data?.detail ??
            "Recalculation failed."}
        </span>
      )}
    </div>
  );
}
