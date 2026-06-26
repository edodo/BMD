// 환자 상세: X-ray 이력 목록 + 업로드.
import { useRef } from "react";
import { useStudies, useUploadStudy } from "@/features/patients/api/queries";
import type { XrayStudyListItem } from "@/lib/types";

const statusLabel: Record<string, string> = {
  uploaded: "대기",
  processing: "분석 중",
  completed: "완료",
  failed: "실패",
};

interface Props {
  patientId: string;
  selectedStudyId?: string;
  onSelect: (s: XrayStudyListItem) => void;
}

export function StudyHistory({ patientId, selectedStudyId, onSelect }: Props) {
  const { data } = useStudies(patientId);
  const upload = useUploadStudy(patientId);
  const fileRef = useRef<HTMLInputElement>(null);

  return (
    <div className="study-history">
      <div className="toolbar">
        <h3>X-ray 이력</h3>
        <button onClick={() => fileRef.current?.click()}>
          DICOM 업로드
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".dcm,application/dicom"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) upload.mutate(f);
          }}
        />
      </div>
      {upload.isPending && <p className="muted">업로드 중…</p>}
      <ul>
        {data?.map((s) => (
          <li
            key={s.id}
            className={s.id === selectedStudyId ? "selected" : ""}
            onClick={() => onSelect(s)}
          >
            <span className="date">
              {new Date(s.uploaded_at).toLocaleDateString()}
            </span>
            <span className={`status status-${s.status}`}>
              {statusLabel[s.status]}
            </span>
            <span className="bmd">
              {s.bmd_value != null ? `BMD ${s.bmd_value.toFixed(3)}` : "—"}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
