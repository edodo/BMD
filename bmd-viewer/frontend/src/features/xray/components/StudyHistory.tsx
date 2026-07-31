// Patient detail: X-ray history list + upload + delete (paged).
import { useEffect, useRef, useState } from "react";
import {
  useStudies,
  useUploadStudy,
  useDeleteStudy,
} from "@/features/patients/api/queries";
import { config } from "@/lib/config";
import type { XrayStudyListItem } from "@/lib/types";

const statusLabel: Record<string, string> = {
  uploaded: "Pending",
  processing: "Analyzing",
  completed: "Completed",
  failed: "Failed",
};

// Full date-time: YYYY-MM-DD HH:MM:SS
function fmtDateTime(iso: string): string {
  const d = new Date(iso);
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
  );
}

interface Props {
  patientId: string;
  selectedStudyId?: string;
  onSelect: (s: XrayStudyListItem) => void;
  onDeleted?: (id: string) => void;
}

export function StudyHistory({
  patientId,
  selectedStudyId,
  onSelect,
  onDeleted,
}: Props) {
  const { data } = useStudies(patientId);
  const upload = useUploadStudy(patientId);
  const del = useDeleteStudy(patientId);
  const fileRef = useRef<HTMLInputElement>(null);

  // 페이징 (config.historyPageSize). 환자 전환 시 1페이지로 초기화.
  const [page, setPage] = useState(0);
  useEffect(() => setPage(0), [patientId]);
  const pageSize = config.historyPageSize;
  const total = data?.length ?? 0;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const safePage = Math.min(page, pageCount - 1);
  const pageItems = (data ?? []).slice(
    safePage * pageSize,
    safePage * pageSize + pageSize
  );

  const handleDelete = (e: React.MouseEvent, s: XrayStudyListItem) => {
    e.stopPropagation();
    const ok = window.confirm(
      `Delete this X-ray study?\n` +
        `${s.original_filename ?? s.id}\n` +
        `Uploaded ${fmtDateTime(s.uploaded_at)}\n` +
        `This will permanently remove the image and its measurement.`
    );
    if (!ok) return;
    del.mutate(s.id, { onSuccess: () => onDeleted?.(s.id) });
  };

  return (
    <div className="study-history">
      <div className="toolbar">
        <h3>X-ray History</h3>
        <button onClick={() => fileRef.current?.click()}>Upload DICOM</button>
        <input
          ref={fileRef}
          type="file"
          accept=".dcm,.dicom,application/dicom"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) upload.mutate(f);
            e.target.value = "";
          }}
        />
      </div>
      {upload.isPending && <p className="muted">Uploading…</p>}
      {upload.isError && (
        <p className="muted" style={{ color: "var(--danger, #d33)" }}>
          Upload failed:{" "}
          {(upload.error as any)?.response?.status === 413
            ? "file too large (server limit)"
            : (upload.error as any)?.response?.data?.detail ??
              (upload.error as Error)?.message ??
              "unknown error"}
        </p>
      )}
      <ul>
        {pageItems.map((s) => (
          <li
            key={s.id}
            className={s.id === selectedStudyId ? "selected" : ""}
            onClick={() => onSelect(s)}
          >
            <div className="study-row-top">
              <span className="datetime">{fmtDateTime(s.uploaded_at)}</span>
              <button
                className="row-delete"
                title="Delete study"
                onClick={(e) => handleDelete(e, s)}
              >
                🗑
              </button>
            </div>
            <div className="study-row-bottom">
              <span className={`status status-${s.status}`}>
                {statusLabel[s.status]}
              </span>
              <span className="bmd">
                {s.bmd_value != null ? `BMD ${s.bmd_value.toFixed(3)}` : "—"}
              </span>
              {s.original_filename && (
                <span className="fname" title={s.original_filename}>
                  {s.original_filename}
                </span>
              )}
            </div>
          </li>
        ))}
      </ul>
      {total > pageSize && (
        <div className="pager">
          <button
            disabled={safePage <= 0}
            onClick={() => setPage(safePage - 1)}
          >
            ‹ Prev
          </button>
          <span className="pager-info">
            {safePage + 1} / {pageCount}
          </span>
          <button
            disabled={safePage >= pageCount - 1}
            onClick={() => setPage(safePage + 1)}
          >
            Next ›
          </button>
        </div>
      )}
    </div>
  );
}
