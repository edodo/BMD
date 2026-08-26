// Patient detail: X-ray history list + upload (multi-file) + date edit + delete (paged).
import { useEffect, useRef, useState } from "react";
import {
  useStudies,
  useUploadStudy,
  useUpdateStudyDate,
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

// <input type=datetime-local> value: YYYY-MM-DDTHH:MM
function fmtDateTimeInput(iso: string): string {
  const d = new Date(iso);
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}` +
    `T${p(d.getHours())}:${p(d.getMinutes())}`
  );
}

function extractErrMsg(err: unknown): string {
  const e = err as { response?: { status?: number; data?: { detail?: string } } };
  if (e?.response?.status === 413) return "file too large (server limit)";
  return e?.response?.data?.detail ?? (err as Error)?.message ?? "unknown error";
}

interface UploadItem {
  name: string;
  status: "pending" | "done" | "error";
  message?: string;
}

// Inline date editor for a single row (pencil toggle -> date input + save/cancel).
function DateEditor({ study }: { study: XrayStudyListItem }) {
  const update = useUpdateStudyDate(study.id);
  const [editing, setEditing] = useState(false);
  const shown = study.acquired_at ?? study.uploaded_at;
  const [value, setValue] = useState(() => fmtDateTimeInput(shown));

  if (!editing) {
    return (
      <span className="datetime">
        {fmtDateTime(shown)}
        <button
          className="date-edit-btn"
          title="Edit acquisition date & time"
          onClick={(e) => {
            e.stopPropagation();
            setValue(fmtDateTimeInput(shown));
            setEditing(true);
          }}
        >
          🖊
        </button>
      </span>
    );
  }

  return (
    <span className="datetime date-editing" onClick={(e) => e.stopPropagation()}>
      <input
        type="datetime-local"
        step="1"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <button
        className="date-save-btn"
        disabled={update.isPending}
        onClick={() => {
          if (!value) return;
          update.mutate(value, { onSuccess: () => setEditing(false) });
        }}
      >
        ✓
      </button>
      <button className="date-cancel-btn" onClick={() => setEditing(false)}>
        ✕
      </button>
    </span>
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
  const [uploadQueue, setUploadQueue] = useState<UploadItem[]>([]);
  const uploadPendingCount = uploadQueue.filter((i) => i.status === "pending").length;
  const uploadStillPending = uploadPendingCount > 0;
  const uploadErrors = uploadQueue.filter((i) => i.status === "error");

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

  const handleFiles = (files: FileList) => {
    const list = Array.from(files);
    if (list.length === 0) return;
    setUploadQueue(list.map((f) => ({ name: f.name, status: "pending" })));
    list.forEach((f, idx) => {
      upload.mutate(f, {
        onSuccess: () =>
          setUploadQueue((q) =>
            q.map((item, i) => (i === idx ? { ...item, status: "done" } : item))
          ),
        onError: (err) =>
          setUploadQueue((q) =>
            q.map((item, i) =>
              i === idx
                ? { ...item, status: "error", message: extractErrMsg(err) }
                : item
            )
          ),
      });
    });
  };

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
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {uploadQueue.length > 0 && uploadStillPending && (
        <p className="upload-progress-line">
          Uploading {uploadQueue.length - uploadPendingCount}/{uploadQueue.length}…
        </p>
      )}
      {/* 성공한 업로드는 History가 몇 초 안에 알아서 보여주므로 따로 목록으로
          안 남긴다. 여기 남기는 건 실패한 것뿐 -- 업로드 자체가 실패하면
          (413 등) 서버에 레코드가 안 생겨 History엔 아예 안 뜨니, 여기가
          유일하게 보이는 곳이다. */}
      {uploadErrors.length > 0 && (
        <div className="upload-queue-wrap">
          <ul className="upload-queue">
            {uploadErrors.map((item, i) => (
              <li key={`${item.name}-${i}`} className="uq-error">
                <span className="uq-icon">✕</span>
                <span className="uq-name" title={item.name}>{item.name}</span>
                <span className="uq-error-msg">{item.message}</span>
              </li>
            ))}
          </ul>
          <button className="uq-clear" onClick={() => setUploadQueue([])}>
            Clear
          </button>
        </div>
      )}
      <ul>
        {pageItems.map((s) => (
          <li
            key={s.id}
            className={s.id === selectedStudyId ? "selected" : ""}
            onClick={() => onSelect(s)}
          >
            <div className="study-row-top">
              <DateEditor study={s} />
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
