// Per-study clinician note editor (lives inside the DICOM metadata block).
import { useEffect, useState } from "react";
import { useUpdateStudyNote } from "@/features/patients/api/queries";

interface Props {
  studyId: string;
  initialNote: string | null;
  initialUpdatedAt: string | null;
}

// Format ISO timestamp → "YYYY-MM-DD HH:MM:SS"
function fmtDateTime(iso: string): string {
  const d = new Date(iso);
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
  );
}

export function StudyNote({ studyId, initialNote, initialUpdatedAt }: Props) {
  const [text, setText] = useState(initialNote ?? "");
  const [dirty, setDirty] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<string | null>(initialUpdatedAt);
  const save = useUpdateStudyNote(studyId);

  // Reset editor when switching studies
  useEffect(() => {
    setText(initialNote ?? "");
    setUpdatedAt(initialUpdatedAt);
    setDirty(false);
  }, [studyId, initialNote, initialUpdatedAt]);

  const handleSave = () => {
    save.mutate(text, {
      onSuccess: (detail) => {
        setDirty(false);
        setUpdatedAt(detail.note_updated_at);
      },
    });
  };

  return (
    <div className="study-note">
      <div className="note-header">
        <h4>Notes</h4>
      </div>
      <textarea
        value={text}
        rows={3}
        placeholder="Add a note for this X-ray (clinical observations, follow-up, etc.)"
        onChange={(e) => {
          setText(e.target.value);
          setDirty(true);
        }}
      />
      <div className="note-actions">
        <button disabled={!dirty || save.isPending} onClick={handleSave}>
          {save.isPending ? "Saving…" : "Save note"}
        </button>
        {updatedAt && (
          <span className="note-timestamp">
            Last saved: {fmtDateTime(updatedAt)}
          </span>
        )}
      </div>
    </div>
  );
}
