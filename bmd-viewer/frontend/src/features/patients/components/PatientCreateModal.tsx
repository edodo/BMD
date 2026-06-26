// 환자 등록 모달.
// 환자번호(MRN)와 이름은 필수, Sex/Date of birth/Notes는 선택.
import { useState } from "react";
import { useCreatePatient } from "../api/queries";

interface Props {
  onClose: () => void;
}

export function PatientCreateModal({ onClose }: Props) {
  const create = useCreatePatient();
  const [mrn, setMrn] = useState("");
  const [name, setName] = useState("");
  const [sex, setSex] = useState("");
  const [birth, setBirth] = useState("");
  const [notes, setNotes] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setError(null);
    if (!mrn.trim() || !name.trim()) {
      setError("Patient number and name are required.");
      return;
    }
    try {
      await create.mutateAsync({
        medical_record_no: mrn.trim(),
        full_name: name.trim(),
        sex: sex || null,
        birth_date: birth ? `${birth}T00:00:00` : null,
        notes: notes.trim() || null,
      });
      onClose();
    } catch (e: unknown) {
      const msg =
        (e as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ?? "Registration failed. The patient number may already exist.";
      setError(typeof msg === "string" ? msg : "Registration failed.");
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Register Patient</h3>
          <button className="modal-close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <label>
            Patient No. (MRN) <span className="req">*</span>
            <input
              value={mrn}
              onChange={(e) => setMrn(e.target.value)}
              placeholder="MRN-001"
            />
          </label>
          <label>
            Name <span className="req">*</span>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="John Doe"
            />
          </label>
          <div className="form-row">
            <label>
              Sex
              <select value={sex} onChange={(e) => setSex(e.target.value)}>
                <option value="">Not specified</option>
                <option value="M">Male</option>
                <option value="F">Female</option>
              </select>
            </label>
            <label>
              Date of birth
              <input
                type="date"
                value={birth}
                onChange={(e) => setBirth(e.target.value)}
              />
            </label>
          </div>
          <label>
            Notes
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
              placeholder="Clinical notes (optional)"
            />
          </label>

          {error && <p className="auth-error">{error}</p>}
        </div>

        <div className="modal-footer">
          <button className="btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button onClick={submit} disabled={create.isPending}>
            {create.isPending ? "Registering…" : "Register"}
          </button>
        </div>
      </div>
    </div>
  );
}
