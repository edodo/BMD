// Patient list + search + registration + delete.
import { useState } from "react";
import { usePatients, useDeletePatient } from "../api/queries";
import { PatientCreateModal } from "./PatientCreateModal";
import type { PatientListItem } from "@/lib/types";

interface Props {
  selectedId?: string;
  onSelect: (p: PatientListItem) => void;
  onDeleted?: (id: string) => void;
}

export function PatientList({ selectedId, onSelect, onDeleted }: Props) {
  const [query, setQuery] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const { data, isLoading } = usePatients(query);
  const del = useDeletePatient();

  const handleDelete = (e: React.MouseEvent, p: PatientListItem) => {
    e.stopPropagation();
    const ok = window.confirm(
      `Delete patient "${p.full_name}" (${p.medical_record_no})?\n` +
        `All X-ray studies and measurements for this patient will be permanently removed.`
    );
    if (!ok) return;
    del.mutate(p.id, { onSuccess: () => onDeleted?.(p.id) });
  };

  return (
    <div className="patient-list">
      <div className="patient-list-header">
        <h3>Patients</h3>
        <button className="btn-sm" onClick={() => setShowCreate(true)}>
          + Registration
        </button>
      </div>
      <input
        className="search"
        placeholder="Search for name or patient number"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {isLoading && <p className="muted">Loading…</p>}
      <ul>
        {data?.map((p) => (
          <li
            key={p.id}
            className={p.id === selectedId ? "selected" : ""}
            onClick={() => onSelect(p)}
          >
            <div className="row-between">
              <div className="name">{p.full_name}</div>
              <button
                className="row-delete"
                title="Delete patient"
                onClick={(e) => handleDelete(e, p)}
              >
                🗑
              </button>
            </div>
            <div className="meta">
              <span>{p.medical_record_no}</span>
              <span>{p.sex ?? "-"}</span>
              <span>Recent BMD: {p.latest_bmd?.toFixed(3) ?? "—"}</span>
              <span>{p.study_count} studies</span>
            </div>
          </li>
        ))}
        {data?.length === 0 && (
          <li className="empty">
            No patients registered. Use "+ Registration" at the top right to
            add one.
          </li>
        )}
      </ul>

      {showCreate && (
        <PatientCreateModal onClose={() => setShowCreate(false)} />
      )}
    </div>
  );
}
