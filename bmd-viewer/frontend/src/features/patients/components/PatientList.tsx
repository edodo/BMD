// 환자 리스트 + 검색.
import { useState } from "react";
import { usePatients } from "../api/queries";
import type { PatientListItem } from "@/lib/types";

interface Props {
  selectedId?: string;
  onSelect: (p: PatientListItem) => void;
}

export function PatientList({ selectedId, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const { data, isLoading } = usePatients(query);

  return (
    <div className="patient-list">
      <input
        className="search"
        placeholder="이름 또는 환자번호 검색"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {isLoading && <p className="muted">불러오는 중…</p>}
      <ul>
        {data?.map((p) => (
          <li
            key={p.id}
            className={p.id === selectedId ? "selected" : ""}
            onClick={() => onSelect(p)}
          >
            <div className="name">{p.full_name}</div>
            <div className="meta">
              <span>{p.medical_record_no}</span>
              <span>{p.sex ?? "-"}</span>
              <span>
                최근 BMD: {p.latest_bmd?.toFixed(3) ?? "—"}
              </span>
              <span>{p.study_count}건</span>
            </div>
          </li>
        ))}
        {data?.length === 0 && (
          <li className="empty">검색 결과가 없습니다.</li>
        )}
      </ul>
    </div>
  );
}
