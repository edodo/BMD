// 다중 환자 비교 — CompareView와 같은 카드형 대조(이미지+밀도히트맵+픽셀격자+
// 5구역격자)를 여러 환자에 걸쳐 쓴다. 차이는 딱 하나: "Pre-op/Post-op"(같은
// 환자의 시간 축) 대신 "Reference/Compare"(임의의 환자 임의의 검사 1건을
// 기준으로, 다른 환자들의 검사를 그 기준과 비교)로 고른다. 카드/격자 렌더링은
// DensityCompareCard.tsx를 그대로 재사용 — 이 컴포넌트가 하는 일은 "여러
// 환자의 스터디 목록에서 기준 1개 + 비교 여러 개를 고르는 UI"뿐이다.
import { useState } from "react";
import { useQueries } from "@tanstack/react-query";
import {
  usePatients,
  useMultiPatientBmd,
  useCreateComparison,
  useListComparisons,
  useDeleteComparison,
} from "@/features/patients/api/queries";
import { getDataProvider } from "@/lib/data";
import { Modal } from "@/components/ui/Modal";
import { CompareCard, shortDate } from "./DensityCompareCard";
import type { SavedComparison, XrayStudyDetail, XrayStudyListItem } from "@/lib/types";

const dp = getDataProvider();

function PatientStudyPicker({
  patientName,
  studies,
  referenceId,
  compareIds,
  onPickReference,
  onToggleCompare,
}: {
  patientName: string;
  studies: XrayStudyListItem[] | undefined;
  referenceId: string | undefined;
  compareIds: string[];
  onPickReference: (studyId: string) => void;
  onToggleCompare: (studyId: string) => void;
}) {
  const completed = (studies ?? []).filter((s) => s.status === "completed");
  return (
    <div className="mpcmp-study-picker">
      <div className="mpcmp-study-picker-head">{patientName}</div>
      {completed.length === 0 ? (
        <p className="muted">No completed studies.</p>
      ) : (
        <>
        <div className="mpcmp-study-picker-cols">
          <span className="col-ref">Ref</span>
          <span className="col-inc">Incl.</span>
          <span>Date / BMD</span>
        </div>
        <ul>
          {completed.map((s) => (
            <li key={s.id}>
              <label className="mpcmp-ref">
                <input
                  type="radio"
                  name="mpcmp-reference"
                  checked={referenceId === s.id}
                  onChange={() => onPickReference(s.id)}
                  title="Set as reference"
                />
                <input
                  type="checkbox"
                  checked={compareIds.includes(s.id)}
                  disabled={referenceId === s.id}
                  onChange={() => onToggleCompare(s.id)}
                  title="Include in comparison"
                />
                <span>{shortDate(s.acquired_at ?? s.uploaded_at)}</span>
                <span className="muted">
                  BMD {s.bmd_value != null ? s.bmd_value.toFixed(3) : "—"}
                </span>
              </label>
            </li>
          ))}
        </ul>
        </>
      )}
    </div>
  );
}

function SaveMultiPatientModal({
  patientIds,
  referenceId,
  compareIds,
  contrast,
  onClose,
}: {
  patientIds: string[];
  referenceId: string | undefined;
  compareIds: string[];
  contrast: boolean;
  onClose: () => void;
}) {
  const create = useCreateComparison();
  const [title, setTitle] = useState("");
  const submit = () => {
    if (!title.trim()) return;
    create.mutate(
      {
        type: "multi_patient",
        title: title.trim(),
        patient_ids: patientIds,
        config: { referenceId, compareIds, contrast },
      },
      { onSuccess: onClose }
    );
  };
  return (
    <Modal
      title="Save comparison"
      onClose={onClose}
      footer={
        <>
          <button className="btn-ghost" onClick={onClose}>Cancel</button>
          <button onClick={submit} disabled={create.isPending}>
            {create.isPending ? "Saving…" : "Save"}
          </button>
        </>
      }
    >
      <label>
        Title
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g. Ward A, Q1 review cohort"
          autoFocus
        />
      </label>
    </Modal>
  );
}

function SavedMultiPatientModal({
  onClose,
  onLoad,
}: {
  onClose: () => void;
  onLoad: (c: SavedComparison) => void;
}) {
  const { data, isLoading } = useListComparisons({ type: "multi_patient" });
  const del = useDeleteComparison();
  return (
    <Modal title="Saved patient comparisons" onClose={onClose} size="wide">
      {isLoading && <p className="muted">Loading…</p>}
      {data?.length === 0 && <p className="muted">No saved comparisons yet.</p>}
      <ul className="saved-cmp-list">
        {data?.map((c) => (
          <li key={c.id}>
            <div>
              <div className="saved-cmp-title">{c.title}</div>
              <div className="saved-cmp-meta">
                {c.patient_ids?.length ?? 0} patient(s)
              </div>
            </div>
            <div className="saved-cmp-actions">
              <button onClick={() => { onLoad(c); onClose(); }}>Load</button>
              <button
                className="saved-cmp-delete"
                title="Delete this saved comparison"
                disabled={del.isPending}
                onClick={() => {
                  if (window.confirm(`Delete saved comparison "${c.title}"?`)) {
                    del.mutate(c.id);
                  }
                }}
              >
                🗑
              </button>
            </div>
          </li>
        ))}
      </ul>
    </Modal>
  );
}

export function MultiPatientCompareView() {
  const [query, setQuery] = useState("");
  const { data: patients } = usePatients(query);
  const [selected, setSelected] = useState<string[]>([]);
  const [referenceId, setReferenceId] = useState<string>();
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [contrast, setContrast] = useState(true);
  const [showSave, setShowSave] = useState(false);
  const [showSaved, setShowSaved] = useState(false);

  // patient_name 조회용 (검색어로 걸러진 patients 목록과 무관하게, 선택된
  // 환자는 항상 이름을 알아야 하므로 -- 이미 있는 엔드포인트 재사용).
  const { data: nameSeries } = useMultiPatientBmd(selected);
  const nameById = new Map(nameSeries?.map((s) => [s.patient_id, s.patient_name]));

  const studyResults = useQueries({
    queries: selected.map((id) => ({
      queryKey: ["studies", id],
      queryFn: () => dp.listStudies(id),
      enabled: !!id,
    })),
  });
  const studiesByPatient = new Map<string, XrayStudyListItem[]>();
  selected.forEach((id, i) => {
    if (studyResults[i]?.data) studiesByPatient.set(id, studyResults[i].data!);
  });

  const togglePatient = (id: string) =>
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );

  const pickReference = (studyId: string) => {
    setReferenceId(studyId);
    setCompareIds((prev) => prev.filter((x) => x !== studyId));
  };
  const toggleCompare = (studyId: string) =>
    setCompareIds((prev) =>
      prev.includes(studyId) ? prev.filter((x) => x !== studyId) : [...prev, studyId]
    );

  // 기준 1건 + 비교 대상 스터디들의 상세(이미지/히트맵/밀도격자 포함)를 가져온다.
  const detailIds = Array.from(
    new Set([referenceId, ...compareIds].filter((x): x is string => !!x))
  );
  const detailResults = useQueries({
    queries: detailIds.map((id) => ({
      queryKey: ["study", id],
      queryFn: () => dp.getStudy(id),
      enabled: !!id,
    })),
  });
  const detailById = new Map<string, XrayStudyDetail>();
  detailResults.forEach((r, i) => {
    if (r.data) detailById.set(detailIds[i], r.data as XrayStudyDetail);
  });

  const reference = referenceId ? detailById.get(referenceId) : undefined;
  const referenceGrid = reference?.measurement?.l4_density_grid ?? null;
  const referenceBmd = reference?.measurement?.bmd_value ?? null;
  const compares = compareIds
    .map((id) => detailById.get(id))
    .filter((d): d is XrayStudyDetail => !!d);

  const canCompare = !!reference && compares.length > 0;

  return (
    <div className="mpcmp-view">
      <div className="mpcmp-toolbar">
        <h3>Compare patients</h3>
        <div className="mpcmp-toolbar-actions">
          <label className="cmp-contrast">
            <input
              type="checkbox"
              checked={contrast}
              onChange={(e) => setContrast(e.target.checked)}
            />
            Contrast mode (density-change by region)
          </label>
          <button disabled={!canCompare} onClick={() => setShowSave(true)}>
            💾 Save
          </button>
          <button className="btn-secondary" onClick={() => setShowSaved(true)}>
            📂 Saved
          </button>
        </div>
      </div>

      <div className="mpcmp-body">
        <div className="mpcmp-picker">
          <input
            className="search"
            placeholder="Search for name or patient number"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <ul>
            {patients?.map((p) => (
              <li key={p.id}>
                <label>
                  <input
                    type="checkbox"
                    checked={selected.includes(p.id)}
                    onChange={() => togglePatient(p.id)}
                  />
                  <span>{p.full_name}</span>
                  <span className="muted">{p.medical_record_no}</span>
                </label>
              </li>
            ))}
          </ul>
        </div>

        <div className="mpcmp-chart-col">
          {selected.length === 0 ? (
            <div className="cmp-empty">
              Select patients, then pick one reference X-ray and one or more to
              compare against it.
            </div>
          ) : (
            <>
              <div className="mpcmp-study-pickers">
                {selected.map((id) => (
                  <PatientStudyPicker
                    key={id}
                    patientName={nameById.get(id) ?? id}
                    studies={studiesByPatient.get(id)}
                    referenceId={referenceId}
                    compareIds={compareIds}
                    onPickReference={pickReference}
                    onToggleCompare={toggleCompare}
                  />
                ))}
              </div>

              {canCompare ? (
                <div className="cmp-strip">
                  <CompareCard
                    study={reference}
                    role="pre"
                    preGrid={null}
                    preBmd={null}
                    contrast={false}
                    patientLabel={nameById.get(reference.patient_id)}
                    preTagText="Reference"
                    postTagText="Compare"
                  />
                  {compares.map((s) => (
                    <CompareCard
                      key={s.id}
                      study={s}
                      role="post"
                      preGrid={referenceGrid}
                      preBmd={referenceBmd}
                      preStudyId={referenceId}
                      contrast={contrast}
                      patientLabel={nameById.get(s.patient_id)}
                      preTagText="Reference"
                      postTagText="Compare"
                    />
                  ))}
                </div>
              ) : (
                <div className="cmp-empty">
                  Pick a reference X-ray (radio) and at least one to compare
                  (checkbox) above.
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {showSave && (
        <SaveMultiPatientModal
          patientIds={selected}
          referenceId={referenceId}
          compareIds={compareIds}
          contrast={contrast}
          onClose={() => setShowSave(false)}
        />
      )}
      {showSaved && (
        <SavedMultiPatientModal
          onClose={() => setShowSaved(false)}
          onLoad={(c) => {
            if (c.patient_ids) setSelected(c.patient_ids);
            const cfg = c.config as
              | { referenceId?: string; compareIds?: string[]; contrast?: boolean }
              | null
              | undefined;
            setReferenceId(cfg?.referenceId);
            setCompareIds(cfg?.compareIds ?? []);
            if (typeof cfg?.contrast === "boolean") setContrast(cfg.contrast);
          }}
        />
      )}
    </div>
  );
}
