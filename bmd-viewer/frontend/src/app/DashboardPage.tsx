// Main dashboard: patient list → patient detail (history) → study viewer + trend.
import { useState } from "react";
import { PatientList } from "@/features/patients/components/PatientList";
import { StudyHistory } from "@/features/xray/components/StudyHistory";
import { XrayViewer } from "@/features/xray/components/XrayViewer";
import { BmdTrendChart } from "@/features/bmd/components/BmdTrendChart";
import { CompareView } from "@/features/bmd/components/CompareView";
import type { PatientListItem, XrayStudyListItem } from "@/lib/types";

export function DashboardPage() {
  const [patient, setPatient] = useState<PatientListItem>();
  const [studyId, setStudyId] = useState<string>();
  const [compare, setCompare] = useState(false);

  return (
    <div className="dashboard">
      <aside className="col col-patients">
        <PatientList
          selectedId={patient?.id}
          onSelect={(p) => {
            setPatient(p);
            setStudyId(undefined);
          }}
          onDeleted={(id) => {
            if (patient?.id === id) {
              setPatient(undefined);
              setStudyId(undefined);
            }
          }}
        />
      </aside>

      {patient ? (
        <>
          <section className="col col-history">
            <header className="patient-header">
              <div>
                <h2>{patient.full_name}</h2>
                <span className="muted">{patient.medical_record_no}</span>
              </div>
              <button
                className={`compare-toggle${compare ? " active" : ""}`}
                onClick={() => setCompare((v) => !v)}
              >
                {compare ? "✕ Close compare" : "⇄ Compare"}
              </button>
            </header>
            {/* Trend를 History 위로 (한눈에 추세 먼저 확인) */}
            <BmdTrendChart patientId={patient.id} />
            <StudyHistory
              patientId={patient.id}
              selectedStudyId={studyId}
              onSelect={(s: XrayStudyListItem) => setStudyId(s.id)}
              onDeleted={(id) => {
                if (studyId === id) setStudyId(undefined);
              }}
            />
          </section>

          <main className="col col-viewer">
            {compare ? (
              <CompareView patientId={patient.id} />
            ) : studyId ? (
              <XrayViewer studyId={studyId} />
            ) : (
              <div className="viewer empty">
                Select an X-ray to view analysis results.
              </div>
            )}
          </main>
        </>
      ) : (
        <div className="col placeholder-col">Select a patient.</div>
      )}
    </div>
  );
}
