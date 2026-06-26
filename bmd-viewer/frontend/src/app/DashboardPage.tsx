// 메인 대시보드: 환자 리스트 → 환자 상세(이력) → 스터디 뷰어 + 추세.
import { useState } from "react";
import { PatientList } from "@/features/patients/components/PatientList";
import { StudyHistory } from "@/features/xray/components/StudyHistory";
import { XrayViewer } from "@/features/xray/components/XrayViewer";
import { BmdTrendChart } from "@/features/bmd/components/BmdTrendChart";
import type { PatientListItem, XrayStudyListItem } from "@/lib/types";

export function DashboardPage() {
  const [patient, setPatient] = useState<PatientListItem>();
  const [studyId, setStudyId] = useState<string>();

  return (
    <div className="dashboard">
      <aside className="col col-patients">
        <PatientList
          selectedId={patient?.id}
          onSelect={(p) => {
            setPatient(p);
            setStudyId(undefined);
          }}
        />
      </aside>

      {patient ? (
        <>
          <section className="col col-history">
            <header className="patient-header">
              <h2>{patient.full_name}</h2>
              <span className="muted">{patient.medical_record_no}</span>
            </header>
            <StudyHistory
              patientId={patient.id}
              selectedStudyId={studyId}
              onSelect={(s: XrayStudyListItem) => setStudyId(s.id)}
            />
            <BmdTrendChart patientId={patient.id} />
          </section>

          <main className="col col-viewer">
            {studyId ? (
              <XrayViewer studyId={studyId} />
            ) : (
              <div className="viewer empty">
                X-ray를 선택하면 분석 결과가 표시됩니다.
              </div>
            )}
          </main>
        </>
      ) : (
        <div className="col placeholder-col">환자를 선택하세요.</div>
      )}
    </div>
  );
}
