// 환자/스터디/추세 데이터 훅. 컴포넌트는 이 훅만 사용한다.
import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getDataProvider } from "@/lib/data";

const dp = getDataProvider();

export function usePatients(query?: string) {
  return useQuery({
    queryKey: ["patients", query ?? ""],
    queryFn: () => dp.listPatients(query),
  });
}

export function usePatient(id: string | undefined) {
  return useQuery({
    queryKey: ["patient", id],
    queryFn: () => dp.getPatient(id!),
    enabled: !!id,
  });
}

export function useStudies(patientId: string | undefined) {
  return useQuery({
    queryKey: ["studies", patientId],
    queryFn: () => dp.listStudies(patientId!),
    enabled: !!patientId,
  });
}

export function useStudy(studyId: string | undefined) {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["study", studyId],
    queryFn: () => dp.getStudy(studyId!),
    enabled: !!studyId,
    // 추론 진행 중이면 폴링
    refetchInterval: (q) =>
      q.state.data?.status === "processing" ||
      q.state.data?.status === "uploaded"
        ? 2000
        : false,
  });

  // 추론이 완료되면 BMD 추세(포인트/변화량/LSC)와 환자 목록(최근 BMD/검사 수)이
  // 새로 반영돼야 하는데, 업로드는 study 목록만 무효화한다 -- 완료 전환 시점에 갱신.
  const status = query.data?.status;
  const patientId = query.data?.patient_id;
  useEffect(() => {
    if (status === "completed" && patientId) {
      qc.invalidateQueries({ queryKey: ["bmd-trend", patientId] });
      qc.invalidateQueries({ queryKey: ["patients"] });
    }
  }, [status, patientId, qc]);

  return query;
}

export function useBmdTrend(patientId: string | undefined) {
  return useQuery({
    queryKey: ["bmd-trend", patientId],
    queryFn: () => dp.getBmdTrend(patientId!),
    enabled: !!patientId,
  });
}

export function useUploadStudy(patientId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => dp.uploadStudy(patientId, file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["studies", patientId] });
    },
  });
}

export function useCreatePatient() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: {
      medical_record_no: string;
      full_name: string;
      sex?: string | null;
      birth_date?: string | null;
      notes?: string | null;
    }) => dp.createPatient(input),
    onSuccess: () => {
      // 모든 환자 목록 쿼리 갱신 (검색어별 캐시 포함)
      qc.invalidateQueries({ queryKey: ["patients"] });
    },
  });
}

export function useDeletePatient() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (patientId: string) => dp.deletePatient(patientId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["patients"] });
    },
  });
}

export function useDeleteStudy(patientId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (studyId: string) => dp.deleteStudy(studyId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["studies", patientId] });
      qc.invalidateQueries({ queryKey: ["bmd-trend", patientId] });
      qc.invalidateQueries({ queryKey: ["patients"] });
    },
  });
}

export function useUpdateStudyNote(studyId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (note: string) => dp.updateStudyNote(studyId, note),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["study", studyId] });
    },
  });
}

export function useOverrideView(studyId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (view: "AP" | "LA") => dp.overrideView(studyId, view),
    onSuccess: (updated) => {
      qc.setQueryData(["study", studyId], updated);
      qc.invalidateQueries({ queryKey: ["bmd-trend", updated.patient_id] });
      qc.invalidateQueries({ queryKey: ["patients"] });
    },
  });
}

export function useLscCalibration() {
  return useQuery({
    queryKey: ["lsc-calibration"],
    queryFn: () => dp.getLscCalibration(),
  });
}

export function useCalibratePrecision() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => dp.calibratePrecision(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["lsc-calibration"] });
      // 보정값이 바뀌면 모든 환자의 추세 경고 판정이 달라질 수 있음
      qc.invalidateQueries({ queryKey: ["bmd-trend"] });
    },
  });
}
