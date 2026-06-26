// 환자/스터디/추세 데이터 훅. 컴포넌트는 이 훅만 사용한다.
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
  return useQuery({
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
