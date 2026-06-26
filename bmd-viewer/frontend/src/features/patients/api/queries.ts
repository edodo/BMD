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
