// REST(FastAPI + SQLite) 구현.
import axios, { type AxiosInstance } from "axios";
import type { DataProvider } from "./provider";
import type {
  AuthSession,
  BmdTrend,
  Patient,
  PatientListItem,
  PrecisionCalibration,
  XrayStudyDetail,
  XrayStudyListItem,
} from "@/lib/types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

export class RestDataProvider implements DataProvider {
  private http: AxiosInstance;

  constructor() {
    this.http = axios.create({ baseURL: API_BASE });
    this.http.interceptors.request.use((config) => {
      const token = localStorage.getItem("bmd_token");
      if (token) config.headers.Authorization = `Bearer ${token}`;
      return config;
    });
    // 인증 만료(401) 시 토큰 제거 후 로그인 화면으로
    this.http.interceptors.response.use(
      (res) => res,
      (err) => {
        const status = err?.response?.status;
        const url: string = err?.config?.url ?? "";
        // 로그인 시도 자체의 401은 폼에서 처리하므로 제외
        if (status === 401 && !url.includes("/auth/login")) {
          localStorage.removeItem("bmd_token");
          // 컨텍스트가 상태를 못 받으므로 새로고침으로 게이트 재평가
          if (typeof window !== "undefined") window.location.reload();
        }
        return Promise.reject(err);
      }
    );
  }

  async login(email: string, password: string): Promise<AuthSession> {
    // OAuth2 password flow는 form-encoded
    const form = new URLSearchParams();
    form.set("username", email);
    form.set("password", password);
    const { data } = await this.http.post<AuthSession>("/auth/login", form);
    localStorage.setItem("bmd_token", data.access_token);
    return data;
  }

  async register(input: {
    email: string;
    password: string;
    full_name: string;
    role?: string;
  }): Promise<void> {
    await this.http.post("/auth/register", {
      email: input.email,
      password: input.password,
      full_name: input.full_name,
      role: input.role ?? "doctor",
    });
  }

  logout(): void {
    localStorage.removeItem("bmd_token");
  }

  isAuthenticated(): boolean {
    return !!localStorage.getItem("bmd_token");
  }

  async listPatients(query?: string): Promise<PatientListItem[]> {
    const { data } = await this.http.get<PatientListItem[]>("/patients", {
      params: query ? { q: query } : undefined,
    });
    return data;
  }

  async getPatient(id: string): Promise<Patient> {
    const { data } = await this.http.get<Patient>(`/patients/${id}`);
    return data;
  }

  async createPatient(input: Partial<Patient>): Promise<Patient> {
    const { data } = await this.http.post<Patient>("/patients", input);
    return data;
  }

  async deletePatient(id: string): Promise<void> {
    await this.http.delete(`/patients/${id}`);
  }

  async listStudies(patientId: string): Promise<XrayStudyListItem[]> {
    const { data } = await this.http.get<XrayStudyListItem[]>(
      `/patients/${patientId}/studies`
    );
    return data;
  }

  async uploadStudy(patientId: string, file: File): Promise<XrayStudyDetail> {
    const form = new FormData();
    form.append("file", file);
    const { data } = await this.http.post<XrayStudyDetail>(
      `/patients/${patientId}/studies`,
      form
    );
    return data;
  }

  async getStudy(studyId: string): Promise<XrayStudyDetail> {
    const { data } = await this.http.get<XrayStudyDetail>(`/studies/${studyId}`);
    return data;
  }

  async deleteStudy(studyId: string): Promise<void> {
    await this.http.delete(`/studies/${studyId}`);
  }

  async updateStudyNote(
    studyId: string,
    note: string
  ): Promise<XrayStudyDetail> {
    const { data } = await this.http.patch<XrayStudyDetail>(
      `/studies/${studyId}/note`,
      { note }
    );
    return data;
  }

  async getBmdTrend(patientId: string): Promise<BmdTrend> {
    const { data } = await this.http.get<BmdTrend>(
      `/patients/${patientId}/bmd-trend`
    );
    return data;
  }

  async overrideView(
    studyId: string,
    view: "AP" | "LA"
  ): Promise<XrayStudyDetail> {
    const { data } = await this.http.post<XrayStudyDetail>(
      `/studies/${studyId}/view`,
      { view }
    );
    return data;
  }

  async getLscCalibration(): Promise<PrecisionCalibration | null> {
    const { data } = await this.http.get<PrecisionCalibration | null>(
      "/precision/lsc"
    );
    return data;
  }

  async calibratePrecision(): Promise<PrecisionCalibration> {
    const { data } = await this.http.post<PrecisionCalibration>(
      "/precision/calibrate"
    );
    return data;
  }

  assetUrl(path: string | null): string | null {
    if (!path) return null;
    return `/derived/${path}`;
  }
}
