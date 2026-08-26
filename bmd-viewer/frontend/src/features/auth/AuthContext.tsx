// 인증 상태 전역 관리.
// 로그인/로그아웃 시 앱이 로그인 화면 ↔ 대시보드를 전환한다.
import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { getDataProvider } from "@/lib/data";
import type { Doctor } from "@/lib/types";

const dp = getDataProvider();

interface AuthContextValue {
  isAuthenticated: boolean;
  doctor: Doctor | null;
  login: (email: string, password: string) => Promise<void>;
  register: (input: {
    email: string;
    password: string;
    full_name: string;
  }) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setAuthed] = useState(dp.isAuthenticated());
  const [doctor, setDoctor] = useState<Doctor | null>(null);

  // 토큰이 있는 상태(첫 로드 시 이미 로그인돼 있던 경우 포함)면 상단바에
  // 보여줄 이름을 가져온다. 실패해도(토큰 만료 등) 조용히 넘어간다 -- 401
  // 인터셉터가 이미 로그아웃 처리를 담당한다.
  useEffect(() => {
    if (!isAuthenticated) {
      setDoctor(null);
      return;
    }
    dp.getCurrentDoctor()
      .then(setDoctor)
      .catch(() => setDoctor(null));
  }, [isAuthenticated]);

  const login = async (email: string, password: string) => {
    await dp.login(email, password);
    setAuthed(true);
  };

  const register = async (input: {
    email: string;
    password: string;
    full_name: string;
  }) => {
    await dp.register(input);
    // 가입 후 곧바로 로그인하여 토큰 확보
    await dp.login(input.email, input.password);
    setAuthed(true);
  };

  const logout = () => {
    dp.logout();
    setAuthed(false);
  };

  return (
    <AuthContext.Provider
      value={{ isAuthenticated, doctor, login, register, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
