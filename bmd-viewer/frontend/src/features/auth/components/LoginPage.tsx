// Login / 회원가입 화면.
// 의사 계정이 없으면 가입 탭에서 만들고, 있으면 Login한다.
import { useState } from "react";
import { useAuth } from "../AuthContext";

type Mode = "login" | "register";

export function LoginPage() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setError(null);
    setBusy(true);
    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        if (!fullName.trim()) throw new Error("Please enter your name.");
        await register({ email, password, full_name: fullName });
      }
    } catch (e: unknown) {
      const msg =
        (e as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ??
        (e as Error)?.message ??
        "An error occurred.";
      setError(typeof msg === "string" ? msg : "An error occurred.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth-screen">
      <div className="auth-card">
        <div className="auth-brand">
          <span className="brand">BMD Viewer</span>
          <span className="subtitle">BMD Result Viewer</span>
        </div>

        <div className="auth-tabs">
          <button
            className={mode === "login" ? "active" : ""}
            onClick={() => {
              setMode("login");
              setError(null);
            }}
          >
            Login
          </button>
          <button
            className={mode === "register" ? "active" : ""}
            onClick={() => {
              setMode("register");
              setError(null);
            }}
          >
            Create account
          </button>
        </div>

        <div className="auth-form">
          {mode === "register" && (
            <label>
              Name
              <input
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="John Doe"
                autoComplete="name"
              />
            </label>
          )}
          <label>
            Email
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="doctor@hospital.com"
              autoComplete="username"
              onKeyDown={(e) => e.key === "Enter" && submit()}
            />
          </label>
          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === "register" ? "8+ characters" : "Password"}
              autoComplete={
                mode === "register" ? "new-password" : "current-password"
              }
              onKeyDown={(e) => e.key === "Enter" && submit()}
            />
          </label>

          {error && <p className="auth-error">{error}</p>}

          <button className="auth-submit" onClick={submit} disabled={busy}>
            {busy
              ? "Processing…"
              : mode === "login"
                ? "Login"
                : "Create account & start"}
          </button>
        </div>
      </div>
    </div>
  );
}
