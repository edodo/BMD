import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DashboardPage } from "./DashboardPage";
import { AuthProvider, useAuth } from "@/features/auth/AuthContext";
import { LoginPage } from "@/features/auth/components/LoginPage";
import "@/styles.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

function AppContent() {
  const { isAuthenticated, doctor, logout } = useAuth();
  const [comparePatients, setComparePatients] = useState(false);

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <span className="brand">BMD Viewer</span>
        <span className="subtitle">BMD Result Viewer</span>
        {doctor && <span className="current-user">{doctor.full_name}</span>}
        <div className="topbar-actions">
          <button
            className={`compare-toggle${comparePatients ? " active" : ""}`}
            onClick={() => setComparePatients((v) => !v)}
          >
            {comparePatients ? "✕ Close" : "⇄ Compare patients"}
          </button>
          <button className="logout-btn" onClick={logout}>
            Logout
          </button>
        </div>
      </header>
      <DashboardPage comparePatients={comparePatients} />
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </QueryClientProvider>
  );
}
