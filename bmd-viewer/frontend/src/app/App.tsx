import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DashboardPage } from "./DashboardPage";
import { AuthProvider, useAuth } from "@/features/auth/AuthContext";
import { LoginPage } from "@/features/auth/components/LoginPage";
import "@/styles.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

function AppContent() {
  const { isAuthenticated, logout } = useAuth();

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <span className="brand">BMD Viewer</span>
        <span className="subtitle">BMD Result Viewer</span>
        <button className="logout-btn" onClick={logout}>
          Logout
        </button>
      </header>
      <DashboardPage />
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
