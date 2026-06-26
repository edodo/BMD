import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DashboardPage } from "./DashboardPage";
import "@/styles.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="app-shell">
        <header className="topbar">
          <span className="brand">BMD Viewer</span>
          <span className="subtitle">골밀도 측정 결과 뷰어</span>
        </header>
        <DashboardPage />
      </div>
    </QueryClientProvider>
  );
}
