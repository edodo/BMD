import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      // 개발 중 백엔드 프록시 (CORS 회피)
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/derived": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
