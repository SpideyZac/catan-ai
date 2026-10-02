import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the Python server runs on :8000; Vite proxies API and WebSocket traffic to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
