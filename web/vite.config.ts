// Vite builds the dashboard into dist/; the workflow copies it next to the
// exported data/ folder and deploys both to GitHub Pages.
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Relative paths ("./assets/...") so the same build works at
  // crystalge07.github.io/F1_Predictor/, at a custom domain, or locally.
  base: "./",
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
