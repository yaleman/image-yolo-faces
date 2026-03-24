import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tailwindcss()],
  build: {
    emptyOutDir: true,
    manifest: "manifest.json",
    outDir: "image_yolo_faces/static/dist",
    rollupOptions: {
      input: "frontend/main.ts",
    },
  },
});
