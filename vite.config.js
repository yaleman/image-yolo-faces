import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tailwindcss()],
  build: {
    emptyOutDir: true,
    manifest: "manifest.json",
    sourcemap: true,
    outDir: "image_yolo_faces/static/dist",
    rollupOptions: {
      input: "frontend/main.ts",
      output: {
        entryFileNames: "assets/[name].js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: (assetInfo) => {
          const assetName = assetInfo.name;
          if (!assetName) {
            return "assets/[name][extname]";
          }

          const extname = path.extname(assetName);
          const basename = path.basename(assetName, extname);
          return `assets/${basename}${extname}`;
        },
      },
    },
  },
});
