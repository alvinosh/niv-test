import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { viteStaticCopy } from "vite-plugin-static-copy"

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), viteStaticCopy({
    targets: [
      {
        src: "node_modules/onnxruntime-web/dist/*.wasm",
        dest: "./public",
      },
    ],
  }),],
  optimizeDeps: {
    exclude: ["@ffmpeg/ffmpeg", "@ffmpeg/util"],
  },

  assetsInclude: ["**/*.onnx"],
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
})
