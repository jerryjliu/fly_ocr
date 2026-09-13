# Fly OCR replay viewer

A static React/Vite application that draws recorded inference. Requires Node.js 22.13+.

```sh
npm ci
npm run dev
npm run build
npm run check:replay
```

The prebuild/predev step copies the repository's source recordings into generated `public/demo` and `public/examples` directories. No graph download, server, cloud credential or document upload is required. `dist` is a static site. See the root README for full reproduction and video export instructions.

Input UV coordinates and schematic eye-atlas coordinates are separate. Samples, neural counts and predictions remain tied to the same recorded event. The renderer never corrects strings or uses evaluation truth as a recognition input.
