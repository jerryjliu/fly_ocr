# A fly attempts to read a PDF

The separate **105-second social cut** starts on the first recognized character,
without the original compilation's opening slides. It retains all seven OCR
examples, with a short hold on each completed result. The original 176-second
compilation and research report are preserved.

[Watch / download the social cut](https://github.com/jerryjliu/fly_ocr/releases/download/social-v1/fly-ocr-social.mp4)

![First frame: character C, eye samples, real prediction and the animated fly](../demo/social/trained-labels-start.png)

## What is rendered

The right panel is original procedural 3D artwork: a fly with red textured eyes,
six articulated legs, a segmented abdomen, bristles, antennae, halteres and a
pair of translucent veined wings. The paper uses the attributed source-page
image, with the actual processed crop inserted in its selected region. That
includes the deliberately tilted crop in the last scene.

The visual idea follows fly-body demonstrations such as
[DOOMFLY](https://github.com/nftechie/doomfly) and
[NeuroMechFly](https://neuromechfly.org/). No external fly mesh, biomechanics
engine or locomotion controller is imported. In particular, this is **not a
NeuroMechFly simulation** and does not add a neural body-control claim to OCR.

The page marker uses the current recorded glyph box. The fly's position moves
between successive box centers, with a head-to-marker offset; leg, head and
wing motion use deterministic time functions. It is an illustrative animated
scan follower. It has no effect on sensory sampling, circuit dynamics or the
decoder. Its eye texture is decorative, independent of the separate retinal
panel. Fly scale, shape, optics and gait are not biologically calibrated.

## What remains measured

The glyph image, receptor samples, downstream counts, prediction, class scores
and measured per-glyph wall time are read from each existing `run.json`.
Text is assembled from recorded character and spacing events. Table positions
come from the existing geometric layout. Reference transcriptions are not
used to produce the displayed OCR; evaluation metadata colors completed table
cells and reports the saved scores.

The eye panel uses the same receptor-identity atlas as the research compilation.
The activity panel sums adjacent neurons into 128 display rows: eight neurons
per row for letter recordings, four for numeric recordings. It retains all
four time bins. This grouping changes the visualization only. Numeric runs
use 512 downstream cells; letter runs use 1,024.

IBM Plex Mono makes capital `I`, lowercase `l` and digit `1` distinguishable in
the displayed text, so errors such as `GoodwiII` remain visible. Timing is
edited for the video, not a real-time performance demonstration. The video is
captioned and silent, with disclosures embedded in every scene.

## Edit list

| Start | Duration | Recorded example |
| --- | --- | --- |
| 0:00 | 13 s | Original-input letter labels |
| 0:13 | 19 s | Calibrated letter labels |
| 0:32 | 10 s | Uppercase heading |
| 0:42 | 17 s | Five mixed-case labels |
| 0:59 | 17 s | Income table: 21/21 exact cells |
| 1:16 | 16 s | Cash-flow table: 44/45 exact cells |
| 1:32 | 13 s | 3-degree tilt: segmentation failure |

Each chapter begins on glyph 1, exposes every event at least once, and reserves
roughly 2.4 seconds for the completed raw result. The saved inference has not
been rerun or modified for this edit.

## Reproduce

Install the Python video extra and viewer dependencies described in the main
README. Use Node.js 22.13 or newer and Chrome/Chromium with WebGL support.

```sh
cd viewer
npm ci
npx playwright install chromium  # optional if installed macOS Chrome is available
cd ..
uv run --extra video python scripts/export-social.py
```

`FLYOCR_CHROME` selects another installed Chrome executable.
`FLYOCR_PREVIEW_ONLY=1` renders first, middle and last stills of each chapter
without replacing the MP4. Rendering takes place on a loopback-only temporary
server and uses local bundled assets. The server and browser close after export.

The output is `demo/social/fly-ocr-social.mp4`, H.264, 1920 × 1080, 24 fps,
2,520 frames, with fast-start metadata and no audio. `demo/social/manifest.json`
records source and renderer SHA-256 fingerprints. Run
`uv run --extra video python scripts/verify-social-media.py` to decode all
frames, check those fingerprints, verify event coverage and confirm that the
original compilation remains unchanged.

Implementation: `viewer/lib/social-fly.ts`, `social-layout.ts`,
`social-timeline.ts`, and `viewer/scripts/export-social.mjs`.
