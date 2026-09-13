# Recorded experiment — 40 seconds

- **0–4 seconds:** Source PDF numeric column and first normalized glyph. No predictions or activity are revealed before an inference event.
- **4–32 seconds:** Reveal all 120 recorded glyph events in order. For each event, the source cursor, glyph image, sampled retinal brightness, downstream count matrix, class scores, predicted character and measured wall time use the same event record. Raw strings accumulate solely from those predictions.
- **32–40 seconds:** Hold all 19 raw strings, mark exact cells using the separate evaluation, and display actual cell exact match, character error rate and held-out numeric-glyph accuracy. Mistakes remain visible.

The viewer and MP4 use `viewer/lib/replay.ts`. Recorded inference is explicit in every frame. Neural activity uses four 25 ms spike-count bins across the selected downstream cells; there are no invented event timestamps. The fly is a schematic presentation cursor. The recorded retinal panel shows brightness before the configured contrast/current transform. Replay timing is edited for readability; measured wall time and simulated duration are labeled.

The final clip is an offline deterministic render, not a claim of live desktop capture or real-time embodied fly behavior. There is no audio track.
