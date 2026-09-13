---
date: 2026-09-12T17:04:12-07:00
git_commit: 20849477963f12a9beaa29d96f949a8b6fb4d33b
branch: main
repository: 2026_09_12_fly_parsing_pdfs
topic: "Character segmentation, reconstruction, and predefined output structure"
tags: [research, codebase]
status: complete
last_updated: 2026-09-12
last_updated_by: codex
---

# Research: Segmentation and output schema

## Research Question

Which algorithm scans and reconstructs document characters, which operations belong to the fly circuit, and do the demos require a predefined output schema?

## Summary

Conventional pixel geometry locates lines and glyphs before circuit inference. A frozen circuit processes one normalized glyph at a time, and an external trained readout predicts its character. Conventional code concatenates those predictions and inserts geometrically inferred spaces. Numeric table mode infers columns and row alignment from pixels.

Recognition does not require an expected transcription, field schema, or row/column count. The selected crop, recognition mode, and trained alphabet are supplied beforehand. Video placeholders use the detected structure from a completed recorded run. Expected values and shapes belong to separate post-inference evaluation.

## Detailed Findings

### Pixel segmentation and reconstruction

- `src/flyocr/pdf/segment.py:14` thresholds grayscale pixels, removes long horizontal rules, finds horizontal ink bands, merges nearby bands, and finds vertical ink runs within each band. These runs become candidate glyph boxes, ordered by position.
- `src/flyocr/pdf/text.py:10` adds a narrow geometric rule for splitting some touching glyph pairs. It explicitly excludes general ligature and cursive segmentation.
- `src/flyocr/pdf/text.py:43` estimates a line baseline and cap height, normalizes glyphs, and inserts a space prefix when an inter-glyph gap exceeds a height- and spacing-based threshold. Spaces are not predicted classes.
- `src/flyocr/demo/record.py:28` iterates through the previously detected rows and glyphs. Line text is the concatenation of each glyph's space prefix and predicted character. No dictionary or language model participates.

### Circuit boundary

- `src/flyocr/readout/predict.py:25` encodes the glyph through the brain, applies the trained readout to spike counts, and chooses a character from the checkpoint alphabet.
- `src/flyocr/brain/model.py:133` restores the canonical circuit state before each glyph by default. Glyphs do not share a document-reading memory.
- `src/flyocr/demo/record.py:47` declares fly motion a presentation effect. The animated fly does not determine the segmentation, scanning order, or reading direction.

### Numeric table structure

- `src/flyocr/pdf/table.py:17` removes horizontal rules, estimates character height, and identifies wide vertical whitespace as column separators. Partial rules constrain separators to keep some separated accounting symbols with their amounts.
- `src/flyocr/pdf/table.py:57` aligns independently segmented columns by vertical position. Row and column counts are derived from these detections.
- `src/flyocr/pdf/table.py:103` writes raw strings into the inferred rectangular matrix. It does not interpret semantic headers, repair numbers, or infer merged cells.
- `src/flyocr/demo/record.py:16` rejects combining the current letter checkpoint with table mode.

### Why the demo looks predefined

- `src/flyocr/demo/record.py:12` accepts an image, graph, recognition artifact, output directory, and table-mode flag. There is no truth or output-schema argument. Segmentation precedes the inference loop.
- `viewer/lib/replay.ts:52` reveals characters from recorded events. The grid at line 54 uses the saved run's detected dimensions, and the text placeholders at line 62 use its saved detected rows. Consequently, the preview already knows the layout before showing the first prediction.
- `viewer/lib/replay.ts:61` displays an expected shape only as an evaluation annotation. Check marks and exact-cell counts also come from evaluation.
- `scripts/record-letters.py:21` selects and renders a declared PDF crop, then records inference. The separate truth file is passed to the evaluator at line 28 after the run has been saved. Line 29 explicitly notes that PDF pixels informed segmentation development; these examples are not a general OCR benchmark.

### Other input crops

`src/flyocr/cli.py:50` accepts a user-supplied image and selects the numeric or letter artifact. The modes can process new crops without a new transcription or output schema, but preserve their documented assumptions: upright text regions, isolated numeric columns, or whitespace-separated numeric tables. They do not implement arbitrary page layout analysis, semantic document extraction, or a general multilingual alphabet.

## Code References

- `src/flyocr/pdf/segment.py:14` — pixel projections and glyph boxes.
- `src/flyocr/pdf/text.py:43` — line normalization and geometric spaces.
- `src/flyocr/pdf/table.py:17` — inferred numeric table structure.
- `src/flyocr/demo/record.py:12` — preprocessing, per-glyph inference, and concatenation.
- `src/flyocr/readout/predict.py:25` — activity-to-character readout.
- `src/flyocr/brain/model.py:133` — canonical reset per glyph.
- `viewer/lib/replay.ts:52` — replay placeholders and evaluation overlays.
- `scripts/record-letters.py:21` — separate inference and truth-based evaluation.

## Architecture Notes

The trained character recognizer is embedded inside a conventional layout pipeline. A fixed output file format (rows, boxes, raw strings, and optionally a table matrix) is distinct from a user-defined extraction schema. The former exists; the latter is not required by this implementation.

## Open Questions

None for the implementation boundary above. Accuracy on an arbitrary new document crop must be measured on that crop; the code's ability to accept an image is not evidence of reliable recognition across layouts.
