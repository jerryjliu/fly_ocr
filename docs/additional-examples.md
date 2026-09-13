# Additional annual-report examples

Four 40-second videos use the **same frozen numeric checkpoint** as the original demo. They were recorded locally on the M2 Pro. There was no letter training, connectome change, decoder refit, language model, spellcheck, numeric repair, or embedded PDF text extraction.

| Video | Input | Exact cells | Structure | Measured inference |
|---|---|---:|---|---:|
| [Comprehensive income](../demo/examples/income-table/video.mp4) | PDF page 41; all three year columns, including numeric year headings | 21/21 | 7 × 3, correct | 35.4 s |
| [Cash-flow formatting](../demo/examples/cash-flow/video.mp4) | PDF page 43; complete numeric financing and investing blocks | 44/45 | 15 × 3, correct | 99.3 s |
| [Tilted scan](../demo/examples/tilted-table/video.mp4) | First crop rotated 3° counterclockwise | 2/21 at the expected cell positions | 19 × 3 instead of 7 × 3 | 38.9 s |
| [English letters](../demo/examples/letters/video.mp4) | Actual “BALANCE SHEETS” heading, PDF page 42 | 0/1 | One line; unsupported alphabet | 4.4 s |

These are selected demonstrations, not an independent general OCR/table benchmark. The layout heuristic was developed while inspecting these source images. Regions and transcriptions were declared before the new predictions; the two initial year cells were printed by an interrupted development run before a geometry-test fix. That fix used rule spans and whitespace, not class predictions. The numeric checkpoint and all recognition settings remained unchanged. No region or cell was removed after seeing results.

## What is trained?

The first head recognizes ten digits. The checkpoint used in all these videos recognizes 16 classes: `0123456789,.-()$`. It cannot emit an English letter. For the 13 heading glyphs it returned `848.568588875`, despite sometimes high class scores. Scores compare the available classes and are not a detector for unsupported input.

Recognizing letters would require a new labeled character corpus, readout training on the resulting circuit activity, and separate held-out evaluation. The implementation could support such an experiment, but letter accuracy has not been established. The current pipeline also loses spaces and has no paragraph/word decoder.

## What formatting worked, and what failed?

The upright crops contain bold and regular text, comma-separated values, dollar signs separated from their amounts, parenthesized negatives, zeros, horizontal rules, and uneven row spacing. The cash-flow error is `(28,103)` → `(28,107)` at row 9, column 3. Its glyph error rate is 1/299 (0.334%). Comprehensive income has no errors across 108 glyphs. The earlier income-statement example also includes decimals, but neither font variation nor these examples establishes support for arbitrary formatting.

The 3° stress test preserves the source values but leaves out deskewing. Angled accounting rules survive horizontal rule removal and become false glyphs, often decoded as `)`. Baselines in different columns no longer align. Many numeric strings survive, but appear in the wrong table positions. Its 170.4% position-sensitive character error rate counts missing, substituted, and extra strings; edit-error rates can exceed 100% when there are insertions. It is a layout failure, not a claim that 170% of individual glyphs are wrong.

No tests here establish handwriting, arbitrary rotations, touching characters, multi-column prose, superscripts, complex merged cells, or noisy scans.

## What “read a table” means here

The new `--table` mode takes a **manually selected numeric table region**. A conventional geometry step removes long horizontal rules, finds wide vertical whitespace, uses per-cell rule spans to keep spaced currency symbols with values, and aligns row baselines across columns. It does not receive the expected number of columns, expected strings, or evaluator labels. Each isolated glyph then goes through the fixed circuit and trained head. The geometry step assembles the raw strings into a matrix.

[Income CSV](../demo/examples/income-table/table.csv) · [Income JSON](../demo/examples/income-table/table.json) · [Cash-flow CSV](../demo/examples/cash-flow/table.csv) · [Cash-flow JSON](../demo/examples/cash-flow/table.json)

CSV preserves raw text, punctuation, and empty positions. JSON also includes boxes, row/column indices, and artifact identity. There is no conversion of parentheses into signed numbers, semantic row naming, unit inference, merged-header interpretation, or automatic discovery of tables on a full page. “C1/C2/C3” in the video are display coordinates. The year numbers in the first table are actual predictions.

## Reproduce

The pinned report is already fetched by `scripts/fetch-example.py`. Example specifications and separately stored labels are in `examples/microsoft-2025/`. Source checksums, crop geometry, and the declared rotation are saved with every recorded run.

```sh
uv run python scripts/more-examples.py
uv run --extra video python scripts/export-video.py --examples income-table,cash-flow,tilted-table,letters
```

To apply the same numeric table pipeline to another upright raster crop:

```sh
uv run flyocr recognize your-table-crop.png --table --output output/your-table
uv run flyocr verify-replay --run output/your-table
```

Inference imports no evaluator. Scoring runs only after the entire event record is saved. Replay verification recomputes predictions from recorded activity and the checkpoint, then checks table JSON/CSV against those same predictions and positions. The viewer bundle contains correctness flags and aggregate scores, but no expected transcription strings. Videos use edited replay timing and show measured inference time; the fly is an animated cursor, not a simulated body. There is no audio track.
