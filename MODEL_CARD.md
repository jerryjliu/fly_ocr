# Model card

## Intended use and scope

A reproducible, exploratory demonstration of printed-character recognition through a fixed connectome model. It is not a production OCR system or evidence that a biological fly reads. Source pixels pass through heuristic segmentation and normalization before the circuit. Reconstruction of words and tables is geometric, outside the network.

## Released checkpoints

| Artifact | Input and trained decoder | Recorded glyph result |
| --- | --- | --- |
| `artifacts/letters-v2` (default `--letters`) | Calibrated UV adapter; square-root counts; 4,096 to 64 to 68 MLP, 266,628 parameters | 1,429/1,632 overall (87.6%); 1,061/1,248 letters (85.0%) |
| `artifacts/letters` | Original approximate UV map; 4,096 to 64 to 68 MLP | 1,222/1,632 overall (74.9%); 897/1,248 letters (71.9%) |
| `artifacts/numeric` (default numeric mode) | Original UV map; linear readout, 16 classes | 751/800 (93.9%) |
| `artifacts/digits` | Original UV map; linear readout, 10 classes | 445/500 (89.0%) |

Spaces are inferred from image gaps, not output classes. The letter alphabet is `0123456789,.-()$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`. The system has no reliable detector for arbitrary unsupported characters. Scores are not calibrated confidence probabilities.

## Fixed versus learned

The retained MaleCNS v1.0 graph contains 166,700 nodes and 25,582,938 edges. Internal contact-count weights and inferred transmitter signs remain fixed, as do simplified LIF dynamics, stimulus settings and the canonical reset. The letter input adapter is a label-free engineered mapping, not trained optics. Feature-cell selection, training normalization and the small decoder are fitted outside the circuit. There is no internal synaptic learning, dictionary, LLM, OCR service, embedded PDF text, semantic correction or learned eye/body movement.

The compound-eye atlas is a **display-only sidecar**. It shows recorded sampled brightness at positions derived from the original left/right column map. Ovals and hexagons are schematic; they are not reconstructed ommatidia or visual angles. Every recorded prediction is unchanged by the atlas. [Atlas methods](docs/retinal-atlas.md).

## Data and evaluation

Synthetic glyphs use six training, two validation and two benchmark font families, with disjoint families and pinned openly licensed files. The letter split contains 6,528 training, 1,088 validation and 1,632 benchmark samples. Configuration selection uses validation, but the benchmark had already been inspected during earlier iterations; it is not a new blind test. A fresh two-family challenge on 544 glyphs reaches 84.9% overall and 88.5% on its 416-letter subset. Shared fonts limit independence and generalization claims.

Three Microsoft 2025 Annual Report crops supply eight letter lines: the calibrated model makes 10 edits in 176 reference characters (5.7% CER), with only 1/8 exact lines. Numeric examples reach 18/19, 21/21 and 44/45 exact cells on selected upright regions. A controlled 3-degree tilt produces the wrong table grid. Expected strings and schemas are never inference inputs.

## Limitations and controls

Uniform cell dynamics, default-positive unknown signs, contrast inversion, tonic lamina drive and inferred visual projection are engineering assumptions. The calibration improves pixel support but weakens correspondence to original approximate visual fields. Manual crop selection, touching glyphs, skew, ambiguous I/l and S/5 shapes, unusual fonts and closed-alphabet errors remain important limitations. Letter mode and numeric table mode are separate; neither provides general whole-page parsing or semantic table understanding.

Mismatched neural responses reduce the calibrated benchmark to 2.3%. Earlier digit controls include edge lesions and separately retrained randomized-target graphs; activity-scale changes limit causal interpretation. A raw-pixel linear baseline scores 99% on the original digit test, above the circuit's 89%. These experiments do not establish a biological or practical OCR advantage. See the [full research report](docs/research-report.md).

## Runtime, integrity and licenses

Tested on Apple M2 Pro, 32 GiB RAM. The native C++ CPU circuit takes about 0.29 seconds per glyph on a small timing sample; each simulated presentation is 100 ms. The selected readout takes about 1 ms. Training uses PyTorch MPS; inference needs no PyTorch or GPU. No cloud deployment is claimed. Linux CI checks the source and saved replays; full-graph cross-platform equivalence has not been measured.

Model cards record graph, stimulus, kernel, readout and adapter identities. Replay verification checks source pixels, geometry, recorded feature-to-score computation, raw strings and atlas provenance. The public export's `PUBLIC_MANIFEST.json` fingerprints included files without including local Git history. The fly is an animated cursor; videos have edited timing.

Project code is MIT. DOOMFLY adaptations retain their MIT notice; MaleCNS is CC BY; bundled fonts retain SIL OFL terms; Microsoft report pixels retain separate rights. See [third-party notices](THIRD_PARTY_NOTICES.md).
