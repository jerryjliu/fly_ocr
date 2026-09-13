# Calibrated inputs and a compact nonlinear readout

The new experiment preserves all 166,700 retained neurons, 25,582,938 connections, internal weights, electrical dynamics, and 100 ms stimulus timing. A fixed geometric input adapter improves image coverage. Only the external character readout is trained.

## Input adapter

The original retinal projection has 3,335 inputs but only 825 distinct sample locations. Its bilinear sampler reaches 52.0% of the 48×48 input pixels and 12.7% of the lower-left quadrant. About 51.1% of training glyph ink intensity lies outside its sampling support. See the [coverage diagnosis](../research/2026-09-12-retinal-coverage-and-accuracy.md).

The adapter assigns those 825 locations to a 33×25 rectangular grid covering the image. A minimum-cost assignment minimizes total squared displacement from the original coordinates. Receptors that shared a sampling location still share one. The assignment uses coordinates only, with no character labels, source text, or fitted image encoder.

Every input pixel can now influence at least one bilinear sample. This does not mean all 2,304 pixel values can be independently recovered from 825 locations: sampling still limits spatial resolution. It eliminates the large region whose pixels previously had no influence at all.

This is an engineered pixel-to-retina adapter, not a validated reconstruction of fly optics. The original graph files and original models are retained. The adapter has its own checksum and changes the model identity. Training and inference load the same frozen coordinate array.

Each glyph still starts from the same canonical circuit state, runs once for 100 ms, and produces four bins of spike counts. There is no additional circuit pass, simulated eye movement, learned scan policy, or pixel shortcut into the decoder.

## Bounded readout comparison

The experiment keeps 1,024 downstream feature cells, selected by the same training-only variance method as the original letter model. It compares four heads on cached responses:

| Candidate | Count transform | Hidden layers | Learned parameters |
|---|---|---|---:|
| Original architecture on new inputs | Identity | 64 ReLU units | 266,628 |
| Compressed counts | Square root | 64 ReLU units | 266,628 |
| Small second layer | Square root | 64 → 32 ReLU units | 266,532 |
| Larger bounded candidate | Square root | 128 → 64 ReLU units | 537,092 |

The square root compresses large spike counts before training-derived standardization. Two small nonlinear layers can express interactions among circuit responses without a large decoder. All heads receive only the 4,096 recorded spike-count features.

Training uses 6,528 glyphs from the original six training font families. The 1,088 validation glyphs choose the head and stopping epoch. Each candidate uses AdamW, fixed seed 817, at most 120 epochs, 18-epoch patience, and the same learning rate, regularization, and dropout. The best validation accuracy wins; ties prefer fewer parameters. Two local workers generate responses, and training reuses cached responses for all four heads.

The selected NumPy artifact must exactly reproduce Torch's validation predictions before it can be frozen. Inference requires neither Torch nor a GPU.

## Evaluation boundaries

The original 1,632-character test split provides a matched regression comparison. Its fonts and errors were already observed during earlier work, so it is not a fresh independent test for the entire development process. No v2 head selection uses those labels.

A second test uses two previously unused font families, Verdana and Georgia, with four generated examples per character per family (544 images). It is generated only after the new checkpoint is frozen, and both original and new models process the same images. Font binaries remain local and are not redistributed. Family-level diversity is still small.

The three document examples repeat exactly the same source crops, segmenter, and word-space heuristics as the earlier demonstrations. Their raw outputs are scored afterward against the same separate transcriptions. These are useful comparisons, not independent general OCR benchmarks.

Checks also verify unchanged graph arrays, saved input samples against actual glyph pixels, readout predictions against recorded counts, and the loss of recognition when neural responses are mismatched to glyph labels. Decoder latency is measured separately from circuit latency.

## Reproduce locally

After the original graph and letter corpus have been prepared:

```sh
uv run --extra benchmarks python scripts/train-letters-v2.py --workers 2
```

The stages can also be run separately with `extract-letters-v2.py`, `fit-letters-v2.py`, `evaluate-letters-v2.py`, and `record-letters-v2.py`. Extraction resumes checked shards. A completed classifier is frozen; changing the declared protocol requires a separate experiment directory. The fresh-font challenge currently expects the listed macOS system fonts.

With the optional video dependencies and the viewer dependencies installed, add `--videos` to render and fully decode-check the three updated demonstrations.

The original numeric checkpoint and numeric table pipeline remain separate. The calibrated letter checkpoint accepts upright text regions and retains the same closed alphabet and segmentation limitations.
