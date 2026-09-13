# The compound-eye view

The video now plots recorded sensory signals in a **schematic two-eye column atlas**. Input sampling UVs and drawing positions are independent. The calibrated model still samples the full glyph; each sample retains its receptor identity when drawn in the atlas. No retraining or extra circuit pass is needed, and all recorded predictions remain identical.

![Two-eye atlas and input coordinates](../demo/compilation/mapping.png)

The atlas derives left/right sides from MaleCNS annotations and uses the original inferred R1-R6 to lamina-column coordinates. The renderer separates overlapping eyes and applies a square-to-oval drawing warp. Hexagons are schematic glyphs for shared column positions, not reconstructed ommatidia. The scale maps sampled brightness monotonically to copper luminance. Blank areas are missing atlas positions, not modeled blind spots; dark facets are sampled dark ink. This is not a retinal firing plot or a reconstruction of a fly's subjective visual field.

There are 3,335 retained receptor inputs (1,107 left, 2,228 right) and 825 distinct image sites. Photoreceptors sharing an original site continue to share an input under the calibration. `artifacts/retinal-atlas.json` records receptor IDs, sides, coordinates and its identity; new graph preparation writes the same sidecar under `data/graph`. Neither file changes the connectivity or trained model identity.

`run.retina_uv` remains the actual input sampler. `run.retinal_atlas` supplies display geometry. Every event's `retina` vector remains its measured brightness samples. For older runs without the atlas, the renderer falls back to the original point-map view. The original-input letter model is available with `--artifact artifacts/letters`; the calibrated model with `--letters`. Comparing them changes the actual input and checkpoint, not merely the drawing.

After preparing the source graph, `uv run python scripts/build-retinal-atlas.py` rebuilds the atlas and attaches it to existing recordings while checking that every prior inference field remains unchanged. The checked report is in `reports/retinal-atlas/verification.json`. This operation checks source sampling values, not the biological validity of the approximate projection.

Real fly neural superposition distinguishes eye-surface ommatidia from lamina cartridges: [Langen et al., Cell 2015](https://doi.org/10.1016/j.cell.2015.05.055). This view deliberately labels itself as a column atlas.
