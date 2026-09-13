"""Small explicit command interface; heavy experiments are opt-in."""
import argparse
from pathlib import Path
from flyocr.common import read_json


def main():
    parser = argparse.ArgumentParser(description="Fixed fly circuit + trained glyph readout")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download"); p.add_argument("--source", default="data/source"); p.add_argument("--config", default="configs/malecns-v1.json")
    p = sub.add_parser("prepare"); p.add_argument("--source", default="data/source"); p.add_argument("--graph", default="data/graph"); p.add_argument("--config", default="configs/malecns-v1.json")
    p = sub.add_parser("glyphs"); group=p.add_mutually_exclusive_group(); group.add_argument("--numeric", action="store_true"); group.add_argument("--letters", action="store_true"); p.add_argument("--output"); p.add_argument("--train-per-class", type=int); p.add_argument("--validation-per-class", type=int); p.add_argument("--test-per-class", type=int)
    p = sub.add_parser("pilot"); p.add_argument("--graph", default="data/graph"); p.add_argument("--corpus", default="data/glyphs-digits"); p.add_argument("--output", default="reports/pilot"); p.add_argument("--contrast", action="store_true")
    p = sub.add_parser("train"); p.add_argument("--graph", default="data/graph"); p.add_argument("--corpus", default="data/glyphs-digits"); p.add_argument("--pilot", default="reports/frozen-pilot"); p.add_argument("--output", default="artifacts/digits")
    p = sub.add_parser("controls"); p.add_argument("--graph", default="data/graph"); p.add_argument("--corpus", default="data/glyphs-digits"); p.add_argument("--artifact", default="artifacts/digits"); p.add_argument("--output", default="reports/controls")
    p = sub.add_parser("render"); p.add_argument("pdf"); p.add_argument("--page", type=int, default=40); p.add_argument("--dpi", type=int, default=300); p.add_argument("--region", nargs=4, type=float, default=[.685,.207,.780,.642]); p.add_argument("--output", default="data/pdf-raster")
    p = sub.add_parser("segment"); p.add_argument("image"); p.add_argument("--output", default="data/pdf-segmented")
    p = sub.add_parser("recognize"); p.add_argument("image", help="Raster image of an upright text or numeric region"); p.add_argument("--graph", default="data/graph"); group=p.add_mutually_exclusive_group(); group.add_argument("--artifact"); group.add_argument("--letters", action="store_true", help="Use calibrated retinal inputs and the compact nonlinear letter readout"); p.add_argument("--output", default="demo/run"); p.add_argument("--table", action="store_true", help="Infer numeric columns from pixel whitespace and emit a raw table")
    p = sub.add_parser("evaluate-pdf"); p.add_argument("--run", default="demo/run/run.json"); p.add_argument("--truth", default="examples/microsoft-2025/evaluation.json"); p.add_argument("--output", default="reports/pdf-crop.json")
    p = sub.add_parser("verify-replay"); p.add_argument("--run", default="demo/run"); p.add_argument("--artifact", default="artifacts/numeric")
    a = parser.parse_args()
    if a.command == "download":
        from flyocr.data.download import fetch
        for name, f in read_json(a.config)["files"].items():
            print(fetch(f["url"], Path(a.source)/name, f["sha256"], f["bytes"]))
    elif a.command == "prepare":
        from flyocr.data.connectome import prepare
        prepare(a.source, a.graph, a.config)
    elif a.command == "glyphs":
        from flyocr.data.glyphs import generate, DIGITS, NUMERIC, ALPHANUMERIC
        counts = {"train": a.train_per_class or (96 if a.letters else 150 if a.numeric else 100), "validation": a.validation_per_class or (16 if a.letters else 30 if a.numeric else 20), "test": a.test_per_class or (24 if a.letters else 50)}
        generate(a.output or ("data/glyphs-letters" if a.letters else "data/glyphs-numeric" if a.numeric else "data/glyphs-digits"), chars=ALPHANUMERIC if a.letters else NUMERIC if a.numeric else DIGITS, per_class=counts, normalization="line-box-v1" if a.letters else "ink-box-v1", seed=20260913 if a.letters else 20260912)
    elif a.command == "pilot":
        from flyocr.experiments.pilot import run
        from flyocr.brain.model import StimulusConfig
        configs = [StimulusConfig(sensory_gain=g, lamina_drive=l, half_saturation=.2, invert=inv) for inv in [False,True] for g in [12.,20.] for l in [12.,20.]] if a.contrast else None
        run(a.graph, a.corpus, a.output, configs=configs)
    elif a.command == "train":
        from flyocr.experiments.train import run
        run(a.graph, a.corpus, a.pilot, a.output)
    elif a.command == "controls":
        from flyocr.experiments.controls import run
        run(a.graph, a.corpus, a.artifact, a.output)
    elif a.command == "render":
        from flyocr.pdf.render import render
        render(a.pdf, a.output, a.page, a.dpi, a.region)
    elif a.command == "segment":
        from flyocr.pdf.segment import prepare
        prepare(a.image, a.output)
    elif a.command == "recognize":
        from flyocr.demo.record import record
        record(a.image, a.graph, a.artifact or ("artifacts/letters-v2" if a.letters else "artifacts/numeric"), a.output, table=a.table)
    elif a.command == "evaluate-pdf":
        from flyocr.eval.pdf import evaluate
        print(evaluate(a.run, a.truth, a.output))
    elif a.command == "verify-replay":
        from flyocr.demo.schema import verify
        print(verify(a.run, a.artifact))


if __name__ == "__main__": main()
