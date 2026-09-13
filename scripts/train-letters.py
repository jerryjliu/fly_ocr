"""Generate the declared corpus once, then resume local readout training."""
import argparse
from pathlib import Path
from flyocr.common import read_json
from flyocr.data.glyphs import generate, contact_sheet, ALPHANUMERIC
from flyocr.experiments.letters import run


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.workers<=4: parser.error("Choose 1–4 local workers")
    if Path('artifacts/letters/recognition.json').exists() and read_json('artifacts/letters/model-card.json').get('readout_type')=='mlp':
        print('The original letter checkpoint is complete. Use --artifact artifacts/letters, or train-letters-v2.py for the calibrated model.');return
    corpus=Path("data/glyphs-letters")
    if not (corpus/"manifest.json").exists():
        generate(corpus,chars=ALPHANUMERIC,per_class={"train":96,"validation":16,"test":24},
                 seed=20260913,normalization="line-box-v1")
    expected={"train":96,"validation":16,"test":24}
    cm=read_json(corpus/"manifest.json")
    if cm["characters"]!=ALPHANUMERIC or cm["per_class"]!=expected or cm["normalization"]!="line-box-v1":
        raise ValueError("Letter corpus differs from declared protocol")
    Path("reports/letters").mkdir(parents=True,exist_ok=True)
    contact_sheet(corpus,"train","reports/letters/training-samples.png")
    run(corpus=corpus,workers=args.workers)


if __name__=="__main__": main()
