"""Run the bounded local calibration/readout experiment and its document checks."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--videos",action="store_true",help="Also export and verify the three updated demo videos")
    args=parser.parse_args()
    if not 1<=args.workers<=2:parser.error("Choose one or two local workers")
    root=Path(__file__).resolve().parents[1]
    for name in ["extract-letters-v2.py","fit-letters-v2.py","evaluate-letters-v2.py","record-letters-v2.py"]:
        command=[sys.executable,str(root/'scripts'/name)]
        if name.startswith(("extract-","evaluate-")):command += ["--workers",str(args.workers)]
        subprocess.run(command,cwd=root,check=True)
    if args.videos:
        subprocess.run([sys.executable,str(root/'scripts/export-video.py'),"--examples",
            "calibrated-heading,calibrated-labels,calibrated-labels-more"],cwd=root,check=True)
        subprocess.run([sys.executable,str(root/'scripts/verify-example-videos.py'),"--calibrated"],cwd=root,check=True)
    subprocess.run([sys.executable,str(root/'scripts/write-letter-v2-results.py')],cwd=root,check=True)


if __name__=="__main__":main()
