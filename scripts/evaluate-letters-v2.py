"""Evaluate frozen input/readout changes, plus two fresh local-font families."""
import argparse
from pathlib import Path
import time
import numpy as np
from flyocr.brain.model import Brain, StimulusConfig
from flyocr.brain.batch import pool, extract
from flyocr.common import read_json, save_json, digest_file, identity
from flyocr.readout.train import probabilities
from flyocr.eval.metrics import classification
from flyocr.vision.letterbox import render_letter


def results(labels, predicted, characters, groups):
    classes = np.arange(len(characters))
    out = {"neural": classification(labels,predicted,classes),"by_group":{},"by_font":{}}
    for name, alphabet in {"uppercase":"ABCDEFGHIJKLMNOPQRSTUVWXYZ","lowercase":"abcdefghijklmnopqrstuvwxyz",
                            "letters":"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz","digits":"0123456789","punctuation":",.-()$"}.items():
        mask = np.isin(labels,[characters.index(c) for c in alphabet])
        out["by_group"][name] = classification(labels[mask],predicted[mask],classes)
    for font in sorted({g["family"] for g in groups}):
        mask = np.array([g["family"]==font for g in groups])
        out["by_font"][font] = classification(labels[mask],predicted[mask],classes)
    return out


def fresh_corpus(card, protocol):
    output = Path("data/glyphs-letters-fresh"); output.mkdir(parents=True,exist_ok=True)
    fonts = {"Verdana":Path("/System/Library/Fonts/Supplemental/Verdana.ttf"),
             "Georgia":Path("/System/Library/Fonts/Supplemental/Georgia.ttf")}
    declaration = protocol["fresh_holdout"]
    manifest = {"fonts":{k:{"path":str(v),"sha256":digest_file(v)} for k,v in fonts.items()},
        "seed":declaration["seed"],"samples_per_font_per_class":declaration["samples_per_font_per_class"],
        "checkpoint_frozen_before_generation":card["artifact_id"],
        "normalization":"line-box-v1","characters":card["characters"],
        "scope":"Two previously unused font families; local font files are not redistributed"}
    if (output/"manifest.json").exists():
        if read_json(output/"manifest.json") != manifest: raise ValueError("Fresh challenge protocol mismatch")
        return output
    rng = np.random.default_rng(declaration["seed"])
    images,labels,groups = [],[],[]
    for label,char in enumerate(card["characters"]):
        for family,path in fonts.items():
            for _ in range(declaration["samples_per_font_per_class"]):
                images.append(np.asarray(render_letter(char,path,rng),np.uint8));labels.append(label)
                groups.append({"family":family,"character":char})
    order = rng.permutation(len(images))
    np.savez_compressed(output/"test.npz",images=np.asarray(images)[order],labels=np.asarray(labels,np.int16)[order])
    save_json(output/"test.json",[groups[i] for i in order]);save_json(output/"manifest.json",manifest)
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--workers",type=int,default=2);args=parser.parse_args()
    if not 1<=args.workers<=2:parser.error("Capped at two local workers")
    output=Path("artifacts/letters-v2");report_dir=Path("reports/letters-v2");report_dir.mkdir(parents=True,exist_ok=True)
    if all(p.exists() for p in [output/"recognition.json",output/"fresh-recognition.json",output/"fresh-reference.json",report_dir/"verification.json"]):
        print("Frozen checkpoint already evaluated.");return
    card=read_json(output/"model-card.json");protocol=read_json(output/"training-protocol.json")
    if identity({k:v for k,v in card.items() if k!="artifact_id"})!=card["artifact_id"]:raise ValueError("Card changed")
    if digest_file(output/"readout.npz")!=card["readout_sha256"]:raise ValueError("Head changed")
    config=StimulusConfig(**card["config"]);uv=np.load(output/"retina-uv.npy")
    payload=dict(np.load(output/"readout.npz",allow_pickle=False));neurons=payload["neurons"]
    previous=read_json("artifacts/letters/model-card.json")
    old_payload=dict(np.load("artifacts/letters/readout.npz",allow_pickle=False))
    fresh=fresh_corpus(card,protocol)
    begin=time.perf_counter()
    with pool("data/graph",config,args.workers,retina_uv=uv) as executor:
        for name,corpus in [("matched",Path("data/glyphs-letters")),("fresh",fresh)]:
            x,benchmark=extract(executor,card["model_id"],corpus,"test",neurons,root="data/letters-v2-shards")
            labels=np.load(corpus/"test.npz")["labels"];p=probabilities(payload,x);pred=p.argmax(1)
            groups=read_json(corpus/"test.json")
            report={"artifact_id":card["artifact_id"],"characters":card["characters"],
                **results(labels,pred,card["characters"],groups),"benchmark":benchmark,
                "scope":"Previously used font benchmark; no v2 selection on test" if name=="matched" else "New font families generated after checkpoint freeze",
                "validation":card["fit"],"head_comparison":{}}
            for candidate in protocol["head_candidates"]:
                head=dict(np.load(output/("readout-"+candidate["name"]+".npz"),allow_pickle=False))
                candidate_pred=probabilities(head,x).argmax(1)
                report["head_comparison"][candidate["name"]]={"accuracy":float((candidate_pred==labels).mean()),"correct":int((candidate_pred==labels).sum()),"n":len(labels)}
            rng=np.random.default_rng(817)
            shuffled=probabilities(payload,x[rng.permutation(len(x))]).argmax(1)
            report["mismatched_neural_responses"]={"accuracy":float((shuffled==labels).mean()),
                "purpose":"Confirm that predictions depend on the matching glyph's circuit response; no pixels enter this readout"}
            destination=output/("recognition.json" if name=="matched" else "fresh-recognition.json")
            np.savez_compressed(output/("heldout-predictions.npz" if name=="matched" else "fresh-predictions.npz"),
                labels=labels,predictions=pred,probabilities=p,features=x)
            save_json(destination,report)
            print(name,report["neural"]["accuracy"],report["by_group"]["letters"]["accuracy"],flush=True)
    # The original model is evaluated on the same new images for a paired comparison.
    with pool("data/graph",StimulusConfig(**previous["config"]),args.workers) as executor:
        old_x,old_benchmark=extract(executor,previous["model_id"],fresh,"test",old_payload["neurons"],root="data/letters-v2-reference-shards")
    labels=np.load(fresh/"test.npz")["labels"];groups=read_json(fresh/"test.json")
    old_pred=probabilities(old_payload,old_x).argmax(1)
    new_pred=np.load(output/"fresh-predictions.npz")["predictions"]
    old_report={"artifact_id":previous["artifact_id"],**results(labels,old_pred,card["characters"],groups),"benchmark":old_benchmark,
        "paired":{"new_correct_old_wrong":int(((new_pred==labels)&(old_pred!=labels)).sum()),
                  "old_correct_new_wrong":int(((new_pred!=labels)&(old_pred==labels)).sum())}}
    save_json(output/"fresh-reference.json",old_report)
    graph_manifest=read_json("data/graph/manifest.json")
    preserved={k:digest_file(Path("data/graph")/(k+".npy"))==graph_manifest["array_hashes"][k] for k in ["ptr","post","weight","uv"]}
    if not all(preserved.values()):raise ValueError("Source graph changed")
    # Benchmark the actual NumPy decoder separately from circuit timing.
    sample=np.load(output/"heldout-predictions.npz")["features"][:128]
    timings={}
    for name,head in [("original",old_payload),("selected",payload)]:
        for row in sample[:5]:probabilities(head,row[None,:])
        start=time.perf_counter()
        for _ in range(5):
            for row in sample:probabilities(head,row[None,:])
        timings[name]=(time.perf_counter()-start)/(5*len(sample))
    save_json(report_dir/"verification.json",{"artifact_id":card["artifact_id"],"source_arrays_unchanged":preserved,
        "decoder_seconds_per_glyph":timings,"evaluation_wall_seconds":time.perf_counter()-begin,
        "new_fresh_accuracy":float((new_pred==labels).mean()),"old_fresh_accuracy":float((old_pred==labels).mean())})
    print("Fresh comparison",old_report["neural"]["accuracy"],"->",float((new_pred==labels).mean()),flush=True)


if __name__=="__main__":main()
