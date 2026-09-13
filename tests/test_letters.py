from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flyocr.vision.letterbox import render_letter
from flyocr.pdf.text import prepare_text, split_bridges
from flyocr.brain.model import Brain, StimulusConfig
from flyocr.brain.batch import pool, encode_job, extract
from flyocr.common import save_json, digest_file

FONT = "assets/fonts/lato/Lato-Regular.ttf"


def test_normalization_preserves_case_height_and_descenders():
    def bounds(char):
        im = np.asarray(render_letter(char,FONT,np.random.default_rng(23)))
        y,x = np.where(im<160)
        return y.min(),y.max()
    cap,lower,desc = bounds("C"),bounds("c"),bounds("g")
    assert lower[0]>cap[0]+4
    assert abs(lower[1]-cap[1])<=2
    assert desc[1]>lower[1]+3


def test_printed_text_preserves_dots_case_and_word_gaps(tmp_path):
    image=Image.new("L",(750,150),255)
    draw=ImageDraw.Draw(image);font=ImageFont.truetype(FONT,38)
    draw.text((15,20),"Big small",font=font,fill=0)
    draw.text((15,80),"Goodwill",font=font,fill=0)
    source=tmp_path/"input.png";image.save(source)
    report=prepare_text(source,tmp_path/"segmented")
    assert [len(r["glyphs"]) for r in report["rows"]]==[8,8]
    assert [[i for i,g in enumerate(r["glyphs"]) if g["prefix"]] for r in report["rows"]]==[[3],[]]


def test_thin_arch_is_not_split_but_unequal_height_bridge_is():
    im=Image.new("L",(30,30),255);d=ImageDraw.Draw(im)
    d.rectangle((1,10,5,28),fill=0);d.rectangle((5,10,20,11),fill=0);d.rectangle((20,2,24,28),fill=0)
    assert len(split_bridges(im,[[1,2,25,29]]))==2
    d.rectangle((1,2,5,28),fill=0)
    assert len(split_bridges(im,[[1,2,25,29]]))==1


def test_parallel_circuit_state_is_isolated_and_shards_resume(tmp_path):
    graph=tmp_path/"graph";graph.mkdir()
    arrays={"ptr":np.array([0,1,2,2],np.int64),"post":np.array([1,2],np.int32),
        "weight":np.array([12.,6.],np.float32),"retina":np.array([0],np.int32),
        "lamina":np.array([1],np.int32),"uv":np.array([[.5,.5]],np.float32),"candidates":np.array([1,2],np.int32)}
    for k,v in arrays.items():np.save(graph/(k+".npy"),v)
    save_json(graph/"manifest.json",{"graph_id":"letter-worker-fixture","array_hashes":{k:digest_file(graph/(k+".npy")) for k in arrays}})
    config=StimulusConfig(sensory_gain=20,lamina_drive=20,invert=True,half_saturation=.2)
    serial=Brain(graph,config);neurons=np.array([1,2],np.int32)
    images=np.array([np.full((48,48),v,np.uint8) for v in [0,255,128,0]])
    expected=np.array([serial.encode(im,neurons)[0].ravel() for im in images])
    corpus=tmp_path/"corpus";corpus.mkdir();np.savez(corpus/"train.npz",images=images)
    with pool(graph,config,2) as executor:
        got,meta=extract(executor,serial.model_id,corpus,"train",neurons,tmp_path/"cache",shard_size=2)
        assert np.array_equal(got,expected)
        # A resumed extraction must read completed shards without submitting jobs.
        class NoWorker:
            def map(self,*args,**kwargs):raise AssertionError("Recomputed a cached shard")
        resumed,_=extract(NoWorker(),serial.model_id,corpus,"train",neurons,tmp_path/"cache",shard_size=2)
        assert np.array_equal(resumed,expected)
