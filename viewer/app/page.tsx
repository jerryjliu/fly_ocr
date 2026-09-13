'use client';
import {useEffect,useRef,useState} from 'react';
import {Button} from '@/components/ui/button';
import {Slider} from '@/components/ui/slider';
import {render,progressAt,WIDTH,HEIGHT,type Run,type Assets,type Evaluation,type Presentation} from '@/lib/replay';
type Example={id:string;title:string;path:string};
export default function Home(){
 const canvas=useRef<HTMLCanvasElement>(null);
 const [catalog,setCatalog]=useState<Example[]>([]),[path,setPath]=useState('/examples/calibrated-labels');
 const [bundle,setBundle]=useState<{run:Run;assets:Assets;score?:Evaluation;glyphs?:number;presentation?:Presentation}>();
 const [error,setError]=useState(''),[playing,setPlaying]=useState(false),[time,setTime]=useState(0);
 useEffect(()=>{fetch('/catalog.json').then(r=>r.json()).then(data=>{setCatalog(data as Example[]);const id=new URLSearchParams(window.location.search).get('example');const selected=(data as Example[]).find(x=>x.id===id);if(selected)setPath(selected.path);}).catch(()=>{});},[]);
 useEffect(()=>{let disposed=false;setBundle(undefined);setError('');setPlaying(false);setTime(0);
 async function load(){
  const response=await fetch(path+'/run.json');if(!response.ok)throw Error('The recorded run is missing. Run the bundle command in the README.');
  const run:Run=await response.json();if(!run.events?.length)throw Error('This run has no recognition events.');
  const image=(src:string)=>new Promise<HTMLImageElement>((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(Error(`Could not load ${src}`));im.src=src;});
  const crop=await image(path+'/crop.png'),glyphs:Record<string,HTMLImageElement>={};await Promise.all(run.events.map(async e=>{glyphs[e.image]=await image(path+'/'+e.image);}));
  const presentation:Presentation|undefined=path==='/demo'?undefined:await fetch(path+'/presentation.json').then(r=>r.json());
  const sourcePage=presentation?await image(path+'/source-page.png'):undefined;
  const scores:{pdf?:Evaluation;glyphs?:number}=await fetch(path+'/scores.json').then(r=>r.ok?r.json():{});await document.fonts.ready;
  if(!disposed)setBundle({run,assets:{crop,glyphs,sourcePage},score:scores.pdf,glyphs:scores.glyphs,presentation});
 }load().catch(e=>{if(!disposed)setError(String(e.message));});return()=>{disposed=true;};},[path]);
 useEffect(()=>{if(!playing)return;let previous=performance.now(),frame:number;const tick=(now:number)=>{const delta=(now-previous)/1000;previous=now;setTime(t=>Math.min(40,t+delta));frame=requestAnimationFrame(tick);};frame=requestAnimationFrame(tick);return()=>cancelAnimationFrame(frame);},[playing]);
 useEffect(()=>{if(time>=40&&playing)setPlaying(false);if(bundle&&canvas.current){const c=canvas.current.getContext('2d');if(c)render(c,bundle.run,bundle.assets,progressAt(time,bundle.run.events.length),time,bundle.score,bundle.glyphs,bundle.presentation);}else if(canvas.current){canvas.current.getContext('2d')?.clearRect(0,0,WIDTH,HEIGHT);}},[bundle,time,playing]);
 return <main><h1 className="sr-only">Fly OCR — recorded fly circuit recognition experiment</h1>
 {catalog.length>1&&<label className="example-picker">Recorded example <select value={path} onChange={e=>setPath(e.target.value)}>{catalog.map(item=><option key={item.id} value={item.path}>{item.title}</option>)}</select></label>}
 {error?<div role="alert" className="notice">{error}</div>:!bundle?<div className="notice">Loading recorded experiment…</div>:null}
 <canvas ref={canvas} width={WIDTH} height={HEIGHT} aria-label="PDF source, current glyph, recorded neural counts, decoder scores and raw predictions"/>
 <div className="controls"><Button size="lg" disabled={!bundle} onClick={()=>{if(time>=40)setTime(0);setPlaying(!playing);}}>{playing?'Pause replay':time>=40?'Replay':'Play replay'}</Button><Button size="lg" variant="outline" disabled={!bundle} onClick={()=>{setPlaying(false);setTime(0);}}>Reset</Button>
 <Slider aria-label="Replay position" min={0} max={40} step={.05} value={[time]} onValueChange={v=>{setPlaying(false);setTime(Array.isArray(v)?v[0]:v);}}/><span>{time.toFixed(1)} / 40 s</span>
 <Button size="lg" variant="outline" disabled={!bundle} onClick={()=>{setPlaying(false);setTime(40);}}>Show results</Button></div>
 <p className="disclosure">Replay timing is edited for viewing; measured compute time appears with each glyph. The fly is an animated cursor. Activity and class scores come from the same saved inference event. Green cells match the separate evaluation; orange cells contain errors.</p>
 {bundle&&time>=32&&<details><summary>Accessible results and experiment details</summary><p>This experiment uses a fixed MaleCNS connectome and a learned {bundle.run.readout_type==='mlp'?'small neural':'linear'} decoder. It does not establish that a biological fly reads. The compound-eye panel displays recorded sampled light at schematic positions derived from the original left/right column map. Calibrated examples keep their engineered input sampling; the eye drawing does not change predictions. Its output alphabet is <code>{bundle.run.characters}</code>. Spaces in text examples come from pixel gaps between words.</p>{bundle.presentation&&<p>{bundle.presentation.note}</p>}
 {bundle.run.layout?<table className="results-table"><caption>Raw strings arranged by pixel geometry; no semantic labels</caption><tbody>{Array.from({length:bundle.run.layout.n_rows},(_,r)=><tr key={r}>{Array.from({length:bundle.run.layout!.n_columns},(_,c)=><td key={c}>{bundle.run.rows.find(cell=>cell.table_row===r&&cell.table_column===c)?.raw||''}</td>)}</tr>)}</tbody></table>:<ol>{bundle.run.rows.map(r=><li key={r.row_id}>{r.row_id}: <code>{r.raw}</code></li>)}</ol>}
 {bundle.score&&<p>Exact cells: {bundle.score.correct_cells}/{bundle.score.n_cells}. Character error rate: {(100*bundle.score.character_error_rate).toFixed(1)}%.</p>}
 <a href={path+'/predictions.csv'} download>Download raw predictions</a>{bundle.run.layout&&<> · <a href={path+'/table.csv'} download>Download table CSV</a> · <a href={path+'/table.json'} download>Download table JSON</a></>}</details>}</main>;
}
