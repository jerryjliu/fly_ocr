// Shared browser/video renderer. Predictions come exclusively from recorded events.
import {drawRetina,type RetinalAtlas} from './retina';
export type Event={id:string;index:number;row_id:string;image:string;box:number[];character:string;prefix?:string;confidence:number;probabilities:number[];counts:number[][];retina:number[];network_spikes_per_bin:number[];wall_seconds:number};
export type Run={retinal_atlas?:RetinalAtlas;events:Event[];rows:{row_id:string;raw:string;box:number[];table_row?:number;table_column?:number}[];layout?:{n_rows:number;n_columns:number};input_projection?:{pixel_support_fraction:number};readout_architecture?:{hidden:number[];count_transform:string};normalization?:string;source_ink_box?:number[];readout_type?:string;characters:string;nodes:number;edges:number;retina_uv:number[][];feature_neuron_ids:string[];bin_ms:number;wall_seconds:number;artifact_id:string};
export type Evaluation={correct_cells:number;n_cells:number;cell_exact_match:number;character_error_rate:number;expected_shape?:number[];predicted_shape?:number[];rows:{row_id:string;exact:boolean}[]};
export type Presentation={title:string;caption:string;source:string;note:string;region:number[];rotation?:number};
export type Assets={crop:CanvasImageSource;glyphs:Record<string,CanvasImageSource>;sourcePage?:CanvasImageSource};
export const WIDTH=1600,HEIGHT=900;
const C={bg:'#081018',panel:'#101d28',line:'#273c4b',ink:'#f1f6f8',muted:'#94acbc',accent:'#ff8756',blue:'#5cd5ef',green:'#a7e6be'};
function text(c:CanvasRenderingContext2D,s:string,x:number,y:number,size=20,color=C.ink,weight=400){c.fillStyle=color;c.font=`${weight} ${size}px FlySans, Arial, sans-serif`;c.fillText(s,x,y);}
function fittedText(c:CanvasRenderingContext2D,s:string,x:number,y:number,maxWidth:number,size=20,color=C.ink){c.font=`400 ${size}px FlySans, Arial, sans-serif`;const width=c.measureText(s)?.width||s.length*size*.5;text(c,s,x,y,Math.min(size,size*maxWidth/Math.max(1,width)),color);}
function line(c:CanvasRenderingContext2D,x:number,y:number,x2:number,y2:number){c.strokeStyle=C.line;c.lineWidth=1;c.beginPath();c.moveTo(x,y);c.lineTo(x2,y2);c.stroke();}
function status(c:CanvasRenderingContext2D,x:number,y:number,good:boolean){c.strokeStyle=good?C.green:C.accent;c.lineWidth=2;c.beginPath();if(good){c.moveTo(x-4,y-4);c.lineTo(x,y);c.lineTo(x+8,y-10);}else{c.moveTo(x-4,y-9);c.lineTo(x+6,y+1);c.moveTo(x+6,y-9);c.lineTo(x-4,y+1);}c.stroke();}
function panel(c:CanvasRenderingContext2D,x:number,y:number,w:number,h:number){c.fillStyle=C.panel;c.fillRect(x,y,w,h);c.strokeStyle=C.line;c.strokeRect(x+.5,y+.5,w-1,h-1);}
function fly(c:CanvasRenderingContext2D,x:number,y:number,t:number){
 c.save();c.translate(x,y);c.rotate(-.25);c.strokeStyle=C.accent;c.lineWidth=2;
 for(const side of [-1,1]){c.fillStyle='#b9deedaa';c.beginPath();c.ellipse(side*11,-8,8,18,side*.7+Math.sin(t*28)*.13,0,Math.PI*2);c.fill();for(let k=0;k<3;k++){c.beginPath();c.moveTo(side*5,k*5);c.lineTo(side*(15+k*2),k*10-7);c.stroke();}}
 c.fillStyle=C.accent;c.beginPath();c.ellipse(0,4,6,13,0,0,Math.PI*2);c.fill();c.fillStyle=C.ink;c.beginPath();c.arc(0,-11,6,0,Math.PI*2);c.fill();c.restore();
}
export function render(c:CanvasRenderingContext2D,run:Run,assets:Assets,progress:number,seconds=0,evaluation?:Evaluation,glyphAccuracy?:number,presentation?:Presentation){
 const done=Math.max(0,Math.min(run.events.length,Math.floor(progress))),e=run.events[Math.max(0,done-1)],has=done>0,finished=done===run.events.length;
 c.fillStyle=C.bg;c.fillRect(0,0,WIDTH,HEIGHT);
 text(c,'FLY / OCR',38,45,22,C.accent,700);text(c,'A fixed fly circuit. A trained character decoder.',224,45,22,C.ink,500);text(c,'RECORDED REPLAY',1320,45,17,C.blue,700);line(c,38,66,1562,66);
 text(c,presentation?.title||'Can a fly circuit read a PDF?',38,120,42,C.ink,600);
 text(c,presentation?.source||`${run.nodes.toLocaleString('en-US')} neurons  /  ${(run.edges/1e6).toFixed(2)}M connections  /  frozen internal weights`,40,155,20,C.muted);
 panel(c,38,185,325,622);panel(c,385,185,745,622);panel(c,1152,185,410,622);
 text(c,'01  SOURCE PIXELS',58,219,18,C.muted,600);text(c,presentation?.caption||'2025 numeric column',58,247,20);
 const overview=!!(presentation&&assets.sourcePage&&seconds<4);
 const crop=(overview?assets.sourcePage:assets.crop) as HTMLImageElement;
 const box=!overview&&run.source_ink_box?run.source_ink_box:undefined;
 const sx=box?Math.max(0,box[0]-12):0,sy=box?Math.max(0,box[1]-12):0;
 const cw=box?Math.min(crop.width,box[2]+12)-sx:crop.width,ch=box?Math.min(crop.height,box[3]+12)-sy:crop.height;
 const fit=Math.min((presentation?259:265)/cw,(presentation?421:509)/ch),dh=ch*fit,dw=cw*fit,dx=185-dw/2,dy=276+(presentation?(421-dh)/2:0);
 c.fillStyle='#fff';c.fillRect(dx-9,dy-6,dw+18,dh+12);c.drawImage(crop,sx,sy,cw,ch,dx,dy,dw,dh);
 if(overview&&presentation){const b=presentation.region;c.strokeStyle=C.accent;c.lineWidth=3;c.strokeRect(dx+b[0]*dw,dy+b[1]*dh,(b[2]-b[0])*dw,(b[3]-b[1])*dh);}
 else if(e){const b=e.box,s=dh/ch;c.strokeStyle=C.accent;c.lineWidth=2;c.strokeRect(dx+(b[0]-sx)*s-2,dy+(b[1]-sy)*s-2,(b[2]-b[0])*s+4,(b[3]-b[1])*s+4);fly(c,dx+dw+25,dy+((b[1]+b[3])/2-sy)*s,seconds);}
 if(presentation){text(c,overview?'Selected region in source PDF':'Selected region → isolated glyphs',58,733,16,C.muted);text(c,presentation.rotation?'Controlled tilt · no deskew':'300 dpi · pixels only',58,761,17,C.blue);}
 text(c,'02  GLYPH → RETINA → CIRCUIT',408,219,18,C.muted,600);
 const glyph=e&&assets.glyphs[e.image];if(glyph)c.drawImage(glyph,411,253,135,135);text(c,'48 × 48 pixels',411,417,17,C.muted);text(c,'→',563,329,33,C.muted);
 c.fillStyle='#02070b';c.fillRect(608,239,245,174);
 if(run.retinal_atlas)drawRetina(c,run.retinal_atlas,has?e.retina:undefined,611,244,239,153);
 else if(has){for(let i=0;i<run.retina_uv.length;i++){const [u,v]=run.retina_uv[i],lum=Math.round(e.retina[i]*235+20);c.fillStyle=`rgb(${lum},${lum},${lum})`;c.fillRect(624+u*214,254+v*140,2.4,2.4);}}
 text(c,run.retinal_atlas?'Compound-eye atlas':'Retinal samples',616,417,17,C.muted);text(c,'→',866,329,33,C.muted);text(c,has?e.character:'—',941,347,78,C.accent,600);text(c,has?`${Math.round(e.confidence*100)}% class score`:'Awaiting event',926,389,18,C.muted);line(c,407,438,1108,438);
 text(c,'RECORDED NEURAL ACTIVITY',408,473,18,C.muted,600);text(c,`${run.feature_neuron_ids.length} downstream cells  ·  ${run.bin_ms} ms bins`,408,499,17,C.muted);
 const ax=411,ay=519,aw=425,ah=175;c.fillStyle='#060d14';c.fillRect(ax,ay,aw,ah);
 if(has){let max=1;for(const bin of e.counts)for(const n of bin)max=Math.max(max,n);for(let b=0;b<e.counts.length;b++)for(let i=0;i<e.counts[b].length;i++){const v=e.counts[b][i];if(!v)continue;c.fillStyle=`rgba(92,213,239,${.2+.8*v/max})`;c.fillRect(ax+b*aw/e.counts.length,ay+i*ah/e.counts[b].length,aw/e.counts.length-2,Math.max(1,ah/e.counts[b].length));}}
 for(let b=0;b<4;b++)text(c,`${b*run.bin_ms}–${(b+1)*run.bin_ms}`,ax+b*aw/4,718,15,C.muted);
 text(c,'Spike counts by cell and time bin',411,749,17,C.muted);text(c,has?`${e.network_spikes_per_bin.reduce((a,b)=>a+b,0).toLocaleString('en-US')} network spikes / 100 ms`:'Activity appears with each recorded event',411,778,17);
 text(c,'CLASS SCORES',868,474,17,C.muted,600);
 const ranks=has?e.probabilities.map((p,i)=>({p,ch:run.characters[i]})).sort((a,b)=>b.p-a.p).slice(0,4):[];
 ranks.forEach(({p,ch},i)=>{const y=517+i*52;text(c,ch,870,y+17,26);c.fillStyle=C.line;c.fillRect(903,y,137,18);c.fillStyle=i===0?C.accent:C.blue;c.fillRect(903,y,137*p,18);text(c,`${Math.round(p*100)}%`,1051,y+16,16,C.muted);});
 text(c,run.readout_architecture?`${run.readout_architecture.count_transform==='sqrt'?'√ counts · ':''}${run.readout_architecture.hidden.join(' → ')} units`:run.readout_type==='mlp'?'Small neural readout':'Linear readout only',868,749,17,C.muted);text(c,has?`${(e.wall_seconds*1000).toFixed(0)} ms measured wall time`:'100 ms simulated / glyph',868,778,16);
 text(c,run.layout?'03  RAW TABLE OF VALUES':run.normalization?'03  RAW TEXT':'03  RAW RECOGNITION',1175,219,18,C.muted,600);
 const outputs:Record<string,string>={};for(const event of run.events.slice(0,done))outputs[event.row_id]=(outputs[event.row_id]||'')+(event.prefix||'')+event.character;
 if(run.layout){
  const cols=run.layout.n_columns,rows=run.layout.n_rows,colw=332/Math.max(1,cols),pitch=Math.min(36,440/Math.max(1,rows));
  for(let col=0;col<cols;col++)text(c,`C${col+1}`,1210+col*colw,251,15,C.muted);
  for(let row=0;row<rows;row++){text(c,String(row+1).padStart(2,'0'),1175,283+row*pitch,14,C.muted);line(c,1208,290+row*pitch,1539,290+row*pitch);}
  run.rows.forEach(r=>{const value=outputs[r.row_id],x=1210+(r.table_column||0)*colw,y=283+(r.table_row||0)*pitch;
   const exact=evaluation?.rows.find(s=>s.row_id===r.row_id)?.exact;
   text(c,value||'·',x,y,Math.min(17,colw/6.8),!value?C.line:finished&&evaluation?(exact?C.green:C.accent):C.ink);
  });
  text(c,finished&&evaluation?.expected_shape?`Grid: ${rows} × ${cols}  ·  expected ${evaluation.expected_shape.join(' × ')}`:`${rows} rows × ${cols} columns · geometry only`,1175,735,16,C.muted);
 }else run.rows.forEach((r,i)=>{const y=256+i*(run.normalization?54:26),v=outputs[r.row_id];text(c,String(i+1).padStart(2,'0'),1175,y,16,C.muted);if(run.normalization)fittedText(c,v||'·',1218,y,278,20,v?C.ink:C.line);else text(c,v||'·',1218,y,20,v?C.ink:C.line,500);if(finished&&evaluation){const row=evaluation.rows.find(x=>x.row_id===r.row_id);status(c,1528,y,!!row?.exact);}});
 line(c,1175,753,1538,753);
 if(finished&&evaluation)text(c,`${evaluation.correct_cells}/${evaluation.n_cells} exact ${run.normalization?'lines':'cells'}  ·  ${(evaluation.character_error_rate*100).toFixed(1)}% CER`,1175,782,19,C.accent,600);else text(c,`${done} / ${run.events.length} characters revealed`,1175,782,19,C.blue);
 c.fillStyle=C.line;c.fillRect(38,826,1524,3);c.fillStyle=C.accent;c.fillRect(38,826,1524*done/run.events.length,3);
 const score=glyphAccuracy===undefined?'':`  ·  ${run.input_projection?'Benchmark':'Held-out'} ${run.normalization?'68-class':'numeric'} glyphs: ${(glyphAccuracy*100).toFixed(1)}%`;
 text(c,finished?`Actual predictions, including errors. ${run.wall_seconds.toFixed(1)} s total inference.${score}`:'One glyph at a time. No embedded PDF text. No language-model correction.',38,858,19);
 text(c,run.retinal_atlas?`${run.input_projection?'Calibrated sampling':'Original sampling'} · Eyes: schematic column atlas, sampled light · Circuit: spike counts · Fly: animated cursor`:presentation?.note||'Approximate fly vision · Closed numeric alphabet · Fly motion is a cursor animation',38,886,16,C.muted);
}
export function progressAt(seconds:number,n:number){return seconds<4?0:seconds>=32?n:Math.min(n,Math.floor((seconds-4)/28*n)+1);}
