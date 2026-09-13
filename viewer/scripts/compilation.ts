import {createCanvas,loadImage,GlobalFonts} from '@napi-rs/canvas';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createHash} from 'node:crypto';
import {render,progressAt,type Run,type Assets} from '../lib/replay';
import {drawRetina} from '../lib/retina';
const out='../demo/compilation';mkdirSync(out,{recursive:true});
const read=(p:string)=>JSON.parse(readFileSync(p,'utf8'));
const sha=(p:string)=>createHash('sha256').update(readFileSync(p)).digest('hex');
const chapters=[
 {id:'intro',seconds:8,title:'Can a fly circuit read?'},
 {id:'mapping',seconds:14,title:'The input and the eye atlas'},
 {id:'trained-labels',seconds:14,title:'Original input mapping'},
 {id:'calibrated-labels',seconds:26,title:'Improved input, same fixed circuit'},
 {id:'calibrated-heading',seconds:16,title:'An uppercase heading'},
 {id:'calibrated-labels-more',seconds:20,title:'Five lines, errors included'},
 {id:'income-table',seconds:22,title:'A table of numbers'},
 {id:'cash-flow',seconds:18,title:'A larger numeric table'},
 {id:'tilted-table',seconds:16,title:'A formatting failure'},
 {id:'results',seconds:14,title:'Measured results'},
 {id:'credits',seconds:8,title:'Try the experiment'},
];
async function main(){
 GlobalFonts.registerFromPath('public/fonts/Lato-Regular.ttf','FlySans');
 const data:Record<string,any>={};
 for(const ch of chapters.filter(x=>!['intro','mapping','results','credits'].includes(x.id))){
  const p='public/examples/'+ch.id,run:Run=read(p+'/run.json');
  const glyphs:Record<string,unknown>={};for(const e of run.events)glyphs[e.image]=await loadImage(p+'/'+e.image);
  data[ch.id]={run,assets:{crop:await loadImage(p+'/crop.png'),sourcePage:await loadImage(p+'/source-page.png'),glyphs} as unknown as Assets,scores:read(p+'/scores.json'),presentation:read(p+'/presentation.json'),sha256:sha(p+'/run.json')};
 }
 const canvas=createCanvas(1920,1080),c=canvas.getContext('2d');c.scale(1.2,1.2);
 const ctx=c as unknown as CanvasRenderingContext2D;
 const txt=(s:string,x:number,y:number,size=24,color='#f1f6f8')=>{c.fillStyle=color;c.font=`${size}px FlySans`;c.fillText(s,x,y);};
 function card(id:string,t:number){
  c.fillStyle='#081018';c.fillRect(0,0,1600,900);
  txt('FLY / OCR',65,67,25,'#ff8756');txt('A CONNECTOME EXPERIMENT',1120,67,20,'#5cd5ef');
  const d=data['calibrated-labels'],e=d.run.events[Math.floor(t*2)%d.run.events.length];
  if(id==='intro'){
   txt('Can a fly circuit',70,225,68);txt('learn to read?',70,306,68);
   txt('166,700 simulated neurons. Frozen connections.',74,395,28,'#94acbc');
   txt('A tiny trained decoder. Actual PDF pixels.',74,441,28,'#94acbc');
   drawRetina(ctx,d.run.retinal_atlas,e.retina,930,175,580,410);
   txt('Recorded sensory samples',1020,640,24,'#94acbc');
   txt('Letters, numbers, tables - and the mistakes.',74,640,35,'#ff8756');
   txt('Research demo, not biological evidence that flies read.',74,785,24,'#94acbc');
  }else if(id==='mapping'){
   txt('Same signals. A fly-like view.',65,166,55);
   txt('Input coordinates and receptor positions are different things.',67,215,27,'#94acbc');
   c.drawImage(d.assets.glyphs[e.image],100,340,210,210);txt('Isolated glyph',110,603,24);
   for(let i=0;i<d.run.retina_uv.length;i++){const [u,v]=d.run.retina_uv[i],b=e.retina[i];const z=Math.round(25+230*b);c.fillStyle=`rgb(${z},${z},${z})`;c.fillRect(467+u*275,323+v*248,3,3);}
   txt('Calibrated sampling',455,603,24);txt('→',352,467,50,'#94acbc');txt('→',815,467,50,'#94acbc');
   drawRetina(ctx,d.run.retinal_atlas,e.retina,920,295,560,300);
   txt('Original column atlas',1040,647,24);txt('3,335 retained receptors / 825 mapped sites',925,699,22,'#94acbc');
   txt('Each signal keeps its receptor identity. No extra inference or accuracy loss.',66,774,27,'#ff8756');
   txt('Eye outlines and facets are schematic; colors encode sampled light, not neural spikes.',66,827,23,'#94acbc');
  }else if(id==='results'){
   txt('What improved - and what did not',65,175,52);
   const rows=[['68-class glyph benchmark','74.9%','87.6%'],['Letters only, same benchmark','71.9%','85.0%'],['Fresh fonts, all 68 classes','77.8%','84.9%'],['PDF character error rate','19.3%','5.7%']];
   txt('Original',980,271,25,'#94acbc');txt('Calibrated',1240,271,25,'#5cd5ef');
   rows.forEach((r,i)=>{const y=345+i*82;txt(r[0],75,y,29);txt(r[1],980,y,35);txt(r[2],1240,y,35,'#ff8756');});
   txt('Same 64-unit decoder size. One 100 ms circuit presentation per glyph.',75,723,27,'#94acbc');
   txt('1,632 benchmark glyphs; 544 fresh-font glyphs; 176 PDF text characters.',75,770,24,'#94acbc');
   txt('These are small experiments. Layout remains heuristic; no word correction.',75,818,24,'#94acbc');
  }else{
   txt('Keep the fly. Keep the errors.',65,230,61);
   txt('Code, checkpoints, replays and the full research report',70,351,31,'#94acbc');
   txt('github.com/jerryjliu/fly_ocr',70,440,43,'#ff8756');
   txt('Data: MaleCNS v1.0 / HHMI Janelia and collaborators',70,571,24);
   txt('Code adapted from DOOMFLY (MIT). Document: Microsoft 2025 Annual Report.',70,617,24);
   txt('Fixed connectome + simplified dynamics + trained readout.',70,712,27,'#5cd5ef');
   txt('Edited replay timing. The fly cursor and eye surfaces are presentation graphics.',70,811,23,'#94acbc');
  }
 }
 const total=chapters.reduce((s,x)=>s+x.seconds,0),preview=process.env.FLYOCR_PREVIEW_ONLY;
 let encoder:any,done:any,errors='';
 if(!preview){encoder=spawn(process.env.FFMPEG_BIN||'ffmpeg',['-y','-f','image2pipe','-vcodec','png','-r','24','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',out+'/fly-ocr.mp4'],{stdio:['pipe','ignore','pipe']});encoder.stderr.on('data',(d:Buffer)=>{errors=(errors+d.toString()).slice(-6000);});encoder.stdin.on('error',()=>{});done=once(encoder,'close');}
 let offset=0;const manifest:any[]=[];
 for(const ch of chapters){
  const frames=ch.seconds*24;
  const renderFrame=(f:number)=>{const local=f/24;
   if(data[ch.id]){const d=data[ch.id],replay=local/ch.seconds*40;render(ctx,d.run,d.assets,progressAt(replay,d.run.events.length),replay,d.scores.pdf,d.scores.glyphs,d.presentation);}
   else card(ch.id,local);
  };
  for(const f of preview?[Math.floor(frames*.48),frames-1]:Array.from({length:frames},(_,i)=>i)){
   renderFrame(f);const png=canvas.toBuffer('image/png');
   if(f===Math.floor(frames*.48)||f===frames-1)writeFileSync(`${out}/${ch.id}${f===frames-1?'-end':''}.png`,png);
   if(!preview){if(encoder.exitCode!==null)throw Error(errors);if(!encoder.stdin.write(png))await once(encoder.stdin,'drain');}
  }
  manifest.push({...ch,start_seconds:offset,run_sha256:data[ch.id]?.sha256});offset+=ch.seconds;console.log('Rendered chapter:',ch.title);
 }
 if(!preview){encoder.stdin.end();const [code]=await done;if(code!==0)throw Error(errors);
 writeFileSync(out+'/manifest.json',JSON.stringify({format:1,seconds:total,fps:24,width:1920,height:1080,frames:total*24,audio:false,chapters:manifest,video_sha256:sha(out+'/fly-ocr.mp4'),renderer_sha256:sha('lib/replay.ts'),atlas_renderer_sha256:sha('lib/retina.ts'),compilation_renderer_sha256:sha('scripts/compilation.ts'),disclosure:'Recorded inference with edited pacing; eye surfaces and fly cursor are schematic. Errors preserved.'},null,2));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
