import {createCanvas,loadImage,GlobalFonts} from '@napi-rs/canvas';
import {readFileSync,writeFileSync,mkdirSync,copyFileSync,existsSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createHash} from 'node:crypto';
import {render,progressAt,WIDTH,HEIGHT,type Run,type Assets,type Presentation} from '../lib/replay';
const ids=process.env.FLYOCR_EXAMPLES?.split(',')||['original'];
async function main(){
 GlobalFonts.registerFromPath('public/fonts/Lato-Regular.ttf','FlySans');
 for(const id of ids){
  if(!/^[a-z0-9-]+$/.test(id))throw Error('Invalid example id');
  const base=id==='original'?'public/demo':'public/examples/'+id;
  const source=id==='original'?'demo/run':'demo/examples/'+id;
  const name=id==='original'?'flyocr-demo':'flyocr-'+id;
  const framesDir=id==='original'?'../reports/video-frames':'../reports/video-frames/'+id;
  const run:Run=JSON.parse(readFileSync(base+'/run.json','utf8'));
  const scores=JSON.parse(readFileSync(base+'/scores.json','utf8'));
  const presentation:Presentation|undefined=existsSync(base+'/presentation.json')?JSON.parse(readFileSync(base+'/presentation.json','utf8')):undefined;
  const crop=await loadImage(base+'/crop.png'),glyphs:Record<string,unknown>={};
  const sourcePage=presentation?await loadImage(base+'/source-page.png'):undefined;
  for(const e of run.events)glyphs[e.image]=await loadImage(base+'/'+e.image);
  const assets={crop,glyphs,sourcePage} as unknown as Assets;
  const canvas=createCanvas(1920,1080),context=canvas.getContext('2d');
  context.scale(1920/WIDTH,1080/HEIGHT);
  mkdirSync('../output',{recursive:true});mkdirSync(framesDir,{recursive:true});
  if(process.env.FLYOCR_PREVIEW_ONLY){
   for(const f of [0,360,959]){render(context as unknown as CanvasRenderingContext2D,run,assets,progressAt(f/24,run.events.length),f/24,scores.pdf,scores.glyphs,presentation);writeFileSync(`${framesDir}/${String(f).padStart(4,'0')}.png`,canvas.toBuffer('image/png'));}
   continue;
  }
  const video=spawn(process.env.FFMPEG_BIN||'ffmpeg',['-y','-f','image2pipe','-vcodec','png','-r','24','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart','../output/'+name+'.mp4'],{stdio:['pipe','ignore','pipe']});
  let errors='';video.stderr.on('data',d=>{errors+=d.toString();if(errors.length>8000)errors=errors.slice(-8000);});
  video.stdin.on('error',()=>{});
  const completion=once(video,'close');
  const frames=40*24;
  for(let f=0;f<frames;f++){
   const t=f/24;
   render(context as unknown as CanvasRenderingContext2D,run,assets,progressAt(t,run.events.length),t,scores.pdf,scores.glyphs,presentation);
   const png=canvas.toBuffer('image/png');
   if([0,120,360,600,780,959].includes(f))writeFileSync(`${framesDir}/${String(f).padStart(4,'0')}.png`,png);
   if(video.exitCode!==null)throw Error(errors);
   if(!video.stdin.write(png))await once(video.stdin,'drain');
   if(f%240===0)console.log(`${id}: Video ${f/24}/40 seconds`);
  }
  video.stdin.end();const [code]=await completion;
  if(code!==0)throw Error(errors);
  const target=id==='original'?'../demo':'../demo/examples/'+id;
  copyFileSync('../output/'+name+'.mp4',target+'/'+(id==='original'?'flyocr-demo':'video')+'.mp4');
  copyFileSync(framesDir+'/0959.png',target+'/preview.png');
  const sha=(file:string)=>createHash('sha256').update(readFileSync(file)).digest('hex');
  writeFileSync(target+'/video-manifest.json',JSON.stringify({format:1,seconds:40,fps:24,width:1920,height:1080,frames,artifact_id:run.artifact_id,source:source+'/run.json',run_sha256:sha(base+'/run.json'),video_sha256:sha('../output/'+name+'.mp4'),renderer:'viewer/lib/replay.ts',renderer_sha256:sha('lib/replay.ts'),atlas_renderer_sha256:sha('lib/retina.ts'),timing:'4 s source context, 28 s edited event replay, 8 s measured results',video:'output/'+name+'.mp4',motion:'presentation cursor, not a simulated body',measured_inference_seconds:run.wall_seconds},null,2));
  console.log('Saved output/'+name+'.mp4');
 }
}
main().catch(e=>{console.error(e);process.exitCode=1;});
