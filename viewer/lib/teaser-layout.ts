import {drawRetina} from './retina';
import type {SocialData} from './social-layout';
import {teaserChapters,teaserSeconds,teaserState} from './teaser-timeline';

const C={bg:'#080f15',panel:'#10212c',border:'#29404d',ink:'#f5f2e9',muted:'#9db2be',orange:'#ff9764',blue:'#77deee',green:'#aff2c3'};
function text(c:CanvasRenderingContext2D,s:string,x:number,y:number,size=24,color=C.ink,bold=false){c.font=`${bold?700:400} ${size}px FlySans, sans-serif`;c.fillStyle=color;c.fillText(s,x,y);}
function panel(c:CanvasRenderingContext2D,x:number,y:number,w:number,h:number){c.fillStyle=C.panel;c.beginPath();c.roundRect(x,y,w,h,18);c.fill();c.strokeStyle=C.border;c.lineWidth=1;c.stroke();}
function fitText(c:CanvasRenderingContext2D,s:string,x:number,y:number,w:number,size=26,color=C.ink,font='FlySans'){c.font=`${size}px ${font}`;const width=c.measureText(s).width;c.font=`${Math.min(size,size*w/Math.max(1,width))}px ${font}`;c.fillStyle=color;c.fillText(s,x,y);}

export function renderTeaser(c:CanvasRenderingContext2D,d:SocialData,chapterIndex:number,local:number,global:number,flyCanvas:HTMLCanvasElement){
  const chapter=teaserChapters[chapterIndex],{run,assets,presentation,scores}=d;
  const {done,finished}=teaserState(local,run.events.length,chapter),e=run.events[done-1];
  const closing=chapterIndex===teaserChapters.length-1&&local>=9;
  c.fillStyle=C.bg;c.fillRect(0,0,1920,1080);
  text(c,closing?'Try Fly OCR. Read the full experiment.':chapter.headline,34,67,46,C.ink,true);
  text(c,'FLY / OCR',1698,66,28,C.orange,true);
  text(c,chapter.kicker,36,114,19,C.orange,true);
  text(c,closing?'Code, research report, methods and ablations.':chapter.subtitle,313,114,26,C.muted);

  panel(c,32,138,440,292);panel(c,488,138,610,292);
  text(c,'THE DOCUMENT',53,173,20,C.muted,true);
  const crop=assets.crop as HTMLImageElement,ink=run.source_ink_box||[0,0,crop.width,crop.height];
  const sx=Math.max(0,ink[0]-9),sy=Math.max(0,ink[1]-9),sw=Math.min(crop.width,ink[2]+9)-sx,sh=Math.min(crop.height,ink[3]+9)-sy;
  const scale=Math.min(392/sw,186/sh),dw=sw*scale,dh=sh*scale,dx=56+(392-dw)/2,dy=191+(186-dh)/2;
  c.fillStyle='#fff';c.fillRect(52,187,400,194);c.drawImage(crop,sx,sy,sw,sh,dx,dy,dw,dh);
  c.fillStyle='#ff7a4330';c.strokeStyle='#f45e2a';c.lineWidth=2.5;
  const b=e.box;c.fillRect(dx+(b[0]-sx)*scale-2,dy+(b[1]-sy)*scale-2,(b[2]-b[0])*scale+4,(b[3]-b[1])*scale+4);
  c.strokeRect(dx+(b[0]-sx)*scale-2,dy+(b[1]-sy)*scale-2,(b[2]-b[0])*scale+4,(b[3]-b[1])*scale+4);
  text(c,'Actual pixels from a PDF',53,410,20,C.muted);
  text(c,'CHARACTER',510,173,20,C.muted,true);text(c,'THROUGH THE FLY EYE ATLAS',714,173,19,C.muted,true);
  c.drawImage(assets.glyphs[e.image],511,208,137,137);text(c,'→',665,293,30,C.orange);
  if(run.retinal_atlas)drawRetina(c,run.retinal_atlas,e.retina,699,189,378,200);
  text(c,'One at a time',515,409,19,C.muted);text(c,'Recorded samples · schematic eye view',695,409,18,C.muted);

  panel(c,32,447,1066,166);
  text(c,'THE CIRCUIT RESPONDS',54,479,19,C.muted,true);
  const ax=55,ay=495,aw=340,ah=90;
  c.fillStyle='#050e14';c.fillRect(ax,ay,aw,ah);
  const groupSize=Math.ceil(run.feature_neuron_ids.length/128);
  const grouped=e.counts.map(bin=>Array.from({length:128},(_,i)=>bin.slice(i*groupSize,i*groupSize+groupSize).reduce((a,v)=>a+v,0)));
  const peak=Math.max(1,...grouped.flat());
  grouped.forEach((bin,j)=>bin.forEach((v,i)=>{if(!v)return;c.fillStyle=`rgba(119,222,238,${.13+.87*v/peak})`;c.fillRect(ax+j*aw/4,ay+i*ah/128,aw/4-3,Math.max(.8,ah/128));}));
  text(c,'→',433,548,38,C.orange);
  text(c,'PREDICTION',510,480,19,C.muted,true);fitText(c,e.character,553,583,109,94,C.orange,'FlyMono');
  text(c,'→',711,548,38,C.orange);
  text(c,run.layout?'Characters become values.':'Characters become text.',790,525,25,C.ink);
  text(c,'Actual model output',790,563,23,C.muted);

  panel(c,32,630,1066,358);
  text(c,run.layout?'RECOGNIZED TABLE':'RECOGNIZED TEXT',55,665,20,C.muted,true);
  text(c,finished?'COMPLETE':'READING…',944,665,18,C.blue);
  const outputs:Record<string,string>={};for(const event of run.events.slice(0,done))outputs[event.row_id]=(outputs[event.row_id]||'')+(event.prefix||'')+event.character;
  if(run.layout){
    const rows=run.layout.n_rows,cols=run.layout.n_columns,pitch=Math.min(32,263/rows),colWidth=922/cols;
    for(let col=0;col<cols;col++)text(c,`COLUMN ${col+1}`,126+col*colWidth,691,14,C.muted);
    for(let row=0;row<rows;row++){
      const y=723+row*pitch;text(c,String(row+1).padStart(2,'0'),58,y,16,C.muted);
      c.strokeStyle='#25404c';c.beginPath();c.moveTo(116,y+7);c.lineTo(1069,y+7);c.stroke();
    }
    for(const row of run.rows){const value=outputs[row.row_id],exact=scores.pdf.rows.find(r=>r.row_id===row.row_id)?.exact;
      text(c,value||'·',126+(row.table_column||0)*colWidth,723+(row.table_row||0)*pitch,24,!value?C.border:finished?(exact?C.green:C.orange):C.ink);
    }
    text(c,'Rows and columns reconstructed from page geometry.',57,970,18,C.muted);
  }else{
    const pitch=run.rows.length<=2?65:50;
    run.rows.forEach((row,i)=>{
      const y=728+i*pitch,value=outputs[row.row_id]||'';
      text(c,String(i+1).padStart(2,'0'),57,y,20,C.muted);
      fitText(c,value||'·',107,y,934,run.rows.length<=2?37:31,C.ink,'FlyMono');
      if(e.row_id===row.row_id&&!finished){c.font=`${run.rows.length<=2?37:31}px FlyMono`;const x=107+c.measureText(value).width;if(x<1045){c.fillStyle=C.orange;c.fillRect(x+4,y-28,3,35);}}
    });
    if(finished)text(c,'Actual predictions, including the mistakes.',57,958,21,C.muted);
  }

  panel(c,1116,138,772,850);
  text(c,'MEANWHILE, ON THE PDF…',1143,177,23,C.orange,true);
  c.save();c.beginPath();c.roundRect(1133,198,738,611,12);c.clip();c.drawImage(flyCanvas,1133,198,738,611);c.restore();
  text(c,'Illustrative fly animation · recorded OCR',1143,839,19,C.muted);
  c.fillStyle='#1d3440';c.beginPath();c.roundRect(1139,859,726,100,13);c.fill();
  if(closing){
    text(c,'CODE + RESEARCH REPORT',1157,891,19,C.blue,true);
    text(c,'github.com/jerryjliu/fly_ocr',1157,937,34,C.orange,true);
  }else if(finished){
    fitText(c,chapter.result,1156,901,692,28,C.ink);
    text(c,chapterIndex===0?'Mixed case. Spaces. Punctuation.':chapterIndex===1?'Selected PDF crop · uncorrected recognition':'A fun experiment, with limits.',1157,939,22,C.orange);
  }else{
    fitText(c,'“'+e.character+'”',1160,930,141,58,C.orange,'FlyMono');
    text(c,'Please hold. Fly is reading.',1311,917,28,C.ink);
  }
  text(c,presentation.source,34,1023,21,C.muted);
  text(c,'Recorded OCR · edited pacing · fly motion is animation',34,1056,19,C.muted);
  text(c,'github.com/jerryjliu/fly_ocr',1492,1043,23,C.orange);
  c.fillStyle=C.border;c.fillRect(32,998,1856,3);c.fillStyle=C.orange;c.fillRect(32,998,1856*Math.min(1,global/teaserSeconds),3);
}
