import {drawRetina} from './retina';
import type {Assets,Run,Evaluation,Presentation} from './replay';
import {socialChapters,socialSeconds,socialState} from './social-timeline';

export type SocialData={run:Run;assets:Assets;scores:{pdf:Evaluation};presentation:Presentation};
const C={bg:'#080f15',panel:'#10212c',border:'#29404d',ink:'#f5f2e9',muted:'#9db2be',orange:'#ff9764',blue:'#77deee',green:'#aff2c3'};
function text(c:CanvasRenderingContext2D,s:string,x:number,y:number,size=24,color=C.ink,bold=false){c.font=`${bold?700:400} ${size}px FlySans, sans-serif`;c.fillStyle=color;c.fillText(s,x,y);}
function panel(c:CanvasRenderingContext2D,x:number,y:number,w:number,h:number){c.fillStyle=C.panel;c.beginPath();c.roundRect(x,y,w,h,18);c.fill();c.strokeStyle=C.border;c.lineWidth=1;c.stroke();}
function fitText(c:CanvasRenderingContext2D,s:string,x:number,y:number,w:number,size=26,color=C.ink,font='FlySans'){c.font=`${size}px ${font}`;const width=c.measureText(s).width;c.font=`${Math.min(size,size*w/Math.max(1,width))}px ${font}`;c.fillStyle=color;c.fillText(s,x,y);}
function fitImage(c:CanvasRenderingContext2D,img:HTMLImageElement,x:number,y:number,w:number,h:number){const scale=Math.min(w/img.width,h/img.height);c.drawImage(img,x+(w-img.width*scale)/2,y+(h-img.height*scale)/2,img.width*scale,img.height*scale);}

export function renderSocial(c:CanvasRenderingContext2D,d:SocialData,chapterIndex:number,local:number,global:number,flyCanvas:HTMLCanvasElement){
  const chapter=socialChapters[chapterIndex],{run,assets,presentation,scores}=d;
  const {done,finished}=socialState(local,run.events.length,chapter.seconds),e=run.events[done-1];
  c.fillStyle=C.bg;c.fillRect(0,0,1920,1080);
  text(c,'I made a fly read a PDF.',34,68,48,C.ink,true);
  text(c,'166,700 neurons · fixed circuit · tiny trained decoder',665,65,25,C.muted);
  text(c,'FLY / OCR',1698,66,28,C.orange,true);
  // A chapter is a caption over the live recording; there are no intro cards.
  text(c,`${String(chapterIndex+1).padStart(2,'0')} / 07`,36,114,19,C.orange,true);
  text(c,chapter.title,128,115,26,C.ink,true);text(c,chapter.subtitle,610,114,20,C.muted);

  panel(c,32,138,440,292);panel(c,488,138,610,292);
  text(c,'PDF PIXELS',53,173,20,C.muted,true);
  const crop=assets.crop as HTMLImageElement,ink=run.source_ink_box||[0,0,crop.width,crop.height];
  const sx=Math.max(0,ink[0]-9),sy=Math.max(0,ink[1]-9),sw=Math.min(crop.width,ink[2]+9)-sx,sh=Math.min(crop.height,ink[3]+9)-sy;
  const scale=Math.min(392/sw,186/sh),dw=sw*scale,dh=sh*scale,dx=56+(392-dw)/2,dy=191+(186-dh)/2;
  c.fillStyle='#fff';c.fillRect(52,187,400,194);c.drawImage(crop,sx,sy,sw,sh,dx,dy,dw,dh);
  c.fillStyle='#ff7a4330';c.strokeStyle='#f45e2a';c.lineWidth=2.5;
  const b=e.box;c.fillRect(dx+(b[0]-sx)*scale-2,dy+(b[1]-sy)*scale-2,(b[2]-b[0])*scale+4,(b[3]-b[1])*scale+4);
  c.strokeRect(dx+(b[0]-sx)*scale-2,dy+(b[1]-sy)*scale-2,(b[2]-b[0])*scale+4,(b[3]-b[1])*scale+4);
  text(c,'Pixels → geometric character split',53,410,20,C.muted);
  text(c,'ONE GLYPH',510,173,20,C.muted,true);text(c,'SAMPLED THROUGH THE EYE ATLAS',694,173,17,C.muted,true);
  const glyph=assets.glyphs[e.image] as HTMLImageElement;fitImage(c,glyph,511,208,137,137);
  text(c,'→',661,293,30,C.orange);
  if(run.retinal_atlas)drawRetina(c,run.retinal_atlas,e.retina,699,189,378,200);
  text(c,'48 × 48',534,404,20,C.muted);text(c,'Recorded light · schematic eye positions',695,409,18,C.muted);

  panel(c,32,447,1066,166);
  text(c,'CIRCUIT ACTIVITY',54,479,19,C.muted,true);
  const ax=55,ay=495,aw=287,ah=90;
  c.fillStyle='#050e14';c.fillRect(ax,ay,aw,ah);
  const groupSize=Math.ceil(run.feature_neuron_ids.length/128);
  const grouped=e.counts.map(bin=>Array.from({length:128},(_,i)=>bin.slice(i*groupSize,i*groupSize+groupSize).reduce((a,v)=>a+v,0)));
  const peak=Math.max(1,...grouped.flat());
  grouped.forEach((bin,j)=>bin.forEach((v,i)=>{if(!v)return;c.fillStyle=`rgba(119,222,238,${.13+.87*v/peak})`;c.fillRect(ax+j*aw/4,ay+i*ah/128,aw/4-3,Math.max(.8,ah/128));}));
  text(c,`${run.feature_neuron_ids.length.toLocaleString('en-US')} cells`,361,517,20,C.ink);text(c,'4 × 25 ms bins',361,546,19,C.muted);text(c,`${groupSize} cells / display row`,361,578,15,C.muted);text(c,'→',520,543,31,C.orange);
  text(c,'READOUT',571,480,19,C.muted,true);fitText(c,e.character,584,578,93,84,C.orange,'FlyMono');
  text(c,`${Math.round(e.confidence*100)}%`,701,526,35,C.ink,true);text(c,'class score',702,554,18,C.muted);
  const ranks=e.probabilities.map((p,i)=>({p,ch:run.characters[i]})).sort((a,b)=>b.p-a.p).slice(0,3);
  ranks.forEach(({p,ch},i)=>{const y=489+i*34;text(c,ch,859,y+18,23);c.fillStyle=C.border;c.fillRect(887,y,120,13);c.fillStyle=i===0?C.orange:C.blue;c.fillRect(887,y,120*p,13);text(c,`${Math.round(p*100)}%`,1020,y+14,17,C.muted);});

  panel(c,32,630,1066,358);
  text(c,run.layout?'RECONSTRUCTED TABLE':'RAW RECOGNIZED TEXT',55,665,20,C.muted,true);
  text(c,`${done} / ${run.events.length} glyphs`,881,665,20,C.blue);
  const outputs:Record<string,string>={};for(const event of run.events.slice(0,done))outputs[event.row_id]=(outputs[event.row_id]||'')+(event.prefix||'')+event.character;
  if(run.layout){
    const rows=run.layout.n_rows,cols=run.layout.n_columns,pitch=Math.min(32,263/rows),colWidth=922/cols;
    for(let col=0;col<cols;col++)text(c,`COLUMN ${col+1}`,126+col*colWidth,691,14,C.muted);
    for(let row=0;row<rows;row++){
      const y=715+row*pitch;text(c,String(row+1).padStart(2,'0'),58,y,Math.min(16,pitch-1),C.muted);
      c.strokeStyle='#25404c';c.beginPath();c.moveTo(116,y+4);c.lineTo(1069,y+4);c.stroke();
    }
    for(const row of run.rows){const value=outputs[row.row_id],exact=scores.pdf.rows.find(r=>r.row_id===row.row_id)?.exact;
      text(c,value||'·',126+(row.table_column||0)*colWidth,715+(row.table_row||0)*pitch,Math.min(22,pitch-1),!value?C.border:finished?(exact?C.green:C.orange):C.ink);
    }
  }else{
    const pitch=run.rows.length<=2?65:50;
    run.rows.forEach((row,i)=>{
      const y=728+i*pitch;const value=outputs[row.row_id]||'';
      text(c,String(i+1).padStart(2,'0'),57,y,20,C.muted);
      fitText(c,value||'·',107,y,934,run.rows.length<=2?37:31,C.ink,'FlyMono');
      if(e.row_id===row.row_id&&!finished){c.font=`${run.rows.length<=2?37:31}px FlyMono`;const x=107+c.measureText(value).width;if(x<1045){c.fillStyle=C.orange;c.fillRect(x+4,y-28,3,35);}}
    });
    if(finished)text(c,'Uncorrected output. The mistakes stay in.',57,958,21,C.muted);
  }

  // The animated fly gets a full side panel, adjacent to the real decoder output.
  panel(c,1116,138,772,850);
  text(c,'MEANWHILE, ON THE PDF…',1143,177,23,C.orange,true);
  c.save();c.beginPath();c.roundRect(1133,198,738,611,12);c.clip();c.drawImage(flyCanvas,1133,198,738,611);c.restore();
  text(c,'3D fly: illustrative scan-following animation',1143,839,19,C.muted);
  c.fillStyle='#1d3440';c.beginPath();c.roundRect(1139,859,726,100,13);c.fill();
  if(finished){
    fitText(c,chapter.result,1156,895,692,25,C.ink);
    const score=run.layout?`${scores.pdf.correct_cells} / ${scores.pdf.n_cells} exact cells`:`${(scores.pdf.character_error_rate*100).toFixed(1)}% character error`;
    text(c,score,1157,933,23,C.orange,true);
  }else{
    fitText(c,'“'+e.character+'”',1160,930,141,58,C.orange,'FlyMono');
    text(c,run.layout?'One glyph. One tiny decision.':'Please hold. Fly is reading.',1311,903,26,C.ink);
    text(c,`${(e.wall_seconds*1000).toFixed(0)} ms measured inference for this glyph`,1312,935,19,C.muted);
  }
  text(c,presentation.source,34,1023,21,C.muted);
  text(c,'Recorded OCR · edited pacing · animation does not control recognition',34,1056,19,C.muted);
  text(c,'github.com/jerryjliu/fly_ocr',1492,1043,23,C.orange);
  c.fillStyle=C.border;c.fillRect(32,998,1856,3);c.fillStyle=C.orange;c.fillRect(32,998,1856*Math.min(1,global/socialSeconds),3);
}
