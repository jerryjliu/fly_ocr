import {ReadingFly} from '../lib/social-fly';
import {renderTeaser} from '../lib/teaser-layout';
import type {SocialData} from '../lib/social-layout';
import {teaserChapters,teaserState} from '../lib/teaser-timeline';

const canvas=document.createElement('canvas');canvas.width=1920;canvas.height=1080;document.body.appendChild(canvas);
const c=canvas.getContext('2d')!;
const fly=new ReadingFly(776,644);
const data:SocialData[]=[];
async function loadImage(src:string){const image=new Image();image.src=src;await image.decode();return image;}
async function json(path:string){const result=await fetch(path);if(!result.ok)throw new Error(path+' '+result.status);return result.json();}
async function init(){
  for(const [name,file] of [['FlySans','Lato-Regular.ttf'],['FlyMono','IBMPlexMono-Regular.ttf']]){
    const font=new FontFace(name,`url(/public/fonts/${file})`);await font.load();document.fonts.add(font);
  }
  for(const chapter of teaserChapters){
    const path='/public/examples/'+chapter.id;
    const [run,scores,presentation,crop,sourcePage]=await Promise.all([json(path+'/run.json'),json(path+'/scores.json'),json(path+'/presentation.json'),loadImage(path+'/crop.png'),loadImage(path+'/source-page.png')]);
    const glyphs:Record<string,HTMLImageElement>={};
    await Promise.all(run.events.map(async(e:{image:string})=>{glyphs[e.image]=await loadImage(path+'/'+e.image);}));
    data.push({run,scores,presentation,assets:{crop,sourcePage,glyphs}});
  }
  let previous=-1;
  function frame(chapterIndex:number,seconds:number){
    const d=data[chapterIndex],chapter=teaserChapters[chapterIndex],{done,position,finished}=teaserState(seconds,d.run.events.length,chapter);
    const e=d.run.events[done-1],next=d.run.events[Math.min(done,d.run.events.length-1)];
    if(previous!==chapterIndex){fly.prepare(chapter.id,d.assets.sourcePage as HTMLImageElement,d.assets.crop as HTMLImageElement,d.presentation);previous=chapterIndex;}
    const global=teaserChapters.slice(0,chapterIndex).reduce((a,ch)=>a+ch.seconds,0)+seconds;
    const scene=fly.render(global,d.run,e,next,position-Math.floor(position),d.presentation,d.assets.crop as HTMLImageElement,finished);
    renderTeaser(c,d,chapterIndex,seconds,global,scene);
    return {done,event_id:e.id,character:e.character,finished};
  }
  Object.assign(window,{teaser:{canvas,frame,chapters:teaserChapters,ready:true}});
  frame(0,0);
}
init().catch(error=>{console.error(error);Object.assign(window,{teaserError:String(error)});});
