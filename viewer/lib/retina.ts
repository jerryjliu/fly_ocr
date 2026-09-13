// Receptor identity determines display position. Image UV determines input only.
export type RetinalAtlas={atlas_id:string;original_uv:number[][];eye:string[];retina_body_ids:string[]};
type Facet={x:number;y:number;indices:number[];eye:string};
const cache=new WeakMap<RetinalAtlas,Facet[]>();
export function atlasFacets(atlas:RetinalAtlas):Facet[]{
 const prior=cache.get(atlas);if(prior)return prior;
 const grouped=new Map<string,Facet>();
 atlas.original_uv.forEach(([u,v],i)=>{
  const eye=atlas.eye[i],key=`${eye}:${u}:${v}`;
  const existing=grouped.get(key);if(existing){existing.indices.push(i);return;}
  // Undo the original overlapping binocular layout, then curve the drawing.
  // This square-to-oval display warp never enters the recognizer.
  const nx=2*(eye==='L'?u/.6:(u-.4)/.6)-1,ny=2*v-1;
  grouped.set(key,{x:nx*Math.sqrt(1-.45*ny*ny),y:ny*Math.sqrt(1-.45*nx*nx),indices:[i],eye});
 });
 const facets=[...grouped.values()];cache.set(atlas,facets);return facets;
}
export function drawRetina(c:CanvasRenderingContext2D,atlas:RetinalAtlas,values:number[]|undefined,x:number,y:number,w:number,h:number){
 c.save();c.translate(x,y);
 for(const side of ['L','R']){
  const cx=w*(side==='L'?.25:.75),cy=h*.48;
  c.fillStyle='#211822';c.strokeStyle='#78504b';c.lineWidth=1.1;
  c.beginPath();c.ellipse(cx,cy,w*.231,h*.455,0,0,Math.PI*2);c.fill();c.stroke();
 }
 for(const f of atlasFacets(atlas)){
  const cx=w*(f.eye==='L'?.25:.75)+f.x*w*.203,cy=h*.48+f.y*h*.404;
  const b=values?f.indices.reduce((a,i)=>a+values[i],0)/f.indices.length:0;
  // A monotone copper luminance scale: dark facet = less sampled light.
  c.fillStyle=values?`rgb(${Math.round(45+210*b)},${Math.round(24+191*b)},${Math.round(28+147*b)})`:'#49343b';
  const radius=Math.min(w/145,h/62);
  c.beginPath();for(let k=0;k<6;k++){const a=Math.PI/3*k,px=cx+radius*Math.cos(a),py=cy+radius*Math.sin(a);if(k)c.lineTo(px,py);else c.moveTo(px,py);}c.closePath();c.fill();
 }
 c.fillStyle='#94acbc';c.font='11px FlySans, Arial, sans-serif';c.fillText('L',w*.24,h*.995);c.fillText('R',w*.74,h*.995);c.restore();
}
