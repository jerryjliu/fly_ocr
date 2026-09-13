import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
const {render,progressAt}=createRequire(import.meta.url)('../.cache/replay-check.cjs');
const run=JSON.parse(readFileSync('public/demo/run.json','utf8'));
assert.equal(progressAt(0,120),0);assert.equal(progressAt(3.99,120),0);assert.equal(progressAt(32,120),120);
let previous=0;for(let t=0;t<=40;t+=.025){const now=progressAt(t,120);assert(now>=previous&&now<=120);previous=now;}
const captured=[];
const context=new Proxy({fillText:(text,x,y)=>captured.push({text,x,y})},{get:(target,key)=>key in target?target[key]:(()=>{})});
const assets={crop:{width:235,height:1436},glyphs:{}};
function outputAt(n){captured.length=0;render(context,run,assets,n);return captured.filter(x=>x.x===1218).map(x=>x.text);}
assert(outputAt(0).every(x=>x==='·'));
assert.deepEqual(outputAt(120),run.rows.map(r=>r.raw));
assert.deepEqual(outputAt(7),outputAt(7));
assert(outputAt(0).every(x=>x==='·'));
console.log('Timeline reveals only recorded predictions; reset and replay are deterministic.');
const catalog=JSON.parse(readFileSync('public/catalog.json','utf8'));
for(const item of catalog.filter(item=>item.id!=='original')){
 const run=JSON.parse(readFileSync('public'+item.path+'/run.json','utf8'));
 const scores=JSON.parse(readFileSync('public'+item.path+'/scores.json','utf8'));
 const presentation=JSON.parse(readFileSync('public'+item.path+'/presentation.json','utf8'));
 const output=n=>{captured.length=0;render(context,run,assets,n,n?40:0,scores.pdf,scores.glyphs,presentation);return [...captured];};
 const initial=output(0);
 assert(!initial.some(t=>t.text.includes('exact cells')));
 assert(!initial.some(t=>t.text.includes('class score')&&!t.text.includes('CLASS SCORES')));
 const final=output(run.events.length);
 if(run.layout){
  for(const r of run.rows){const x=1210+r.table_column*332/run.layout.n_columns,y=283+r.table_row*Math.min(36,440/run.layout.n_rows);
   assert(final.some(t=>t.text===r.raw&&t.x===x&&t.y===y),'Cell position/raw value mismatch: '+r.row_id);
  }
 }else assert.deepEqual(final.filter(t=>t.x===1218).map(t=>t.text),run.rows.map(r=>r.raw));
 assert.deepEqual(output(0),initial);
}
console.log('All table and unsupported-input replays preserve raw cell positions and reset cleanly.');
