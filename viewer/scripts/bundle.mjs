import {cpSync,readFileSync,writeFileSync,mkdirSync,existsSync,rmSync} from 'node:fs';
import {resolve} from 'node:path';
const root=resolve('..');
function bundle(source,target,report,presentation,artifact='numeric'){
 if(!existsSync(source+'/run.json'))throw Error('No recorded demo at '+source);
 // These are generated viewer assets; discard obsolete glyph files on rebundle.
 rmSync(target,{recursive:true,force:true});mkdirSync(target,{recursive:true});
 for(const name of ['run.json','crop.png','glyphs','predictions.csv','table.csv','table.json','source-page.png']){
  if(existsSync(source+'/'+name))cpSync(source+'/'+name,target+'/'+name,{recursive:true});
 }
 const pdf=JSON.parse(readFileSync(report,'utf8'));
 const recognition=JSON.parse(readFileSync(root+'/artifacts/'+artifact+'/recognition.json','utf8'));
 const run=JSON.parse(readFileSync(source+'/run.json','utf8'));
 if(recognition.artifact_id!==run.artifact_id)throw Error('Run and evaluation use different readouts');
 const scores={glyphs:recognition.neural.accuracy,artifact_id:run.artifact_id,pdf:{correct_cells:pdf.correct_cells,n_cells:pdf.n_cells,cell_exact_match:pdf.cell_exact_match,character_error_rate:pdf.character_error_rate,expected_shape:pdf.expected_shape,predicted_shape:pdf.predicted_shape,rows:pdf.rows.map(r=>({row_id:r.row_id,exact:r.exact}))}};
 // No evaluation truth strings enter the viewer bundle.
 writeFileSync(target+'/scores.json',JSON.stringify(scores));
 if(presentation)writeFileSync(target+'/presentation.json',JSON.stringify(presentation));
}
bundle(root+'/demo/run',resolve('public/demo'),root+'/reports/pdf-crop.json');
const catalog=[{id:'original',title:'Original numeric column',path:'/demo'}];
const specs=JSON.parse(readFileSync(root+'/examples/microsoft-2025/more-examples.json','utf8'));
for(const item of specs.examples){
 const source=root+'/demo/examples/'+item.id;
 if(!existsSync(source+'/run.json'))continue;
 bundle(source,resolve('public/examples/'+item.id),root+'/reports/more-examples/'+item.id+'.json',item);
 catalog.push({id:item.id,title:item.title,path:'/examples/'+item.id});
}
const letterSpecs=root+'/examples/microsoft-2025/letter-examples.json';
if(existsSync(letterSpecs))for(const item of JSON.parse(readFileSync(letterSpecs,'utf8')).examples){
 const source=root+'/demo/examples/'+item.id;
 if(!existsSync(source+'/run.json')||!existsSync(root+'/reports/letters/'+item.id+'.json'))continue;
 bundle(source,resolve('public/examples/'+item.id),root+'/reports/letters/'+item.id+'.json',item,'letters');
 catalog.push({id:item.id,title:item.title,path:'/examples/'+item.id});
}
const calibratedSpecs=root+'/examples/microsoft-2025/calibrated-examples.json';
if(existsSync(calibratedSpecs))for(const item of JSON.parse(readFileSync(calibratedSpecs,'utf8')).examples){
 const source=root+'/demo/examples/'+item.id;
 if(!existsSync(source+'/run.json'))continue;
 bundle(source,resolve('public/examples/'+item.id),root+'/reports/letters-v2/'+item.id+'.json',item,'letters-v2');
 catalog.push({id:item.id,title:item.title,path:'/examples/'+item.id});
}
writeFileSync('public/catalog.json',JSON.stringify(catalog));
console.log('Bundled '+catalog.length+' recorded experiments.');
