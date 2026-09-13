// Results first, then evidence, then context and the report. Development
// history and ablations remain in the original compilation and social cut.
export const teaserChapters = [
  {id:'calibrated-labels',seconds:12,scanSeconds:9.2,startEvent:4,kicker:'PRINTED CHARACTERS',headline:'I made a fly circuit read a PDF.',subtitle:'Letters and numbers, recognized from actual document pixels.',result:'Printed text → recognized characters.'},
  {id:'income-table',seconds:13,scanSeconds:10,startEvent:1,kicker:'TABLES OF VALUES',headline:'Numbers, back in rows and columns.',subtitle:'A real financial table, recognized one character at a time.',result:'21 of 21 cells correct in this example.'},
  {id:'calibrated-labels-more',seconds:15,scanSeconds:7.7,startEvent:1,kicker:'THE IDEA',headline:'166,700 neurons. A tiny trained decoder.',subtitle:'A fixed simulated fly circuit + a learned character readout.',result:'It still makes mistakes. The full results are in the report.'},
] as const;
export const teaserSeconds=teaserChapters.reduce((sum,ch)=>sum+ch.seconds,0);
export function teaserState(seconds:number,n:number,chapter:(typeof teaserChapters)[number]){
  const position=Math.min(n-1,chapter.startEvent-1+Math.max(0,seconds)/chapter.scanSeconds*(n-chapter.startEvent+1));
  return {done:Math.floor(position)+1,position,finished:position>=n-1};
}
