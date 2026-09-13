// An immediate-start edit of the seven inference scenes in the long compilation.
// No titles or reference transcriptions are passed to the decoder.
export const socialChapters = [
  {id:'trained-labels',seconds:13,title:'First attempt: letters',subtitle:'Original retinal sampling',result:'A promising start. Also: “carb equiyalentr”.'},
  {id:'calibrated-labels',seconds:19,title:'Give the fly a better view',subtitle:'Same circuit · calibrated input',result:'One character off. We keep the capital I.'},
  {id:'calibrated-heading',seconds:10,title:'Now read the heading',subtitle:'Uppercase letters',result:'S and 5 are still a problem.'},
  {id:'calibrated-labels-more',seconds:17,title:'Five more lines',subtitle:'Mixed case · punctuation · no word correction',result:'A few letters get away. Close, little guy.'},
  {id:'income-table',seconds:17,title:'Put the numbers back in a table',subtitle:'Numeric checkpoint · geometry reconstructs the grid',result:'21 / 21 cells correct.'},
  {id:'cash-flow',seconds:16,title:'Try a bigger table',subtitle:'45 cells · 299 glyphs',result:'44 / 45 cells. One digit trips it up.'},
  {id:'tilted-table',seconds:13,title:'Rotate the page just 3°…',subtitle:'A controlled formatting failure',result:'The splitter breaks. The fly needs a deskew step.'},
] as const;
export const socialSeconds=socialChapters.reduce((sum,ch)=>sum+ch.seconds,0);
export function socialState(seconds:number,n:number,duration:number){
  // Every scene starts on glyph 1, then reserves 2.4 seconds for its raw result.
  const scanDuration=duration-2.4;
  const position=Math.min(n-1,Math.max(0,seconds)/scanDuration*n);
  return {done:Math.min(n,Math.floor(position)+1),position,finished:position>=n-1};
}
