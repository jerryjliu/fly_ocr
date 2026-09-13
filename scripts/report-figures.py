"""Draw scientific figures from saved measurements; no inferred or retouched data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
out=Path('docs/report-assets');out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
s=json.load(open('reports/letters-v2/summary.json'))
fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
labels=['68-class benchmark\n(n=1,632)','Letters subset\n(n=1,248)','Fresh fonts / all classes\n(n=544)']
a=[s['matched']['original']['accuracy'],s['matched']['original_letters']['accuracy'],s['fresh']['original']['accuracy']]
b=[s['matched']['selected']['accuracy'],s['matched']['selected_letters']['accuracy'],s['fresh']['selected']['accuracy']]
x=np.arange(3)
for values,offset,color,name in [(a,-.19,'#617e91','Original mapping'),(b,.19,'#e77c4c','Calibrated input')]:
 bars=ax.bar(x+offset,np.array(values)*100,.34,label=name,color=color)
 ax.bar_label(bars,fmt='%.1f%%',padding=3)
ax.set_xticks(x,labels);ax.set_ylim(0,105);ax.set_ylabel('Character accuracy (%)');ax.legend(loc='upper left',frameon=False,ncols=2)
fig.savefig(out/'accuracy.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,2.1));ax.axis('off');names=[('Pixels','Heuristic segmentation'),('48 x 48 glyph','Engineered input adapter'),('3,335 receptors','Fixed LIF circuit'),('4 x 1,024 counts','Trained 64-unit readout'),('Character','Geometric reconstruction')]
for i,(title,sub) in enumerate(names):
 x=i*2
 ax.add_patch(FancyBboxPatch((x,.45),1.73,1,boxstyle='round,pad=.04',facecolor='#edf3f5',edgecolor='#5b798c'))
 ax.text(x+.865,1.09,title,ha='center',fontsize=11,fontweight='bold');ax.text(x+.865,.78,sub.replace(' ','\n',1),ha='center',va='center',fontsize=8)
 if i<4:ax.annotate('',xy=(x+1.99,.95),xytext=(x+1.78,.95),arrowprops={'arrowstyle':'->','color':'#d66d3e'})
ax.set_xlim(-.1,9.95);ax.set_ylim(.15,1.8);fig.savefig(out/'pipeline.png',dpi=200,bbox_inches='tight');plt.close(fig)
formulas={
 'adapter':[r'$s_i=\mathrm{bilinear}(I,\,u_i)$',r'$\pi^*=\arg\min_{\pi}\sum_{k=1}^{825}\|u_k-q_{\pi(k)}\|_2^2$'],
 'dynamics':[r'$\tau_m\frac{dv_i}{dt}=-(v_i-E_L)+g_i+I_i,\qquad \tau_s\frac{dg_i}{dt}=-g_i$',r'$g_j\leftarrow g_j+W_{ji},\qquad W_{ji}=0.275\,\sigma_i\,n_{ij}$'],
 'drive':[r'$z_b=z_{b-1}+(1-e^{-25/10})\,[(1-s_i)-z_{b-1}]$',r'$I_i=20\,\frac{z_b}{0.2+z_b},\qquad I_{\mathrm{lamina}}=20$'],
 'readout':[r'$x=\mathrm{clip}\!\left(\frac{\sqrt{\mathrm{vec}(C)}-\mu_{\rm train}}{\sigma_{\rm train}},-10,10\right)$',r'$h=\max(0,W_1x+b_1),\qquad p=\mathrm{softmax}(W_2h+b_2)$'],
 'cer':[r'$\mathrm{CER}=\frac{\mathrm{substitutions}+\mathrm{deletions}+\mathrm{insertions}}{\mathrm{reference\ characters}}$'],
}
for name,lines in formulas.items():
 fig=plt.figure(figsize=(9,len(lines)*.62));fig.patch.set_facecolor('#f1f5f6')
 for i,line in enumerate(lines): fig.text(.035,1-(i+.72)/len(lines),line,fontsize=18)
 fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight',pad_inches=.12);plt.close(fig)
print('Saved measured charts and model equations')
