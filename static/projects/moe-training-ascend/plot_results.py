"""Render standalone research figures from data/plot-data.json; no NPU required."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'data/plot-data.json').read_text()); OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
INK='#162d40';BLUE='#567a9a';TEAL='#008b86';COPPER='#c57945';GRAY='#657787'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.titlelocation':'left','axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#cfdae0','xtick.color':GRAY,'ytick.color':GRAY,'axes.spines.top':False,'axes.spines.right':False,'grid.color':'#e7edf0','grid.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
CMAP=LinearSegmentedColormap.from_list('ocean',['#eff8f6','#8cd2c3','#009a8d','#075258'])
def finish(fig,name):
    for ext in ['svg','pdf','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',dpi=220)
    plt.close(fig)
# 1. Every complete-block case, with the same fixed color scale.
tokens=sorted(set(r['tokens'] for r in D['blocks']));dist=['uniform','hot','multihot','longtail']
actual=list(dict.fromkeys(r['distribution'] for r in D['blocks']))
if set(dist)!=set(actual):dist=actual
matrix=np.array([[next(r['reduction']*100 for r in D['blocks'] if r['tokens']==t and r['distribution']==d) for t in tokens] for d in dist])
fig,ax=plt.subplots(figsize=(8.5,4.7),layout='constrained');im=ax.imshow(matrix,cmap=CMAP,vmin=0,vmax=100,aspect='auto')
for i in range(4):
 for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}%',ha='center',va='center',fontsize=15,fontweight='bold',color='white' if matrix[i,j]>65 else INK)
ax.set(xticks=range(4),xticklabels=[f'{x:,}' for x in tokens],yticks=range(4),yticklabels=[x.replace('hot','hot spot').replace('longtail','long tail') for x in dist],xlabel='Input tokens',ylabel='Expert load distribution',title='Complete MoE block · forward + backward')
fig.colorbar(im,ax=ax,label='Median time reduction (%)',shrink=.85)
finish(fig,'block-performance')
# 2. Paired complete-actor samples; source medians and source paired CI.
s=D['actor_samples'];a=np.array([x['base']/1000 for x in s]);b=np.array([x['candidate']/1000 for x in s]);p=np.arange(1,len(s)+1)
fig,axs=plt.subplots(1,2,figsize=(11,4.1),gridspec_kw={'width_ratios':[1.65,1]},layout='constrained')
for v,c,label in [(a,BLUE,'Reference'),(b,TEAL,'Expert optimization')]:axs[0].plot(p,v,'o-',ms=3,lw=1,color=c,label=label)
axs[0].set(title='All 40 balanced pairs',xlabel='Pair index',ylabel='Slowest-rank update time (s)');axs[0].legend(frameon=False);axs[0].grid(axis='y')
axs[1].axhline(0,color=GRAY,lw=.8);axs[1].scatter(p,a-b,c=TEAL,s=21,alpha=.8)
axs[1].set(title='Absolute savings per pair',xlabel='Pair index',ylabel='Reference − candidate (s)');axs[1].grid(axis='y')
finish(fig,'actor-performance')
# 3. Profile counts are independent of timing, all 16 ranks have these counts.
fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained');x=np.arange(4);keys=['gmm','gmm_as_strided','pack','combine']
for offset,arm,color,label in [(-.18,'base',BLUE,'Reference'),(.18,'candidate',TEAL,'Expert optimization')]:
 vals=[D['profile'][arm][k] for k in keys];bars=ax.bar(x+offset,vals,.34,color=color,label=label)
 ax.bar_label(bars,padding=5,fontsize=10)
ax.set(xticks=x,xticklabels=['Grouped matmul','Layout materialization','Branch packing','Branch addition'],ylabel='Device kernel launches per rank',ylim=(0,1340),title='Full-model trace · mechanism behind the timing result');ax.legend(frameon=False);ax.grid(axis='y');ax.set_axisbelow(True)
finish(fig,'execution-mechanism')
# 4. Snapshot-derived longitudinal quality, including every declared seed.
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');q=D['quality']
for col,layers in enumerate([2,48]):
 rows=[r for r in q if r['layers']==layers and r['split']=='valid'];seeds=sorted(set(r['seed'] for r in rows));steps=sorted(set(r['step'] for r in rows))
 lookup={(r['seed'],r['step'],r['arm']):r['nll'] for r in rows}
 for arm,color,label in [('A',BLUE,'Reference A'),('B',COPPER,'Reference B'),('C',TEAL,'Combined candidate C')]:
  curves=np.array([[lookup[seed,step,arm] for step in steps] for seed in seeds])
  for curve in curves:axes[0,col].plot(steps,curve,color=color,alpha=.18,lw=.8)
  axes[0,col].plot(steps,curves.mean(0),'o-',color=color,label=label,ms=3,lw=1.8)
  if arm!='A':
   diff=np.array([[lookup[seed,step,arm]-lookup[seed,step,'A'] for step in steps] for seed in seeds])
   for curve in diff:axes[1,col].plot(steps,curve,color=color,alpha=.3,lw=.8)
   axes[1,col].plot(steps,diff.mean(0),'o-',color=color,ms=3,lw=1.8,label=f'{arm} − A')
 axes[0,col].set(title=f'{layers} layers · {len(seeds)} independent seeds',ylabel='Validation NLL (lower is better)')
 axes[1,col].axhline(0,color=GRAY,lw=.7);axes[1,col].set(xlabel='Training update',ylabel='Validation NLL difference')
 for row in range(2):axes[row,col].grid(axis='y');axes[row,col].legend(frameon=False,fontsize=8)
finish(fig,'quality-trajectories')
# 5. One-sided confidence limits, not two-sided error bars.
fig,axes=plt.subplots(1,2,figsize=(11,4.9),layout='constrained')
labels=[f'{l} layers · {s.replace("train_holdout","train holdout")}' for l in [2,48] for s in ['valid','test','train_holdout']]
for ax,key,limit,title in zip(axes,['candidate_nll','excess_routing'],[.005,.01],['Candidate − reference NLL','Excess route disagreement (C − B)']):
 vals=[D['quality_assessment'][f'layers{l}']['splits'][s][key]['upper95'] for l in [2,48] for s in ['valid','test','train_holdout']]
 ax.axvline(0,color=GRAY,lw=.8);ax.axvline(limit,color=COPPER,ls='--',label='Declared acceptance limit')
 ax.scatter(vals,np.arange(6),color=[TEAL]*3+[BLUE]*3,s=55,zorder=3)
 for i,v in enumerate(vals):ax.annotate(f'{v:.5f}',(v,i),xytext=(7,-3),textcoords='offset points',fontsize=8)
 ax.set(yticks=range(6),yticklabels=labels,title=title,xlabel='One-sided 95% upper confidence bound',xlim=(min(-.001,min(vals)*1.4),limit*1.22),ylim=(5.6,-.6));ax.legend(frameon=False,fontsize=8,loc='lower right');ax.grid(axis='x')
finish(fig,'quality-confidence')
# 6. Training-state differences, reduced over parameters within each decoder layer.
fig,axes=plt.subplots(2,1,figsize=(11,5.3),layout='constrained')
for ax,kind,title in zip(axes,['parameter','gradient'],['Final parameter differences','Final gradient differences']):
 mat=np.array([[np.median([r['relative_l2'] for r in D['state'] if r['kind']==kind and r['arm']==arm and r['layer']==layer]) for layer in range(48)] for arm in ['B','C']])
 assert np.isfinite(mat).all() and (mat>0).all()
 im=ax.imshow(mat,aspect='auto',cmap=CMAP,norm=LogNorm(vmin=mat.min(),vmax=mat.max()))
 ax.set(title=title,yticks=[0,1],yticklabels=['Reference B vs A','Candidate C vs A'],xticks=[0,7,15,23,31,39,47],xlabel='Decoder layer (zero-based)');fig.colorbar(im,ax=ax,label='Relative L2 · median of 3 seeds',shrink=.85)
finish(fig,'snapshot-state-differences')
# 7. Pair-complete controlled arrival experiment, no filtered samples.
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for ax,metric,title in zip(axes,['rank0_preparation_ms','full_pipeline_ms'],['Local preparation savings','Collective pipeline completion savings']):
 for j,regime in enumerate(['rank0_first','rank0_last']):
  rows=[r for r in D['arrival'] if r['regime']==regime];lookup={(r['pair_index'],r['arm']):r[metric] for r in rows}
  diff=np.array([lookup[i,'base']-lookup[i,'candidate'] for i in range(100)])
  # Deterministic offsets separate points without changing their values.
  ax.scatter(j+np.linspace(-.16,.16,100),diff,color=[BLUE,TEAL][j],alpha=.4,s=15)
  ax.plot([j-.22,j+.22],[np.median(diff)]*2,color=INK,lw=2)
  stat=D['arrival_statistics'][regime][metric];ci=stat.get('paired_delta_ci95_ms')
  if ci:ax.plot([j,j],ci,color=COPPER,lw=3);ax.plot([j-.07,j+.07],[ci[0]]*2,color=COPPER);ax.plot([j-.07,j+.07],[ci[1]]*2,color=COPPER)
  ax.text(j,4.6,f'{np.median(diff):.3f} ms',ha='center',fontsize=11,fontweight='bold')
 ax.axhline(0,color=GRAY,lw=.8);ax.set(title=title,xticks=[0,1],xticklabels=['Optimized rank arrives first','Optimized rank arrives last'],ylabel='Reference − candidate (ms)',ylim=(-.6,5.2));ax.grid(axis='y')
finish(fig,'arrival-sensitivity')
print('Rendered 7 figures in SVG, PDF and PNG.')
