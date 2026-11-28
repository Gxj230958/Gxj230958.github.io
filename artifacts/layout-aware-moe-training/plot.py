"""Rebuild the figures for the combined Ascend MoE study from its two public datasets."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EARLY = ROOT / "static/projects/moe-training-ascend/data"
MATCH = ROOT / "artifacts/moe-branch-rounding/analysis"
OUT = ROOT / "static/projects/layout-aware-moe-training/figures"
INK, SOFT, RULE = "#302820", "#63584f", "#ded5c9"
PLUM, OCHRE, RUST, OLIVE = "#744966", "#ac722c", "#a04c34", "#526443"
PALE, PAPER = "#f6f0e6", "#fffdf9"
plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": SOFT, "ytick.color": SOFT,
                     "axes.edgecolor": RULE, "axes.spines.right": False, "axes.spines.top": False,
                     "svg.fonttype": "path", "pdf.fonttype": 42, "savefig.dpi": 320})

def rows(file):
    with file.open(newline="") as stream:
        return list(csv.DictReader(stream))

def save(fig, name):
    for suffix in ("svg", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", facecolor=fig.get_facecolor())
    plt.close(fig)

def title(ax, label, heading):
    ax.set_title(f"({label})  {heading}", fontsize=14, fontweight="bold", loc="left", pad=15)

def cover():
    f, a = plt.subplots(figsize=(7.8, 5.2), facecolor=PALE)
    f.subplots_adjust(0, 0, 1, 1)
    a.set(xlim=(0, 100), ylim=(0, 100))
    a.axis("off")
    def box(x, y, w, h, fc, ec=RULE):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.2",
                                   linewidth=0.9, edgecolor=ec, facecolor=fc))
    def label(x, y, s, size=13, color=INK, weight="normal", ha="center"):
        a.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, weight=weight)
    def arrow(start, end):
        a.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": SOFT,
                                                        "lw": 1.2, "mutation_scale": 11})
    label(6, 88, "One call, two rounding points", 26, weight="bold", ha="left")
    label(6, 80, "Input-gradient schedule for a gated expert", 15, color=SOFT, ha="left")
    a.plot([6,94], [73,73], color=RULE, lw=1)
    label(22, 67, "Packed gradient rows", 14)
    label(76, 67, "Paired weight views", 14)
    for x, first, second in [(7, "dGate", "dUp"), (61, "Gate", "Up")]:
        for i,(s,c,fill) in enumerate([(first,PLUM,"#eee3ec"), (second,OCHRE,"#f6e6cf")]):
            xx=x+15*i
            box(xx,54,14,9,fill,c)
            label(xx+7,58.4,s,15,c,"bold")
    arrow((22,52),(35,46))
    arrow((76,52),(63,46))
    box(21,33.5,56,12,INK,INK)
    label(49,39.5,"Native grouped matmul  ·  2E groups",16,"white","bold")
    arrow((39,33),(29.5,26.3))
    arrow((60,33),(68.5,26.3))
    for x,s,c,fill in [(11,"Gate product",PLUM,"#eee3ec"),(58,"Up product",OCHRE,"#f6e6cf")]:
        box(x,15.3,29,10,fill,c)
        label(x+14.5,20.4,s+"  ·  BF16",13.7,c)
    arrow((33,14.6),(43.4,8.6))
    arrow((65,14.6),(54.6,8.6))
    label(49,6.4,"Σ  →  dX",17,INK,"bold")
    save(f,"grouped-backward")

def block():
    original = rows(EARLY/"block-cases.csv")
    matched = [r for r in rows(MATCH/"block.csv") if r["model"] == "qwen"]
    f, (a,b) = plt.subplots(1,2,figsize=(10.2,3.7),gridspec_kw={"width_ratios":[1,1.06]})
    f.subplots_adjust(left=.077, right=.947, bottom=.205, top=.80, wspace=.51)
    loads=["uniform","hot","multihot","longtail"]
    counts=sorted({int(r["tokens"]) for r in original})
    grid=np.array([[100*float(next(r["reduction"] for r in original if r["distribution"]==d and int(r["tokens"])==n)) for d in loads] for n in counts])
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap=LinearSegmentedColormap.from_list("clay",["#f6eada","#d7b493","#a65c43","#743a36"])
    norm=Normalize(40,82)
    a.imshow(grid,cmap=cmap,norm=norm,aspect="auto",interpolation="nearest")
    for i,row in enumerate(grid):
        for j,v in enumerate(row):
            a.text(j,i,f"{v:.1f}",va="center",ha="center",fontsize=12,weight="bold",color="white" if v>68 else INK)
    a.set(xticks=range(4),xticklabels=["Uniform","Hot","Multi-hot","Long-tail"],yticks=range(4),yticklabels=[f"{n:,}" for n in counts],ylabel="Input rows")
    a.set_xlabel("Expert load distribution")
    a.tick_params(length=0,pad=5)
    title(a,"a","Complete layout patch")
    for i,r in enumerate(matched):
        v,lo,hi=map(float,(r["savings_pct"],r["low_pct"],r["high_pct"]))
        b.hlines(i,0,v,lw=1.7,color=RULE)
        b.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="D" if r["route"]=="hot" else "o",color=PLUM if i<2 else OLIVE,capsize=4,ms=5)
        b.text(v+1.2,i,f"{v:.2f}%",va="center",fontsize=11,color=INK)
    b.set(xlim=(0,46),ylim=(3.5,-.5),yticks=range(4),yticklabels=[f'{int(r["tokens"]):,} · {r["route"]}' for r in matched],xticks=[0,10,20,30,40],xlabel="Complete block time saved (%)")
    b.tick_params(axis="y",length=0)
    title(b,"b","Input-gradient schedule alone")
    save(f,"block-baselines")

def actor():
    original=rows(EARLY/"actor-pairs.csv")
    matched=[r for r in rows(MATCH/"actor.csv") if r["layers"]=="24"]
    f,(a,b)=plt.subplots(1,2,figsize=(10.2,3.65),gridspec_kw={"width_ratios":[1.12,1]})
    f.subplots_adjust(left=.082,right=.975,top=.8,bottom=.22,wspace=.41)
    for r in original:
        a.plot([0,1],[float(r["base"])/1000,float(r["candidate"])/1000],color="#b8a895",lw=.8,alpha=.52)
    for i,(k,c) in enumerate([("base",OCHRE),("candidate",PLUM)]):
        vals=np.array([float(r[k])/1000 for r in original])
        a.scatter(np.full(len(vals),i),vals,s=6,color=c,alpha=.4)
        m=float(np.median(vals))
        a.plot([i-.1,i+.1],[m,m],color=INK,lw=2)
        a.text(i+(-.13 if i==0 else .12),m,f"{m:.3f}",va="center",ha="right" if i==0 else "left",fontsize=11)
    a.set(xlim=(-.58,1.54),xticks=[0,1],xticklabels=["Split reference","Layout patch"],ylabel="Slowest-rank update time (s)")
    title(a,"a","48 layers · 16 NPUs")
    for i,r in enumerate(matched):
        val,lo,hi=map(float,(r["savings_pct"],r["low_pct"],r["high_pct"]))
        b.errorbar(val,i,xerr=[[val-lo],[hi-val]],fmt="D",color=PLUM,capsize=5,ms=6,lw=1.9)
        b.text(hi+.035,i,f"{val:.2f}%",va="center",fontsize=11)
    b.set(xlim=(1.7,2.73),ylim=(2.6,-.6),xticks=[1.8,2,2.2,2.4,2.6],yticks=range(3),yticklabels=["Session 1","Session 2","Session 3"],xlabel="Update time saved by grouping dX (%)")
    b.tick_params(axis="y",length=0)
    title(b,"b","24 layers · 8 NPUs")
    save(f,"actor-replay")

def trace():
    tasks=rows(MATCH/"qwen-hot-dx-tasks.csv")
    colors={"Slice":"#b7694e","GMM":PLUM,"Add":"#68625c","Pack":OCHRE,"Combine":OLIVE}
    f,(a,b)=plt.subplots(2,1,figsize=(10,4.1),gridspec_kw={"height_ratios":[1.7,1]})
    f.subplots_adjust(left=.17,right=.94,top=.79,bottom=.17,hspace=.85)
    for ax,arms in [(a,["B","C"]),(b,["C"])]:
        for j,arm in enumerate(arms):
            selected=[r for r in tasks if r["arm"]==arm]
            for r in selected:
                start=float(r["relative_start_us"])/1000
                duration=float(r["duration_us"])/1000
                ax.broken_barh([(start,duration)],(j-.24,.48),facecolors=colors[r["task"]],edgecolors="white",linewidth=.45)
                if ax==a and r["task"]=="Slice" and duration>.5:
                    ax.text(start+duration/2,j,f"{duration:.3f} ms",ha="center",va="center",color="white",size=11)
                if ax==b:
                    ax.text(start+duration/2,j,r["task"],ha="center",va="center",color="white",size=11,weight="bold")
            end=(float(selected[-1]["relative_start_us"])+float(selected[-1]["duration_us"]))/1000
            if ax==a: ax.text(end+.026,j,f"{end:.3f} ms",va="center",fontsize=11)
        ax.set(ylim=(len(arms)-.46,-.54),xlim=(0,1.63 if ax==a else .23))
        ax.tick_params(axis="y",length=0)
    a.set(yticks=[0,1],yticklabels=["Separate dX","Grouped dX"],xticks=np.arange(0,1.51,.25))
    b.set(yticks=[0],yticklabels=["Grouped detail"],xticks=[0,.05,.1,.15,.2],xlabel="Time since first dX task in each trace (ms)")
    a.legend(handles=[Rectangle((0,0),1,1,fc=c,label=l) for l,c in colors.items()],ncol=5,frameon=False,loc="lower left",bbox_to_anchor=(0,1.18),borderaxespad=0,handlelength=1.15,columnspacing=2)
    save(f,"device-timeline")

def arrival():
    stats=json.loads((EARLY/"plot-data.json").read_text())["arrival_statistics"]
    f,a=plt.subplots(figsize=(8.2,2.95))
    f.subplots_adjust(left=.185,right=.83,top=.81,bottom=.22)
    for i,key in enumerate(["rank0_first","rank0_last"]):
        for j,(stat,color,mark) in enumerate([("rank0_preparation_ms",PLUM,"o"),("full_pipeline_ms",OLIVE,"D")]):
            rec=stats[key][stat]
            v=rec["paired_delta_median_ms"]
            lo,hi=rec["paired_delta_ci95_ms"]
            y=i+(j-.5)*.26
            a.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt=mark,color=color,capsize=4,label=["Rank 0 preparation","Collective completion"][j] if i==0 else None,ms=6)
            a.text(v+.1,y,f"{v:.3f}",va="center",fontsize=11)
    a.axvline(0,color=RULE,lw=1,zorder=0)
    a.set(yticks=[0,1],yticklabels=["Rank 0 first","Rank 0 last"],ylim=(1.52,-.52),xlim=(-.25,4.5),xticks=[0,1,2,3,4],xlabel="Median paired time saved (ms)")
    a.legend(frameon=False,ncol=2,loc="lower left",bbox_to_anchor=(0,1),borderaxespad=0)
    a.tick_params(axis="y",length=0)
    save(f,"collective-arrival")

if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    cover()
    block()
    trace()
    actor()
    arrival()
