"""Publication figures for the matched branch-gradient measurements."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
BLUE, ORANGE, GREEN, PURPLE, INK = "#0072B2", "#D55E00", "#00805D", "#89528C", "#192B36"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.right": False, "axes.spines.top": False, "axes.edgecolor": "#63717A",
    "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "path",
})


def records(name):
    with (ROOT / "analysis" / (name + ".csv")).open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=360, facecolor="white", bbox_inches="tight", pad_inches=.09)
    plt.close(fig)


def method():
    fig, ax = plt.subplots(figsize=(7.15, 2.8))
    ax.set(xlim=(0, 10), ylim=(0, 4.5))
    ax.axis("off")

    def box(x, y, w, label, fill, edge):
        ax.add_patch(FancyBboxPatch((x, y-.33), w, .66, boxstyle="round,pad=0.04,rounding_size=.08",
                                    edgecolor=edge, facecolor=fill, linewidth=1.05))
        ax.text(x+w/2, y, label, ha="center", va="center", fontsize=9)

    def arrow(x, y, u, v):
        ax.annotate("", (u,v), (x,y), arrowprops=dict(arrowstyle="-|>", lw=1, color="#53626B"))

    ax.text(0,4.24,"B   Separate branch products",weight="bold",size=11)
    ax.text(0,2.14,"C   Group by (expert, branch)",weight="bold",size=11)
    for y, branch, color, light in [(3.64,"gate",BLUE,"#E4F1F8"), (2.88,"up",ORANGE,"#FDF0E6")]:
        box(.12,y,2.08,f"{branch} gradient slice",light,color)
        box(2.72,y,1.60,"GMM · E groups","#F5F6F6","#819099")
        box(4.90,y,2.10,f"BF16 {branch} product",light,color)
        arrow(2.25,y,2.65,y)
        arrow(4.38,y,4.82,y)
        arrow(7.08,y,7.96,3.26)
    box(8.03,3.26,1.69,"Add → BF16 dX","#E8F5EF",GREEN)
    box(.12,1.19,2.08,"Pack gate, up\nwithin each expert","#EDF0F7",PURPLE)
    box(2.72,1.19,1.60,"One GMM\n2E groups","#F5F6F6","#819099")
    box(4.90,1.59,2.10,"BF16 gate product","#E4F1F8",BLUE)
    box(4.90,.78,2.10,"BF16 up product","#FDF0E6",ORANGE)
    arrow(2.25,1.19,2.65,1.19)
    for y in (1.59,.78):
        arrow(4.38,1.19,4.82,y)
        arrow(7.08,y,7.96,1.19)
    box(8.03,1.19,1.69,"Add → BF16 dX","#E8F5EF",GREEN)
    ax.text(.12,.09,"Example: two experts with 3 and 1 rows  →  grouped row counts [3, 3, 1, 1].",size=9)
    save(fig,"branch-schedule")


def regimes(blocks):
    fig, axes = plt.subplots(1,2,figsize=(7.45,3.48),layout="constrained",sharex=True)
    styles = {"uniform":(BLUE,"o"), "hot":(ORANGE,"s"), "text":(GREEN,"D")}
    for ax, rows, title in zip(axes, [[r for r in blocks if r["model"]!="mixtral"],
                                      [r for r in blocks if r["model"]=="mixtral"]],
                               ["a   Qwen3 and routing control","b   Pretrained Mixtral layer"]):
        ax.set_title(title,loc="left",fontweight="bold")
        ax.axvline(0,lw=.8,color=INK)
        labels=[]
        for j,r in enumerate(rows):
            val,lo,hi = (float(r[k]) for k in ("savings_pct","low_pct","high_pct"))
            c,m=styles[r["route"]]
            ax.errorbar(val,j,xerr=[[val-lo],[hi-val]],color=c,fmt=m,ms=4,capsize=2)
            labels.append(f"{int(r['tokens']):,}  {r['route']}")
            ax.annotate(f"{val:+.1f}",(hi,j),xytext=(4,0),textcoords="offset points",va="center",fontsize=8)
        ax.set_yticks(range(len(rows)),labels)
        ax.set_ylim(len(rows)-.48,-.6)
        ax.set_xlim(-8,52)
        ax.set_xticks([0,10,20,30,40,50])
        ax.set_xlabel("Reduction in block time (%)")
        ax.grid(axis="x",alpha=.15)
    save(fig,"routing-and-length")


def traces():
    d={(r["stage"],r["case"],r["arm"]):r for r in records("profile")}
    fig, (a,b) = plt.subplots(1,2,figsize=(7.1,2.8),layout="constrained")
    x=np.arange(2)
    for j,(arm,c,hat,lab) in enumerate([("base",BLUE,"","Separate · slices"),("candidate",ORANGE,"//","Grouped · pack + combine")]):
        rows=[d["trace-boundary","32768-"+route,arm] for route in ("uniform","hot")]
        y=[float(r["slice_ms"]) if j==0 else float(r["pack_ms"])+float(r["combine_ms"]) for r in rows]
        bars=a.bar(x+(j-.5)*.35,y,.34,color=c,hatch=hat,edgecolor=INK,lw=.5,label=lab)
        a.bar_label(bars,fmt="%.2f",fontsize=8,padding=2)
    a.set(xticks=x,xticklabels=["Uniform","One hot expert"],ylabel="Device task time (ms)",ylim=(0,1.85))
    a.legend(frameon=False,loc="upper left",fontsize=8)
    a.set_title("a   Narrow experts · staging time",loc="left",weight="bold")
    for j,(c,lab,hat) in enumerate([(GREEN,"Staging",""),(PURPLE,"Native GMMs","//")]):
        y=[]
        for stage,case in [("trace-mixtral","8192-text"),("trace-mixtral-length","32768-text")]:
            old,new=d[stage,case,"base"],d[stage,case,"candidate"]
            y.append(float(new["pack_ms"])+float(new["combine_ms"])-float(old["slice_ms"]) if j==0
                     else float(new["gmm_ms"])-float(old["gmm_ms"]))
        bars=b.bar(x+(j-.5)*.35,y,.34,color=c,hatch=hat,edgecolor=INK,lw=.5,label=lab)
        b.bar_label(bars,fmt="%+.2f",fontsize=8,padding=2)
    b.axhline(0,color=INK,lw=.7)
    b.set(xticks=x,xticklabels=["8,192 rows","32,768 rows"],ylabel="Grouped − separate (ms)",ylim=(-5,14))
    b.legend(frameon=False,loc="upper left",fontsize=8)
    b.set_title("b   Wide experts · real text",loc="left",weight="bold")
    save(fig,"device-costs")


def actor():
    fig,(a,b)=plt.subplots(1,2,figsize=(7.2,2.5),layout="constrained",width_ratios=[1.4,1])
    sessions=[r for r in records("actor") if r["layers"]=="24"]
    for j,r in enumerate(sessions):
        samples=(ROOT/"results"/r["session"]/"samples.jsonl").read_text().splitlines()
        pairs=[json.loads(s) for s in samples]
        y=[s["base"]-s["candidate"] for s in pairs]
        a.scatter(y,j+np.linspace(-.2,.2,len(y)),color="#738494",alpha=.6,s=8,lw=0)
        a.plot([float(r["low_ms"]),float(r["high_ms"])],[j,j],color=GREEN,lw=2)
        a.scatter(float(r["median_paired_ms"]),j,color=GREEN,marker="D",s=23)
        for k,c,m in [("b_p95",BLUE,"o"),("c_p95",ORANGE,"s")]:
            b.scatter(float(r[k])/1000,j,marker=m,s=32,color=c,
                      label=("Separate" if k=="b_p95" else "Grouped") if j==0 else None)
        b.plot([float(r[k])/1000 for k in ("b_p95","c_p95")],[j,j],zorder=0,color=".6")
    for ax in (a,b):
        ax.set_yticks(range(3),["Session 1","Session 2","Session 3"])
        ax.set_ylim(2.55,-.5)
        ax.grid(axis="x",alpha=.15)
    a.axvline(0,color=INK,lw=.75)
    a.set_title("a   All 120 paired updates",loc="left",weight="bold")
    a.set_xlabel("Separate − grouped update time (ms)")
    b.set_title("b   Tail latency",loc="left",weight="bold")
    b.set_xlabel("Session p95 (s)")
    b.legend(loc="upper center",bbox_to_anchor=(.5,-.19),ncol=2,fontsize=8,frameon=False)
    save(fig,"actor-updates")


def cover(blocks):
    rows=[r for r in blocks if r["route"]=="text"]
    fig,ax=plt.subplots(figsize=(4.4,2.7),layout="constrained")
    vals=[float(r["savings_pct"]) for r in rows]
    bars=ax.barh(range(3),vals,color=[BLUE,GREEN,ORANGE],edgecolor=INK,lw=.6,height=.46)
    bars[-1].set_hatch("//")
    ax.axvline(0,lw=.8,color=INK)
    ax.set_yticks(range(3),[f"{int(r['tokens']):,} rows" for r in rows],fontsize=12)
    ax.set_xlim(-4,43)
    ax.invert_yaxis()
    ax.set_title("Mixtral backward grouping",loc="left",size=16,weight="bold",pad=9)
    ax.set_xlabel("Reduction in expert-block time (%)",fontsize=12)
    for j,v in enumerate(vals):
        ax.text(max(v,0)+1.3,j,f"{v:+.1f}%",va="center",weight="bold",size=13)
    ax.set_xticks([0,10,20,30,40])
    save(fig,"mixtral-cover")


if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    blocks=records("block")+records("length-extrapolation")
    method()
    regimes(blocks)
    traces()
    actor()
    cover(blocks)
