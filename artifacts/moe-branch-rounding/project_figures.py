"""Project-page illustrations of branch grouping and its measured cost."""
import csv

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from paper_figures import ROOT, OUT, records, save

INK, MUTED = "#183343", "#4d6471"
GATE, UP = "#07689a", "#a34b12"
ICE, PEACH = "#d9ecf6", "#ffe6ce"
TEAL, PURPLE = "#007568", "#74418b"


def mechanism():
    fig, ax = plt.subplots(figsize=(6.8, 4.65))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 100), ylim=(0, 69), aspect="auto")
    ax.axis("off")

    def label(x, y, s, size=13, color=INK, bold=False, align="center"):
        ax.text(x, y, s, ha=align, va="center", fontsize=size, color=color,
                weight="bold" if bold else "normal")

    def box(x, y, w, h, fill, edge, radius=1.5):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=fill,edgecolor=edge,lw=1.15))

    def arrow(x, y, u, v):
        ax.annotate("", xy=(u,v), xytext=(x,y), arrowprops=dict(
            arrowstyle="-|>", color=MUTED, lw=1.25, mutation_scale=12, shrinkA=0, shrinkB=0))

    label(50,63.8,"One GMM, two BF16 products",25,bold=True)
    # Two input arrangements match the two native GMM operands.
    for x,title,sub in [(4,"Pack gradients","gate / up rows per expert"),
                        (53,"Reshape weight view","[2E, I, H] · shared storage")]:
        box(x,45.0,43,11.9,"#f6f8f8","#b9c7cc")
        for row in range(4):
            for col in range(2):
                branch_color = (GATE if row >= 2 else UP)
                ax.add_patch(Rectangle((x+1.8+col*2.65,47.6+row*1.55),2.35,1.28,
                                       color=branch_color,linewidth=0))
        label(x+25.1,52.9,title,14.4,bold=True)
        label(x+25.1,48.2,sub,10.8,color=MUTED)
    arrow(25.5,44.7,25.5,39.8)
    arrow(64.2,44.7,64.2,39.8)
    box(4,9.5,65.5,29.8,"white","#7f96a2",2)
    label(36.8,35.2,"E experts → 2E groups",18,bold=True)
    for y,branch,product,fill,color in [
        (23.0,"Gate · group 2e",r"$dG_e W_{g,e}$",ICE,GATE),
        (12.8,"Up · group 2e+1",r"$dU_e W_{u,e}$",PEACH,UP)]:
        box(6.2,y,61.0,8.1,fill,color)
        label(8.1,y+4.1,branch,12.8,color=color,bold=True,align="left")
        label(43,y+4.1,product,14.5)
        label(60.6,y+4.1,"BF16",12.7,color=color,bold=True)
    ax.plot([69.6,75.3,75.3],[27.05,27.05,22.0],color=GATE,lw=1.4)
    ax.plot([69.6,75.3,75.3],[16.85,16.85,22.0],color=UP,lw=1.4)
    arrow(75.3,22,79.2,22)
    box(79.6,16.8,16.2,10.6,"#def2ed",TEAL)
    label(87.7,22.1,r"$dX_e$",18,color=TEAL,bold=True)
    label(87.7,12.8,"Combine (+)",13,color=TEAL,bold=True)
    label(50,4.4,"Each branch rounds before the final sum.",14.5)
    save(fig,"expert-branch-cover")


def matched_baseline():
    rows=[r for r in records("block") if r["model"]=="qwen"]
    fig=plt.figure(figsize=(7.2,4.65))
    fig.subplots_adjust(0,0,1,1)
    top=fig.add_axes([0,0,1,1])
    top.axis("off")
    top.text(.05,.925,"Isolating the input-gradient schedule",fontsize=20,weight="bold",color=INK)
    top.text(.05,.838,"Held fixed",weight="bold",size=12,color=INK)
    top.text(.235,.838,"Routing   ·   packed forward   ·   physical weight gradients",size=12,color=MUTED)
    top.plot([.05,.96],[.80,.80],color="#b9c7cc",lw=.8,transform=top.transAxes)
    top.set(xlim=(0,1),ylim=(0,1))
    top.text(.05,.741,"Changed",weight="bold",size=12,color=INK)
    top.text(.235,.741,"B   Two native GMMs",size=12,color=GATE,weight="bold")
    top.text(.59,.741,"C   One 2E-group GMM",size=12,color=TEAL,weight="bold")
    ax=fig.add_axes([.24,.154,.44,.46])
    for j,r in enumerate(rows):
        b,c=map(float,(r["b_ms"],r["c_ms"]))
        ax.plot([c,b],[j,j],color="#85979d",lw=2)
        ax.plot(b,j,"s",color=GATE,ms=6)
        ax.plot(c,j,"o",color=TEAL,ms=7)
        ax.annotate(f"{c:.2f}",(c,j),(-4,6),textcoords="offset points",ha="right",fontsize=9,color=TEAL)
        ax.annotate(f"{b:.2f}",(b,j),(3,6),textcoords="offset points",fontsize=9,color=GATE)
        top.text(.77,.154+.46*(1-(j+.58)/4.12),f"{float(r['savings_pct']):.2f}%",
                 weight="bold",fontsize=13,va="center",color=INK)
        top.text(.901,.154+.46*(1-(j+.58)/4.12),
                 f"{float(r['low_pct']):.2f}–{float(r['high_pct']):.2f}",
                 fontsize=9.5,ha="center",va="center",color=MUTED)
    top.text(.765,.635,"Reduction",size=10,weight="bold")
    top.text(.903,.635,"95% CI",size=10,ha="center",weight="bold")
    ax.set(yticks=range(len(rows)),yticklabels=[f"{int(r['tokens']):,}  ·  {r['route']}" for r in rows],
           ylim=(3.54,-.58),xlim=(0,18.6),xticks=[0,4,8,12,16])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y",length=0,labelsize=11)
    ax.grid(axis="x",alpha=.16)
    ax.set_xlabel("Median block forward + backward (ms)",fontsize=11)
    save(fig,"matched-qwen")


def trace_timeline():
    with (ROOT/"analysis/qwen-hot-dx-tasks.csv").open() as f:
        tasks=list(csv.DictReader(f))
    colors={"Slice": "#48688f","GMM": "#cc7726","Add": "#684478","Pack": TEAL,"Combine": PURPLE}
    fig=plt.figure(figsize=(7.3,4.25))
    fig.text(.05,.93,"Where the backward pass saves time",fontsize=20,weight="bold",color=INK)
    fig.text(.05,.855,"Qwen3   ·   128 token rows   ·   fixed top-8 hotspot   ·   Ascend 910B2C",
             size=11,color=MUTED)
    ax=fig.add_axes([.21,.455,.74,.30])
    for arm,j in [("B",1),("C",0)]:
        arm_rows=[r for r in tasks if r["arm"]==arm]
        for r in arm_rows:
            name=r["task"]
            start,dt=(float(r[k])/1000 for k in ("relative_start_us","duration_us"))
            ax.barh(j,dt,left=start,height=.45,color=colors[name],edgecolor="white",lw=.35)
            if name=="Slice" and dt>.3:
                ax.text(start+dt/2,j,f"Slice   {dt:.3f} ms",ha="center",va="center",size=10,color="white")
        end=max(float(r["relative_start_us"])+float(r["duration_us"]) for r in arm_rows)/1000
        ax.text(end+.02,j,f"{end:.3f}",va="center",size=10,weight="bold")
    ax.set(yticks=[1,0],yticklabels=["B · Separate","C · Grouped"],ylim=(-.55,1.55),
           xlim=(0,1.60),xticks=[0,.25,.5,.75,1,1.25,1.5])
    ax.set_xlabel("Time from each arm’s first dX task (ms)",labelpad=2)
    ax.grid(axis="x",alpha=.14)
    ax.tick_params(axis="y",length=0,labelsize=11)
    ax.spines["left"].set_visible(False)
    fig.text(.05,.278,"Inside C",size=13,weight="bold")
    lower=fig.add_axes([.21,.18,.74,.105])
    for r in [r for r in tasks if r["arm"]=="C"]:
        x,dt=map(float,(r["relative_start_us"],r["duration_us"]))
        lower.barh(0,dt,left=x,color=colors[r["task"]],edgecolor="white",height=.76)
        lower.text(x+dt/2,0,f"{r['task']}   {dt:.1f} µs",ha="center",va="center",
                   fontsize=11,weight="bold",color="white" if r["task"]!="GMM" else INK)
    lower.set(xlim=(0,225),ylim=(-.5,.5),yticks=[],xticks=[])
    lower.set_frame_on(False)
    fig.text(.21,.12,"Magnified C timeline",fontsize=9,color=MUTED)
    fig.text(.05,.026,"B tasks",fontsize=10,weight="bold")
    for x,name in [(0.21,"Slice"),(.39,"GMM"),(.57,"Add")]:
        fig.add_artist(Rectangle((x,.023),.014,.021,transform=fig.transFigure,
                                 facecolor=colors[name],edgecolor="none"))
        fig.text(x+.021,.026,name,fontsize=9,color=MUTED)
    save(fig,"qwen-dx-profile")


if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    mechanism()
    matched_baseline()
    trace_timeline()
