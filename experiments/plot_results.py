"""Generate publication-sized PDF and PNG figures from the included result tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parent
JOINT=BASE/"joint_stopping"
RES=BASE/"extensions"/"results"
INK="#183248"; GREY="#60717f"; BLUE="#236389"; TEAL="#087966"; ORANGE="#ad5827"; GRID="#dce3e7"
plt.rcParams.update({"font.size":10, "axes.titlesize":11, "axes.titleweight":"bold",
                     "axes.labelcolor":INK, "text.color":INK, "xtick.color":GREY, "ytick.color":GREY,
                     "axes.spines.top":False, "axes.spines.right":False, "axes.edgecolor":GRID,
                     "legend.frameon":False, "pdf.fonttype":42, "font.family":"DejaVu Sans"})


def save(fig, directory, name):
    for suffix in ("png","pdf"):
        fig.savefig(directory/f"{name}.{suffix}",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def header(fig,title,subtitle):
    fig.text(.04,.975,title,ha="left",va="top",fontsize=17,fontweight="bold")
    fig.text(.04,.875,subtitle,ha="left",va="top",fontsize=10,color=GREY)


def headline(directory):
    df=pd.read_csv(JOINT/"thresholds.csv")
    row=df.loc[(df.family=="set_cover") & (df.realizations==1) & df.chosen].iloc[0]
    timing=pd.read_json(JOINT/"timings.jsonl",lines=True)
    timing=timing[(timing.family=="set_cover") & (timing.realizations==1)].groupby("method").seconds.median()
    fig,axes=plt.subplots(1,3,figsize=(11.0,3.8))
    header(fig,"Two-stage set covering with one demand realization per example",
           f"48 calibration examples  ·  α = 0.10  ·  15th cost percentile  ·  common last-unsafe boundary {row.cutoff:.5f}")
    series=[("Planner LP solves",[row.full_lp,row.full_lp,row.independent_lp,row.joint_lp]),
            ("Loss evaluations",[row.full_losses,row.joint_losses,row.independent_losses,row.joint_losses]),
            ("Elapsed time (s)",[timing["independent"],timing["joint"]])]
    for i,(ax,(label,values)) in enumerate(zip(axes,series)):
        methods=["Exhaustive","Full path + sweep","Independent","Joint"] if i<2 else ["Independent","Joint"]
        colors=["#a9b7bf","#80919e",BLUE,TEAL] if i<2 else [BLUE,TEAL]
        ys=np.arange(len(values))
        ax.barh(ys,values,color=colors,height=.55)
        ax.set_yticks(ys,methods);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.35)
        ax.set_title(label,loc="left",pad=10)
        for y,v in zip(ys,values):
            ax.text(v+max(values)*.025,y,(f"{v:,.0f}" if i<2 else f"{v:.2f}"),va="center",fontsize=9)
        ax.tick_params(axis="both",length=0);ax.set_xticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        annotation=(f"{100*(1-values[-1]/values[0]):.1f}% fewer than exhaustive" if i<2
                    else f"{values[0]/values[-1]:.2f}× speedup vs. independent")
        ax.text(0,-.15,annotation,transform=ax.transAxes,color=TEAL,fontweight="bold",fontsize=9)
    fig.subplots_adjust(left=.14,right=.985,top=.68,bottom=.25,wspace=.82)
    fig.text(.04,.028,"Independent scans stop at each example’s first unsafe decision. Times are medians of three serial runs per method.",fontsize=9,color=GREY)
    save(fig,directory,"set_cover_T1")


def graded(directory):
    data=pd.read_csv(RES/"graded_counts.csv")
    fig,axes=plt.subplots(1,2,sharey=True,figsize=(9.6,3.8))
    header(fig,"How much does joint stopping add for graded losses?",
           "Set covering  ·  loss evaluations saved relative to independent per-example saturation stopping")
    for ax,(realizations,group) in zip(axes,data.groupby("realizations",sort=True)):
        x=100*group["quantile"];y=group.percent_saved
        ax.plot(x,y,marker="o",color=TEAL,ms=3.7)
        selected=group[group.paper_threshold].iloc[0]
        ax.scatter(100*selected["quantile"],selected.percent_saved,s=100,facecolors="none",edgecolors=INK,zorder=4)
        ax.set_title(f"T = {realizations} demand realization{'s' if realizations != 1 else ''} per example",loc="left")
        ax.set_xlabel("Cost percentile used as threshold")
        ax.set_ylim(-3,100);ax.set_xticks([10,30,50,70,90]);ax.grid(axis="y",color=GRID)
        ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    axes[0].set_ylabel("Evaluations saved")
    fig.subplots_adjust(top=.70,bottom=.25,left=.09,right=.985,wspace=.14)
    fig.text(.04,.025,"Open circles indicate the paper’s binary operating thresholds. All 17 settings on each curve reuse the same 48 examples.",fontsize=9,color=GREY)
    save(fig,directory,"graded_thresholds")


def scaling(directory):
    data=pd.read_csv(RES/"scaling.csv").query("J == 8193")
    groups=list(data.groupby("Regime",sort=False))
    fig,axes=plt.subplots(1,3,figsize=(11,4.3),sharey=True)
    header(fig,"Grid-query costs depend on where the unsafe indices lie",
           "8,193 grid points  ·  40 independent threshold-vector draws per point  ·  fewer queries is better")
    specs=[("CertBisect","Certified bisection",ORANGE,"o"),("AnchQS_3w","Anchored three-way",GREY,"s"),
           ("AnchQS_2w","Anchored two-way",TEAL,"^"),("UnanchQS_2w","Unanchored two-way",BLUE,"D")]
    for ax,((_,g),title) in zip(axes,zip(groups,["Sparse, small cutoff","Dense, small cutoff","Dense, uniform indices"])):
        g=g.sort_values("n")
        for col,label,color,marker in specs:
            ax.plot(g.n,g[col]/g.n,c=color,marker=marker,ms=4,lw=1.6,label=label)
        ax.set_title(title,loc="left");ax.set_xscale("log");ax.set_xticks([48,250,1000,2000],["48","250","1k","2k"])
        ax.tick_params(axis="x",which="minor",bottom=False)
        ax.set_xlabel("Calibration examples");ax.grid(axis="y",color=GRID)
    top=max(data[c].div(data.n).max() for c,*_ in specs)
    axes[0].set_ylim(0,np.ceil(top+.15))
    axes[0].set_ylabel("Mean queries per example")
    fig.subplots_adjust(top=.70,bottom=.25,left=.08,right=.98,wspace=.12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=4,fontsize=9)
    save(fig,directory,"grid_scaling")


def synthetic(directory):
    df=pd.read_csv(RES/"planner_study.csv")
    fig,ax=plt.subplots(figsize=(8.4,3.8))
    header(fig,"On-demand paths reduce work for LP and integer planners",
           "100 synthetic examples per setting  ·  operation counts relative to full path enumeration")
    y=np.arange(len(df))
    for offset,quantity,color,label in [(-.16,"solves",BLUE,"Planner solves"),(.16,"losses",TEAL,"Loss evaluations")]:
        change=100*(1-df["joint_"+quantity]/df["full_"+quantity])
        ax.barh(y+offset,change,height=.26,color=color,label=label)
        for yi,val in zip(y+offset,change):
            ax.text(val+.8,yi,f"{val:.1f}%",va="center",fontsize=9)
    labels=[f"{r.planner.upper()}, {'graded' if r.loss=='continuous' else 'binary'} loss" for r in df.itertuples()]
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,73)
    ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.set_xlabel("Reduction from full enumeration")
    ax.set_axisbelow(True);ax.grid(axis="x",color=GRID)
    fig.subplots_adjust(top=.70,left=.20,bottom=.25,right=.96)
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",ncol=2)
    save(fig,directory,"planner_savings")


def heldout(directory):
    df=pd.read_csv(RES/"pool_splits.csv")
    fig,ax=plt.subplots(figsize=(9,3.8))
    header(fig,"Held-out losses across repeated calibration splits",
           "300 synthetic LP scenes  ·  150 calibration and 150 test scenes per split  ·  500 splits per target")
    arr=[];levels=[]
    for alpha,g in df.groupby("alpha"):
        arr.append(100*g.heldout_loss.to_numpy());levels.append(100*alpha)
    y=np.arange(len(arr))
    ax.boxplot(arr,orientation="horizontal",positions=y,widths=.40,patch_artist=True,showfliers=False,
               whis=(10,90),boxprops=dict(facecolor="#d6e9ed",edgecolor=BLUE),medianprops=dict(color=BLUE),
               whiskerprops=dict(color=BLUE),capprops=dict(color=BLUE))
    ax.scatter(levels,y,s=90,marker="|",color=ORANGE,linewidth=2.6,zorder=4,label="Target")
    means=[np.mean(a) for a in arr]
    ax.scatter(means,y,s=38,marker="D",color=TEAL,zorder=5,label="Mean realized loss")
    for yi,values in zip(y,arr):
        ax.text(np.percentile(values,90)+.7,yi,f"mean {np.mean(values):.2f}%",va="center",fontsize=9,color=TEAL)
    ax.set_yticks(y,[f"α = {a:.0f}%" for a in levels]);ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.set_xlim(0,max(np.percentile(a,90) for a in arr)+8)
    ax.set_xlabel("Mean realized loss on a held-out split");ax.grid(axis="x",color=GRID)
    fig.subplots_adjust(top=.69,bottom=.30,left=.11,right=.96)
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.055),ncol=2)
    fig.text(.04,.02,"Boxes show quartiles, whiskers the 10th–90th percentiles. All splits reuse the same pool and are dependent.",fontsize=9,color=GREY)
    save(fig,directory,"heldout_risk")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=BASE/"figures")
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    for draw in (headline,graded,scaling,synthetic,heldout):
        draw(args.out)
