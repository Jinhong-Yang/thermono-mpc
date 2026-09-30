"""Rebuild publication-size figures from the frozen aggregate CSVs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

INK, BLUE, ORANGE, GREY, LIGHT = "#263238", "#23609c", "#b76323", "#606b73", "#d7dde1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})

def read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

def save(fig, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.04,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)

def box(ax, xy, wh, label, color=INK):
    x, y = xy
    w, h = wh
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, lw=1.2))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=10.3, linespacing=1.22)

def arrow(ax, start, end, color=GREY, style="-"):
    ax.annotate("", xy=end, xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1,
                    "linestyle": style, "shrinkA": 0, "shrinkB": 0})

def architecture(output):
    # Match the approximately five-inch elsarticle text column so labels
    # retain their intended point size when included at linewidth.
    fig, ax = plt.subplots(figsize=(5.0, 3.15))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 6.5), ylim=(0, 3.15))
    ax.axis("off")
    # This panel depicts B4, the only scored comparator using the full
    # asynchronous RuntimeSupervisor. Other comparator paths are synchronous.
    for x, label in ((.12, "Thermal\nplant"), (1.75, "Probes +\nobserver"),
                     (3.38, "PINO MPC\ncontroller"), (5.01, "Runtime\nvalidator")):
        box(ax, (x, 1.83), (1.32, .69), label)
    box(ax, (.12, .22), (1.32, .63), "Full-state\nscoring", GREY)
    box(ax, (3.38, .22), (1.32, .63), "PINO\npredictor", BLUE)
    box(ax, (5.01, .22), (1.32, .63), "PID\nfallback", ORANGE)
    for x in (1.44, 3.07, 4.70):
        arrow(ax, (x, 2.175), (x+.31, 2.175))
    # This feedback carries applied power, not a measured state.
    ax.plot([6.33, 6.43, 6.43, .78, .78], [2.175, 2.175, 2.91, 2.91, 2.68],
            color=GREY, lw=1.1)
    arrow(ax, (.78, 2.68), (.78, 2.52))
    ax.text(3.25, 3.00, "B4: power applied at the next control boundary", ha="center", fontsize=9.5)
    arrow(ax, (.78, 1.83), (.78, .85), style="--")
    ax.text(.90, 1.32, "truth", fontsize=9.2, color=GREY)
    ax.text(.78, .06, "Evaluation only", ha="center", fontsize=9.2, color=GREY)
    arrow(ax, (3.70, 1.83), (3.70, .85), BLUE)
    arrow(ax, (4.40, .85), (4.40, 1.83), BLUE)
    ax.text(3.56, 1.26, "state +\nplans", ha="right", fontsize=9.2, color=BLUE)
    ax.text(4.53, 1.32, "forecasts", ha="left", fontsize=9.2, color=BLUE)
    arrow(ax, (5.67, .85), (5.67, 1.83), ORANGE)
    # Shared estimate reaches fallback independently of the neural predictor.
    ax.plot([2.41, 2.41, 5.67, 5.67], [1.83, .08, .08, .16], color=GREY, lw=1.1)
    arrow(ax, (5.67, .16), (5.67, .22), GREY)
    ax.text(2.22, 1.34, "state", ha="right", fontsize=9.2, color=GREY)
    save(fig, output/"architecture")

def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=.65)
    ax.set_axisbelow(True)

def control_differences(analysis, output):
    rows = read(analysis/"scenario_metrics.csv")
    values = {(r["scenario_id"], r["controller"]): float(r["seed_mean"])
              for r in rows if r["metric"] == "rmse_K"}
    scenarios = [f"N{i:02d}" for i in range(1,7)] + [f"S{i:02d}" for i in range(1,5)] + ["F01","F02"]
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(5.0,3.35))
    fig.subplots_adjust(left=.105,right=.985,bottom=.13,top=.83)
    plotted=[]
    for label, comparator, color, marker, offset in (
        ("PINO - data-only FNO","B2_DATA_NO_MPC",BLUE,"o",-.10),
        ("PINO - physical ROM","B1_ROM_CEM",GREY,"s",.10)):
        diff=[values[(s,"B3_PINO_MPC")]-values[(s,comparator)] for s in scenarios]
        ax.scatter(x+offset,diff,marker=marker,s=31,color=color,label=label,zorder=3)
        plotted.extend({"scenario":s,"comparator":comparator,"difference_K":v}
                       for s,v in zip(scenarios,diff))
    ax.axhline(0,color=INK,lw=1)
    for boundary in (5.5,9.5):
        ax.axvline(boundary,color=LIGHT,lw=1)
    ax.set_xticks(x,scenarios)
    ax.set_yticks([0,5,10,15,20])
    ax.set(ylabel="PINO - comparator RMSE (K)",ylim=(-.8,21.6),xlim=(-.6,11.6))
    for center,label in ((2.5,"Nominal"),(7.5,"Material shift"),(10.5,"Sensor\nstress")):
        ax.text(center,20.6,label,ha="center",fontsize=9.2,color=GREY)
    ax.legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,1.05),
              ncol=2,handletextpad=.45,columnspacing=1.1)
    clean_axes(ax)
    save(fig,output/"control_paired_rmse")
    return plotted

def prediction(analysis,output):
    group={}
    for r in read(analysis/"prediction_trajectory_metrics.csv"):
        group.setdefault((int(r["trajectory_seed"]),r["variant"]),[]).append(float(r["field_rmse_K"]))
    seeds=sorted({s for s,_ in group})
    assert len(seeds)==8
    names=["ROM","data_only","physics_informed"]
    colors=[GREY,BLUE,ORANGE]
    aggregate={r["variant"]:r for r in read(analysis/"prediction_aggregate.csv")
               if r["metric"]=="field_rmse_K"}
    fig,ax=plt.subplots(figsize=(5.0,3.30))
    fig.subplots_adjust(left=.105,right=.985,bottom=.14,top=.82)
    plotted=[]
    # Horizontal offsets separate nearly coincident marks without altering error values.
    for offset,seed in zip(np.linspace(-.065,.065,len(seeds)),seeds):
        ys=[float(np.mean(group[(seed,n)])) for n in names]
        xs=np.arange(3)-.09+offset
        ax.plot(xs,ys,color=LIGHT,lw=.8,zorder=1)
        for j,value in enumerate(ys):
            ax.scatter(xs[j],value,s=22,facecolor="white",edgecolor=colors[j],linewidth=1.1,zorder=2)
            plotted.append({"trajectory_seed":seed,"variant":names[j],"field_rmse_K":value})
    estimates=[]
    for j,name in enumerate(names):
        mean,low,high=[float(aggregate[name][k]) for k in ("mean","ci95_low","ci95_high")]
        assert np.isclose(mean,np.mean([np.mean(group[(s,name)]) for s in seeds]))
        ax.errorbar(j+.14,mean,yerr=[[mean-low],[high-mean]],fmt="D",markersize=5.5,
                    color=colors[j],elinewidth=1.6,capsize=3.5,zorder=4)
        estimates.append({"variant":name,"mean":mean,"ci95_low":low,"ci95_high":high})
    ax.set_xticks(range(3),["Physical ROM","Data-only FNO","PINO"])
    ax.set(ylabel="12-step field forecast RMSE (K)",ylim=(0,1.65),xlim=(-.4,2.4))
    ax.set_yticks([0,.4,.8,1.2,1.6])
    legend=[Line2D([],[],marker="o",markersize=5,markerfacecolor="white",color=GREY,
                   linestyle="none",label="Trajectory mean"),
            Line2D([],[],marker="D",markersize=5,color=GREY,linestyle="none",
                   label="Mean and 95% interval")]
    ax.legend(handles=legend,frameon=False,loc="lower center",bbox_to_anchor=(.5,1.02),
              ncol=2,handletextpad=.4)
    clean_axes(ax)
    save(fig,output/"prediction_field_rmse")
    return {"trajectories":plotted,"estimates":estimates,
            "interval":"95% percentile bootstrap over eight whole trajectories; 10000 resamples"}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    architecture(args.output)
    controls=control_differences(args.analysis_dir,args.output)
    forecasts=prediction(args.analysis_dir,args.output)
    provenance={"control":controls,"prediction":forecasts,
        "input_sha256":{name:hashlib.sha256((args.analysis_dir/name).read_bytes()).hexdigest()
            for name in ("scenario_metrics.csv","prediction_trajectory_metrics.csv","prediction_aggregate.csv")},
        "renderer_sha256_lf":hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest()}
    (args.output/"figure_data.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(f"Saved three PDF/PNG figures and plotted-data provenance under {args.output}")
