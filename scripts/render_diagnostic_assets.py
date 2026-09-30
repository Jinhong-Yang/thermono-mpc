"""Render descriptive post-freeze figures without changing frozen evidence."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
    'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42})
COLORS={'ROM':'#606b73','data_only':'#23609c','physics_informed':'#b76323'}
LABELS={'ROM':'ROM','data_only':'FNO','physics_informed':'PINO'}

def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))

def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#dde1e5',linewidth=.6);ax.set_axisbelow(True)

def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',pad_inches=.04,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.04);plt.close(fig)

def main(root,output):
    output.mkdir(parents=True,exist_ok=True);provenance={};inputs={}
    def rows(name):
        p=root/name;inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest();return read(p)
    d1=rows('D1/episode_summary.csv');assert len(d1)==21
    groups=[('ROM','')]+[(v,str(s)) for v in ['data_only','physics_informed'] for s in [101,202,303]]
    fig,ax=plt.subplots(figsize=(5.0,3.35));fig.subplots_adjust(left=.15,right=.98,bottom=.18,top=.83)
    plotted=[]
    for offset,scenario,color,marker in [(-.16,'N03','#23609c','o'),(0,'S01','#b76323','s'),(.16,'F01','#606b73','^')]:
        x=[];y=[]
        for j,(v,seed) in enumerate(groups):
            r=next(r for r in d1 if r['variant']==v and r['scenario_id']==scenario and (v=='ROM' or r['model_seed']==seed))
            x.append(j+offset);y.append(float(r['end_mean_bias_K']));plotted.append(r)
        ax.scatter(x,y,color=color,marker=marker,s=35,label=scenario,zorder=3)
    ax.axhline(0,color='#263238',linewidth=.9)
    ax.set_xticks(range(7),['ROM']+[LABELS[v]+'\n'+seed for v,seed in groups[1:]])
    ax.set_ylabel('Selected-plan endpoint bias (K)');ax.set_xlim(-.5,6.5)
    ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02));style(ax)
    save(fig,output,'diagnostic_selected_bias');provenance['selected_bias']=plotted

    trace=rows('D1/N03_B3_PINO_MPC_seed101/selected_plans.csv')
    t=[float(r['selection_time_s']) for r in trace]
    fig,axes=plt.subplots(3,1,figsize=(5.0,6.4),sharex=True)
    fig.subplots_adjust(left=.17,right=.98,bottom=.09,top=.86,hspace=.48)
    for key,label,color,linestyle in [('reference_end_K','Horizon-end reference','#263238',':'),
        ('predicted_end_mean_K','Selected-plan forecast','#b76323','-'),
        ('realized_end_mean_K','Same-plan cloned plant','#23609c','--')]:
        axes[0].plot(t,[float(r[key]) for r in trace],label=label,color=color,linestyle=linestyle)
    axes[0].set_ylabel('Endpoint mean (K)');axes[0].legend(frameon=False,fontsize=9,loc='lower left',bbox_to_anchor=(0,1.03),borderaxespad=0)
    for key,label,color in [('end_mean_bias_K','Estimated start','#b76323'),('oracle_end_mean_bias_K','True aligned start','#23609c')]:
        axes[1].plot(t,[float(r[key]) for r in trace],label=label,color=color,
            linestyle='--' if key=='oracle_end_mean_bias_K' else '-')
    axes[1].set_ylabel('Endpoint bias (K)');axes[1].set_ylim(0,4)
    axes[1].legend(frameon=False,fontsize=9,loc='lower right')
    axes[2].plot(t,[float(r['selected_first_power_total_W']) for r in trace],color='#606b73')
    axes[2].set(xlabel='Plan selection time (s)',ylabel='Selected power (W)')
    for ax in axes:style(ax)
    save(fig,output,'diagnostic_selected_trace');provenance['selected_trace']={'run':'N03_B3_PINO_MPC_seed101','rows':trace,
        'alignment':'All endpoint quantities shown against selection time; endpoint is 130 s after selection. The power panel is the sum of the first action in the selected plan.'}

    hist=rows('D2/energy_histograms.csv')
    family={r['run_id']:r['variant'] for r in d1}
    fig,ax=plt.subplots(figsize=(5.0,3.1));fig.subplots_adjust(left=.14,right=.98,bottom=.18,top=.84)
    plotted=[]
    for population,label,color in [('training','Training','#263238'),('ROM','ROM selected',COLORS['ROM']),('data_only','FNO selected',COLORS['data_only']),('physics_informed','PINO selected',COLORS['physics_informed'])]:
        subset=[r for r in hist if (r['population']=='training' if population=='training' else r['population']=='selected' and family[r['run_id']]==population)]
        bins=sorted({float(r['left_J']) for r in subset});counts=np.array([sum(int(r['count']) for r in subset if float(r['left_J'])==b) for b in bins]);fraction=counts/counts.sum()
        right=max(float(r['right_J']) for r in subset)
        edges=np.array(bins+[right])/1e6
        ax.stairs(fraction,edges,baseline=None,label=label,color=color,linewidth=1.5)
        plotted.append({'population':population,'bin_edges_MJ':edges.tolist(),'fraction':fraction.tolist(),'occurrences':int(counts.sum())})
    ax.set(xlabel='12-step plan energy (MJ)',ylabel='Fraction of plans',xlim=(0,.72),ylim=(0,1));style(ax)
    ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.01));save(fig,output,'diagnostic_plan_energy');provenance['plan_energy']=plotted

    d4=rows('D4/summary.csv');assert len(d4)==4
    fig,axes=plt.subplots(2,1,figsize=(5.0,4.9),sharex=True);fig.subplots_adjust(left=.15,right=.98,bottom=.13,top=.97,hspace=.2)
    x=np.arange(4)
    for ax,key,label in [(axes[0],'mean_control_rmse_K','Control RMSE (K)'),(axes[1],'physics_residual_normalized','Normalized residual')]:
        ax.plot(x,[float(r[key]) for r in d4],'-o',color=COLORS['physics_informed']);ax.set_ylabel(label);style(ax)
    axes[1].set_xticks(x,[f"{float(r['physics_weight']):g}" for r in d4]);axes[1].set_xlabel('Physics residual weight (new seed-101 fits)')
    save(fig,output,'diagnostic_residual_weight');provenance['residual_weight']=d4
    provenance['input_sha256']=inputs
    provenance['scope']='Post-freeze descriptive diagnostics. Episode/plan occurrences are correlated; no cycle-level significance or population uncertainty is inferred.'
    (output/'diagnostic_figure_data.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Saved four diagnostic figures and their plotted values/input hashes.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnostic-dir',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();main(a.diagnostic_dir,a.output)
