#!/usr/bin/env python3
"""Study *published* daily Heston calibrations and quantum-state continuity.

No new calibration is performed: the publicly released daily SPX estimates are
from 2013 (Mikkila/Kanniainen research code) and Jan-Apr 2020 (nlal46).
Download data on first use; use --input-2013/--input-2020 for offline snapshots.

Examples:
  python heston_calibrations.py
  python heston_calibrations.py --input-2013 local2013.csv --input-2020 local2020.csv

Fixed-spot, fixed-drift, fixed-bin Heston state comparisons isolate parameter
changes (not changing underlying spot or the option surface). No transport
circuit is fitted here; fidelity is the IDENTITY baseline, not transport fidelity.
An 8-qubit terminal distribution is only a marginal, not a multi-time path state.
Do not interpret medoid cover counts as a theorem about reference count.

Public data sources (author credits preserved):
  https://github.com/oskarimikkila/Empirical-Deep-Hedging/blob/master/data/heston_params.csv
  https://github.com/nlal46/Heston-Numerical-Calibration/blob/main/calibration_results_2020.csv
The 2013 repository labels its file as 2013 daily values; the associated
published paper discusses a different 2012 daily-calibration sample.
Do not conflate the two, or claim these records came from our own calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SOURCES = {
    "2013": "https://raw.githubusercontent.com/oskarimikkila/Empirical-Deep-Hedging/master/data/heston_params.csv",
    "2020": "https://raw.githubusercontent.com/nlal46/Heston-Numerical-Calibration/main/calibration_results_2020.csv",
}
PARAMETERS = ["v0", "kappa", "theta", "xi", "rho"]


def read_panel(source: str, path: Path | None, cache_dir: Path):
    if path is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"daily_heston_{source}.csv"
        if not path.exists():
            print(f"Downloading published {source} data: {SOURCES[source]}", flush=True)
            try:
                urllib.request.urlretrieve(SOURCES[source], path)
            except Exception as e:
                raise RuntimeError(f"Could not download {source}; pass --input-{source} with a local copy: {e}") from e
    data = pd.read_csv(path)
    if "eta" in data: data = data.rename(columns={"eta": "xi"})
    if "sigma" in data: data = data.rename(columns={"sigma": "xi"})
    required = ["date", *PARAMETERS]
    missing = [x for x in required if x not in data]
    if missing: raise ValueError(f"Missing {missing} in {path}")
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if "degenerate" not in data: data["degenerate"] = False
    data["degenerate"] = data["degenerate"].astype(str).str.lower().isin(("true", "1", "yes"))
    for name in PARAMETERS:
        data[name] = pd.to_numeric(data[name], errors="raise")
    if not np.all(np.isfinite(data[PARAMETERS].to_numpy())):
        raise ValueError(f"Nonfinite values in {path}")
    if (data[["v0", "kappa", "theta", "xi"]] <= 0).any().any() or (data.rho.abs() > 1).any():
        raise ValueError(f"Out-of-domain Heston parameters in {path}")
    data["source_index"] = np.arange(len(data))
    return data


class FixedHestonEncoder:
    """FFT inversion of log(S_T/S_0); same discretisation for every day.

    Inversion uses the Heston little-trap CF as in the project's heston.py.
    The broad FFT domain is normalised first; truncation at +/-cutoff is
    accounted for separately. Negative numerical artifacts are clipped and
    their total mass reported. Large violations reject the result.
    """
    def __init__(self, qubits=8, n_fft=16384, fft_halfwidth=3.5, cutoff=0.7,
                 maturity=1.0, drift=0.05):
        self.qubits = qubits
        self.n_fft = n_fft
        self.maturity = maturity
        self.drift = drift
        self.cutoff = cutoff
        self.n_bins = 1 << qubits
        self.dx = 2 * fft_halfwidth / n_fft
        self.origin = -fft_halfwidth
        self.xgrid = self.origin + self.dx * np.arange(n_fft+1)
        self.edges = np.linspace(-cutoff, cutoff, self.n_bins+1)
        self.freq = 2 * np.pi * np.fft.fftfreq(n_fft, d=self.dx)
        self.shift = np.exp(-1j*self.freq*self.origin)

    def state(self, row):
        v0, kappa, theta, xi, rho = (float(row[p]) for p in PARAMETERS)
        u = self.freq
        b = kappa - rho*xi*1j*u
        d = np.sqrt(b*b + xi**2*(u*u + 1j*u))
        g = (b-d)/(b+d)
        e = np.exp(-d*self.maturity)
        c = (kappa*theta/xi**2)*((b-d)*self.maturity - 2*np.log((1-g*e)/(1-g)))
        D = (b-d)/xi**2 * (1-e)/(1-g*e)
        cf = np.exp(1j*u*self.drift*self.maturity + c + v0*D)
        density = np.real(np.fft.fft(cf*self.shift)) / (self.n_fft*self.dx)
        negative_mass = float(np.sum(np.maximum(-density, 0)) * self.dx)
        masses = np.maximum(density, 0)*self.dx
        total = masses.sum()
        if not np.isfinite(total) or total <= 0 or negative_mass > 0.02:
            raise ArithmeticError(f"FFT unstable: total={total}, clipped negative mass={negative_mass}")
        masses /= total
        edge_mass = float(masses[:32].sum()+masses[-32:].sum())
        cdf = np.r_[0.0, np.cumsum(masses)]
        within = np.diff(np.interp(self.edges,self.xgrid,cdf))
        within = np.maximum(within, 0)
        keep = float(within.sum())
        if keep < 1e-6:
            raise ArithmeticError(f"Almost no mass retained ({keep:.6g})")
        psi = np.sqrt(within/keep)
        psi /= np.linalg.norm(psi)
        return psi,keep,negative_mass,edge_mass


def percentile(a):
    a = np.asarray(a, dtype=float)
    if len(a) == 0: return {"n":0}
    return {"n":int(len(a)), "median":float(np.median(a)), "mean":float(np.mean(a)),
            "p90":float(np.quantile(a,.9)), "p95":float(np.quantile(a,.95)),
            "max":float(np.max(a))}


def medoid_cover(psi, dates, minimum_fidelity):
    """Greedy observed-state reference cover (identity fidelity only)."""
    similarity = np.clip(psi @ psi.T, 0, 1)**2
    eligible = similarity >= minimum_fidelity
    remaining = np.ones(len(psi), dtype=bool)
    centres, assignments = [], np.full(len(psi),-1,dtype=int)
    while remaining.any():
        candidate = eligible[:,remaining].sum(axis=1)
        best = int(np.argmax(candidate))
        hit = remaining & eligible[best]
        assignments[hit] = len(centres)
        centres.append({"date":str(dates.iloc[best].date()),"index":best,
                        "newly_covered":int(hit.sum())})
        remaining[hit] = False
    return centres, assignments


def sequential_regions(psi, fidelity_min, clean_df):
    """Sequential anchor refresh needed for identity transport baseline only."""
    segments = []
    anchor = 0
    start=0
    for j in range(1,len(psi)):
        consecutive = clean_df.source_index.iloc[j] == clean_df.source_index.iloc[j-1]+1
        f = float(np.clip(np.dot(psi[j],psi[anchor]),0,1)**2)
        if f < fidelity_min or not consecutive:
            segments.append({"start":str(clean_df.date.iloc[start].date()),
                             "end":str(clean_df.date.iloc[j-1].date()),
                             "n_days":int(j-start),"start_v0":float(clean_df.v0.iloc[start]),
                             "end_v0":float(clean_df.v0.iloc[j-1])})
            start=anchor=j
    segments.append({"start":str(clean_df.date.iloc[start].date()),
                     "end":str(clean_df.date.iloc[-1].date()),
                     "n_days":int(len(psi)-start),"start_v0":float(clean_df.v0.iloc[start]),
                     "end_v0":float(clean_df.v0.iloc[-1])})
    return segments


def analyse(name, panel, args, out):
    clean = panel.loc[~panel.degenerate].reset_index(drop=True).copy()
    if clean.empty:raise ValueError(f"No clean dates: {name}")
    # Raw calibration estimates: measured day-to-day changes are not unit-invariant.
    consecutive = clean.source_index.diff() == 1
    delta = clean[PARAMETERS].diff().where(consecutive)
    parameter_summary = {k: {"value_min":float(clean[k].min()),"value_max":float(clean[k].max()),
                             "value_median":float(clean[k].median()),
                             "absolute_daily_change":percentile(delta[k].abs().dropna())}
                         for k in PARAMETERS}
    enc = FixedHestonEncoder(qubits=args.qubits, n_fft=args.n_fft,
                            cutoff=args.cutoff, fft_halfwidth=args.fft_halfwidth,
                            maturity=args.maturity, drift=args.drift)
    states=[]; mass=[]; negatives=[]; edge=[]
    for _,r in clean.iterrows():
        a,b,c,d = enc.state(r)
        states.append(a);mass.append(b);negatives.append(c);edge.append(d)
    states=np.stack(states)
    clean["retained_mass"]=mass
    clean["lost_mass"]=1-clean.retained_mass
    clean["fft_negative_mass_clipped"]=negatives
    clean["fft_edge_mass"]=edge
    # The same 256 discrete encoding bins are compared at every date.
    overlap = np.sum(states[1:]*states[:-1],axis=1)
    fidelity = np.clip(overlap,0,1)**2
    clean["adjacent_identity_fidelity"] = np.r_[np.nan,fidelity]
    clean.loc[~consecutive,"adjacent_identity_fidelity"]=np.nan
    clean["adjacent_identity_infidelity"] = 1-clean.adjacent_identity_fidelity
    clean["delta_v0"] = delta["v0"]
    clean["delta_kappa"] = delta["kappa"]
    clean["delta_theta"] = delta["theta"]
    clean["delta_xi"] = delta["xi"]
    clean["delta_rho"] = delta["rho"]
    clean["sqrt_v0"]=np.sqrt(clean.v0)
    clean["state_distance_from_first"] = np.sqrt(np.maximum(0,1-np.clip(states @ states[0],0,1)**2))
    # Principal components of actual quantum amplitude states (not raw parameters).
    shifted=states-states.mean(axis=0)
    _,singular,vt=np.linalg.svd(shifted,full_matrices=False)
    pcs=shifted @ vt[:2].T
    clean["state_pc1"] = pcs[:,0]
    clean["state_pc2"] = pcs[:,1]
    explained=(singular**2)/(np.sum(singular**2))
    # Covering observed daily states by *identity-only* references.
    centres,cluster=medoid_cover(states,clean.date,args.fidelity_min)
    clean["reference_cluster"]=cluster
    seq=sequential_regions(states,args.fidelity_min,clean)
    # Sensitivity of the region picture to the fidelity requirement.
    sensitivity={}
    for fmin in sorted(set([0.99,0.995,args.fidelity_min,0.9999])):
        cc,labels=medoid_cover(states,clean.date,fmin)
        segments=sequential_regions(states,fmin,clean)
        sensitivity[str(fmin)]={
            "greedy_identity_medoid_count":len(cc),
            "largest_medoid_cluster_days":int(np.bincount(labels).max()),
            "chronological_identity_reference_refreshes":len(segments)-1,
            "median_chronological_segment_days":float(np.median([z["n_days"] for z in segments])),
            "fraction_adjacent_identity_fidelity_ge_threshold":float(np.mean(fidelity[consecutive.iloc[1:].to_numpy()]>=fmin)),
        }
    # The 2013 panel contains obviously unstable fitted parameters near box
    # bounds; report them explicitly rather than silently treating as market
    # regime changes. Near-bound values come from the published CSV itself.
    fit_flags={"rho_positive_count":int((clean.rho>0).sum()),
               "v0_near_0p01_count":int((clean.v0<=.010001).sum()),
               "xi_near_1_count":int(((clean.xi>=.999)&(clean.xi<=1.00001)).sum())}
    spearman={}
    allinf=(1-clean.adjacent_identity_fidelity).to_numpy()
    for par in PARAMETERS:
        x=np.abs(delta[par].to_numpy())
        valid=np.isfinite(x)&np.isfinite(allinf)
        spearman[par]=float(spearmanr(x[valid],allinf[valid]).statistic) if valid.sum()>5 else None
    daily_ok=clean.adjacent_identity_fidelity.dropna()
    # Per-day *state* changes can be small despite large parameter changes.
    p_summary={
        "source": SOURCES[name],"first_date":str(panel.date.iloc[0].date()),
        "last_date":str(panel.date.iloc[-1].date()),"n_total":len(panel),"n_clean":len(clean),
        "n_flagged":int(panel.degenerate.sum()),"n_consecutive_clean_pairs":int(consecutive.sum()),
        "flagged_dates":[str(d.date()) for d in panel.loc[panel.degenerate,"date"]],
        "parameter_changes":parameter_summary,
        "fit_quality_warning_counts":fit_flags,
        "absolute_daily_parameter_change_vs_state_infidelity_spearman":spearman,
        "identity_threshold_sensitivity":sensitivity,
        "daily_identity_infidelity":percentile(1-daily_ok.to_numpy()),
        "fraction_adjacent_identity_fidelity_ge_threshold":float(np.mean(daily_ok>=args.fidelity_min)),
        "lost_mass":percentile(clean.lost_mass.to_numpy()),
        "fraction_dates_lost_mass_below_threshold":float(np.mean(clean.lost_mass<=args.max_lost_mass)),
        "maximum_fft_negative_mass":float(clean.fft_negative_mass_clipped.max()),
        "maximum_fft_edge_mass":float(clean.fft_edge_mass.max()),
        "pca_explained_first_two":list(map(float,explained[:2])),
        "identity_medoid_cover":{
           "threshold":args.fidelity_min,"centres_needed_greedy":len(centres),
           "representative_centres":centres[:min(len(centres),20)]},
        "identity_sequential_regions":{
          "n_segments":len(seq),"median_segment_days":float(np.median([z["n_days"] for z in seq])),
          "longest_segment_days":int(max(z["n_days"] for z in seq)),
          "regions":seq},
        "notes":["Published parameter estimates; no option-chain repricing or new calibration performed.",
                 "All state fidelities compare identical fixed spot/drift/maturity/bin definitions.",
                 "Identity overlaps are not transported circuit fidelities.",
                 "Greedy medoid cover counts are upper bounds on needed centres for an identity-only observed-day cover, not optimal counts or a transport limitation.",
                 "Lost mass is a separate fixed-window encoding error."]}
    if name=="2020":
        p_summary["pre_vs_stress_regime"]={}
        for label,sub in [("before_2020_02_24",clean[clean.date<pd.Timestamp("2020-02-24")]),
                          ("from_2020_02_24",clean[clean.date>=pd.Timestamp("2020-02-24")])]:
            ff=sub.adjacent_identity_fidelity.dropna()
            p_summary["pre_vs_stress_regime"][label]={
                "days":len(sub),"clean_consecutive_pairs":len(ff),
                "median_daily_identity_infidelity":float(np.median(1-ff)),
                "fraction_pairs_identity_fidelity_ge_threshold":float(np.mean(ff>=args.fidelity_min)),
                "median_lost_mass":float(sub.lost_mass.median()),
                "fraction_days_lost_mass_below_threshold":float(np.mean(sub.lost_mass<=args.max_lost_mass))}
    clean.to_csv(out/f"{name}_daily_quantum_continuity.csv",index=False)
    (out/f"{name}_summary.json").write_text(json.dumps(p_summary,indent=2)+"\n")
    print(f"{name}: {len(clean)} clean days, median |Δv0|={parameter_summary['v0']['absolute_daily_change']['median']:.5g}, "
          f"median daily identity infidelity={p_summary['daily_identity_infidelity']['median']:.5g}, "
          f"fraction next-day identity F≥{args.fidelity_min}: {p_summary['fraction_adjacent_identity_fidelity_ge_threshold']:.1%}, "
          f"identity medoids={len(centres)}, "
          f"median chronological region={p_summary['identity_sequential_regions']['median_segment_days']:.0f} days",flush=True)
    return clean,p_summary


def plots(datasets, summaries, out, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for row,name in enumerate(["2013","2020"]):
        d=datasets[name]
        ax=axes[row,0]
        ax.plot(d.date,d.sqrt_v0,label="sqrt(v0)",lw=1.5)
        ax.plot(d.date,np.sqrt(d.theta),label="sqrt(theta)",lw=1.15)
        ax.set_ylabel("Annualised volatility proxy")
        ax.set_title(f"{name}: calibrated short-run and long-run volatility")
        ax.grid(alpha=.23);ax.legend(fontsize=8)
        ax=axes[row,1]
        ax.plot(d.date,np.maximum(1-d.adjacent_identity_fidelity,1e-8),label="One-day identity infidelity",lw=1)
        ax.axhline(1-args.fidelity_min,color="tab:red",linestyle="--",lw=1,label="Identity threshold")
        ax.set_yscale("log")
        ax.set_ylabel("Squared state infidelity")
        ax.set_title(f"{name}: daily changes of encoded terminal states")
        ax.grid(alpha=.23);ax.legend(fontsize=8)
        for a in axes[row]:a.tick_params(axis='x',rotation=20)
    fig.suptitle("Public daily Heston calibration panels: parameters and quantum-state continuity")
    fig.savefig(out/"daily_parameters_and_state_continuity.png",dpi=170)
    plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for ax,name in zip(axes,["2013","2020"]):
        d=datasets[name]
        sc=ax.scatter(d.state_pc1,d.state_pc2,c=np.arange(len(d)),s=22,cmap="viridis")
        ax.set_xlabel("State amplitude principal component 1")
        ax.set_ylabel("State amplitude principal component 2")
        ax.set_title(f"{name}: distribution states, coloured by trading-day order")
        fig.colorbar(sc,ax=ax,label="Trading day index")
    fig.savefig(out/"quantum_state_geometry.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for ax,name in zip(axes,["2013","2020"]):
        d=datasets[name]
        ax.plot(d.date,d.lost_mass,label="Lost mass outside window")
        ax.axhline(args.max_lost_mass,color='tab:red',ls='--',label="Lost-mass threshold")
        ax.set_title(f"{name}: encoding window failure is separate from fidelity")
        ax.set_ylabel("Lost probability mass")
        ax.tick_params(axis="x",rotation=20);ax.grid(alpha=.23);ax.legend(fontsize=8)
    fig.savefig(out/"encoding_window_mass_loss.png",dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for row,name in enumerate(["2013","2020"]):
        d=datasets[name]
        axes[row,0].plot(d.date,d.kappa,label="kappa")
        axes[row,0].plot(d.date,d.xi,label="xi")
        axes[row,0].set_title(f"{name}: daily mean reversion and vol-of-vol")
        axes[row,0].legend(fontsize=8);axes[row,0].grid(alpha=.2)
        axes[row,1].plot(d.date,d.rho,label="rho")
        axes[row,1].set_ylim(-1.05,1.05)
        axes[row,1].set_title(f"{name}: calibrated price/variance correlation")
        axes[row,1].grid(alpha=.2)
        for ax in axes[row]:ax.tick_params(axis="x",rotation=20)
    fig.savefig(out/"structural_parameter_variability.png",dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.6),constrained_layout=True)
    for name in ("2013","2020"):
        sens=summaries[name]["identity_threshold_sensitivity"]
        levels=sorted((float(x),val) for x,val in sens.items())
        ax.plot([x for x,_ in levels],
                [v["greedy_identity_medoid_count"] for _,v in levels],
                marker="o",label=name)
    ax.set_title("Greedy number of reference states required for identity-only fidelity coverage")
    ax.set_xlabel("Minimum identity-state fidelity")
    ax.set_ylabel("Reference states covering observed dates")
    ax.grid(alpha=.23);ax.legend()
    fig.savefig(out/"identity_reference_cover_sensitivity.png",dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-2013",type=Path)
    parser.add_argument("--input-2020",type=Path)
    parser.add_argument("--output",type=Path,default=Path("results/heston_calibrations"))
    parser.add_argument("--qubits",type=int,default=8)
    parser.add_argument("--n-fft",type=int,default=16384)
    parser.add_argument("--fft-halfwidth",type=float,default=3.5)
    parser.add_argument("--cutoff",type=float,default=0.7)
    parser.add_argument("--maturity",type=float,default=1.0)
    parser.add_argument("--drift",type=float,default=0.05)
    parser.add_argument("--fidelity-min",type=float,default=0.999)
    parser.add_argument("--max-lost-mass",type=float,default=0.01)
    args=parser.parse_args()
    if not(0<args.fidelity_min<1 and 0<args.max_lost_mass<1):
        parser.error("Fidelity and mass thresholds must lie between zero and one")
    if not(2<=args.qubits<=12 and args.n_fft >=2**args.qubits and args.cutoff < args.fft_halfwidth):
        parser.error("Invalid FFT/bin configuration")
    args.output.mkdir(parents=True,exist_ok=True)
    data={};summaries={}
    for source,path in [("2013",args.input_2013),("2020",args.input_2020)]:
        panel=read_panel(source,path,args.output/"cache")
        data[source],summaries[source]=analyse(source,panel,args,args.output)
    plots(data,summaries,args.output,args)
    (args.output/"analysis_settings.json").write_text(json.dumps({
        "qubits":args.qubits,"n_fft":args.n_fft,"fft_halfwidth":args.fft_halfwidth,
        "cutoff":args.cutoff,"maturity":args.maturity,"drift":args.drift,
        "identity_fidelity_min":args.fidelity_min,"max_lost_mass":args.max_lost_mass,
        "data_sources":SOURCES,
        "definition":"Normalised square-root binned probabilities of log(S_T/S0) on fixed log window"},indent=2)+"\n")
    print(f"Saved plots, daily diagnostics and summaries in {args.output.resolve()}")

if __name__=="__main__":
    main()