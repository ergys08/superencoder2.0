#!/usr/bin/env python3
"""Interactive Heston v0 continuity explorer.

Run from anywhere:
    python "heston_coninuity/v0_continuity.py"

Open http://127.0.0.1:8765/ if the browser does not open automatically.
Requires NumPy and the repository's heston.py. No Plotly or web APIs required.

The smooth terminal-price PDF is obtained by numerical Fourier inversion of
Heston's analytic characteristic function, NOT a Monte Carlo histogram.
Probabilities are integrated over fixed log-price bins, with both tails shown
separately rather than silently renormalised. The optional conditional
amplitude fidelity renormalises *only for its stated comparison*.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np

# Permit execution directly from this subfolder without installing the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heston import HestonModel, HestonParameters

REFERENCE_V0 = 0.04
V0_MIN, V0_MAX = 0.005, 0.30
S0 = 100.0
N_FFT = 16384
FOURIER_HALF_WIDTH = 2.5
LOG_S0 = float(np.log(S0))


@lru_cache(maxsize=192)
def full_distribution(v0: float):
    """PDF and CDF on a broad, fixed, uniform log-price Fourier grid."""
    model = HestonModel(HestonParameters(
        s0=S0, mu=0.05, v0=v0, kappa=2.0, theta=0.04,
        xi=0.30, rho=-0.70, maturity=1.0
    ))
    dx = 2.0 * FOURIER_HALF_WIDTH / N_FFT
    origin = LOG_S0 - FOURIER_HALF_WIDTH
    u = 2 * np.pi * np.fft.fftfreq(N_FFT, d=dx)
    transform = model.characteristic_function(u) * np.exp(-1j * u * origin)
    raw = np.real(np.fft.fft(transform)) / (N_FFT * dx)

    # The numerical inversion is periodised on a deliberately wide window.
    # Tiny negative Gibbs/roundoff oscillations cannot be probability masses.
    neg_mass = float(np.maximum(-raw, 0.0).sum() * dx)
    if neg_mass > 1e-4:
        raise ArithmeticError(
            "Fourier inversion is inaccurate (negative mass too large). "
            "Increase N_FFT / FOURIER_HALF_WIDTH."
        )
    density = np.maximum(raw, 0.0)
    area = float(density.sum() * dx)
    if not np.isfinite(area) or area <= 0:
        raise ArithmeticError("Invalid Fourier-inverted density")

    density /= area
    # Piecewise-constant integration on fine-grid cells: no random sampling.
    cdf = np.concatenate(([0.0], np.cumsum(density) * dx))
    cdf[-1] = 1.0
    log_edges = origin + np.arange(N_FFT + 1) * dx
    edge_mass = float((density[:N_FFT // 100].sum()
                       + density[-N_FFT // 100:].sum()) * dx)
    if edge_mass > 2e-3:
        raise ArithmeticError(
            "Fourier window is too narrow; edge mass is {:.3g}".format(edge_mass)
        )
    return log_edges, density, cdf, edge_mass


def binned_data(v0: float, cutoff: float, n_qubits: int):
    x, density, cdf, edge_mass = full_distribution(v0)
    edges = np.linspace(LOG_S0 - cutoff, LOG_S0 + cutoff, 2**n_qubits + 1)
    cumulative = np.interp(edges, x, cdf)
    raw_probs = np.maximum(np.diff(cumulative), 0.0)

    left = float(cumulative[0])
    right = float(1.0 - cumulative[-1])
    inside = float(raw_probs.sum())
    if inside <= 0:
        raise ArithmeticError("No probability remains inside the cutoff")

    # Return price-density values p_S(s) = p_logS(log(s)) / s.
    view_half = min(FOURIER_HALF_WIDTH - 0.05, max(0.70, 1.65 * cutoff))
    log_plot = np.linspace(LOG_S0 - view_half, LOG_S0 + view_half, 401)
    price_plot = np.exp(log_plot)
    price_density = np.interp(log_plot, x[:-1], density) / price_plot

    # The reference is evaluated with the SAME bin edges / cutoff.
    x_ref, d_ref, cdf_ref, _ = full_distribution(REFERENCE_V0)
    ref_cumulative = np.interp(edges, x_ref, cdf_ref)
    ref_probs = np.maximum(np.diff(ref_cumulative), 0.0)
    ref_inside = float(ref_probs.sum())
    ref_pdf = np.interp(log_plot, x_ref[:-1], d_ref) / price_plot

    # Fidelity between the *conditional*, normalised in-register states.
    overlap = float(np.sqrt(
        (raw_probs / inside) * (ref_probs / ref_inside)
    ).sum())
    fidelity = float(np.clip(overlap * overlap, 0.0, 1.0))
    return {
        "v0": v0,
        "cutoff": cutoff,
        "n_qubits": n_qubits,
        "bins": int(2**n_qubits),
        "cutoff_prices": [float(np.exp(edges[0])), float(np.exp(edges[-1]))],
        "bin_edges": np.exp(edges).tolist(),
        "bin_probability": raw_probs.tolist(),
        "reference_bin_probability": ref_probs.tolist(),
        "kept_mass": inside,
        "left_lost": left,
        "right_lost": right,
        "lost_mass": left + right,
        "reference_kept_mass": ref_inside,
        "conditional_amplitude_fidelity": fidelity,
        "pdf_prices": price_plot.tolist(),
        "pdf": price_density.tolist(),
        "reference_pdf": ref_pdf.tolist(),
        "fft_window_edge_mass": edge_mass
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heston | Initial-Variance Continuity</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#182536;background:#f5f7fa}
*{box-sizing:border-box}body{margin:0}.wrap{max-width:1130px;margin:28px auto;padding:0 20px}
h1{font-size:27px;letter-spacing:-.7px;margin:0 0 5px}p{line-height:1.5}
.sub{color:#516275;margin:0 0 20px}.card{background:white;border:1px solid #dfe6ec;
 border-radius:12px;padding:17px 20px;margin-bottom:16px;box-shadow:0 2px 8px #1a2c4408}
.controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}
label{display:block;font-weight:650;margin-bottom:10px;font-size:14px}
label strong{float:right;color:#176e91;font-variant-numeric:tabular-nums}
input[type=range]{width:100%;accent-color:#1687a8}select{width:100%;padding:7px;border:1px solid #cbd6dd;border-radius:7px}
.small{font-size:12px;color:#647386}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.stat{padding:14px;background:#f4f7fa;border-radius:9px}.stat b{display:block;font-size:23px;font-variant-numeric:tabular-nums}
.stat span{font-size:12px;color:#516275}.head{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;align-items:center}
h2{font-size:17px;margin:0 0 8px}.legend{font-size:12px;color:#536273}
.dot{display:inline-block;width:12px;height:4px;vertical-align:middle;margin:0 5px 2px 12px}
.canvas{width:100%;height:270px;display:block}#binsCanvas{height:280px}
.note{font-size:13px;color:#536273}.warn{color:#a43f30}#error{color:#aa3c2a}
@media(max-width:750px){.controls{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main class="wrap">
<h1>Heston initial-variance continuity</h1>
<p class="sub">Vary <b>v₀</b>. The smooth terminal-price PDF comes from Fourier inversion of the analytic Heston characteristic function; bars contain its integrated probability mass. The <b>price cutoff is deliberately lossy</b>.</p>
<section class="card controls">
 <div><label for="v0">Initial variance v₀ <strong id="v0Text"></strong></label>
 <input id="v0" type="range" min="0.005" max="0.300" step="0.001" value="0.040">
 <div class="small">Other parameters held fixed: κ=2, θ=0.04, ξ=0.3, ρ=−0.7; S₀=100, μ=0.05, T=1</div></div>
 <div><label for="cutoff">Log-price cutoff ±h <strong id="cutoffText"></strong></label>
 <input id="cutoff" type="range" min="0.08" max="1.00" step="0.01" value="0.30">
 <div class="small">Accept prices only between S₀e<sup>−h</sup> and S₀e<sup>h</sup></div></div>
 <div><label for="qubits">Register resolution <strong id="qubitsText"></strong></label>
 <select id="qubits"><option value="4">4 qubits · 16 bins</option><option value="5">5 qubits · 32 bins</option>
 <option value="6" selected>6 qubits · 64 bins</option><option value="7">7 qubits · 128 bins</option>
 <option value="8">8 qubits · 256 bins</option><option value="9">9 qubits · 512 bins</option></select>
 <div class="small">Bins remain fixed while v₀ changes. Uniform spacing in log-price.</div></div>
</section>
<section class="card"><div class="stats">
 <div class="stat"><span>Mass retained in register</span><b id="retained">—</b></div>
 <div class="stat"><span>Lost below left cutoff</span><b id="left">—</b></div>
 <div class="stat"><span>Lost above right cutoff</span><b id="right">—</b></div>
 <div class="stat"><span>Conditional fidelity to v₀=0.04</span><b id="fidelity">—</b></div>
 </div><p class="note" id="bounds"></p></section>
<section class="card">
 <div class="head"><h2>Analytical-model terminal-price PDF</h2><div class="legend"><span class="dot" style="background:#177da6"></span>Current v₀<span class="dot" style="background:#d09251"></span>Reference v₀=0.04</div></div>
 <canvas class="canvas" id="pdfCanvas"></canvas><p class="note">Fourier-inverted PDF in price units, plotted against a logarithmic price axis. Shaded regions are outside the cutoff; the curve is not a Monte Carlo histogram.</p>
</section>
<section class="card">
 <div class="head"><h2>Probability mass in each finite-register bin</h2><div class="legend"><span class="dot" style="background:#177da6"></span>Current unnormalised bin mass<span class="dot" style="background:#d09251"></span>Reference</div></div>
 <canvas class="canvas" id="binsCanvas"></canvas>
 <p class="note">Each bar is <b>P(Sₜ in this bin)</b>, not a normalised conditional probability. Bar heights add to the retained mass, not 100%. No tail probability is silently added to edge bins. If you condition on being inside the cutoff to form a unit-norm qubit state, you change the represented distribution.</p>
</section><p class="small">Numerical note: Fourier inversion uses a fixed, broad log-price window (16,384 points) and small negative numerical oscillations are clipped. The plotted “analytical” PDF is a numerical inverse of the exact Heston characteristic function. Results should be checked for numerical convergence before research use.</p>
<div id="error"></div>
</main><script>
const $ = id=>document.getElementById(id);
const v0=$("v0"), cutoff=$("cutoff"), qubits=$("qubits");
let timer=null, seq=0;
const fmt=(x,n=2)=>Number(x).toFixed(n);
function paintLabels(){
 $("v0Text").textContent=fmt(v0.value,3);
 $("cutoffText").textContent=fmt(cutoff.value,2);
 $("qubitsText").textContent=qubits.value+" qubits";
}
function request(){
 paintLabels(); const id=++seq;
 const url="/api?v0="+v0.value+"&cutoff="+cutoff.value+"&qubits="+qubits.value;
 fetch(url).then(async r=>{const j=await r.json();if(!r.ok)throw Error(j.error||"Request failed");return j})
 .then(d=>{if(id!==seq)return; $("error").textContent=""; render(d)})
 .catch(e=>{if(id===seq)$("error").textContent=e.message});
}
function schedule(){paintLabels();clearTimeout(timer);timer=setTimeout(request,85)}
[v0,cutoff,qubits].forEach(el=>el.addEventListener("input",schedule));
function setup(canvas){
 const rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
 canvas.width=Math.max(100,Math.floor(rect.width*ratio));
 canvas.height=Math.floor(rect.height*ratio);
 const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);
 return {ctx,w:rect.width,h:rect.height,left:59,right:17,top:17,bottom:39};
}
function axes(g,ymax,ylab){
 const {ctx,w,h,left,right,top,bottom}=g,plotH=h-top-bottom;
 ctx.font="11px system-ui";ctx.strokeStyle="#dde5eb";ctx.fillStyle="#536577";
 ctx.textAlign="right";
 for(let i=0;i<5;i++){
   const y=top+plotH*i/4;
   ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();
   ctx.fillText((ymax*(1-i/4)).toPrecision(3),left-7,y+4);
 }
 ctx.fillStyle="#334659";ctx.textAlign="left";ctx.fillText(ylab,left,11);
}
function drawPDF(d){
 const g=setup($("pdfCanvas")), {ctx,w,h,left,right,top,bottom}=g;
 const xs=d.pdf_prices,ys=d.pdf,ref=d.reference_pdf;
 const xl=Math.log(xs[0]),xr=Math.log(xs[xs.length-1]),wplot=w-left-right,hplot=h-top-bottom;
 const X=s=>left+(Math.log(s)-xl)/(xr-xl)*wplot;
 const ymax=Math.max(...ys,...ref)*1.09||1;const Y=v=>top+hplot*(1-v/ymax);
 axes(g,ymax,"PDF / £");
 const a=X(d.cutoff_prices[0]),b=X(d.cutoff_prices[1]);
 ctx.fillStyle="#c86a641c";ctx.fillRect(left,top,Math.max(0,a-left),hplot);
 ctx.fillRect(b,top,Math.max(0,w-right-b),hplot);
 ctx.strokeStyle="#be6c62";ctx.setLineDash([5,4]);
 [a,b].forEach(x=>{ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+hplot);ctx.stroke()});
 ctx.setLineDash([]);
 [[ref,"#d09251",2],[ys,"#177da6",2.6]].forEach(pair=>{
 ctx.beginPath();ctx.strokeStyle=pair[1];ctx.lineWidth=pair[2];
 pair[0].forEach((v,i)=>{const x=X(xs[i]),y=Y(v);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y)});
 ctx.stroke();});
 ctx.fillStyle="#536577";ctx.textAlign="center";ctx.font="11px system-ui";
 for(let i=0;i<=5;i++){let t=i/5;let x=left+t*wplot;
 ctx.fillText(fmt(Math.exp(xl+t*(xr-xl)),0),x,h-bottom+17)}
 ctx.fillText("Terminal price (log-scale ticks)",(left+w-right)/2,h-4);
}
function drawBins(d){
 const g=setup($("binsCanvas")),{ctx,w,h,left,right,top,bottom}=g;
 const p=d.bin_probability,ref=d.reference_bin_probability,n=p.length;
 const hplot=h-top-bottom,wplot=w-left-right;
 const ymax=Math.max(...p,...ref)*1.13||1;axes(g,ymax,"Bin probability");
 const bw=wplot/n;
 for(let i=0;i<n;i++){
  const x=left+i*bw;
  ctx.fillStyle="#d092517a";
  const rh=hplot*ref[i]/ymax;ctx.fillRect(x,top+hplot-rh,Math.max(.2,bw),rh);
  ctx.fillStyle="#177da6bc";
  const ph=hplot*p[i]/ymax;ctx.fillRect(x+bw*.10,top+hplot-ph,Math.max(.35,bw*.8),ph);
 }
 ctx.fillStyle="#536577";ctx.textAlign="center";ctx.font="11px system-ui";
 for(let i=0;i<=4;i++){
   let s=d.cutoff_prices[0]*Math.pow(d.cutoff_prices[1]/d.cutoff_prices[0],i/4);
   ctx.fillText(fmt(s,0),left+wplot*i/4,h-bottom+17);
 }
 ctx.fillText("Terminal-price bin edges (uniform in log price)",(left+w-right)/2,h-4);
}
let lastData;
function render(d){
 lastData=d;
 $("retained").textContent=(100*d.kept_mass).toFixed(2)+"%";
 $("left").textContent=(100*d.left_lost).toFixed(2)+"%";
 $("right").textContent=(100*d.right_lost).toFixed(2)+"%";
 $("fidelity").textContent=d.conditional_amplitude_fidelity.toFixed(6);
 $("bounds").textContent="Register cutoffs: £"+fmt(d.cutoff_prices[0],2)+" to £"+
 fmt(d.cutoff_prices[1],2)+". Total lost: "+fmt(100*d.lost_mass,2)+"%. "+
 "Reference retained: "+fmt(100*d.reference_kept_mass,2)+"%.";
 drawPDF(d);drawBins(d);
}
window.addEventListener("resize",()=>{if(lastData){drawPDF(lastData);drawBins(lastData)}});
request();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif url.path == "/api":
            try:
                args = parse_qs(url.query)
                v0 = round(float(args.get("v0", [REFERENCE_V0])[0]), 5)
                cutoff = float(args.get("cutoff", [0.3])[0])
                n_qubits = int(args.get("qubits", [6])[0])
                if not (V0_MIN <= v0 <= V0_MAX and
                        0.08 <= cutoff <= 1.0 and 4 <= n_qubits <= 9):
                    raise ValueError("Slider inputs out of supported range")
                body = json.dumps(binned_data(v0, cutoff, n_qubits),
                                  allow_nan=False).encode("utf-8")
                self.send_response(200)
            except (ValueError, ArithmeticError, OverflowError) as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        else:
            self.send_error(404, "Not found")
            return
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # Avoid flooding the terminal while sliders are dragged.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true",
                        help="Do not automatically open the browser")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    address = "http://127.0.0.1:{}/".format(server.server_port)
    print("Heston v0 continuity explorer: " + address, flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_open:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
