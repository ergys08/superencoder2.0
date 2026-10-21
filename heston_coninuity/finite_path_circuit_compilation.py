#!/usr/bin/env python3
"""Compile the finite Heston v0 tangent-generator paths into gate circuits.

Run from repository root AFTER v0_finite_tangent_transport.py:
    python heston_coninuity/finite_path_circuit_compilation.py

Requires only numpy/scipy/matplotlib (no Qiskit dependency). Reads the ACTUAL
fixed Pauli-generator selections and midpoint coefficients from the preceding
experiment; does not refit coefficients. Compiles each step into exp(-i dv H)
using either first-order Lie product formula or symmetric second-order Strang.
Each Pauli exponential is then decomposed exactly into H, S, Sdg, RZ and CX.

Reports cumulative native gate counts, CX count and circuit depth by v0,
under all-to-all or a linear nearest-neighbour CX layout. An elementary,
conservative cancellation pass merges rotations and cancels adjacent inverses
across disjoint wires. Exports OpenQASM 2 gate circuits (no reference state
preparation); validates exported gates against Pauli-rotation statevectors.
Fidelity of the *compiled product formula* is compared to both the true
Heston target and the original exact-Hamiltonian-exponential path.

The source path uses oracle target tangents to fit coefficients, and the
trajectory-selected generator architectures use an oracle training sweep.
Circuit counts are logical, noiseless, non-hardware-calibrated estimates;
we do not infer any economic advantage or choose a cheapness threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v0_finite_tangent_transport import matrix_of_generator
from v0_naive_tangent_generators import HestonFamily, Settings as FamilySettings


@dataclass(frozen=True)
class Gate:
    kind: str
    qubits: tuple[int, ...]
    angle: float = 0.0


class Node:
    __slots__ = ("gate", "prev", "alive")
    def __init__(self, gate, prev):
        self.gate = gate
        self.prev = prev
        self.alive = True


class CircuitAccumulator:
    """Offline logical compiler: cancel adjacent inverses across idle wires."""
    def __init__(self, n: int):
        self.n = n
        self.nodes: list[Node] = []
        self.last: list[Node | None] = [None] * n
        self.counts = Counter()
        self.active_count = 0

    def _remove(self, node: Node):
        node.alive = False
        self.active_count -= 1
        self.counts[node.gate.kind] -= 1
        for q in node.gate.qubits:
            self.last[q] = node.prev[q]

    def add(self, gate: Gate):
        if gate.kind == "rz" and abs(gate.angle) < 1e-13:
            return
        qs = gate.qubits
        prev = self.last[qs[0]]
        can_touch = prev is not None and all(self.last[q] is prev for q in qs)
        if can_touch and prev.gate.qubits == gate.qubits:
            old = prev.gate
            inverse = ((old.kind == gate.kind and gate.kind in ("h", "cx"))
                       or (old.kind, gate.kind) in (("s", "sdg"), ("sdg", "s")))
            if inverse:
                self._remove(prev)
                return
            if gate.kind == "rz" and old.kind == "rz":
                theta = old.angle + gate.angle
                if abs(theta) < 1e-13:
                    self._remove(prev)
                else:
                    prev.gate = Gate("rz", qs, theta)
                return
        node = Node(gate, {q: self.last[q] for q in qs})
        self.nodes.append(node)
        for q in qs:
            self.last[q] = node
        self.active_count += 1
        self.counts[gate.kind] += 1

    def extend(self, gates: Iterable[Gate]):
        for gate in gates:
            self.add(gate)

    def active(self):
        return (node.gate for node in self.nodes if node.alive)

    def metrics(self):
        depths = [0]*self.n
        for g in self.active():
            layer = 1 + max(depths[q] for q in g.qubits)
            for q in g.qubits:
                depths[q] = layer
        return {"gates": self.active_count, "cx":self.counts["cx"],
                "rz":self.counts["rz"], "h":self.counts["h"],
                "s_sdg":self.counts["s"]+self.counts["sdg"],
                "single_qubit":self.active_count-self.counts["cx"],
                "depth":max(depths, default=0)}


def remote_cx(control: int, target: int, topology: str):
    """CX sequence; on a line route via SWAPs and restore layout exactly."""
    if topology == "all_to_all" or abs(control-target) == 1:
        return [Gate("cx", (control, target))]
    if topology != "linear":
        raise ValueError("topology must be all_to_all or linear")
    if abs(control-target)==2:
        # Exact remote CX in four native nearest-neighbour CXs, cheaper than
        # routing a SWAP there and back (seven CXs).
        mid=(control+target)//2
        return [Gate("cx",(control,mid)),Gate("cx",(mid,target)),
                Gate("cx",(control,mid)),Gate("cx",(mid,target))]
    # Bring control beside target using SWAPs, perform the CX, and restore.
    sign = 1 if target > control else -1
    swaps = [(i, i+sign) for i in range(control, target-sign, sign)]
    # Above range stops one edge before target: for control=0,target=2 -> (0,1).
    def swap_gates(a,b):
        return [Gate("cx",(a,b)),Gate("cx",(b,a)),Gate("cx",(a,b))]
    gates = []
    for a,b in swaps:
        gates.extend(swap_gates(a,b))
    moved_control = target-sign
    gates.append(Gate("cx", (moved_control, target)))
    for a,b in reversed(swaps):
        gates.extend(swap_gates(a,b))
    return gates


def pauli_rotation(word: str, theta: float, topology="all_to_all"):
    """Circuit for exp(-i theta P) with big-endian word -> internal q ordering."""
    active = [q for q,c in enumerate(word) if c != "I"]
    if not active or abs(theta) < 1e-14:
        return []
    gates = []
    for q, letter in enumerate(word):
        if letter == "X":
            gates.append(Gate("h",(q,)))
        elif letter == "Y":
            gates.extend((Gate("sdg",(q,)),Gate("h",(q,))))
    target = active[-1]
    parity = []
    if topology=="linear" and len(active)==3 and active[2]-active[0]==2:
        # On an interval of length 3, CX(0,1) then CX(1,2) collects
        # Z0*Z1*Z2 at the last qubit in just two nearest-neighbour CXs.
        parity = [Gate("cx",(active[0],active[1])),
                  Gate("cx",(active[1],active[2]))]
    elif topology=="linear" and len(active)==2 and active[1]-active[0]==2:
        # Gap in support, e.g. Z I Z: temporary flip of middle qubit
        # is uncomputed by the inverse parity network after the RZ.
        c,t=active; m=c+1
        parity=[Gate("cx",(m,t)),Gate("cx",(c,m)),Gate("cx",(m,t))]
    else:
        for q in active[:-1]:
            parity.extend(remote_cx(q, target, topology))
    gates.extend(parity)
    gates.append(Gate("rz",(target,),2*theta))
    gates.extend(reversed(parity))  # each CX is self-inverse
    for q, letter in reversed(list(enumerate(word))):
        if letter == "X":
            gates.append(Gate("h",(q,)))
        elif letter == "Y":
            gates.extend((Gate("h",(q,)), Gate("s",(q,))))
    return gates


def pauli_step(words, coeffs, dv, formula="lie", repetitions=1):
    """List of (word, theta) whose exact rotations approximate e^-i dv sum cP."""
    base = [(word, float(c)*float(dv)/repetitions) for word,c in zip(words,coeffs)
            if abs(c*dv/repetitions) > 1e-14]
    if formula == "lie":
        pattern = base
    elif formula == "strang":
        pattern = [(w,theta/2) for w,theta in base]
        pattern += [(w,theta/2) for w,theta in reversed(base)]
    else:
        raise ValueError(formula)
    return pattern * repetitions


def pauli_rotate_state(state, word, theta):
    """Apply exp(-i theta P) exactly; independent of the gate compiler."""
    n = len(word)
    ids = np.arange(len(state),dtype=np.int64)
    xmask=zmask=ycount=0
    for pos,ch in enumerate(word):
        mask=1<<(n-1-pos)
        if ch in "XY": xmask |= mask
        if ch in "YZ": zmask |= mask
        if ch=="Y": ycount+=1
    src = ids ^ xmask
    parity = np.zeros(len(state),dtype=np.int8)
    for bit in range(n):
        if zmask & (1<<bit):
            parity ^= ((src>>bit)&1).astype(np.int8)
    # P_{row,source}=i^y (-1)^[Z/Y on source].
    signs = (1-2*parity)*(1j**ycount)
    return np.cos(theta)*state -1j*np.sin(theta)*signs*state[src]


def simulate_gates(state, gates):
    """Direct matrix-free simulator for H/S/Sdg/RZ/CX (independent check)."""
    out=np.asarray(state,dtype=np.complex128).copy()
    n=round(np.log2(len(out)))
    for g in gates:
        if g.kind=="cx":
            c,t=g.qubits
            ctrl=1<<(n-1-c); targ=1<<(n-1-t)
            indices=np.arange(len(out))
            source=indices[((indices&ctrl)!=0)&((indices&targ)==0)]
            other=source^targ
            a=out[source].copy();out[source]=out[other];out[other]=a
        else:
            q=g.qubits[0]
            a=out.reshape(1<<q,2,1<<(n-q-1))
            if g.kind=="h":
                x=a[:,0,:].copy();y=a[:,1,:].copy()
                a[:,0,:]=(x+y)/np.sqrt(2)
                a[:,1,:]=(x-y)/np.sqrt(2)
            elif g.kind=="s": a[:,1,:]*=1j
            elif g.kind=="sdg": a[:,1,:]*=-1j
            elif g.kind=="rz":
                a[:,0,:]*=np.exp(-.5j*g.angle)
                a[:,1,:]*=np.exp(.5j*g.angle)
            else: raise ValueError(g.kind)
    return out


def qasm_for_gates(gates, n, metadata=""):
    """Qiskit-compatible OpenQASM2 qubit indexing: rightmost Pauli letter=q[0]."""
    lines=["OPENQASM 2.0;", 'include "qelib1.inc";',f"qreg q[{n}];",
           "// Transport only: prepare |psi(reference_v0)> before applying this circuit."]
    for line in metadata.splitlines():
        lines.append("// "+line)
    for g in gates:
        q=[n-1-x for x in g.qubits]
        if g.kind=="cx": lines.append(f"cx q[{q[0]}],q[{q[1]}];")
        elif g.kind=="rz": lines.append(f"rz({g.angle:.16g}) q[{q[0]}];")
        else: lines.append(f"{g.kind} q[{q[0]}];")
    return "\n".join(lines)+"\n"


def read_csv(path):
    with open(path,newline="") as f:
        return list(csv.DictReader(f))


def get_source(input_dir):
    with (input_dir/"settings.json").open() as f:
        settings=json.load(f)
    selection_rows=read_csv(input_dir/"selected_generators.csv")
    selection=defaultdict(list)
    for row in selection_rows:
        key=(int(row["support"]),int(row["budget"]),row["method"])
        selection[key].append((int(row["generator_rank"]),row["pauli_string"]))
    selection={k:[p for _,p in sorted(v)] for k,v in selection.items()}
    coefficients=defaultdict(dict)
    for row in read_csv(input_dir/"generator_coefficients.csv"):
        key=(int(row["support"]),int(row["budget"]),row["method"],row["branch"],
             round(float(row["v0_midpoint"]),10))
        coefficients[key][row["pauli_string"]]=float(row["coefficient"])
    trajectory=defaultdict(list)
    for row in read_csv(input_dir/"trajectory.csv"):
        key=(int(row["support"]),int(row["budget"]),row["method"],row["branch"])
        trajectory[key].append(row)
    for key in trajectory:
        trajectory[key].sort(key=lambda row:int(row["step_index"]))
    return settings,selection,coefficients,trajectory


def fidelity(a,b):
    return float(np.clip(abs(np.vdot(a,b))**2,0,1))


def csv_write(path, rows):
    if rows:
        with path.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)


def compile_case(case, words, series, coeffs, family, options, outputs, sampled):
    """Return per-checkpoint rows, plus representative OpenQASM exports."""
    support,budget,method,branch=case
    first_v=float(series[0]["v0"])
    origin=family.psi(first_v).astype(np.complex128)
    target_cache={}
    def exact_target(v):
        k=round(v,9)
        if k not in target_cache:
            target_cache[k]=family.psi(v)
        return target_cache[k]
    states={formula:origin.copy() for formula in options.formulas}
    accumulated={word:0.0 for word in words}
    oracle=origin.copy()
    topology_acc={(formula,layout): CircuitAccumulator(options.qubits)
                  for formula in options.formulas for layout in options.layouts}
    generators={word:matrix_of_generator(word) for word in words}
    records=[]
    for k,row in enumerate(series):
        v=float(row["v0"])
        if k>0:
            prev=float(series[k-1]["v0"])
            dv=v-prev
            mid=round((prev+v)/2,10)
            fields=coeffs[(support,budget,method,branch,mid)]
            coeff=np.asarray([fields[w] for w in words])
            K=sum((float(c)*generators[w] for w,c in zip(words,coeff)),
                  csr_matrix((len(origin),len(origin))))
            oracle=expm_multiply(dv*K,oracle,traceA=0.0)
            for word,c in zip(words,coeff):
                accumulated[word]+=float(c*dv)
            for formula in options.formulas:
                if formula=="folded":
                    # Single-pass ansatz: one time-integrated rotation per
                    # generator. Unlike Lie/Strang this deliberately discards
                    # Hamiltonian time ordering; it is a low-cost baseline.
                    rotations=[(word,accumulated[word]) for word in words]
                    states[formula]=origin.copy()
                    for layout in options.layouts:
                        topology_acc[(formula,layout)]=CircuitAccumulator(options.qubits)
                else:
                    rotations=pauli_step(words,coeff,dv,formula,options.repetitions)
                for word,theta in rotations:
                    states[formula]=pauli_rotate_state(states[formula],word,theta)
                for layout in options.layouts:
                    compiled=topology_acc[(formula,layout)]
                    for word,theta in rotations:
                        compiled.extend(pauli_rotation(word,theta,layout))
        exact=exact_target(v)
        for formula in options.formulas:
            for layout in options.layouts:
                accumulator=topology_acc[(formula,layout)]
                cost=accumulator.metrics()
                record={"support":support,"budget":budget,"method":method,
                        "branch":branch,"formula":formula,"repetitions":options.repetitions,
                        "connectivity":layout,"step_index":k,"v0":v,
                        "v0_distance":abs(v-options.reference_v0),"generator_count":len(words),
                        "fidelity_compiled_target":fidelity(states[formula],exact),
                        "fidelity_oracle_target":fidelity(oracle,exact),
                        "fidelity_compiled_oracle":fidelity(states[formula],oracle),
                        "infidelity_compiled_target":1-fidelity(states[formula],exact),
                        "original_experiment_fidelity":float(row["fidelity"]),
                        **cost}
                records.append(record)
                if (method=="trajectory" and support in (2,3) and budget in (8,16)
                    and abs(v-options.reference_v0)>1e-10
                    and any(abs(v-x)<abs(float(series[1]["v0"])-first_v)/2+1e-10
                            for x in options.export_at)):
                    gates=list(accumulator.active())
                    tag=(f"s{support}_k{budget}_{branch}_{formula}_{layout}"
                         f"_v0_{v:.4f}").replace(".","p")
                    name=f"transport_{tag}.qasm"
                    metadata=(f"support={support}, budget={budget}, method={method}, "
                              f"branch={branch}, formula={formula}, routing={layout}, "
                              f"reference_v0={options.reference_v0}, target_v0={v}.")
                    (outputs/"qasm"/name).write_text(qasm_for_gates(gates,options.qubits,metadata))
                    sampled.append({"qasm_file":name, "step_index":k,**record})
                    # Verify *decomposed and locally simplified gates* against
                    # rotations represented by states[formula].
                    from_gates=simulate_gates(origin,gates)
                    err=float(np.linalg.norm(from_gates-states[formula]))
                    if err>3e-8:
                        raise AssertionError(f"Gate sequence mismatch {name}: {err}")
    return records


def make_figures(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r["method"]=="trajectory"
              and r["branch"]=="increasing" and r["support"] in (2,3)]
    colors={4:"#3b82a0",8:"#db8734",16:"#589965"}
    for formula in sorted(set(r["formula"] for r in selected)):
        fig,axs=plt.subplots(1,3,figsize=(16.0,4.7),constrained_layout=True)
        for layout,style in [("all_to_all","-"),("linear","--")]:
            for support in (2,3):
                for budget in sorted(set(r["budget"] for r in selected)):
                    series=sorted((r for r in selected if r["formula"]==formula
                                   and r["connectivity"]==layout
                                   and r["support"]==support and r["budget"]==budget),
                                  key=lambda r:r["step_index"])
                    if not series:continue
                    label=f"r≤{support}, K={budget}, {layout}"
                    for ax,metric in zip(axs,("cx","depth","infidelity_compiled_target")):
                        ax.plot([r["v0"] for r in series],
                                [r[metric] for r in series],label=label,
                                linestyle=style,color=colors.get(budget),
                                alpha=0.96 if support==3 else 0.48,
                                linewidth=1.8 if support==3 else 1.1)
        for ax,metric,ylabel in zip(axs,("cx","depth","infidelity_compiled_target"),
                                    ("Accumulated CX count","Accumulated circuit depth",
                                     "Endpoint infidelity (1-F)")):
            ax.set_xlabel("Target initial variance v₀"); ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            if metric=="infidelity_compiled_target": ax.set_yscale("log")
        axs[0].legend(fontsize=7,ncol=2)
        fig.suptitle(f"Compiled finite Heston transport; {formula} product formula; no reset")
        fig.savefig(output/f"compilation_cost_vs_v0_{formula}.png",dpi=160)
        plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(12.5,5.0),constrained_layout=True)
    for formula,color in (("lie","#377ca6"),("strang","#d58c41"),("folded","#5f9965")):
        for support,style in ((2,"--"),(3,"-")):
            series=sorted((r for r in selected if
                           r["formula"]==formula and r["connectivity"]=="linear"
                           and r["support"]==support and r["budget"]==16),
                          key=lambda r:r["step_index"])
            if series:
                label=f"{formula}, support≤{support}, K=16"
                axs[0].plot([r["v0"] for r in series],
                            [r["fidelity_compiled_target"] for r in series],
                            label=label,color=color,linestyle=style)
                axs[1].plot([r["v0"] for r in series],
                            [1-r["fidelity_compiled_oracle"] for r in series],
                            label=label,color=color,linestyle=style)
    axs[0].set_ylabel("Fidelity vs actual Heston target")
    axs[1].set_ylabel("Product-formula error: 1−F(compiled, exact exponentials)")
    axs[1].set_yscale("log")
    for ax in axs:
        ax.set_xlabel("Target initial variance v₀"); ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Physical gate-product error vs finite-path generator error")
    fig.savefig(output/"compiled_fidelity_vs_oracle.png",dpi=160)
    plt.close(fig)


def diagram_sample(output):
    """Draw a sample actual four-qubit span Pauli exponential compiled into gates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    word="IIXYIIII"
    gates=pauli_rotation(word,theta=0.125,topology="linear")
    # Only diagram active span for readability; labels show original 8 qubits.
    visible=[2,3]
    fig,ax=plt.subplots(figsize=(max(10,len(gates)*.52),2.9),constrained_layout=True)
    xpositions=np.arange(len(gates))+.8
    for q in visible:
        ax.hlines(q,0,len(gates)+.6,color="#6b7280",linewidth=1.2)
        ax.text(-.05,q,f"q{q}",ha="right",va="center",fontsize=12)
    for x,g in zip(xpositions,gates):
        if g.kind=="cx":
            c,t=g.qubits
            ax.plot([x,x],[c,t],color="#222",linewidth=1.3)
            ax.plot(x,c,"o",ms=6,color="#222")
            ax.plot(x,t,marker="o",markerfacecolor="white",markeredgecolor="#222",ms=12)
            ax.plot([x-.10,x+.10],[t,t],color="#222")
            ax.plot([x,x],[t-.1,t+.1],color="#222")
        else:
            q=g.qubits[0]
            label=("Rz(0.25)" if g.kind=="rz" else g.kind.upper())
            ax.text(x,q,label,ha="center",va="center",fontsize=8,
                    bbox={"boxstyle":"round,pad=.25","facecolor":"white","edgecolor":"#3b81a1"})
    ax.set(xlim=(-.55,len(gates)+.65),ylim=(3.65,1.35))
    ax.axis("off")
    ax.set_title("Exact gate decomposition of exp(−i·0.125·X₂Y₃) (H, S, RZ, CX)")
    fig.savefig(output/"example_pauli_generator_circuit.png",dpi=180)
    plt.close(fig)


def smoke_tests():
    rng=np.random.default_rng(7)
    n=5
    psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    psi/=np.linalg.norm(psi)
    for word in ("YIIII","IIYII","IXYII","IYIXI","XYIII","YIYII","IYZYI",
                 "YZYII", "XYXII", "YIYII", "IYIXI"):
        for layout in ("all_to_all","linear"):
            gates=pauli_rotation(word,.319,layout)
            a=pauli_rotate_state(psi,word,.319)
            b=simulate_gates(psi,gates)
            err=np.linalg.norm(a-b)
            if err>2e-13:
                raise AssertionError(f"Wrong decomposition {word}/{layout}: {err}")
            simplify=CircuitAccumulator(n)
            simplify.extend(gates)
            err=np.linalg.norm(simulate_gates(psi,simplify.active())-a)
            if err>2e-13:
                raise AssertionError(f"Wrong optimisation {word}/{layout}: {err}")
    # Pauli products with noncommuting terms and optimiser cancellation at edges.
    for layout in ("all_to_all","linear"):
        strings=[("IYIYI",.11),("ZYIII",.07),("IXYII",-.23)]
        builder=CircuitAccumulator(n)
        v=psi.copy()
        for w,theta in strings:
            v=pauli_rotate_state(v,w,theta)
            builder.extend(pauli_rotation(w,theta,layout))
        err=np.linalg.norm(simulate_gates(psi,builder.active())-v)
        if err>2e-13:raise AssertionError(f"Wrong product compiler: {err}")
    return True


def run(options):
    smoke_tests()
    output=Path(options.output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"qasm").mkdir(exist_ok=True)
    source_settings,selections,coeffs,trajectories=get_source(Path(options.input))
    if options.qubits != int(source_settings["qubits"]):
        raise ValueError("qubit count must match saved generator trajectory")
    if abs(options.reference_v0-float(source_settings["reference_v0"]))>1e-10:
        raise ValueError("reference must match saved trajectory")
    family=HestonFamily(FamilySettings(qubits=options.qubits,
        n_fft=int(source_settings["n_fft"]),cutoff=float(source_settings["cutoff"]),
        reference_v0=options.reference_v0))
    cases=[key for key in trajectories if key[:3] in selections
           and key[0] in options.supports and key[1] in options.budgets
           and key[2] in options.methods and key[3] in options.branches]
    if not cases:raise ValueError("No cases matched the saved trajectory")
    results=[];exports=[]
    for index,case in enumerate(sorted(cases),1):
        print(f"[{index}/{len(cases)}] compiling {case}...",flush=True)
        results.extend(compile_case(case,selections[case[:3]],trajectories[case],
                                    coeffs,family,options,output,exports))
    csv_write(output/"compiled_trajectory.csv",results)
    csv_write(output/"exported_qasm_index.csv",exports)
    summary=[]
    for key in sorted({(r["support"],r["budget"],r["method"],r["branch"],
                       r["formula"],r["connectivity"]) for r in results}):
        series=sorted((r for r in results if
            (r["support"],r["budget"],r["method"],r["branch"],
             r["formula"],r["connectivity"])==key),key=lambda row:row["step_index"])
        end=series[-1]
        summary.append({"support":key[0],"budget":key[1],"method":key[2],
                        "branch":key[3],"formula":key[4],"connectivity":key[5],
                        "endpoint_v0":end["v0"],"final_fidelity":end["fidelity_compiled_target"],
                        "final_cx":end["cx"],"final_depth":end["depth"],
                        "final_gates":end["gates"],"worst_product_formula_error":max(
                            1-r["fidelity_compiled_oracle"] for r in series),
                        "first_fidelity_below_0_999":next((r["v0"] for r in series[1:]
                            if r["fidelity_compiled_target"]<.999),None)})
    csv_write(output/"compiled_summary.csv",summary)
    (output/"settings.json").write_text(json.dumps(vars(options),indent=2)+"\n")
    make_figures(results,output)
    diagram_sample(output)
    print("\nResults:")
    for row in summary:
        if (row["method"]=="trajectory" and row["branch"]=="increasing"
            and row["connectivity"]=="linear" and row["budget"] in (8,16)):
            print(f"{row['formula']:6} s≤{row['support']} K={row['budget']:2} "
                  f"end F={row['final_fidelity']:.6f}; "
                  f"CX={row['final_cx']:,}; depth={row['final_depth']:,}; "
                  f"first F<.999 at {row['first_fidelity_below_0_999']}")
    print(f"\nSaved {len(results)} cost checkpoints and {len(exports)} validated QASM circuits to {output}")
    return summary


def main():
    default_input="heston_coninuity/results/v0_finite_tangent_transport"
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",default=default_input)
    parser.add_argument("--output",default="heston_coninuity/results/finite_path_circuit_compilation")
    parser.add_argument("--supports",nargs="+",type=int,default=[2,3])
    parser.add_argument("--budgets",nargs="+",type=int,default=[4,8,16])
    parser.add_argument("--methods",nargs="+",default=["trajectory","reference"])
    parser.add_argument("--branches",nargs="+",default=["decreasing","increasing"])
    parser.add_argument("--formulas",nargs="+",choices=["lie","strang","folded"],default=["lie","strang","folded"])
    parser.add_argument("--layouts",nargs="+",choices=["all_to_all","linear"],default=["all_to_all","linear"])
    parser.add_argument("--repetitions",type=int,default=1)
    parser.add_argument("--export-at",nargs="+",type=float,default=[0.02,0.08,0.12,0.20])
    args=parser.parse_args()
    args.qubits=8
    args.reference_v0=0.04
    if args.repetitions<1:raise ValueError("repetitions must be positive")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits=None
    if threadpool_limits:
        with threadpool_limits(limits=1,user_api="blas"):
            run(args)
    else:
        run(args)


if __name__=="__main__":
    main()