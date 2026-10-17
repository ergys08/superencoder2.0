"""Pauli-rotation transport primitives and independent statevector simulator.

Conventions
-----------
Words are strings in increasing qubit order: word[0] acts on qubit 0
(the least significant bit of Qiskit's computational-basis index).

For an odd number of Y's, P is imaginary and the generator A=-iP is
REAL anti-symmetric. Therefore exp(-i theta P/2) preserves the real
amplitudes of our Heston encodings.
"""
from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Primitive:
    word: str  # Characters I, X, Y, Z; q0 at index zero.

    def __post_init__(self):
        if not self.word or any(c not in 'IXYZ' for c in self.word):
            raise ValueError('Pauli words must use I, X, Y, Z')
        if self.word.count('Y') % 2 != 1:
            raise ValueError('Use an odd number of Y gates to preserve real amplitudes')

    @property
    def support(self):
        return sum(p != 'I' for p in self.word)

    @property
    def cx_upper_bound(self):
        """CNOTs for standard parity ladder synthesis, before cancellations."""
        return 2*(self.support-1)


def primitive_pool(n: int, max_support: int = 4, full_two_qubit: bool = False) -> list[Primitive]:
    """Contiguous k-local odd-Y Pauli strings, plus optional long-range 2-local.

    Contiguity refers to q indices, not adjacent price-bin indices.
    Every primitive has an exact standard CX+single-qubit decomposition.
    """
    from itertools import product, combinations
    words = set()
    for k in range(1, min(n, max_support)+1):
        for start in range(n-k+1):
            for paulis in product('XYZ', repeat=k):
                if paulis.count('Y') % 2:
                    word = ['I']*n
                    word[start:start+k] = paulis
                    words.add(''.join(word))
    if full_two_qubit and max_support >= 2:
        for a,b in combinations(range(n),2):
            for left,right in product('XYZ',repeat=2):
                if (left=='Y')+(right=='Y') == 1:
                    word = ['I']*n
                    word[a],word[b] = left,right
                    words.add(''.join(word))
    return [Primitive(w) for w in sorted(words)]


@lru_cache(maxsize=10000)
def real_action(word: str) -> tuple[np.ndarray,np.ndarray]:
    """Return permutation and +/- signs for A=-i P on real statevectors.

    output[j] = signs[j] * input[permutation[j]].  This is exact for
    any Pauli word with odd number of Y letters and uses Qiskit ordering.
    """
    if word.count('Y') % 2 != 1:
        raise ValueError('real_action requires odd number of Y')
    n=len(word)
    flip=sum((1<<j) for j,p in enumerate(word) if p in 'XY')
    phase=sum((1<<j) for j,p in enumerate(word) if p in 'YZ')
    basis=np.arange(1<<n,dtype=np.int64)
    indices=basis ^ flip
    # P|x> = i^m (-1)^(sum x_j over Y/Z positions) |x xor flip>
    # Thus -i P has sign (-1)^((m-1)/2 + above parity).
    par=np.zeros(len(basis),dtype=np.int8)
    for j in range(n):
        if (phase>>j)&1:
            par ^= ((indices>>j)&1).astype(np.int8)
    sign=np.where(par==0,1.,-1.) * (-1.)**((word.count('Y')-1)//2)
    indices.setflags(write=False)
    sign.setflags(write=False)
    return indices,sign


def apply_numpy(state: np.ndarray, primitive: Primitive, angle: float) -> np.ndarray:
    """Apply exp(-i angle P/2) without constructing a 2**n square matrix."""
    state = np.asarray(state)
    indices,sign = real_action(primitive.word)
    if state.shape[-1] != len(indices):
        raise ValueError('State dimension and Pauli word disagree')
    return np.cos(angle/2)*state + np.sin(angle/2)*(state[...,indices]*sign)


def simulate(reference: np.ndarray, primitives: list[Primitive], angles: np.ndarray) -> np.ndarray:
    if len(primitives) != len(angles):
        raise ValueError('Each primitive needs one rotation angle')
    state=np.asarray(reference)
    for primitive,theta in zip(primitives, angles):
        state=apply_numpy(state,primitive,float(theta))
    return state


def reference_tangents(reference: np.ndarray, primitives: list[Primitive]) -> np.ndarray:
    """Columns of Jacobian d(U(theta)ref)/dtheta at theta=0."""
    if not primitives:
        return np.empty((len(reference),0))
    return np.stack([.5*reference[real_action(p.word)[0]]*real_action(p.word)[1]
                     for p in primitives],axis=1)


def compile_qiskit(primitives: list[Primitive], angles: np.ndarray):
    """Build a concrete Qiskit circuit; dependency required only for gate metrics.

    Each e^(-i theta P/2) is compiled by Pauli basis changes, a CNOT
    parity ladder, RZ and exact inverse changes. No oracle optimiser.
    """
    from qiskit import QuantumCircuit
    if len(primitives) != len(angles) or not primitives:
        raise ValueError('Require nonempty primitive list and matching angles')
    qc=QuantumCircuit(len(primitives[0].word))
    for prim, theta in zip(primitives, angles):
        if len(prim.word) != qc.num_qubits:
            raise ValueError('Incompatible primitive qubit counts')
        active=[j for j,p in enumerate(prim.word) if p!='I']
        for j in active:
            if prim.word[j]=='X': qc.h(j)
            elif prim.word[j]=='Y':
                qc.sdg(j)
                qc.h(j)
        for a,b in zip(active, active[1:]): qc.cx(a,b)
        qc.rz(float(theta), active[-1])
        for a,b in reversed(list(zip(active,active[1:]))): qc.cx(a,b)
        for j in active:
            if prim.word[j]=='X': qc.h(j)
            elif prim.word[j]=='Y':
                qc.h(j)
                qc.s(j)
    return qc


def compiled_cx_count(primitives: list[Primitive], optimization_level: int = 2) -> tuple[int,int]:
    """Transpiled CNOT count, depth (measured at generic NONZERO angles).

    If Qiskit isn't installed, return conservative structural CX count
    and depth=-1 to *explicitly* mark depth unavailable. In normal
    experiments, install requirements.txt to get actual transpile metrics.
    """
    if not primitives:
        return 0,0
    try:
        from qiskit import transpile
    except ImportError:
        return sum(p.cx_upper_bound for p in primitives),-1
    qc=compile_qiskit(primitives, np.linspace(.19,.57,len(primitives)))
    transpiled=transpile(qc,basis_gates=['u','cx'],optimization_level=optimization_level)
    return int(transpiled.count_ops().get('cx',0)),int(transpiled.depth())
