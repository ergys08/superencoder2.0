"""Architecture grammar, reproducible mutations and JSON serialisation."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from primitives import Primitive, primitive_pool


@dataclass(frozen=True)
class Candidate:
    """An ordered circuit *topology*; angles are NOT part of the architecture."""
    words: tuple[str,...]

    @property
    def primitives(self) -> list[Primitive]:
        return [Primitive(w) for w in self.words]

    @property
    def n_angles(self) -> int:
        return len(self.words)

    @property
    def cx_upper_bound(self) -> int:
        return sum(p.cx_upper_bound for p in self.primitives)

    def validate(self, n_qubits: int, max_cx: int, max_primitives: int):
        if not 1 <= len(self.words) <= max_primitives:
            raise ValueError('Invalid primitive count')
        if any(len(w) != n_qubits for w in self.words):
            raise ValueError('Wrong number of qubits')
        if self.cx_upper_bound > max_cx:
            raise ValueError('Architecture exceeds conservative CNOT limit')

    def to_dict(self) -> dict:
        return {'words':list(self.words),'n_angles':self.n_angles,
                'structural_cx_bound':self.cx_upper_bound}

    @classmethod
    def from_dict(cls,d):
        return cls(tuple(d['words']))


def seed_candidates(n_qubits: int, max_support: int, max_cx: int,
                    max_primitives: int) -> list[Candidate]:
    """Human-interpretable controls to compare evolutionary search against."""
    pool=primitive_pool(n_qubits,max_support)
    single=[p.word for p in pool if p.support==1]
    double=[p.word for p in pool if p.support==2]
    triple=[p.word for p in pool if p.support==3]
    four=[p.word for p in pool if p.support==4]
    schemes=[single,double,triple,four,single+double, double+triple, single+triple+four]
    unique={}
    for scheme in schemes:
        if not scheme:continue
        words=[]
        for word in scheme:
            trial=Candidate(tuple(words+[word]))
            if trial.cx_upper_bound <= max_cx and len(trial.words)<=max_primitives:
                words.append(word)
        if words:
            c=Candidate(tuple(words))
            unique[c.words]=c
    return list(unique.values())


def mutate(parent: Candidate, pool: list[Primitive], rng: np.random.Generator,
           max_cx: int, max_primitives: int, trials: int = 100) -> Candidate:
    """Add, replace, remove or reorder a primitive; enforce the hard budget."""
    pool_words=[p.word for p in pool]
    for _ in range(trials):
        words=list(parent.words)
        action=int(rng.integers(0,4))
        if action==0 and len(words)<max_primitives:
            words.insert(int(rng.integers(len(words)+1)),str(rng.choice(pool_words)))
        elif action==1:
            words[int(rng.integers(len(words)))] = str(rng.choice(pool_words))
        elif action==2 and len(words)>1:
            words.pop(int(rng.integers(len(words))))
        elif action==3 and len(words)>1:
            i,j=rng.choice(len(words),size=2,replace=False)
            words[int(i)],words[int(j)]=words[int(j)],words[int(i)]
        c=Candidate(tuple(words))
        if 1 <= len(c.words)<=max_primitives and c.cx_upper_bound<=max_cx and c!=parent:
            return c
    return parent


def save_candidate(candidate: Candidate,path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(candidate.to_dict(),indent=2))
