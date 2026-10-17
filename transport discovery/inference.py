"""Deploy the learned fixed-reference transport circuit on NEW Heston inputs.

Uses only: saved reference state, winner topology, frozen NN weights, 6 inputs.
No target Heston FFT inversion and NO per-target angle optimisation.

Example:
 python 'transport discovery/inference.py' \
    --run 'transport discovery/results/main' \
    --eta 0.05 0.04 2.0 0.04 0.30 -0.70 \
    --qpy 'transport discovery/results/main/transport.qpy'
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from candidate import Candidate
from config import BOUNDS,PARAMETER_NAMES
from learner import TransportNN,predict
from primitives import compile_qiskit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='Completed results directory')
    p.add_argument('--eta',type=float,nargs=6,required=True,metavar='PARAM')
    p.add_argument('--qpy',default=None,help='Optional Qiskit QPY circuit output')
    args=p.parse_args()
    out=Path(args.run)
    eta=np.asarray(args.eta,dtype=float)
    if not np.isfinite(eta).all() or (eta<BOUNDS[:,0]).any() or (eta>BOUNDS[:,1]).any():
        raise ValueError(f'Input must respect original Heston domain: {dict(zip(PARAMETER_NAMES,BOUNDS.tolist()))}')
    settings=json.loads((out/'settings.json').read_text())
    candidate=Candidate.from_dict(json.loads((out/'winner.json').read_text()))
    ref=np.load(out/'reference.npz')['amplitudes']
    model=TransportNN(candidate,width=settings['nn_width'])
    try:
        weights=torch.load(out/'winner_model'/'nn_state_dict.pt',map_location='cpu',weights_only=True)
    except TypeError:
        weights=torch.load(out/'winner_model'/'nn_state_dict.pt',map_location='cpu')
    model.load_state_dict(weights)
    predictions,angles=predict(model,eta[None,:],ref)
    result={'heston_parameters':dict(zip(PARAMETER_NAMES,eta.tolist())),
            'architecture':candidate.to_dict(),
            'angles_radians':angles[0].tolist(),
            'output_norm':float(np.linalg.norm(predictions[0])),
            'note':'Angles predicted with frozen NN; no Heston FFT or per-target optimisation.'}
    if args.qpy:
        from qiskit import qpy
        circuit=compile_qiskit(candidate.primitives,angles[0])
        path=Path(args.qpy);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as f:qpy.dump(circuit,f)
        result['qpy_path']=str(path)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
