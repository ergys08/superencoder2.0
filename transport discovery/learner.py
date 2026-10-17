"""Train an amortised Heston-parameter-to-transport-angles map.

NO target-specific variational optimisation is performed at inference.
Optimisation occurs once across shared training states, then weights are
frozen for the validation/test sets. Angles vanish exactly at the reference.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy
import math
import numpy as np
import torch
from torch import nn
from config import BOUNDS, REFERENCE
from candidate import Candidate
from primitives import real_action


class TransportNN(nn.Module):
    """Differentiable real-amplitude simulator with an identity-anchored MLP."""
    def __init__(self,candidate:Candidate,width:int=48):
        super().__init__()
        n=len(candidate.words)
        self.net=nn.Sequential(nn.Linear(6,width),nn.Tanh(),
                               nn.Linear(width,width),nn.Tanh(),nn.Linear(width,n))
        # Small angle initialisation; keep the gates initially near identity.
        nn.init.normal_(self.net[-1].weight,std=.01)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('centre',torch.tensor(REFERENCE,dtype=torch.float64))
        self.register_buffer('scale',torch.tensor(BOUNDS[:,1]-BOUNDS[:,0],dtype=torch.float64))
        self.actions=[]
        for i,w in enumerate(candidate.words):
            idx,sign=real_action(w)
            self.register_buffer(f'idx_{i}',torch.from_numpy(idx.copy()))
            self.register_buffer(f'sgn_{i}',torch.from_numpy(sign.copy()))
        self.n_ops=n
        self.double()

    def angles(self,eta):
        x=(eta-self.centre)/self.scale
        # Reference maps exactly to theta=0 for any network weights.
        zero=torch.zeros((1,6),dtype=x.dtype,device=x.device)
        return np.pi*(torch.tanh(self.net(x))-torch.tanh(self.net(zero)))

    def forward(self,eta,reference):
        theta=self.angles(eta)
        psi=reference.expand(len(eta),-1)
        for k in range(self.n_ops):
            perm=getattr(self,f'idx_{k}')
            signs=getattr(self,f'sgn_{k}')
            action=psi.index_select(1,perm)*signs
            psi=torch.cos(theta[:,k:k+1]/2)*psi+torch.sin(theta[:,k:k+1]/2)*action
        return psi


@dataclass
class TrainingResult:
    model:TransportNN
    history:list[dict]
    epochs_run:int
    best_val_loss:float


def robust_infidelity_loss(pred, target, *, tail_fraction: float = .25,
                            tail_weight: float = .75):
    """CVaR-style training objective: mean infidelity + hard-target tail.

    Let e_i = 1 - F_i. We minimise (1-w)*mean(e) + w*mean(top_q(e)).
    The top-q term discourages a network from sacrificing hard Heston
    distributions to optimise easy high-overlap states. The minibatch tail is
    a surrogate, not the actual worst case over the continuous domain.
    """
    infidelities = 1 - torch.sum(pred * target, dim=1).square()
    n_worst = max(1, math.ceil(len(infidelities) * tail_fraction))
    return ((1 - tail_weight) * infidelities.mean() +
            tail_weight * torch.topk(infidelities, n_worst).values.mean())


def train(candidate:Candidate,eta_train,psi_train,eta_val,psi_val,reference,
          *,seed=2026,epochs=130,width=48,batch_size=32,patience=22,lr=0.003,
          tail_fraction=.25,tail_weight=.75) -> TrainingResult:
    """Train a shared angle map using a worst-tail-weighted fidelity objective."""
    torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    model=TransportNN(candidate,width=width)
    device='cpu'
    x=torch.as_tensor(eta_train,dtype=torch.float64,device=device)
    y=torch.as_tensor(psi_train,dtype=torch.float64,device=device)
    xv=torch.as_tensor(eta_val,dtype=torch.float64,device=device)
    yv=torch.as_tensor(psi_val,dtype=torch.float64,device=device)
    ref=torch.as_tensor(reference,dtype=torch.float64,device=device).unsqueeze(0)
    optim=torch.optim.Adam(model.parameters(),lr=lr)
    best_val=float('inf');best_state=copy.deepcopy(model.state_dict());since=0
    history=[]
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        perm=rng.permutation(len(x))
        for ids in np.array_split(perm,max(1,int(np.ceil(len(x)/batch_size)))):
            optim.zero_grad()
            pred=model(x[ids],ref)
            loss=robust_infidelity_loss(pred,y[ids],tail_fraction=tail_fraction,tail_weight=tail_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optim.step()
        model.eval()
        with torch.no_grad():
            val_pred=model(xv,ref)
            val_loss=float(robust_infidelity_loss(val_pred,yv,tail_fraction=tail_fraction,tail_weight=tail_weight))
        history.append({'epoch':epoch+1,'validation_infidelity':val_loss})
        if val_loss < best_val-1e-9:
            best_val=val_loss;best_state=copy.deepcopy(model.state_dict());since=0
        else:
            since+=1
        if since >= patience:break
    model.load_state_dict(best_state)
    return TrainingResult(model,history,len(history),best_val)


def predict(model,eta,reference,batch_size=64):
    model.eval()
    ref=torch.as_tensor(reference,dtype=torch.float64).unsqueeze(0)
    out=[];angles=[]
    with torch.no_grad():
        for start in range(0,len(eta),batch_size):
            x=torch.as_tensor(eta[start:start+batch_size],dtype=torch.float64)
            out.append(model(x,ref).cpu().numpy())
            angles.append(model.angles(x).cpu().numpy())
    return np.concatenate(out),np.concatenate(angles)


def save_model(result:TrainingResult,candidate:Candidate,path:Path):
    """Record architecture, weights, and fitting history for later inference."""
    import json
    path.mkdir(parents=True,exist_ok=True)
    torch.save(result.model.state_dict(),path/'nn_state_dict.pt')
    (path/'architecture.json').write_text(json.dumps(candidate.to_dict(),indent=2))
    (path/'training_history.json').write_text(json.dumps(result.history,indent=2))