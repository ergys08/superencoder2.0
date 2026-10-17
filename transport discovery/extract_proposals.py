"""Turn safe LLM-generated architecture source files into a main-run proposal JSON.

Usage: python 'transport discovery/extract_proposals.py' \
  --checkpoints 'transport discovery/results/llm/checkpoints' \
  --out 'transport discovery/results/llm/proposals.json'

Only parses literal WORDS lists via ast.literal_eval. It never executes
untrusted generated Python. Duplicate/illegal architectures are excluded.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from candidate import Candidate
from config import Settings
from primitives import primitive_pool
from openevolve_evaluator import parse_words


def extract(paths: list[Path]) -> tuple[list[dict], dict]:
    s = Settings()
    allowed = {p.word for p in primitive_pool(s.n_qubits, s.max_support, True)}
    accepted: dict[tuple[str, ...], dict] = {}
    rejected = 0
    for path in paths:
        try:
            cand = parse_words(path)
            cand.validate(s.n_qubits, s.max_cx, s.max_primitives)
            if not set(cand.words).issubset(allowed):
                raise ValueError('Unsupported Pauli string')
            accepted.setdefault(cand.words, {'words': list(cand.words)})
        except (ValueError, SyntaxError, OSError, TypeError):
            rejected += 1
    return list(accepted.values()), {'accepted': len(accepted), 'rejected': rejected,
                                     'scanned': len(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, required=True,
                        help='OpenEvolve checkpoint ROOT containing checkpoint_*/programs/')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoints.is_dir():
        parser.error('No checkpoints found; check --checkpoints path')
    paths = sorted(set(args.checkpoints.glob('checkpoint_*/best_program.py'))
                   | set(args.checkpoints.glob('checkpoint_*/programs/*.py')))
    if not paths:
        parser.error('No generated *.py candidates found inside checkpoint folders')
    proposals, summary = extract(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(proposals, indent=2))
    print('Architecture proposals:', summary)
    print('Saved', args.out)


if __name__ == '__main__':
    main()