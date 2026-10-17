"""OpenEvolve candidate: edit ONLY WORDS below.

The independent evaluator parses WORDS using ast.literal_eval without
executing any candidate code. This means the LLM may propose gate type,
location, number and ordering, but cannot change fidelity tests or datasets.

Each word acts on 8 qubits, with index 0 at the LEFT corresponding to q0.
Each word must have ODD number of Y and any number of X/Z/I.
Allowed support <= 4 contiguous positions, or arbitrary 2-local pairs.
The total standard CNOT upper bound must be <= 60.
"""
# EVOLVE-BLOCK-START
WORDS = [
    'YIIIIIII',
    'IYIIIIII',
    'IIZYIIII',
    'IIIIZYII',
    'IIIZZYII',
    'IIIIIIYZ',
]
# EVOLVE-BLOCK-END