# B2.2 routing corpus

`data/b22-prompts.json` defines 64 fixed prompts across science, coding,
mathematics, writing, history, planning, language, and mixed systems reasoning.

Each domain contributes five training prompts, one validation prompt, and two
held-out test prompts, yielding a 40/8/16 split. The split is fixed before route
collection so predictor tuning cannot move difficult examples into training.

The resident Qwen collector records:

- the formatted prompt token count;
- the greedy output token IDs and text;
- one unique expert set per layer for prefill; and
- one unique expert set per layer for each of 31 decode forwards.

Fixed-length decoding intentionally continues through EOS if encountered. This
keeps route-target length constant for the initial predictor comparison. A later
workload study should use natural stopping and longer 128–512-token traces.

The corpus is a systems research fixture, not a comprehensive language-model
evaluation set. Its purpose is to test whether prompt/prefill routing predicts
the later expert working set across visibly different tasks.

