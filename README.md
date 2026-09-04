# Coin Challenge Multimodal Questioner

Submission for the Coin Challenge: an evidence-grounded multimodal questioner for the
interactive image-matching game. Given a textual description of a hidden target and one
candidate image at a time, `YourQuestioner` either asks the Oracle a concise question
about the hidden target or concludes whether the candidate matches.

The repository contains only the core agent code. Datasets, image assets, and runtime
engineering scripts are not part of the submission; the evaluation harness supplies the
data and runs the agent.

## Core Code

| File | Purpose |
| --- | --- |
| `Questioner.py` | `QuestionerInterface` and the active `YourQuestioner` agent |
| `Oracle.py` | Minimal `OracleInterface` contract |
| `env.py` | Gymnasium environment: rewards, transitions, and Oracle calls |
| `utils.py` | Model clients (Doubao/ModelArk, Gemini, local vLLM) and image helpers |
| `eval_model.py` | Evaluation loop, provider setup, and result writing |

## Game and Scoring

| Action | Reward |
| --- | ---: |
| Correct conclusion | `+10` |
| Incorrect conclusion | `-10` (episode ends) |
| Oracle question | `-1` |

A correct conclusion advances to the next candidate. An episode is fully successful only
when every candidate is classified correctly.

## Questioner Design

`YourQuestioner` is a model-plus-controller architecture:

1. The image model inspects the candidate and returns a compact JSON action.
2. A clear category mismatch or a reliable target conflict permits conclusion `0`.
3. Otherwise it asks one atomic yes/no question about a discriminative visible attribute
   of the hidden target.
4. Oracle answers are parsed as `yes` / `no` / `uncertain` and stored with attribute and
   candidate provenance.
5. A direct `no` for the current candidate forces conclusion `0`; under a generic
   category description, conclusion `1` requires two independent grounded matches.
6. Invalid JSON, repeated questions, or unsupported conclusions receive one repair
   attempt before an evidence-gated fallback.

The controller keeps a bounded, structured memory of target evidence and distinguishes
current-candidate evidence from evidence gathered on earlier candidates.

## Protocol

```python
observation = {
    "image": np.ndarray,  # RGB, H x W x 3
    "answer": str | None,  # present only after an Oracle question
}

action = {
    "question": str | None,  # max 300 chars, about the hidden target
    "conclusion": 0 | 1 | None,  # 0 = no match, 1 = match
    "reasoning": str,
}
```

Exactly one of `question` and `conclusion` must be non-`None`. The Oracle is asked with
`ask(*, prompt=..., images=...)` and answers about the hidden target image.

## Interfaces

A custom questioner inherits `QuestionerInterface` and implements
`ask_or_conclude(observation)`. A custom oracle implements the `OracleInterface` in
`Oracle.py`. See `AGENTS.md` for the detailed contract.
