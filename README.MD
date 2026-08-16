# Coin Challenge Multimodal Questioner

An evidence-grounded multimodal agent for an interactive image-matching challenge.

## Project Description

This repository implements and evaluates a multimodal `Questioner`. In each episode, the Questioner receives a textual description of a hidden target and sees one candidate image at a time. It must either ask an `Oracle` a concise visual question about the hidden target or conclude whether the current candidate matches it.

The active implementation, `YourQuestioner`, combines an image-capable language model with deterministic control logic. The model interprets the visible candidate and proposes discriminative attributes. The controller maintains structured evidence, tracks answer polarity and provenance, enforces the action protocol, limits context growth, repairs malformed output, and gates unsupported conclusions.

Questioner and Oracle use Doubao / Volcano Engine ModelArk by default. Gemini and local vLLM backends are also supported. The two roles have separate image permissions: the Questioner sees only the candidate image, while the Oracle sees only the hidden target image.

For the detailed architecture, engineering improvements, limitations, and future work, see the English [technical report](TECHNICAL_REPORT.md) or its [PDF version](TECHNICAL_REPORT.pdf).

## Repository Layout

| Path | Purpose |
| --- | --- |
| `Questioner.py` | Questioner interface and the active `YourQuestioner` implementation |
| `Oracle.py` | Minimal Oracle interface |
| `env.py` | Gymnasium environment, validation, rewards, candidate transitions, and Oracle calls |
| `utils.py` | Doubao, Gemini, and local vLLM clients plus image encoding |
| `eval_model.py` | Evaluation CLI, provider setup, episode loop, summaries, and result writing |
| `episodes_*.jsonl` | Train, validation, and test episode definitions |
| `images/` | Image assets referenced by episode files |
| `tests/` | Offline tests for memory, evidence gates, repair, transitions, and retries |
| `scripts/` | Environment setup, asset download, dataset split, and vLLM launch helpers |
| `results/` | Locally generated gzip JSON evaluation output |

## Game and Scoring

| Action | Reward |
| --- | ---: |
| Correct conclusion | `+10` |
| Incorrect conclusion | `-10` and the episode ends |
| Oracle question | `-1` |

```text
episode_reward = 10 * correct_conclusions - 10 * incorrect_conclusions - oracle_questions
```

A correct conclusion advances to the next candidate. An episode is fully successful only when every candidate is classified correctly. By default, the environment truncates an episode after 60 steps or 600 seconds.

## Quick Start

Run every command from the repository root because episode files contain repository-relative image paths.

### 1. Create the Environment

```bash
bash scripts/install.sh
```

The installer creates `coin_env/` and installs the evaluation dependencies.

### 2. Download the Images

Using the Hugging Face CLI:

```bash
mkdir -p images
hf download --repo-type dataset e-zorzi/images_coin_challenge \
  --local-dir images --force-download
```

If Git LFS pointers remain or Hugging Face is slow, use the validating mirror downloader:

```bash
./coin_env/bin/python scripts/download_images_mirror.py
```

`env.py` rejects Git LFS pointer files so that a text pointer is never passed to Pillow as if it were a PNG image.

### 3. Configure Doubao

```bash
cp .env.example .env
```

At minimum, set:

```dotenv
ARK_API_KEY=your_ark_api_key
ARK_MODEL_ID=your_image_understanding_endpoint_id
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```

The repository ignores `.env`; never commit real credentials. The Questioner uses `QUESTIONER_MODEL_ID` when set and otherwise falls back to `ARK_MODEL_ID`. The Doubao Oracle uses `ARK_MODEL_ID`.

### 4. Run a Small Smoke Evaluation

Start with a small, lower-cost slice to verify credentials, assets, and endpoint compatibility:

```bash
QUESTIONER_PROVIDER=doubao \
./coin_env/bin/python eval_model.py 0 3 \
  --split train \
  --description-type category \
  --oracle-provider doubao
```

The positional range is half-open. `0 3` runs episodes 0, 1, and 2.

### 5. Run the Complete Test Split

Run only the `category` description:

```bash
QUESTIONER_PROVIDER=doubao \
./coin_env/bin/python eval_model.py 0 16 \
  --split test \
  --description-type category \
  --oracle-provider doubao
```

Run all six description types:

```bash
QUESTIONER_PROVIDER=doubao \
./coin_env/bin/python eval_model.py 0 16 \
  --split test \
  --description-type all \
  --oracle-provider doubao
```

Supported description types are `category`, `color`, `context`, `color_context_feature`, `color_feature`, and `color_context`. `all` runs all six in sequence.

Use the live CLI help as the authoritative option reference:

```bash
./coin_env/bin/python eval_model.py --help
```

## Running in the Background

A full six-description evaluation can make many paid API calls. If the terminal must be closed, start the process with `nohup`; applying only `disown` after launch may leave process output attached to the terminal.

```bash
mkdir -p logs

nohup env QUESTIONER_PROVIDER=doubao \
  ./coin_env/bin/python eval_model.py 0 16 \
  --split test \
  --description-type all \
  --oracle-provider doubao \
  > logs/test_all.log 2>&1 < /dev/null &
```

Follow progress with:

```bash
tail -f logs/test_all.log
```

The evaluator currently writes a result only after one complete description type. It does not checkpoint after every episode, so interrupting the current description type loses its unwritten progress.

## Results and Metrics

Results are written to:

```text
results/<QuestionerClass>_<description>_<split>_<start>_<end>.gzip.json
```

Example:

```text
results/YourQuestioner_category_test_0_16.gzip.json
```

Each gzip JSON file includes episode IDs, target images, descriptions, actions, Oracle answers, reasoning, conclusion and question counts, rewards, full-episode success flags, and timing.

The console summary reports:

- `Decision accuracy`: correct conclusions divided by all conclusions;
- `Oracle questions`: total and average questions per episode;
- `Environment reward`: total and average reward per episode;
- `Questioner time`: average Questioner model latency;
- `Full-match rate`: fraction of episodes in which every candidate was classified correctly.

## Model Configuration

### Doubao / Volcano Engine ModelArk

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `ARK_API_KEY` | none | Required Volcano Engine API key |
| `ARK_MODEL_ID` | none | Required image-capable endpoint ID |
| `ARK_BASE_URL` | Beijing API URL | OpenAI-compatible API base URL |
| `ARK_TIMEOUT_SECONDS` | `90` | Per-request timeout |
| `ARK_MAX_OUTPUT_LENGTH` | `800` | Output character budget, approximately 200 tokens |
| `ARK_THINKING` | `disabled` | ModelArk thinking parameter |
| `QUESTIONER_PROVIDER` | `doubao` | Questioner backend |
| `QUESTIONER_MODEL_ID` | `ARK_MODEL_ID` | Questioner model or endpoint |
| `ORACLE_PROVIDER` | `doubao` | Oracle backend when no CLI option is provided |

To use the same Doubao endpoint for Questioner and Oracle:

```bash
QUESTIONER_PROVIDER=doubao \
QUESTIONER_MODEL_ID=doubao-seed-2-1-turbo-260628 \
ARK_MODEL_ID=doubao-seed-2-1-turbo-260628 \
./coin_env/bin/python eval_model.py 0 3 \
  --split test --description-type category --oracle-provider doubao
```

Replace the example model name with the actual ModelArk endpoint ID.

### Gemini

Set `GEMINI_API_KEY`, then run:

```bash
QUESTIONER_PROVIDER=gemini \
./coin_env/bin/python eval_model.py 0 3 \
  --split test --description-type category --oracle-provider gemini
```

`google-genai` is optional and is not required by the default Doubao workflow.

### Local vLLM

Start the local server:

```bash
bash scripts/launch_qwen35_vllm.sh
```

In another terminal, run:

```bash
QUESTIONER_PROVIDER=local \
QUESTIONER_MODEL_ID=local-qwen35-4b \
ORACLE_MODEL_ID=local-qwen35-4b \
./coin_env/bin/python eval_model.py 0 3 \
  --split test --description-type category --oracle-provider local
```

`ClientBasedLLM` currently connects to `http://localhost:8000/v1`. The model name must exactly match the vLLM `--served-model-name` value.

## Current Questioner Strategy

`YourQuestioner` follows this control flow:

1. The multimodal model inspects the current candidate and returns a compact JSON action.
2. A clear category mismatch or reliable conflict permits an immediate conclusion of `0`.
3. For a same-category candidate with insufficient evidence, the model asks an atomic yes/no question about a distinctive visible attribute.
4. The Oracle answer is parsed as `yes`, `no`, or `uncertain` and stored with its attribute and candidate provenance.
5. A direct `no` for the current candidate forces conclusion `0`.
6. For a category-only description, two direct `yes` answers for independent attributes with no conflict permit conclusion `1`.
7. Invalid JSON, repeated questions, or unsupported conclusions receive at most one repair attempt before an evidence-gated fallback.

Default memory and retry limits:

| Environment variable | Default |
| --- | ---: |
| `QUESTIONER_MAX_QUESTIONS_PER_CANDIDATE` | `2` |
| `QUESTIONER_MAX_EVIDENCE_ITEMS` | `12` |
| `QUESTIONER_MAX_MEMORY_CHARS` | `6000` |
| `QUESTIONER_API_ATTEMPTS` | `2` |
| `QUESTIONER_API_RETRY_DELAY` | `1` second |

Candidate-local state is cleared when the candidate image changes, while target evidence persists for the complete episode.

## Extension Interfaces

A custom Questioner must inherit `QuestionerInterface` and implement `ask_or_conclude(observation)`.

```python
observation = {
    "image": np.ndarray,  # RGB, H x W x 3
    "answer": str | None,
}

action = {
    "question": str | None,
    "conclusion": 0 | 1 | None,
    "reasoning": str,
}
```

Exactly one of `question` and `conclusion` must be non-`None`. A question may contain at most 300 characters and must concern the hidden target. The evaluator also reads `questions`, `answers`, `n_questions`, and `time_required` from the Questioner.

A custom Oracle must implement:

```python
def ask(*, prompt, images) -> str:
    ...
```

Current clients support exactly one image per multimodal request.

## Dataset Splits

`split_manifest.json` records a reproducible 80/10/10 split created with seed 2026:

| File | Episodes |
| --- | ---: |
| `episodes_train_split.jsonl` | 134 |
| `episodes_val.jsonl` | 17 |
| `episodes_test.jsonl` | 16 |

Important: `eval_model.py --split train` currently reads `episodes_train.jsonl`, not `episodes_train_split.jsonl`. Do not mix the two files or overwrite the original dataset when split isolation matters.

## Validation

Run the offline checks after code changes:

```bash
./coin_env/bin/python -m unittest discover -s tests

./coin_env/bin/python -m py_compile \
  env.py Questioner.py Oracle.py utils.py eval_model.py

./coin_env/bin/python eval_model.py --help
```

The tests cover Questioner memory, candidate transitions, evidence gates, repeated questions, output repair, context limits, and transient-error retry classification.

## Troubleshooting

### Images Cannot Be Opened

```bash
./coin_env/bin/python scripts/download_images_mirror.py
```

### Doubao Returns `429 SetLimitExceeded`

Check endpoint rate limits, concurrency limits, account balance, and model quota. The current code retries connection failures and timeouts only; quota and parameter errors are not retried indefinitely.

### Results Are Missing After an Interrupted Run

Results are written only when the current description type completes. Use smaller ranges, run description types separately, or launch with `nohup`. The current version does not automatically merge result shards.

### Detailed Description Modes Have Lower Reward

The fixed two-question budget is useful for weak descriptions but can add unnecessary Oracle cost when the description already contains strong visual details. The technical report discusses this tradeoff and an adaptive-questioning direction.

## Security and Repository Hygiene

- Never commit `.env`, API keys, results, logs, virtual environments, or download caches.
- Do not commit downloaded replacements for image assets unless the task explicitly updates those assets.
- `results/` is generated locally and is not part of the competition source.
- Run `git diff --check` and the offline tests before committing code changes.
