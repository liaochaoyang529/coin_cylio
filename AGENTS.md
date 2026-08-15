# Repository Guidelines

## Purpose and Runtime Model

This repository evaluates a multimodal questioner in an interactive image-matching game. For each episode, the questioner receives a textual description of a hidden target and one candidate image at a time. It must either ask the oracle a concise question about the hidden target or conclude whether the candidate matches it.

Run commands from the repository root because episode files contain repository-relative image paths. A correct conclusion earns `+10`, an incorrect conclusion earns `-10`, and each oracle question costs `-1`. An incorrect conclusion ends the episode; a correct conclusion advances to the next candidate. `QAEnv` also truncates at 60 steps or 600 seconds by default.

## Repository Map

- `env.py`: Gymnasium environment, action validation, reward/termination logic, image loading, and oracle calls.
- `Questioner.py`: `QuestionerInterface`, an unfinished local-vLLM template, and the active `YourQuestioner` baseline.
- `Oracle.py`: minimal `OracleInterface` contract.
- `utils.py`: Gemini, OpenAI-compatible/ModelArk, and local vLLM clients plus image encoding.
- `eval_model.py`: CLI entry point, provider selection, episode loop, summaries, and gzip JSON result writing.
- `episodes_train.jsonl`, `episodes_val.jsonl`, `episodes_test.jsonl`: datasets selected by `--split`.
- `episodes_train_split.jsonl`, `split_manifest.json`: reproducible 80/10/10 split artifacts created by `scripts/split_episodes.py`.
- `images/`: image assets referenced by episode JSONL records.
- `scripts/install.sh`: creates `coin_env/` and installs the evaluation dependencies.
- `scripts/download_images_mirror.py`: downloads and validates required images through an HF-compatible mirror.
- `scripts/launch_qwen35_vllm.sh`: current local Qwen vLLM launcher.
- `results/`: generated evaluation output; `logs/`, `downloads/`, `*.orig`, and virtual environments are local artifacts, not source.

Do not modify or commit model wheels, downloaded images, API credentials, logs, results, or virtual environments unless the task explicitly concerns them. The worktree may already contain user changes; inspect it and preserve unrelated edits.

## Required Interfaces

Keep these contracts stable unless the task explicitly changes the protocol:

- A questioner inherits `QuestionerInterface` and implements `ask_or_conclude(observation)`.
- `observation` is `{"image": np.ndarray[H,W,3], "answer": str | None}`. The image is RGB; an answer is present only after an oracle question.
- Every returned action must contain `question`, `conclusion`, and `reasoning`.
- Exactly one of `question` and `conclusion` must be non-`None`.
- A conclusion is `0` (not a match) or `1` (match), not an arbitrary truthy value.
- A question is a non-empty string of at most 300 characters and should concern the hidden target, not the visible candidate.
- Track asked questions in `self.questions`, answers in `self.answers`, count questions in `self.n_questions`, and model latency in `self.time_required`; `eval_model.py` reads these fields directly and calls `add_answer()` and `reset_time()`.
- An oracle exposes `ask(*, prompt=..., images=...) -> str`. Current clients support exactly one image per multimodal request.

When changing the loop, remember that one episode contains multiple candidate decisions. A successful conclusion can replace the candidate image without terminating the episode.

## Setup and Assets

Create the main environment with the maintained installer:

```bash
bash scripts/install.sh
```

Download assets with Hugging Face:

```bash
mkdir -p images
hf download --repo-type dataset e-zorzi/images_coin_challenge \
  --local-dir images --force-download
```

If Git LFS pointers remain or Hugging Face is slow, use the validating mirror downloader:

```bash
./coin_env/bin/python scripts/download_images_mirror.py
```

Do not assume a `.png` path is a real image: `env.py` intentionally rejects Git LFS pointer files.

## Model Configuration

`load_api_keys()` loads repository `.env` first and falls back to `$HOME/.env.ml`. Never commit either file or print key values. Start from `.env.example` for ModelArk/Doubao configuration.

The defaults are:

- Questioner: `QUESTIONER_PROVIDER=doubao`, using `QUESTIONER_MODEL_ID` when set, otherwise `ARK_MODEL_ID`.
- Oracle: `--oracle-provider doubao`, or `ORACLE_PROVIDER` when the CLI option is omitted.
- Doubao requires `ARK_API_KEY` and an image-capable `ARK_MODEL_ID`; optional controls include `ARK_BASE_URL`, `ARK_TIMEOUT_SECONDS`, `ARK_MAX_OUTPUT_LENGTH`, and `ARK_THINKING`.
- Gemini requires `GEMINI_API_KEY`; select it with `QUESTIONER_PROVIDER=gemini` and/or `--oracle-provider gemini`.
- Local questioners require `QUESTIONER_PROVIDER=local` and `QUESTIONER_MODEL_ID`. Local oracles require `--oracle-provider local` and `ORACLE_MODEL_ID`.

`ClientBasedLLM` currently connects to `http://localhost:8000/v1`. Match the vLLM served model name exactly. `QUESTIONER_LOCAL` and `--local` are compatibility switches; prefer the explicit provider settings above.

Questioner memory is bounded by `QUESTIONER_MAX_QUESTIONS_PER_CANDIDATE` (default `2`), `QUESTIONER_MAX_EVIDENCE_ITEMS` (default `12`), and `QUESTIONER_MAX_MEMORY_CHARS` (default `6000`). Keep the memory-character budget conservative for local models with a 4096-token context window.

Questioner model calls retry transient connection/timeouts according to `QUESTIONER_API_ATTEMPTS` (default `2`) and `QUESTIONER_API_RETRY_DELAY` (default `1` second). Non-transient API errors are not retried.

The two `scripts/launch_eval_*_oracle.sh` files are legacy examples and still pass the removed `--task-type` option. Do not use them as authoritative commands without updating them. Check the live CLI instead:

```bash
./coin_env/bin/python eval_model.py --help
```

## Evaluation and Validation

Use a small slice before any longer or paid run:

```bash
./coin_env/bin/python eval_model.py 0 3 \
  --split train --description-type category
```

The positional range is half-open: `start_idx` is included and `end_idx` is excluded. Supported description types are `category`, `color`, `context`, `color_context_feature`, `color_feature`, and `color_context`; `all` runs all six. A complete current test split is:

```bash
./coin_env/bin/python eval_model.py 0 16 \
  --split test --description-type category
```

Results are written to:

```text
results/<QuestionerClass>_<description>_<split>_<start>_<end>.gzip.json
```

There is no formal test suite. For code changes, at minimum:

```bash
./coin_env/bin/python -m py_compile \
  env.py Questioner.py Oracle.py utils.py eval_model.py
./coin_env/bin/python eval_model.py --help
```

Run the 0:3 smoke evaluation only when the configured model endpoints and credentials are available; it makes external/model calls and may incur cost. Verify that completed runs print aggregate metrics and that the gzip JSON contains aligned per-episode lists for actions, answers, reasoning, counts, rewards, and timing.

`scripts/split_episodes.py` preserves `episodes_train.jsonl` and writes the training portion to `episodes_train_split.jsonl`. However, `eval_model.py --split train` currently reads `episodes_train.jsonl`, not `episodes_train_split.jsonl`. Do not claim split isolation or overwrite datasets silently; change the evaluator explicitly if a task requires the split training file.

## Coding and Change Discipline

Use Python with 4-space indentation, descriptive names, and small helpers for parsing, prompt construction, and provider setup. Prefer structured JSON parsing for model output over brittle string slicing. Keep retries and timeouts bounded, and preserve useful errors around missing credentials, unavailable servers, malformed actions, and Git LFS assets.

Keep changes narrowly scoped. Do not refactor provider code, prompts, datasets, or evaluation semantics as incidental cleanup. When behavior changes, update `README.MD`, `.env.example`, and launch scripts only when they are in scope and affected. PR descriptions should state behavioral changes, exact validation commands, required providers/credentials, and dataset-path assumptions. Use short imperative commit messages, for example `Clarify local VLM configuration`.
