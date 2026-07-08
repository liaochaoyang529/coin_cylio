# Repository Guidelines

## Project Structure & Module Organization

This repository implements an interactive image-matching QA challenge. Core code lives at the repository root:

- `env.py`: Gymnasium environment (`QAEnv`) that runs episodes and calls the oracle.
- `Questioner.py`: interfaces and templates for participant questioners.
- `Oracle.py`: oracle interface definition.
- `utils.py`: Gemini, OpenAI-compatible, and vLLM client helpers.
- `eval_model.py`: local evaluation script and result logging.
- `episodes_train.jsonl`: training episode metadata.
- `images/`: downloaded image assets referenced by the JSONL file.
- `scripts/`: setup and example launch scripts.

Generated outputs should go under `results/`. Local virtual environments such as `coin_env/` and `vllm_env/` should not be committed.

## Build, Test, and Development Commands

Create and activate the local environment:

```bash
uv venv coin_env
source coin_env/bin/activate
```

Install runtime dependencies:

```bash
pip install retrying flask attrs gymnasium colorama accelerate transformers==4.43.1 Pillow opencv-python dotenv qwen-vl-utils huggingface_hub google-genai openai
```

Download images if missing:

```bash
mkdir -p images
hf download --repo-type dataset e-zorzi/images_coin_challenge --local-dir images
```

Run evaluation after implementing `YourQuestioner`:

```bash
mkdir -p results
python eval_model.py 0 70 --description-type category
```

Use `--description-type all` to loop over all description variants.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and clear, descriptive names. Keep interfaces stable: custom questioners must inherit `QuestionerInterface` and implement `ask_or_conclude`; custom oracles must expose `ask(prompt=..., images=...)`. Prefer small helper functions over large prompt-building blocks embedded directly in evaluation loops.

## Testing Guidelines

There is no formal test suite. Validate changes by running a small evaluation slice first:

```bash
python eval_model.py 0 3 --description-type category
```

Check that actions always contain exactly one of `question` or `conclusion`, and include `reasoning` for logging. Confirm output files appear in `results/*.gzip.json`.

## Commit & Pull Request Guidelines

The existing history uses short, descriptive commits such as `Cleaning` and `More fixes`; prefer clearer imperative messages, for example `Implement local VLM questioner`. Pull requests should summarize behavior changes, list tested commands, note required API keys or model servers, and mention any data-path assumptions.

## Security & Configuration Tips

Do not commit API keys, `.env.ml`, downloaded model weights, or large generated results. Gemini keys are expected in `$HOME/.env.ml` as `GEMINI_API_KEY="..."`. For local VLM usage, start the vLLM server separately and ensure `ORACLE_MODEL_ID` matches the served model name.
