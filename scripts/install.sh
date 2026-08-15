#!/usr/bin/env bash
set -euo pipefail

python -m venv coin_env
coin_env/bin/python -m ensurepip --upgrade
coin_env/bin/python -m pip install \
  numpy attrs gymnasium colorama Pillow python-dotenv openai retrying huggingface_hub

echo "Install complete. Copy .env.example to .env and configure ARK_API_KEY and ARK_MODEL_ID."
