#!/bin/bash
# 服务器环境搭建：uv + venv + jax(cuda13) + 运行时依赖 + 项目安装 + 验证
set -e
echo "=== [1/5] install uv ==="
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
    pip3 install --user --break-system-packages -q uv
  fi
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

echo "=== [2/5] create venv ==="
cd ~/HyperscaleES
uv venv .venv --python 3.12

echo "=== [3/5] install jax(cuda13) + runtime deps ==="
uv pip install --python .venv/bin/python "jax[cuda13]" optax numpy \
  gymnax distrax flax gymnasium seaborn chex einops huggingface_hub tokenizers \
  importlib_resources pyrwkv-tokenizer transformers datasets reasoning-gym math-verify

echo "=== [4/5] install project (no-deps) ==="
uv pip install --python .venv/bin/python -e . --no-deps

echo "=== [5/5] verify ==="
.venv/bin/python -c "import hyperscalees; from hyperscalees.models.snn import SNNModel; import jax; print('imports ok'); print('jax', jax.__version__); print('devices:', jax.devices())"

echo "SETUP_DONE"
