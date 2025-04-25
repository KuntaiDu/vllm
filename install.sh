export VLLM_COMMIT=dc1b4a6f1300003ae27f033afbdff5e2683721ce # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
uv pip install --editable . -v