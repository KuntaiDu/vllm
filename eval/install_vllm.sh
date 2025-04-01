# copy-paste these commands and execute them under vllm root folder.
export VLLM_COMMIT=9b459eca88b4953586391c14d183574c4d21fca3
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .