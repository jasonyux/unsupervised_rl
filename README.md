
install verl under the verl/ directory, following https://verl.readthedocs.io/en/latest/start/install.html#install-dependencies


Other dependencies:
```
rouge_score
```


To install with v0 engine support during RL:
```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.9.2
pip install flash-attn==2.7.4.post1
pip install transformers==4.51.1
```


To install with megatron
```bash
MAX_JOBS=32

echo "Notice that TransformerEngine installation can take very long time, please be patient"
NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1
pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2

pip install nvidia-cudnn-cu12==9.8.0.87
```

To install with megatraon and sglang
```bash
cd verl_061
./scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
# https://github.com/triton-inference-server/pytriton/issues/51#issuecomment-3569383834
pip install 'multiprocess==0.70.11'
```