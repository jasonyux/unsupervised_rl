
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