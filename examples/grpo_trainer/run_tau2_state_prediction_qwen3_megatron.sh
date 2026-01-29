set -x

# for rerun the task
# pkill -9 sglang
# sleep 3
ray stop --force
pkill -9 ray
# pkill -9 python
sleep 3
# pkill -9 ray
# pkill -9 python
# pkill -9 redis

export WANDB_PROJECT=rl_early_experience
export WANDB_RUN_GROUP=tau2_state_pred-megatron

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export VLLM_USE_V1=1

echo "USING GPU: $CUDA_VISIBLE_DEVICES"
export NCCL_TIMEOUT=1800


N_GPUS=8
N_TENSOR_PARALLEL=4
N_EXPERT_MODEL_PARALLEL=8
N_EXPERT_TENSOR_PARALLEL=1
N_PIPELINE_PARALLEL=1
RO_GPU_UTIL=0.6
# RO_FREE_CACHE_ENGINE=False
RO_FREE_CACHE_ENGINE=True
# N_GPUS=8
# N_TENSOR_PARALLEL=2


# train_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-q30b-a3b-thinking-2507-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/train_noempty_nofterminal_samplt0.0trainedrt0.9.parquet
# # train_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-q30b-a3b-thinking-2507-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/test_noempty_nofterminal_512.parquet
# val_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-q30b-a3b-thinking-2507-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/test_noempty_nofterminal_512.parquet
# dset_short_name=react-alldomains-v2nopanocr-q30b-a3b-tk-uq235bngpt4.1-noept-nofterm-slt0.0tdrt0.9-s60h5_3repeats

# train_dset_fpath=/data/users/baolinpeng/xy/unsupervised_rl/data/state_pred/tau2bench/react-alldomains-v2nopanocr-q30b-a3b-thinking-2507-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/train_q30ba3b-gpt4.1-gptoss120b-noempty_nofterminal_samplt0.0trainedrt0.9.parquet
# train_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-q30b-a3b-thinking-2507-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/train_q30ba3b-gpt4.1-gptoss120b-noempty_nofterminal_samplt0.0trainedrt0.9_5120.parquet
# val_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-q30b-a3b-thinking-2507-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/test_noempty_nofterminal_512.parquet
# dset_short_name=react-alldomains-v2nopanocr-5120-q30ba3b-gpt4.1-gptoss120b-noept-nofterm-slt0.0tdrt0.9-s60h5_3repeats
train_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-q30b-a3b-thinking-2507-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/train_q30ba3b-q245ba22b-noempty_nofterminal_samplt0.0trainedrt0.9.parquet
val_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-q30b-a3b-thinking-2507-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/test_noempty_nofterminal_64.parquet
dset_short_name=react-alldomains-v2nopanocr-q30ba3b-q235ba22b-noept-nofterm-slt0.0tdrt0.9-s60h5_3repeats


# train_batch_size=16
train_batch_size=32
group_size=8


max_prompt_length=10240
# max_response_length=1024
max_response_length=12240  # qwen3-8b model


JUDGE_EMBED_MODEL_API_BASE=http://blp-wmrl6nkbtwl-worker-4.blp-wmrl6nkbtwl:12200/v1
JUDGE_EMBED_MODEL_API_KEY=empty
JUDGE_EMBED_MODEL_NAME=qwen-embedding-8b
JUDGE_GEN_KWARGS='{}'
JUDGE_MAX_TOKEN_TO_JUDGE=512
JUDGE_EMBED_QUERY_TEMPLATE_NAME=v1  # use v2 when we doing wsucc dset
JUDGE_API_CONCURRENCY=4
JUDGE_THRESHOLD=0.6  # default 0.8
reward_fn_path=unsupervised_rl/rewards/embed_next_state_v3_tau2bench.py
reward_short_name=q8b-embed-v3-t2b

reward_manager=batch
reward_fn_name=batched_compute_score


model_path=/data/users/shared/models/Qwen3-30B-A3B-Thinking-2507
has_think_in_prompt=True
DIST_CKPT_PATH=/home/checkpoints/tmp_Qwen3-30B-A3B-Thinking-2507_dist
# cd verl
# python scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH
# cd ../

model_id=qwen3-30b-a3b
lr=5e-7
offload=True
max_token_per_gpu=$(($max_prompt_length + $max_response_length + 512))  # some buffer
ppo_mini_batch_size=32
entropy_coef=0.0
kl_loss_coef=0.05
train_epochs=1

save_freq=500 # about 1 epoch
test_freq=20

val_temperature=1.0
log_val_generations=10


exp_name=tau2-${model_id}-state_pred-grpo-${reward_short_name}-g${group_size}-${dset_short_name}-lr${lr}kl${kl_loss_coef}-bsz${train_batch_size}ppo${ppo_mini_batch_size}-gen${max_response_length}-jdgd${JUDGE_MAX_TOKEN_TO_JUDGE}-trsh${JUDGE_THRESHOLD}-ep${train_epochs}
# exp_name=debug-tau2-${model_id}-state_pred-grpo-${reward_short_name}-g${group_size}-${dset_short_name}-bsz${train_batch_size}-gen${max_response_length}-jdgd${JUDGE_MAX_TOKEN_TO_JUDGE}-trsh${JUDGE_THRESHOLD}-ep${train_epochs}
default_local_dir=/home/checkpoints/$WANDB_RUN_GROUP/$exp_name
mv_dir=checkpoints/$WANDB_RUN_GROUP/


mkdir -p logs/$WANDB_RUN_GROUP
rm -f logs/$WANDB_RUN_GROUP/$exp_name.log

## exit if the output directory already exists
if [ -d $default_local_dir ]; then
    echo "Output directory $default_local_dir already exists. Exiting."
    exit 1
fi


python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=$train_dset_fpath \
    data.val_files=$val_dset_fpath \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.shuffle=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_per_gpu \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_token_per_gpu \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$N_TENSOR_PARALLEL \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$N_PIPELINE_PARALLEL \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$N_EXPERT_MODEL_PARALLEL \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$N_EXPERT_TENSOR_PARALLEL \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coef \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_TENSOR_PARALLEL \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=$RO_GPU_UTIL \
    actor_rollout_ref.rollout.n=$group_size \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=$RO_FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.free_cache_engine=$RO_FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(($max_prompt_length+$max_response_length)) \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$N_TENSOR_PARALLEL \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$N_PIPELINE_PARALLEL \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$N_EXPERT_MODEL_PARALLEL \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$N_EXPERT_TENSOR_PARALLEL \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=$reward_manager \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=$reward_fn_name \
    +custom_reward_function.reward_kwargs.judge_api_base=$JUDGE_EMBED_MODEL_API_BASE \
    +custom_reward_function.reward_kwargs.judge_api_key=$JUDGE_EMBED_MODEL_API_KEY \
    +custom_reward_function.reward_kwargs.judge_embed_model_name=$JUDGE_EMBED_MODEL_NAME \
    +custom_reward_function.reward_kwargs.embed_query_template_name=$JUDGE_EMBED_QUERY_TEMPLATE_NAME \
    +custom_reward_function.reward_kwargs.max_token_to_judge=$JUDGE_MAX_TOKEN_TO_JUDGE \
    +custom_reward_function.reward_kwargs.has_think_in_prompt=$has_think_in_prompt \
    +custom_reward_function.reward_kwargs.judge_api_concurrency=$JUDGE_API_CONCURRENCY \
    +custom_reward_function.reward_kwargs.threshold=$JUDGE_THRESHOLD \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$train_epochs \
    trainer.default_local_dir=$default_local_dir \
    actor_rollout_ref.actor.checkpoint.save_contents="['model', 'extra']" \
    trainer.log_val_generations=$log_val_generations \
    ray_init.num_cpus=32


python scripts/model_merger_bulk.py merge \
--backend megatron \
--local_dir $default_local_dir

cp $0 $default_local_dir/train.sh

mv $default_local_dir $mv_dir