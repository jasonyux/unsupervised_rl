set -x

export WANDB_PROJECT=rl_early_experience
export WANDB_RUN_GROUP=tau2_state_pred

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

echo "USING GPU: $CUDA_VISIBLE_DEVICES"

N_GPUS=4
N_TENSOR_PARALLEL=1
RO_GPU_UTIL=0.7
RO_FREE_CACHE_ENGINE=True
# N_GPUS=8
# N_TENSOR_PARALLEL=2

# train_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-qwen8b-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/train_shortlongusersubp_noempty_nofterminal_samplt0.0trainedrt0.85.parquet
# val_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-qwen8b-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/test_no_empty_no_fterminal_512.parquet
# dset_short_name=react-alldomains-v2nopanocr-qwen8b-uq235bngpt4.1-slusubp-noept-nofterm-slt0.0tdrt0.85-s60h5_3repeats
train_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-qwen8b-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/train_longusersubp_noempty_nofterminal_samplt0.0trainedrt0.85.parquet
val_dset_fpath=data/state_pred/tau2bench/react-alldomains-v2nopanocr-qwen8b-userqwen235bngpt4.1-tmpqwen2.5-step60_h5_3repeats/test_no_empty_no_fterminal_512.parquet
dset_short_name=react-alldomains-v2nopanocr-qwen8b-uq235bngpt4.1-lusubp-noept-nofterm-slt0.0tdrt0.85-s60h5_3repeats




train_batch_size=32
group_size=8


max_prompt_length=10240
# max_response_length=1024
max_response_length=8192  # qwen3-8b model


# export JUDGE_MODEL_API_BASE=http://127.0.0.1:12500/v1
# export JUDGE_MODEL_API_KEY=empty
# export JUDGE_MODEL_NAME=Qwen3-235B-A22B-Instruct-2507
# export JUDGE_GEN_KWARGS='{"temperature": 0.7, "max_completion_tokens": 2048}'
# reward_fn_path=unsupervised_rl/rewards/judge_next_state.py
# reward_short_name=q235b-judge
# JUDGE_EMBED_MODEL_API_BASE=http://blp-wmrl6nkbtwl-worker-4.blp-wmrl6nkbtwl:12200/v1
JUDGE_EMBED_MODEL_API_BASE=http://blp-wmrl6nkbtwl-worker-4.blp-wmrl6nkbtwl:12201/v1
JUDGE_EMBED_MODEL_API_KEY=empty
JUDGE_EMBED_MODEL_NAME=qwen-embedding-8b
JUDGE_GEN_KWARGS='{}'
JUDGE_MAX_TOKEN_TO_JUDGE=512
JUDGE_EMBED_QUERY_TEMPLATE_NAME=v1  # use v2 when we doing wsucc dset
JUDGE_API_CONCURRENCY=4
JUDGE_THRESHOLD=0.6  # default 0.8
# reward_fn_path=unsupervised_rl/rewards/embed_next_state.py
# reward_short_name=q8b-embed
# reward_fn_path=unsupervised_rl/rewards/embed_next_state_v3.py
# reward_short_name=q8b-embed-v3
reward_fn_path=unsupervised_rl/rewards/embed_next_state_v3_tau2bench.py
reward_short_name=q8b-embed-v3-t2b

# reward_manager=naive
# reward_fn_name=compute_score
reward_manager=batch
reward_fn_name=batched_compute_score


# model_path=Qwen/Qwen2.5-7B-Instruct
# model_id=qwen2.5-7b
# model_path=checkpoints/tau2_wm_sft/qwen2.5-7b-instruct-nspred_sft-qwen7b-usergpt4.1-plus-userqwen235b-shortsubp-longsubp-noempty-samp0.0r-nothink-1.0p-2epoch-2e-6lr-10240seq/checkpoint-300
# model_id=qwen2.5-7b-nspred-sft-qwen7b-usergpt4.1nqwen235b-nothink-ckpt300
# model_path=checkpoints/tau2_wm_sft/qwen2.5-7b-instruct-nspred_sft-qwen7b-inst-usergpt4.1-tmpqwen2.5-step60_h5_3repeats-longsubp-noempty-nothink-1.0p-2epoch-2e-6lr-10240seq/checkpoint-164
# model_id=qwen2.5-7b-nspred-sft-qwen7b-inst-usergpt4.1-tmpqwen2.5-ckpt164
# model_path=Qwen/Qwen3-8B
model_path=/data/users/shared/models/Qwen3-8B
model_id=qwen3-8b
lr=1e-6
offload_stuff=False
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=2
log_prob_micro_batch_size_per_gpu=2
# ppo_mini_batch_size=16
# ppo_micro_batch_size_per_gpu=1
# log_prob_micro_batch_size_per_gpu=1
# offload_stuff=True
entropy_coef=0.001
kl_loss_coef=0.001
train_epochs=2
# train_epochs=1
# train_epochs=4

save_freq=500
test_freq=20

val_temperature=1.0
log_val_generations=10


exp_name=tau2-${model_id}-state_pred-grpo-${reward_short_name}-g${group_size}-${dset_short_name}-bsz${train_batch_size}-gen${max_response_length}-jdgd${JUDGE_MAX_TOKEN_TO_JUDGE}-trsh${JUDGE_THRESHOLD}-ep${train_epochs}
default_local_dir=/home/checkpoints/$WANDB_RUN_GROUP/$exp_name
mv_dir=checkpoints/$WANDB_RUN_GROUP/


mkdir -p logs/$WANDB_RUN_GROUP
rm -f logs/$WANDB_RUN_GROUP/$exp_name.log

## exit if the output directory already exists
if [ -d $default_local_dir ]; then
    echo "Output directory $default_local_dir already exists. Exiting."
    exit 1
fi


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_dset_fpath \
    data.val_files=$val_dset_fpath \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coef \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload_stuff \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload_stuff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_TENSOR_PARALLEL \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$RO_GPU_UTIL \
    actor_rollout_ref.rollout.n=$group_size \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=$RO_FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.free_cache_engine=$RO_FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=$offload_stuff \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=$reward_manager \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=$reward_fn_name \
    +custom_reward_function.reward_kwargs.judge_api_base=$JUDGE_EMBED_MODEL_API_BASE \
    +custom_reward_function.reward_kwargs.judge_api_key=$JUDGE_EMBED_MODEL_API_KEY \
    +custom_reward_function.reward_kwargs.judge_embed_model_name=$JUDGE_EMBED_MODEL_NAME \
    +custom_reward_function.reward_kwargs.embed_query_template_name=$JUDGE_EMBED_QUERY_TEMPLATE_NAME \
    +custom_reward_function.reward_kwargs.max_token_to_judge=$JUDGE_MAX_TOKEN_TO_JUDGE \
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
    trainer.log_val_generations=$log_val_generations \
    ray_init.num_cpus=32


python scripts/model_merger_bulk.py merge \
--backend fsdp \
--local_dir $default_local_dir

cp $0 $default_local_dir/train.sh

mv $default_local_dir $mv_dir