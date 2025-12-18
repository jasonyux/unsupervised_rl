set -x
ENGINE=${1:-vllm}
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export WANDB_PROJECT=rl_early_experience
export WANDB_RUN_GROUP=tau2bench_rl_after_state_pred

export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

N_GPUS=4
N_TENSOR_PARALLEL=2
RO_GPU_UTIL=0.7
RO_FREE_CACHE_ENGINE=True


### model
# model_path=Qwen/Qwen2.5-7B-Instruct
# model_id=qwen2.5-7b
# model_path=/data/users/shared/models/Qwen3-8B
# model_id=qwen3-8b
model_path=checkpoints/tau2_state_pred/tau2-qwen3-8b-state_pred-grpo-q8b-embed-v3-t2b-g8-react-alldomains-v2nopanocr-qwen8b-uq235bngpt4.1-slusubp-noempty-nofterminal-slt0.0tdrt0.85-s60h5_3repeats-bsz32-gen8192-jdgd512-trsh0.6-ep2/global_step_148/checkpoint-148-actor
model_id=qwen3-8b-state_pred-embed-v3-t2b-alldomains-v2nopanocr-q8b-uq235bngpt4.1-slusubp-noept-nofterm-slt0.0tdrt0.85-t0.6-ckpt148
disable_mm_preprocessor_cache=False  # use True for VL models
disable_cascade_attn=True # use True for A100
save_intermediate_outputs=True


### env
test_task_set_name=all
test_task_split_name=test
train_task_set_name=all
train_task_split_name=train
# user_llm=gpt-4.1
# user_llm_temperature=0.0
# user_llm_api_base=https://api.openai.com/v1
# user_llm_api_key=${OPENAI_API_KEY}
# user_llm_save_name=gpt4.1
user_llm=hosted_vllm/Qwen3-235B-A22B-Instruct-2507
user_llm_temperature=0.0
# user_llm_api_base=http://blp-wmrlzrmz5-master-1.blp-wmrlzrmz5:12500/v1
user_llm_api_base=http://blp-wmrl6nkbtwl-worker-4.blp-wmrl6nkbtwl:12500/v1
user_llm_api_key=empty
user_llm_max_completion_tokens=512
user_llm_save_name=qwen3-235b-a22b-inst-2507


env_id=tau2bench-$test_task_set_name
env_max_steps=20
env_max_concurrency=8
env_text_template_key='qwen2.5'
# env_text_template_key='qwen2.5_wthink'
max_history_length=5
add_len_penalty=False


max_prompt_length=13332 # 12240 could still go over with h5
response_length=8192


### data and batching
# train_data_size=32
train_data_size=8
val_data_size=64
group_size=8  # default 8
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"

randomize_reset_seed=True
train_dset_fpath=data/verl-agent/text/train_tau2bench_$train_data_size.parquet
val_dset_fpath=data/verl-agent/text/test_tau2bench_$val_data_size.parquet


#### training hparam
ppo_mini_batch_size=64  # after rollout, ppo updates once per ppo_mini_batch_size effectively
# ppo_micro_batch_size_per_gpu=8
# log_prob_micro_batch_size_per_gpu=16
ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=8
lr=1e-6
entropy_coef=0.001
train_wm=False
train_epochs=200


### logging and saving
# save_freq=100
save_freq=100
test_freq=10
log_val_generations=1
val_temperature=0.1


### run
# algo=gigpo
algo=grpo
exp_name=${env_id}-user${user_llm_save_name}-s${env_max_steps}-lp${add_len_penalty}_${algo}_prompt${env_text_template_key}_${model_id}_bsz${train_data_size}
# exp_name=run2-${env_id}s${env_max_steps}_${algo}_prompt${env_text_template_key}_${model_id}_bsz${train_data_size}
# default_local_dir=/home/checkpoints_early_exp/$WANDB_RUN_GROUP/$exp_name
# default_local_dir=checkpoints_early_exp/$WANDB_RUN_GROUP/$exp_name
# default_local_dir=/local2/data/xy2437/verl-agent/checkpoints_early_exp/$WANDB_RUN_GROUP/$exp_name
default_local_dir=/home/checkpoints/$WANDB_RUN_GROUP/$exp_name
mv_dir=checkpoints/$WANDB_RUN_GROUP/
mkdir -p logs/$WANDB_RUN_GROUP
rm -f logs/$WANDB_RUN_GROUP/$exp_name.log

## exit if the output directory already exists
if [ -d $default_local_dir ]; then
    echo "Output directory $default_local_dir already exists. Exiting."
    exit 1
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algo \
    data.train_files=$train_dset_fpath \
    data.val_files=$val_dset_fpath \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coef \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_TENSOR_PARALLEL \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$RO_GPU_UTIL \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=$RO_FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.free_cache_engine=$RO_FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=$disable_mm_preprocessor_cache \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=$disable_cascade_attn \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + response_length)) \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.world_model.enable=$train_wm \
    env.seed=0 \
    env.randomize_reset_seed=$randomize_reset_seed \
    env.max_steps=$env_max_steps \
    env.text_template_key=$env_text_template_key \
    env.max_history_length=$max_history_length \
    env.rollout.n=$group_size \
    env.env_name=tau2bench \
    env.tau2bench.user_llm=$user_llm \
    env.tau2bench.user_llm_args.temperature=$user_llm_temperature \
    env.tau2bench.user_llm_args.api_base=$user_llm_api_base \
    env.tau2bench.user_llm_args.api_key=$user_llm_api_key \
    env.tau2bench.user_llm_args.max_completion_tokens=$user_llm_max_completion_tokens \
    env.tau2bench.test_task_set_name=$test_task_set_name \
    env.tau2bench.test_task_split_name=$test_task_split_name \
    env.tau2bench.task_set_name=$train_task_set_name \
    env.tau2bench.task_split_name=$train_task_split_name \
    env.tau2bench.add_len_penalty=$add_len_penalty \
    env.tau2bench.max_concurrency=$env_max_concurrency \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$train_epochs \
    trainer.val_before_train=True \
    trainer.default_local_dir=$default_local_dir \
    trainer.log_val_generations=$log_val_generations
    # 2>&1 | tee logs/$WANDB_RUN_GROUP/$exp_name.log

python scripts/model_merger_bulk.py merge \
--backend fsdp \
--local_dir $default_local_dir

cp $0 $default_local_dir/train.sh

mv $default_local_dir $mv_dir