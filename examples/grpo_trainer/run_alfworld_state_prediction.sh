set -x

export WANDB_PROJECT=rl_early_experience
export WANDB_RUN_GROUP=alfworld_state_pred

echo "USING GPU: $CUDA_VISIBLE_DEVICES"

N_GPUS=2
N_TENSOR_PARALLEL=1


# train_dset_fpath=$HOME/data/gsm8k/train.parquet
# val_dset_fpath=$HOME/data/gsm8k/test.parquet
train_dset_fpath=data/state_pred/alfworld/train_alfworld_react-qwen3-235b-inst-default_w_refl-step30_hist2_temp1.0.parquet
val_dset_fpath=data/state_pred/alfworld/test_alfworld_react-qwen3-235b-inst-default_w_refl-step30_hist2_temp1.0.parquet


train_batch_size=32
group_size=8


max_prompt_length=1500
# max_response_length=512
max_response_length=128


export JUDGE_MODEL_API_BASE=http://127.0.0.1:12500/v1
export JUDGE_MODEL_API_KEY=empty
export JUDGE_MODEL_NAME=Qwen3-235B-A22B-Instruct-2507
export JUDGE_GEN_KWARGS='{"temperature": 0.7, "max_completion_tokens": 2048}'
reward_fn_path=unsupervised_rl/rewards/judge_next_state.py
# reward_manager=naive
# reward_fn_name=compute_score
reward_manager=batch
reward_fn_name=batched_compute_score


model_path=Qwen/Qwen2.5-7B-Instruct
lr=1e-6
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=4
entropy_coef=0
kl_loss_coef=0.001
train_epochs=2

save_freq=200
test_freq=20

val_temperature=1.0
log_val_generations=5


exp_name=debug2-alfworld-state_pred-grpo-g${group_size}-bsz${train_batch_size}-gen${max_response_length}
default_local_dir=/home/checkpoints/$WANDB_RUN_GROUP/$exp_name
mv_dir=checkpoints/$WANDB_RUN_GROUP/
mkdir -p logs/$WANDB_RUN_GROUP
rm -f logs/$WANDB_RUN_GROUP/$exp_name.log


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
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_TENSOR_PARALLEL \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$group_size \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=$reward_manager \
    custom_reward_function.path=$reward_fn_path \
    custom_reward_function.name=$reward_fn_name \
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