set -x

export WANDB_PROJECT=rl_early_experience
export WANDB_RUN_GROUP=alfworld_state_pred

echo "USING GPU: $CUDA_VISIBLE_DEVICES"

N_GPUS=2
N_TENSOR_PARALLEL=1
# N_GPUS=8
# N_TENSOR_PARALLEL=2


# train_dset_fpath=data/state_pred/alfworld/train_alfworld_react-qwen3-235b-inst-default_w_refl-step30_hist2_temp1.0.parquet
# val_dset_fpath=data/state_pred/alfworld/test_alfworld_react-qwen3-235b-inst-default_w_refl-step30_hist2_temp1.0.parquet
# train_dset_fpath=data/state_pred/alfworld/solver-all-default_w_refl-step30_hist2/train.parquet
# val_dset_fpath=data/state_pred/alfworld/solver-all-default_w_refl-step30_hist2/test.parquet
# dset_short_name=solver-all-default_w_refl-step30_hist2
# train_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1-default_w_refl-step30_hist2/train.parquet
# val_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1-default_w_refl-step30_hist2/test.parquet
# dset_short_name=solver-all-custnsppromptv1-default_w_refl-step30_hist2
# train_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1-default_w_refl-step30_hist2-combined0to2048/train.parquet
# val_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1-default_w_refl-step30_hist2-combined0to2048/test.parquet
# dset_short_name=solver-all-custnsppromptv1-default_w_refl-step30_hist2-combined0to2048
# train_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1_w_choices-default_w_refl-step30_hist2-combined0to2048/train.parquet
# val_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1_w_choices-default_w_refl-step30_hist2-combined0to2048/test.parquet
# dset_short_name=solver-all-custnsppromptv1_w_choices-default_w_refl-step30_hist2-combined0to2048
train_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1-default_w_refl-step30_hist2-combined0to2048/train_shortsubp.parquet
val_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1-default_w_refl-step30_hist2-combined0to2048/test_shortsubp.parquet
dset_short_name=solver-all-custnsppromptv1-shortsubp-default_w_refl-step30_hist2-combined0to2048
# train_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1-default_w_refl-step30_hist5-combined0to2048/train.parquet
# val_dset_fpath=data/state_pred/alfworld/solver-all-custnsppromptv1-default_w_refl-step30_hist5-combined0to2048/test.parquet
# dset_short_name=solver-all-custnsppromptv1-default_w_refl-step30_hist5-combined0to2048


train_batch_size=32
group_size=8


max_prompt_length=1500
max_response_length=512
# max_response_length=128


# export JUDGE_MODEL_API_BASE=http://127.0.0.1:12500/v1
# export JUDGE_MODEL_API_KEY=empty
# export JUDGE_MODEL_NAME=Qwen3-235B-A22B-Instruct-2507
# export JUDGE_GEN_KWARGS='{"temperature": 0.7, "max_completion_tokens": 2048}'
# reward_fn_path=unsupervised_rl/rewards/judge_next_state.py
# reward_short_name=q235b-judge
export JUDGE_EMBED_MODEL_API_BASE=http://adaptation.cs.columbia.edu:55520/v1
export JUDGE_EMBED_MODEL_API_KEY=empty
export JUDGE_EMBED_MODEL_NAME=qwen-embedding-8b
export JUDGE_GEN_KWARGS='{}'
export JUDGE_MAX_TOKEN_TO_JUDGE=128
reward_fn_path=unsupervised_rl/rewards/embed_next_state.py
reward_short_name=q8b-embed

# reward_manager=naive
# reward_fn_name=compute_score
reward_manager=batch
reward_fn_name=batched_compute_score


model_path=Qwen/Qwen2.5-7B-Instruct
model_id=qwen2.5-7b
# model_path=Qwen/Qwen2.5-32B-Instruct
# model_id=qwen2.5-32b
lr=1e-6
offload_stuff=False
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=4
# ppo_mini_batch_size=16
# ppo_micro_batch_size_per_gpu=1
# log_prob_micro_batch_size_per_gpu=1
# offload_stuff=True
entropy_coef=0
kl_loss_coef=0.001
# train_epochs=2
train_epochs=4

save_freq=250
test_freq=20

val_temperature=1.0
log_val_generations=10


exp_name=alfworld-${model_id}-state_pred-grpo-${reward_short_name}-g${group_size}-${dset_short_name}-bsz${train_batch_size}-gen${max_response_length}-jdgd${JUDGE_MAX_TOKEN_TO_JUDGE}-ep${train_epochs}
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=$group_size \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=$offload_stuff \
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