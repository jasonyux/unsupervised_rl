set -x
ENGINE=${1:-vllm}
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export VLLM_USE_V1=False
export WANDB_PROJECT=rl_early_experience
export WANDB_RUN_GROUP=alfworld_rl_after_state_pred

N_GPUS=2


### model
# model_path=Qwen/Qwen2.5-7B-Instruct
# model_id=qwen2.5-7b
# model_path=checkpoints/alfworld_wm_sft/qwen2.5-7b-instruct-nspred_sft-solver_all-custnsppromptv1-2048seeds-1.0_1.0p-2epoch-2e-6lr-2048seq/checkpoint-1114
# model_id=qwen2.5-7b-nspred_sft-solver_all-custnsppromptv1-2048seeds-ckpt1114
# model_path=checkpoints/alfworld_state_pred/alfworld-qwen2.5-7b-state_pred-grpo-q8b-embed-g8-solver-all-custnsppromptv1-shortsubp-default_w_refl-step30_hist2-combined0to2048-bsz32-gen512-jdgd128/global_step_172/checkpoint-172-actor
# model_id=qwen2.5-7b-state_pred-grpo-q8b-embed-solver-all-custnsppromptv1-shortsubp-s30h2-2048s-ckpt172
model_path=checkpoints/alfworld_state_pred/alfworld-qwen2.5-7b-state_pred-grpo-q8b-embed-g8-solver-all-custnsppromptv1-shortsubp-default_w_refl-step30_hist2-combined0to2048-bsz32-gen512-jdgd128-ep4/global_step_344/checkpoint-344-actor
model_id=qwen2.5-7b-state_pred-grpo-q8b-embed-solver-all-custnsppromptv1-shortsubp-s30h2-2048s-ckpt344
disable_mm_preprocessor_cache=False  # use True for VL models
disable_cascade_attn=True # use True for A100
save_intermediate_outputs=True


### env
env_name=alfworld/AlfredTWEnv
env_id=alfworld-text
# env_max_steps=50
env_max_steps=15
# env_max_steps=10
# env_text_template_key='default_w_plan_w_refl'
env_text_template_key='default_w_refl'
max_history_length=2


max_prompt_length=2048
response_length=1024


### data and batching
# train_data_size=32
train_data_size=8
val_data_size=128
group_size=8  # default 8
mode="mean_std_norm" # "mean_norm" or "mean_std_norm"

randomize_reset_seed=True
train_dset_fpath=data/verl-agent/text/train_alfworld_$train_data_size.parquet
val_dset_fpath=data/verl-agent/text/test_alfworld_$val_data_size.parquet


#### training hparam
ppo_mini_batch_size=64  # after rollout, ppo updates once per ppo_mini_batch_size effectively
ppo_micro_batch_size_per_gpu=8
log_prob_micro_batch_size_per_gpu=16
lr=1e-6
entropy_coef=0.001
train_wm=False
train_epochs=300


### logging and saving
# save_freq=100
save_freq=150
test_freq=20
log_val_generations=1
val_temperature=1.0


### run
# algo=gigpo
algo=grpo
exp_name=${env_id}s${env_max_steps}_${algo}_prompt${env_text_template_key}_${model_id}_bsz${train_data_size}
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=$disable_mm_preprocessor_cache \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=$disable_cascade_attn \
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
    env.env_name=alfworld/AlfredTWEnv \
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