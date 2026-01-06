#!/bin/bash
# 실행 커맨드 예시 모음
# 사용법: bash run_commands.sh 또는 각 커맨드를 직접 실행

# ============================================================================
# 기본 실행 (COLMAP 데이터셋)
# ============================================================================

# Single GPU training - Default strategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

# Single GPU training - MCMC strategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc

# ============================================================================
# NeRF 데이터셋 실행
# ============================================================================

# NeRF 데이터셋 - Default strategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background

# NeRF 데이터셋 - MCMC strategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background

# ============================================================================
# COLMAP 데이터셋 실행 (커스텀 경로)
# ============================================================================

# COLMAP 데이터셋 - Default strategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type colmap \
    --data_dir /Bean/data/gwangjin/2025/3dgs/garden

# ============================================================================
# Distributed training (Multi-GPU)
# ============================================================================

# 4 GPUs - Default strategy (batch size 4x, steps 4x 감소)
CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --steps_scaler 0.25

# 4 GPUs - MCMC strategy
CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py mcmc \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --steps_scaler 0.25

# ============================================================================
# 커스텀 설정 예시
# ============================================================================

# 커스텀 result_dir 지정
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --result_dir /Bean/log/gwangjin/2025/gsplat/custom_experiment \
    --white_background

# 다운샘플링 팩터 변경
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --data_factor 2 \
    --white_background

# 최대 스텝 수 변경
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --max_steps 15000 \
    --white_background

# ============================================================================
# Evaluation only (체크포인트에서 평가)
# ============================================================================

# 체크포인트에서 평가만 실행
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --ckpt /Bean/log/gwangjin/2025/gsplat/baseline/lego_type_nerf_factor_1_whitebg_norm/ckpts/ckpt_30000_rank0.pt \
    --white_background

# ============================================================================
# TensorBoard 및 WandB 사용
# ============================================================================

# TensorBoard 활성화
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --use_tensorboard \
    --tb_every 100 \
    --tb_save_image

# WandB 활성화 (wandb가 설치되어 있어야 함)
# CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
#     --dataset_type nerf \
#     --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
#     --white_background \
#     --use_wandb \
#     --wandb_project gsplat \
#     --wandb_run_name lego_experiment

# ============================================================================
# 실험용 설정
# ============================================================================

# Patch training (랜덤 크롭)
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --patch_size 256

# Camera optimization 활성화
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --pose_opt \
    --pose_opt_lr 1e-5

# Appearance optimization 활성화
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --app_opt \
    --app_opt_lr 1e-3

# ============================================================================
# Hierarchical Training (Multigrid Methods)
# ============================================================================

# Hierarchical trainer (always renders at max LOD)
CUDA_VISIBLE_DEVICES=0 python hierarchical_trainer.py \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/ficus \
    --data_factor 1 \
    --max_steps 30000 \
    --batch_size 1 \
    --white_background

# ============================================================================
# Inverted F-Cycle Training
# ============================================================================

# Inverted F-cycle - NeRF dataset (기본 설정)
CUDA_VISIBLE_DEVICES=0 python inverted_fcycle_trainer.py \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/ficus \
    --data_factor 1 \
    --max_steps 30000 \
    --batch_size 1 \
    --white_background

# Inverted F-cycle - COLMAP dataset
CUDA_VISIBLE_DEVICES=0 python inverted_fcycle_trainer.py \
    --dataset_type colmap \
    --data_dir /Bean/data/gwangjin/2025/3dgs/garden \
    --data_factor 4 \
    --max_steps 30000 \
    --batch_size 1

# Inverted F-cycle - 커스텀 smoothing/solving steps
CUDA_VISIBLE_DEVICES=0 python inverted_fcycle_trainer.py \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/ficus \
    --data_factor 1 \
    --max_steps 30000 \
    --batch_size 1 \
    --white_background \
    --smoothing_steps 10 \
    --solving_steps 200

# ============================================================================
# V-Cycle Training
# ============================================================================

# V-cycle - NeRF dataset (smoothing/solving steps = 100 고정)
CUDA_VISIBLE_DEVICES=0 python vcycle_trainer.py \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/ficus \
    --data_factor 1 \
    --max_steps 30000 \
    --batch_size 1 \
    --white_background

# V-cycle - COLMAP dataset
CUDA_VISIBLE_DEVICES=0 python vcycle_trainer.py \
    --dataset_type colmap \
    --data_dir /Bean/data/gwangjin/2025/3dgs/garden \
    --data_factor 4 \
    --max_steps 30000 \
    --batch_size 1

# V-cycle - 커스텀 설정
CUDA_VISIBLE_DEVICES=0 python vcycle_trainer.py \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/ficus \
    --data_factor 1 \
    --max_steps 30000 \
    --batch_size 1 \
    --white_background \
    --opacity_reg 0.001 \
    --max_level 8

