# Simple Trainer 실행 커맨드 가이드

## 기본 사용법

### 1. 기본 실행 (COLMAP 데이터셋)

```bash
# Single GPU - Default strategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

# Single GPU - MCMC strategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc
```

### 2. NeRF 데이터셋 실행

```bash
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
```

### 3. COLMAP 데이터셋 실행

```bash
# COLMAP 데이터셋 - Default strategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type colmap \
    --data_dir /Bean/data/gwangjin/2025/3dgs/garden
```

## Multi-GPU 실행

```bash
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
```

## 커스텀 설정

### Result 디렉토리 지정

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --result_dir /Bean/log/gwangjin/2025/gsplat/custom_experiment \
    --white_background
```

### 다운샘플링 팩터 변경

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --data_factor 2 \
    --white_background
```

### 최대 스텝 수 변경

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --max_steps 15000 \
    --white_background
```

## Evaluation Only

체크포인트에서 평가만 실행:

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --ckpt /Bean/log/gwangjin/2025/gsplat/baseline/lego_type_nerf_factor_1_whitebg_norm/ckpts/ckpt_30000_rank0.pt \
    --white_background
```

## 로깅 옵션

### TensorBoard

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --use_tensorboard \
    --tb_every 100 \
    --tb_save_image
```

### WandB (설치 필요: `pip install wandb`)

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --use_wandb \
    --wandb_project gsplat \
    --wandb_run_name lego_experiment
```

## 실험용 설정

### Patch Training (랜덤 크롭)

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --patch_size 256
```

### Camera Optimization

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --pose_opt \
    --pose_opt_lr 1e-5
```

### Appearance Optimization

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
    --dataset_type nerf \
    --data_dir /Bean/data/gwangjin/2025/3dgs/lego \
    --white_background \
    --app_opt \
    --app_opt_lr 1e-3
```

## 주요 옵션 설명

- `--dataset_type`: 데이터셋 타입 (`colmap` 또는 `nerf`)
- `--data_dir`: 데이터셋 디렉토리 경로
- `--result_dir`: 결과 저장 디렉토리 (자동 생성되지만 수동 지정 가능)
- `--white_background`: NeRF 데이터셋에서 RGBA 이미지의 배경을 흰색으로 설정
- `--data_factor`: 이미지 다운샘플링 팩터
- `--max_steps`: 최대 학습 스텝 수
- `--steps_scaler`: 스텝 수 스케일링 팩터 (Multi-GPU 시 사용)
- `--use_tensorboard`: TensorBoard 로깅 활성화
- `--use_wandb`: WandB 로깅 활성화
- `--pose_opt`: 카메라 포즈 최적화 활성화
- `--app_opt`: Appearance 최적화 활성화

## 자동 생성되는 Result 디렉토리

Result 디렉토리는 자동으로 생성되며, 다음 형식을 따릅니다:

```
/Bean/log/gwangjin/2025/gsplat/baseline/{dataset_name}_{settings_str}
```

예시:
- `lego_type_nerf_factor_1_whitebg_norm`
- `garden_type_colmap_factor_4_norm`

Settings 문자열에는 다음이 포함됩니다:
- `type_{dataset_type}`: 데이터셋 타입
- `factor_{data_factor}`: 다운샘플링 팩터
- `whitebg`: NeRF + white_background인 경우
- `norm`: normalize_world_space인 경우
- `patch_{patch_size}`: patch_size가 설정된 경우

