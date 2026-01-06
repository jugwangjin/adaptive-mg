# READ_THIS_TOMORROW_ME.md

## 사용자 메모

### 1. Opacity Reset/Regularization 문제

**문제점:**
- Reset하고 그냥 두면 coarse level에서 pruning 호출되면 뒷레벨들 싹다 사라짐
- Opacity reg 줄 때도 coarse level에 opacity reg 들어가면 fine level은 가만히 있다가 싹다 투명해짐
- 이건 뭔가 잘못되긴 한듯, 애초에 lower level에서 fine level pruning된거 자체가 문제
- 아마 visibility mask를 다시 확인해야될지도 모르겠다

### 2. Children 생성 후 Densification 문제

**문제점:**
- Children을 만든 것 까지는 좋은데, 그러고나서 다시 densification 하면 결국엔 coarse level에선 똑같은게 아닌가?
- Children이 있을 때의 densification은 달라야할 것 같은데?
- 예를 들면 이미 child가 있다 -> child를 생성하지는 않음 그런 규칙
- 근데 그러면 그래프가 너무 regular해진다는 단점은 있다
- Child 최대 수를 두고 고민하게 하든가 해야될지도 모른다

---

## 깊은 분석 및 해결 방안

### 1. Opacity Reset/Regularization 문제 분석

#### 1.1 문제의 근본 원인

**현재 구조의 문제점:**

1. **Hierarchical Pruning의 연쇄 효과:**
   - Coarse level에서 parent gaussian이 prune되면, 그 parent를 참조하는 모든 child gaussians가 무효화됨
   - `get_splats()`에서 `parent_mean + child_residual * scale_factor`를 계산할 때, parent가 없으면 child의 위치가 정의되지 않음
   - 현재 구현에서는 parent가 prune되면 child들이 "고아(orphan)"가 되어 렌더링에서 제외됨

2. **Opacity Regularization의 레벨 불일치:**
   - 현재 `vcycle_trainer.py`의 opacity regularization은 `visible_mask`를 사용하지만, 이는 특정 레벨에서 렌더링된 gaussians만 포함
   - Coarse level에서 opacity regularization을 적용하면, fine level의 gaussians는 regularization을 받지 않음
   - 하지만 fine level의 gaussians는 여전히 parent의 opacity에 의존적임 (hierarchical structure에서)
   - 결과적으로 coarse level의 opacity가 낮아지면, fine level의 effective opacity도 낮아짐

3. **Visibility Mask의 레벨별 불일치:**
   - `set_visible_mask(level)`은 특정 레벨의 leaf nodes만 visible로 설정
   - 하지만 opacity regularization은 모든 visible gaussians에 적용됨
   - Coarse level에서 opacity regularization을 적용하면, fine level의 gaussians는 regularization을 받지 않지만, parent의 opacity 변화에 영향을 받음

#### 1.2 참고 논문 분석

**"Revising Densification in Gaussian Splatting" (arxiv:2404.06109):**
- Opacity regularization은 pixel-error driven formulation을 사용
- Cloning operation에서 opacity를 올바르게 처리하는 것이 중요
- 현재 opacity를 그대로 복사하면 alpha-compositing logic에 bias가 생김

**"A Hierarchical 3D Gaussian Representation" (arxiv:2406.12080):**
- Hierarchical structure에서 interior nodes를 최적화하는 것이 중요
- Chunk-based training과 hierarchy consolidation을 통해 large scene을 처리
- 하지만 opacity regularization에 대한 구체적인 언급은 없음

#### 1.3 해결 방안

**방안 1: 레벨별 독립적인 Opacity Regularization**
```python
# 각 레벨의 gaussians에 대해 독립적으로 opacity regularization 적용
# Coarse level의 opacity 변화가 fine level에 영향을 주지 않도록
if cfg.opacity_reg > 0.0:
    # 현재 렌더링 레벨의 gaussians만 regularization
    current_level_mask = (self.multigrid_gaussians.levels == target_level)
    if current_level_mask.any():
        current_level_opacities = torch.sigmoid(
            self.splats["opacities"][current_level_mask]
        )
        opacity_reg_loss = cfg.opacity_reg * (current_level_opacities ** 2).mean()
        loss += opacity_reg_loss
```

**방안 2: Parent-Child Opacity Decoupling**
- Child gaussians의 opacity를 parent와 독립적으로 관리
- `get_splats()`에서 opacity는 hierarchical structure를 따르지 않고, 각 gaussian의 독립적인 opacity 사용
- 하지만 이는 hierarchical structure의 의미를 약화시킬 수 있음

**방안 3: Pruning 시 Child 보호**
- Parent가 prune될 때, child gaussians를 root node로 승격 (parent_index = -1, level = 1)
- 또는 child gaussians를 함께 prune하는 대신, parent의 역할을 대체하는 child를 선택

**방안 4: Visibility Mask 기반 Opacity Regularization 개선**
```python
# 현재 렌더링에 실제로 사용된 gaussians만 regularization
if "visible_mask" in info:
    rendering_visible_mask = info["visible_mask"]
    # 추가: 현재 레벨의 gaussians만 필터링
    current_level_mask = (self.multigrid_gaussians.levels == target_level)
    level_specific_mask = rendering_visible_mask & current_level_mask
    
    if level_specific_mask.any():
        visible_opacities = torch.sigmoid(
            self.splats["opacities"][level_specific_mask]
        )
        opacity_reg_loss = cfg.opacity_reg * (visible_opacities ** 2).mean()
        loss += opacity_reg_loss
```

**추천 방안:**
- **방안 1 + 방안 4 조합**: 레벨별 독립적인 opacity regularization + visibility mask 기반 필터링
- 이렇게 하면 각 레벨의 gaussians가 독립적으로 regularization을 받고, coarse level의 opacity 변화가 fine level에 영향을 주지 않음

### 2. Children 생성 후 Densification 문제 분석

#### 2.1 문제의 근본 원인

**현재 Densification 로직의 문제:**

1. **중복 Child 생성:**
   - `_grow_gs()`에서 `grad2d > threshold`인 gaussians에 대해 children을 생성
   - 하지만 이미 children이 있는 parent에 대해서도 다시 children을 생성할 수 있음
   - 결과적으로 같은 parent에 여러 세트의 children이 생성될 수 있음

2. **Graph 구조의 불규칙성:**
   - 현재는 parent가 children을 생성하면, 그 children들이 다시 children을 생성할 수 있음
   - 하지만 parent 자체도 계속 children을 생성할 수 있어서, graph가 불규칙해짐

3. **Coarse Level Densification의 의미:**
   - Coarse level에서 children을 생성하면, fine level의 gaussians가 증가함
   - 하지만 coarse level 자체의 gaussians 수는 변하지 않음
   - V-cycle에서 coarse level을 학습할 때, coarse level의 gaussians만 업데이트되므로, 새로 생성된 children은 학습되지 않음

#### 2.2 참고 논문 분석

**Octree-GS (reference_codes/octree_anygs):**
- Anchor (parent)와 Child의 명확한 구분
- `anchor_growing()`에서 children을 생성할 때, 이미 children이 있는 anchor는 건너뜀
- 하지만 이는 graph를 너무 regular하게 만들 수 있음

**"A Hierarchical 3D Gaussian Representation" (arxiv:2406.12080):**
- Hierarchy를 구축할 때 merging method를 사용
- Interior nodes를 최적화하여 visual quality를 향상
- 하지만 densification에 대한 구체적인 언급은 없음

#### 2.3 해결 방안

**방안 1: Child 존재 여부 확인 후 Densification**
```python
@torch.no_grad()
def _grow_gs(self, ...):
    # Check if gaussians already have children
    is_parent = torch.zeros(N, dtype=torch.bool, device=device)
    valid_parent_mask = (parent_indices != -1)
    if valid_parent_mask.any():
        valid_parent_indices = parent_indices[valid_parent_mask]
        valid_bounds = (valid_parent_indices >= 0) & (valid_parent_indices < N)
        if valid_bounds.any():
            is_parent[valid_parent_indices_in_bounds] = True
    
    # Only create children for gaussians that don't have children yet
    is_create_children = (grads > level_thresholds) & (~is_parent)
```

**방안 2: Child 최대 수 제한**
```python
# 각 parent당 최대 n_children_max개의 children만 허용
n_children_max = 4  # 또는 설정 가능한 파라미터

# Count existing children per parent
children_count = torch.zeros(N, dtype=torch.long, device=device)
valid_parent_mask = (parent_indices != -1)
if valid_parent_mask.any():
    valid_parent_indices = parent_indices[valid_parent_mask]
    children_count.scatter_add_(0, valid_parent_indices, torch.ones_like(valid_parent_indices))

# Only create children if under limit
is_create_children = (grads > level_thresholds) & (children_count < n_children_max)
```

**방안 3: 레벨별 Densification 전략**
```python
# Coarse level에서는 children 생성 금지 (이미 fine level이 있으므로)
# Fine level에서만 children 생성 허용
if target_level >= max_level - 1:  # Fine levels only
    is_create_children = (grads > level_thresholds) & (~is_parent)
else:  # Coarse levels: no children creation
    is_create_children = torch.zeros_like(grads, dtype=torch.bool)
```

**방안 4: Adaptive Child Creation (Gradient 기반)**
```python
# Parent가 이미 children을 가지고 있어도, gradient가 매우 크면 추가 children 생성
# 하지만 기존 children의 gradient도 고려
if is_parent.any():
    # Get children's gradients for parents
    parent_grads = torch.zeros(N, device=device)
    for parent_idx in torch.where(is_parent)[0]:
        child_mask = (parent_indices == parent_idx)
        if child_mask.any():
            parent_grads[parent_idx] = grads[child_mask].max()  # Max child gradient
    
    # Create children if parent's gradient is much larger than children's
    gradient_ratio_threshold = 2.0  # Parent gradient must be 2x larger
    is_create_children_parent = (
        (grads > level_thresholds * gradient_ratio_threshold) & 
        (grads > parent_grads * gradient_ratio_threshold) &
        is_parent
    )
    is_create_children = (grads > level_thresholds) & (~is_parent) | is_create_children_parent
```

**추천 방안:**
- **방안 1 + 방안 2 조합**: Child 존재 여부 확인 + Child 최대 수 제한
- 이렇게 하면 graph가 너무 불규칙해지지 않으면서도, 필요한 경우 추가 children을 생성할 수 있음
- 추가로, V-cycle에서 각 레벨을 학습할 때 해당 레벨의 gaussians만 densification을 수행하도록 제한

### 3. Multigrid Method 관점에서의 개선 사항

#### 3.1 V-cycle과 Densification의 상호작용

**현재 문제:**
- V-cycle에서 각 레벨을 학습할 때, 해당 레벨의 gaussians만 업데이트됨
- 하지만 densification은 모든 레벨의 gaussians에 대해 수행됨
- 결과적으로 새로 생성된 children이 다음 V-cycle까지 학습되지 않음

**해결 방안:**
- Densification을 V-cycle의 특정 단계에서만 수행 (예: coarsest level solving 후)
- 또는 각 레벨 학습 후 해당 레벨의 gaussians에 대해서만 densification 수행

#### 3.2 Hierarchical Structure의 일관성 유지

**문제:**
- Pruning, child creation 등이 hierarchical structure를 깨뜨릴 수 있음
- Parent가 prune되면 child들이 고아가 됨
- Child가 생성되면 parent의 역할이 변경됨

**해결 방안:**
- 모든 hierarchical operation 후 structure consistency check
- Parent가 prune될 때 child들을 root로 승격하거나 함께 prune
- Child 생성 시 parent의 상태를 적절히 업데이트

### 4. 구현 우선순위

1. **즉시 수정 필요:**
   - Opacity regularization을 레벨별로 독립적으로 적용
   - Parent가 prune될 때 child 처리 로직 추가

2. **단기 개선:**
   - Child 존재 여부 확인 후 densification
   - Child 최대 수 제한

3. **장기 개선:**
   - Adaptive child creation (gradient 기반)
   - V-cycle과 densification의 통합 최적화
   - Hierarchical structure consistency check 자동화

### 5. 참고할 추가 논문 및 자료

1. **Multigrid Methods:**
   - "A Multigrid Tutorial" by William L. Briggs
   - Multigrid method의 기본 원리와 V-cycle, F-cycle, W-cycle

2. **Hierarchical Representations:**
   - Octree-GS implementation (reference_codes/octree_anygs)
   - "A Hierarchical 3D Gaussian Representation" (arxiv:2406.12080)

3. **Densification Strategies:**
   - "Revising Densification in Gaussian Splatting" (arxiv:2404.06109)
   - Original 3DGS paper의 Adaptive Density Control

4. **Optimization:**
   - Multigrid optimization techniques
   - Hierarchical optimization methods

---

## 다음 단계

1. Opacity regularization 수정 (레벨별 독립 적용)
2. Parent pruning 시 child 처리 로직 추가
3. Child 존재 여부 확인 후 densification 로직 수정
4. Child 최대 수 제한 추가
5. 테스트 및 검증

