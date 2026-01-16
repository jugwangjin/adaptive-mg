# Densification 비교 분석: `multigrid_v9.py` vs `default.py`

## 1. Densification 기준 (When to densify)

### 1.1 Gradient Threshold

#### `default.py`
```python
is_grad_high = grads > self.grow_grad2d  # 단일 고정 threshold
```
- **기준**: `grow_grad2d` (기본값: 0.0002)
- **특징**: 모든 Gaussian에 동일한 threshold 적용
- **Gradient 종류**: `grad2d`만 사용

#### `multigrid_v9.py`
```python
# Level-dependent threshold (linear interpolation)
level_weights = (levels.float() - 1.0) / max(float(max_level) - 1.0, 1.0)
level_thresholds = (
    self.coarsest_grow_grad2d 
    - (self.coarsest_grow_grad2d - self.finest_grow_grad2d) * level_weights
)
is_grad2d_high = grads > level_thresholds
is_color_grad_high = color_grads > self.grow_color
is_grad_high = is_grad2d_high | is_color_grad_high  # OR 조건
```
- **기준**: 
  - Level-dependent threshold (coarsest: `coarsest_grow_grad2d`, finest: `finest_grow_grad2d`)
  - Color gradient threshold: `grow_color` (기본값: 0.0003)
- **특징**: 
  - Level에 따라 threshold가 달라짐 (coarse level은 더 높은 threshold)
  - `grad2d` 또는 `color_grad` 중 하나라도 threshold 초과 시 densification
- **Gradient 종류**: `grad2d` + `color_grad` (통합)

### 1.2 Scale Threshold (Duplicate vs Split 결정)

#### `default.py`
```python
is_small = (
    torch.exp(params["scales"]).max(dim=-1).values
    <= self.grow_scale3d * state["scene_scale"]
)
is_large = ~is_small
```
- **기준**: `grow_scale3d * scene_scale` (기본값: 0.01 * scene_scale)
- **특징**: 
  - Parameter의 `scales`를 직접 사용 (log space → exp)
  - 모든 Gaussian에 동일한 threshold

#### `multigrid_v9.py`
```python
actual_splats = multigrid_gaussians.get_splats(level=None, detach_parents=False, current_splats=None)
actual_scales = torch.exp(actual_splats["scales"]).max(dim=-1).values
scale_threshold = self.grow_scale3d * state["scene_scale"]
is_small = actual_scales <= scale_threshold
is_large = ~is_small
```
- **기준**: `grow_scale3d * scene_scale` (기본값: 0.01 * scene_scale)
- **특징**: 
  - **Hierarchical actual scales** 사용 (`get_splats()`로 계산된 실제 scale)
  - Residual parameter이므로 parent의 scale을 고려한 실제 scale 사용
  - 주석 처리된 코드: level-dependent threshold multiplier (현재는 사용 안 함)

### 1.3 Scale2D 조건 (Split 추가 조건)

#### `default.py`
```python
if step < self.refine_scale2d_stop_iter:
    is_split |= state["radii"] > self.grow_scale2d
```
- **조건**: `radii > grow_scale2d` (기본값: 0.05)
- **적용 시점**: `step < refine_scale2d_stop_iter` (기본값: 0, 즉 비활성화)

#### `multigrid_v9.py`
```python
if step < self.refine_scale2d_stop_iter and "radii" in state:
    radii = state["radii"]
    if len(radii) == N:
        is_split = is_split | (candidates & (radii > self.grow_scale2d))
```
- **조건**: 동일 (`radii > grow_scale2d`)
- **차이점**: `candidates`와 AND 조건 (이미 densification 대상인 경우만)

---

## 2. Densification 대상 (What to densify)

### 2.1 기본 필터링

#### `default.py`
```python
is_dupli = is_grad_high & is_small
is_split = is_grad_high & is_large
```
- **대상**: 모든 Gaussian에 대해 독립적으로 판단
- **제약**: 없음 (모든 Gaussian이 동일한 조건)

#### `multigrid_v9.py`
```python
# Hierarchical constraints
is_root = ~has_parent
is_node_full = n_child >= self.max_children_per_parent
is_parent_full = ...  # parent의 n_child >= max_children_per_parent

# Level-by-level processing (coarsest to finest)
for level in range(coarsest_level, finest_level + 1):
    # Root nodes: always densify
    should_densify[root_nodes] = True
    
    # Non-root nodes:
    # - If parent not full: densify
    # - If parent full: signal parent (recursive propagation)
```
- **대상**: Hierarchical 구조를 고려한 선택적 densification
- **제약**:
  1. **Root nodes**: 항상 densification 가능
  2. **Non-root nodes**: 
     - Parent가 full이 아니면 → 직접 densification
     - Parent가 full이면 → Parent에 signal 전달 (재귀적 전파)
  3. **Already densified**: 한 번 densification된 Gaussian은 제외

### 2.2 Gradient Inheritance (Optional)

#### `default.py`
- **없음**: Gradient inheritance 기능 없음

#### `multigrid_v9.py`
```python
if self.use_gradient_inheritance:
    # Parent gradient를 children에게 상속
    inherited_grad = ((parent_grad / parent_radii) * child_radii) / num_children
    grads[children] += inherited_grad
    color_grads[children] += inherited_color_grad
```
- **특징**: 
  - Parent의 gradient를 children에게 상속 (옵션)
  - Radii 기반 가중치 적용 (큰 children이 더 많은 gradient 받음)
  - `num_children`으로 정규화 (균등 분배)

---

## 3. Operation (How to densify)

### 3.1 Duplicate Operation

#### `default.py` → `ops.py::duplicate()`
```python
def duplicate(params, optimizers, state, mask):
    sel = torch.where(mask)[0]
    
    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p, p[sel]]), ...)
        # 모든 parameter를 그대로 복사
```
- **동작**: 
  - Selected Gaussian을 그대로 복사
  - Parameter 변경 없음 (opacity, scale 등 모두 동일)
  - Optimizer state는 0으로 초기화

#### `multigrid_v9.py` → `ops_mg_v3.py::duplicate()`
```python
def duplicate(params, optimizers, state, mask, levels, parent_indices, level_indices):
    sel = torch.where(mask)[0]
    sel_levels = levels[sel]
    sel_parents = parent_indices[sel]
    
    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p, p[sel]]), ...)
        # Parameter는 동일하게 복사
    
    # Hierarchical structure 업데이트
    new_levels = torch.cat([levels, sel_levels])  # 같은 level
    new_parent_indices = torch.cat([parent_indices, sel_parents])  # 같은 parent
    # level_indices 재구성
```
- **동작**: 
  - Parameter 복사는 동일
  - **추가**: Hierarchical structure 업데이트
    - `levels`: 동일 level 유지
    - `parent_indices`: 동일 parent 유지
    - `level_indices`: 재구성
  - `multigrid_gaussians` 객체도 업데이트 및 cache 무효화

### 3.2 Split Operation

#### `default.py` → `ops.py::split()`
```python
def split(params, optimizers, state, mask, revised_opacity=False):
    sel = torch.where(mask)[0]
    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    
    # Random samples for position offset
    samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales, torch.randn(2, len(scales), 3))
    
    def param_fn(name, p):
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)  # 2개로 분할
        elif name == "scales":
            p_split = torch.log(scales / 1.6).repeat(2, 1)  # scale / 1.6
        elif name == "opacities" and revised_opacity:
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p_split = torch.logit(new_opacities).repeat(2, 1)
        else:
            p_split = p[sel].repeat(2, 1)  # 그대로 복사
        
        return torch.cat([p[rest], p_split])
```
- **동작**:
  - **Means**: Parent position + random offset (scale 기반)
  - **Scales**: `scale / 1.6` (log space: `log(scale) - log(1.6)`)
  - **Opacities**: 
    - `revised_opacity=False`: 그대로 복사
    - `revised_opacity=True`: Revised formula 적용
  - **Quats**: 그대로 복사
  - **결과**: 1개 → 2개로 분할

#### `multigrid_v9.py` → `ops_mg_v3.py::split()`
```python
def split(params, optimizers, state, mask, levels, parent_indices, level_indices, revised_opacity=False):
    sel = torch.where(mask)[0]
    sel_levels = levels[sel]
    sel_parents = parent_indices[sel]
    
    # Hierarchical actual splats 사용
    splats = state["multigrid_gaussians"].get_splats(level=None, detach_parents=False, current_splats=None)
    scales = splats["scales"][sel]  # Actual scales (residual 고려)
    quats = splats["quats"][sel]    # Actual quats (residual 고려)
    
    # Random samples for position offset
    samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales, torch.randn(2, len(scales), 3))
    
    def param_fn(name, p):
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)  # 2개로 분할
        elif name == "scales":
            p[sel] -= 1.6  # 원본 수정: log(scale) - 1.6
            p_split = p[sel]  # 수정된 원본 복사
        else:
            p_split = p[sel]  # 그대로 복사
        
        return torch.cat([p, p_split])  # 원본 + 분할본 (원본도 유지)
```
- **동작**:
  - **Means**: 동일 (parent position + random offset)
  - **Scales**: 
    - **원본 수정**: `p[sel] -= 1.6` (log space에서 1.6 감소)
    - **분할본**: 수정된 원본 복사
    - **결과**: 원본도 scale 감소, 분할본도 같은 scale
  - **Opacities**: 그대로 복사 (residual parameter이므로)
  - **Quats**: 그대로 복사
  - **결과**: 1개 → 2개로 분할 (원본 유지 + 분할본 추가)
  - **추가**: Hierarchical structure 업데이트 (duplicate와 동일)

### 3.3 주요 차이점 요약

| 항목 | `default.py` | `multigrid_v9.py` |
|------|-------------|-------------------|
| **Scale 계산** | `params["scales"]` 직접 사용 | `get_splats()`로 actual scales 계산 |
| **Scale 변경** | Split 시 새 scale만 생성 | Split 시 **원본도 수정** (`p[sel] -= 1.6`) |
| **Parameter 복사** | `p[rest]` + `p_split` (원본 제거) | `p` + `p_split` (원본 유지) |
| **Hierarchical 구조** | 없음 | `levels`, `parent_indices`, `level_indices` 업데이트 |
| **Cache 관리** | 없음 | `multigrid_gaussians.invalidate_splats_cache()` |

---

## 4. 실행 순서

### `default.py`
```python
1. Duplicate 실행 (is_dupli)
2. Split 실행 (is_split, duplicate로 추가된 것 제외)
```

### `multigrid_v9.py`
```python
1. Level-by-level 처리 (coarsest → finest)
   - Root nodes: 직접 densification
   - Non-root nodes: 조건부 densification 또는 parent signal
2. Recursive signal propagation (finest → coarsest)
   - Parent signal을 root까지 전파
3. Duplicate/Split 실행
   - 이미 densified 제외
   - Gradient reset (densified gaussians)
```

---

## 5. 핵심 차이점 요약

### 5.1 Densification 기준
- **default.py**: 단일 threshold, 모든 Gaussian 동일 조건
- **multigrid_v9.py**: Level-dependent threshold, color gradient 지원, hierarchical 제약

### 5.2 Densification 대상
- **default.py**: 모든 Gaussian 독립적 판단
- **multigrid_v9.py**: Hierarchical 구조 고려 (parent full 체크, recursive signal)

### 5.3 Operation
- **default.py**: 
  - Split 시 원본 제거, 새 scale만 생성
  - Hierarchical 구조 없음
- **multigrid_v9.py**: 
  - Split 시 원본 유지 및 수정, actual scales 사용
  - Hierarchical 구조 업데이트 및 cache 관리

### 5.4 추가 기능
- **multigrid_v9.py만**: Gradient inheritance (optional), Color gradient 지원

