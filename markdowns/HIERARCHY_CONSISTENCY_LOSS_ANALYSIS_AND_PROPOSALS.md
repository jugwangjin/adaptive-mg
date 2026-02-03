## Hierarchy Consistency Loss (v2/v3) 분석 & 제안 카탈로그

이 문서는 `hierarchy_trainer_vcycle_v2.py`, `hierarchy_trainer_vcycle_v3.py`의 hierarchy consistency loss를 **3DGS 렌더링식**과 **가우시안(적분/혼합/투영) 성질**에 기반해 해석하고, 더 수학적으로 근거가 명확한 대안/확장 아이디어를 **다양한 범주**로 정리한 설계 노트입니다.

---

## 0. TL;DR (핵심 요약)

- v2의 핵심은 “자식들의 2D 가우시안(알파 질량) 혼합”을 **단일 2D 가우시안(부모)**로 **moment matching(1·2차 모멘트 일치)**시키는 형태입니다. 이 부분은 통계/가우시안 혼합 축약 관점에서 근거가 강합니다.
- v3는 v2의 **expected covariance + Browser falloff(오파시티)**를 유지하면서,
  - **containment(자식이 부모 타원 안)**, **oversize 페널티**, **depth consistency**, **premultiplied color + depth 가중치**로 “렌더링(정렬/가시성) 가정”을 더 직접 삽입한 형태입니다.
- 가장 근거가 강한 방향(저비용)은, **렌더러가 실제로 적분하는 양**에 맞춰
  - 색을 “unpremultiplied color 평균”이 아니라 **integrated premultiplied mass**로 맞추거나,
  - “alpha-field(coverage field)”를 샘플 포인트에서 직접 맞추는 것.
- 더 faithful하게 가려면 **국소 render-to-render**가 최강이지만 비용이 큽니다.

---

## 1. 표기/기본 가정

### 1.1 2D 투영된 Gaussian

카메라에서 어떤 3D Gaussian이 투영되면 2D에서 대략 다음 형태의 타원 가우시안을 갖습니다.

- 중심: $\mu \in \mathbb{R}^2$
- 공분산: $\Sigma \in \mathbb{R}^{2\times2}$, SPD
- 역공분산(=conic): $Q = \Sigma^{-1}$
- "가우시안 커널":

$$
g(x) = \exp\left(-\frac12 (x-\mu)^\top \Sigma^{-1} (x-\mu)\right)
$$

### 1.2 알파(Opacity) & premultiplied color

일반적인 3DGS alpha blending을 단순화하면, 픽셀 $x$에서 Gaussian $i$의 알파 기여는

$$
a_i(x) = \alpha_i \, g_i(x), \qquad \alpha_i \in [0,1]
$$

색(혹은 SH를 평가해 얻은 RGB 등) $c_i$에 대해 premultiplied 기여는

$$
p_i(x) = a_i(x)\, c_i
$$

정렬(front-to-back) 알파 블렌딩에서 최종 색은

$$
C(x) = \sum_{i} T_i(x)\, p_i(x), \qquad
T_i(x)=\prod_{j<i} \big(1-a_j(x)\big)
$$

즉 “image-space contribution”은 본질적으로 **premultiplied color**와 **transmittance**에 의해 결정됩니다.

---

## 2. “Hierarchy consistency”의 목표를 렌더링 관점에서 다시 정의

부모 Gaussian $P$가 자식 집합 $\{C_i\}$를 대체(혹은 근사)한다고 할 때, 가장 근본적인 목표는:

### 2.1 Field-level equivalence (이상적 목표)

특정 뷰(카메라)에서, 부모-only 렌더와 children-only 렌더가 유사해야 합니다.

- 알파 필드:
$$
a_P(x) \approx a_{\text{children}}(x)
$$
- 색 필드(또는 premul 필드):
$$
T_P(x)\,a_P(x)\,c_P(x) \approx \sum_i T_i(x)\,a_i(x)\,c_i(x)
$$

하지만 위 목표를 픽셀 전부에 대해 정확히 계산하면 비용이 크므로, v2/v3는 **투영 파라미터**를 이용해 “충분통계(모멘트/적분량)”를 맞추는 근사로 접근합니다.

---

## 3. v2: `hierarchy_trainer_vcycle_v2.py` loss 항목별 분석

v2는 크게 아래 4가지로 요약됩니다.

### 3.1 Weight = opacity × area (알파 질량 기반)

v2의 weight는 (코드상)

- $\text{area}_i = \sqrt{\det(\Sigma_i)} = 1/\sqrt{\det(Q_i)}$
- $w_i \propto \alpha_i \cdot \text{area}_i$
- parent별로 normalize

이때 "왜 $\alpha \sqrt{\det\Sigma}$인가?"에 대한 수학적 근거는 아래 적분 성질입니다:

$$
\int_{\mathbb{R}^2} g(x)\,dx = 2\pi\sqrt{\det\Sigma}
$$

그러므로 "integrated alpha mass"는

$$
M_i = \int a_i(x)\,dx = \alpha_i \int g_i(x)\,dx \propto \alpha_i \sqrt{\det\Sigma_i}
$$

즉 v2의 weight는 **0차 모멘트(질량)**에 대한 자연스러운 선택입니다.

### 3.2 expected mean/covariance = moment matching (혼합 가우시안 축약)

자식들의 혼합을 단일 가우시안으로 근사할 때, 1·2차 모멘트를 맞추면:

$$
\mu_\text{mix}=\frac{1}{W}\sum_i w_i \mu_i
$$
$$
\Sigma_\text{mix}=\frac{1}{W}\sum_i w_i\left(\Sigma_i + (\mu_i-\mu_\text{mix})(\mu_i-\mu_\text{mix})^\top\right)
$$

이는 통계적으로 표준적인 혼합 축약이며, v2의 expected covariance는 이 구조를 그대로 구현합니다.

### 3.3 opacity falloff (Browser 스타일)

v2는

- $W=\sum_i w_i$
- "대표 면적" $A=\sqrt{\det(\Sigma_\text{mix})}$
- falloff $f = W/A$
- opacity로 매핑 $\hat{\alpha} = f/(1+f)$

로 구성됩니다.

해석:
- $W$는 "총 질량(coverage capacity)"
- $A$는 "퍼지는 면적"
- 그 비율은 "면적당 농도" 같은 값이며,
- $f/(1+f)$는 saturating behavior로 0~1로 안정적으로 압축합니다.

### 3.4 color

v2는 expected_color를 단순히

$$
\hat{c} = \sum_i w_i c_i
$$

로 평균냅니다(코드상 MSE).

렌더링 관점에서 실제 중요한 것은 $a(x)c$인데, v2는 color와 alpha를 분리해 맞추기 때문에 이 항목은 상대적으로 "근거가 약한 편"입니다(하지만 alpha와 면적이 이미 맞춰진다면 색 평균도 어느 정도 의미를 가짐).

---

## 4. v3: `hierarchy_trainer_vcycle_v3.py` loss 항목별 분석 (v2 대비)

v3는 v2의 “혼합 축약(moment matching)”을 일부 유지하면서, 렌더링/가우시안 기하 구조에 대한 제약을 추가합니다.

### 4.1 Containment(inside) + oversize (가우시안의 ‘지원(support)’ 일치)

부모/자식 중심 차이 벡터 $d=\mu_c-\mu_p$, 단위방향 $u=d/\|d\|$를 두고

- 방향별 표준편차(extent):
$$
e(\Sigma,u) = \sqrt{u^\top \Sigma u}
$$

v3는
- inside margin $m=\max(e_p - e_c, 0)$
- inside loss: $\mathrm{ReLU}(\|d\| - m)$
- oversize loss: $\mathrm{ReLU}(e_c - e_p)$

를 사용합니다.

렌더링 관점 해석:
- “parent를 통과하는 ray는 child도 통과한다” 가정을 강제하려면, child footprint가 parent footprint 안에 들어가는 제약이 자연스럽습니다.
- v2의 mean 매칭은 멀티모달(자식들이 양쪽에 분리)에서 centroid가 빈 곳으로 가는 문제가 있을 수 있는데, containment는 그 리스크를 줄입니다.

### 4.2 expected covariance & opacity falloff는 유지 (v2-style)

v3는 expected mean/covar를 **opacity×area weight**로 계산하고, opacity도 **falloff**로 계산합니다.

차이점:
- v3는 이 weight 계산을 `no_grad()`로 고정(학습이 weight로 치팅하는 것 방지).

### 4.3 color는 premultiplied + depth 기반 가중치 (정렬/가시성 휴리스틱)

v3는
- depth score $s_i=\exp(-\beta z_i)$ 를 parent별로 정규화하여
- approx_transmittance $\tilde{T}_i = s_i/\sum_j s_j$
- expected premul:
$$
\widehat{(ac)} = \sum_i \tilde{T}_i \, (\alpha_i c_i)
$$
- parent premul: $\alpha_p c_p$

로 premultiplied domain에서 비교합니다.

장점:
- 렌더링의 핵심 quantity인 premultiplied를 직접 맞추므로 v2보다 렌더링 연결이 강함.

약점(수학적):
- 실제 $T_i(x)$는 픽셀 위치 $x$와 kernel overlap에 따라 달라지는데, $\tilde{T}_i$는 전역 스칼라로 뭉뚱그립니다.
- footprint(면적)나 위치별 overlap이 color 가중치에 직접 들어가지 않습니다.

### 4.4 depth consistency

depth 가중치가 “비슷한 레이어에 있을수록 합리적”이므로, parent/children depth가 비슷하도록 MSE를 추가합니다.

---

## 5. v2 vs v3 비교 (렌더링/가우시안 관점)

### 5.1 무엇이 “수학적으로 강한가”

- **가장 근거 강함**: v2/v3 공통의 **moment matching(expected mean/covar)** + **opacity×area weight**
  - 가우시안 적분 및 혼합 축약의 표준 공식과 직접 연결됨
- **falloff**: 엄밀한 렌더링 유도라기보다 “면적당 질량→포화 opacity”라는 합리적 요약(근거 중간)
- **color 평균(v2)**: 렌더링의 premul/occlusion 반영 약함(근거 약)
- **premul+depth(v3)**: 렌더링의 방향성은 맞지만 occlusion을 매우 거칠게 근사(근거 중간)
- **containment(v3)**: “지원 영역이 겹쳐야 한다”는 구조적 제약(근거 강~중간: 사용 목적에 따라 매우 유용)

---

## 6. 설계 원칙(Design principles): ‘합리적인 consistency’란?

여기서는 loss 설계 시 자주 충돌하는 목표를 정리합니다.

### 6.1 Renderer-quantity first

렌더러가 실제로 누적하는 양은 (대개)
- $a(x)$ (coverage)
- $T(x)\,a(x)\,c(x)$ (visible premultiplied)

따라서 "파라미터 MSE"보다 "field-level(알파/프리멀)"에 가까운 목적이 더 설득력 있습니다.

### 6.2 Closed-form integrals are gold

가우시안은 적분/모멘트가 닫힌 형태로 나옵니다. 가능한 한
- $\int a(x)\,dx$,
- $\int a(x)\,x\,dx$,
- $\int a(x)\,(x-\mu)(x-\mu)^\top dx$,
- $\int a(x)\,c\,dx$

같은 폐형식 quantity로 consistency를 구성하면 비용/안정성이 좋습니다.

### 6.3 Occlusion is the hard part

정렬/투과율이 들어가면 비선형(곱/누적)이고, 픽셀별로 달라집니다. 이를 얼마나 근사할지(혹은 무시할지)가 설계의 핵심입니다.

---

## 7. 제안 카탈로그 (다양한 대안/확장 아이디어)

아래는 “근거(수학/렌더링)”와 “비용”을 축으로 다양한 아이디어를 나열합니다.

### 7.1 (저비용/강근거) Integrated mass / premultiplied mass matching

#### 7.1.1 Alpha mass conservation

부모/자식의 integrated alpha mass를 맞춥니다:

$$
M(\alpha,\Sigma) = \alpha \cdot 2\pi\sqrt{\det\Sigma}
$$
$$
M_p \approx \sum_i M_i
$$

장점: 매우 싸고, coverage의 “총량”을 직접 맞춤.  
단점: shape mismatch(어디에 퍼지냐)는 별도 항목 필요.

#### 7.1.2 Premultiplied mass matching (색을 더 렌더링-정합적으로)

색을 "평균"이 아니라 "적분 premul 질량"으로 맞춤:

$$
P(\alpha,\Sigma,c)=\alpha\cdot 2\pi\sqrt{\det\Sigma}\cdot c
$$
$$
P_p \approx \sum_i P_i
$$

이건 v2의 weight 정의(질량 기반)와 완전히 정합적이며, v3의 premultiplied 방향성을 "수학적으로" 강화합니다.

옵션: depth를 넣고 싶으면 $\exp(-\beta z_i)$ 같은 감쇠를 우변에 추가(근거는 약해지지만 실용적).

---

### 7.2 (저~중비용/강근거) Gaussian divergence 기반 loss (SPD/기하학)

v2의 covar_loss는 Frobenius relative error입니다. 더 좌표-불변/통계적 의미가 있는 대안들:

#### 7.2.1 KL divergence between Gaussians (폐형식)

2D 가우시안 $p=\mathcal{N}(\mu_0,\Sigma_0)$, $q=\mathcal{N}(\mu_1,\Sigma_1)$에서:

$$
\mathrm{KL}(p\|q)=\frac12\left[
\mathrm{tr}(\Sigma_1^{-1}\Sigma_0)
(\mu_1-\mu_0)^\top \Sigma_1^{-1}(\mu_1-\mu_0)
-2+\ln\frac{\det\Sigma_1}{\det\Sigma_0}
\right]
$$

장점: mean/covar를 동시에 통계적 의미로 묶어 비교.  
단점: 비대칭(원하면 symmetrized KL 사용).

#### 7.2.2 Bhattacharyya distance (겹침 기반)

겹침(overlap) 정도를 반영하는 distance. 위치/shape를 동시에 반영.

#### 7.2.3 SPD manifold metric (Affine-invariant / Log-Euclidean)

$$
d_{\text{AIRM}}(\Sigma_0,\Sigma_1)=\|\log(\Sigma_0^{-1/2}\Sigma_1\Sigma_0^{-1/2})\|_F
$$

장점: SPD 행렬 공간의 자연스러운 거리.  
단점: 행렬 로그/고유분해 필요(2x2라면 충분히 가능).

---

### 7.3 (중비용/렌더링 근거↑) Alpha-field 직접 매칭 (샘플링)

falloff처럼 단일 스칼라로 opacity를 요약하지 말고, **알파 필드 자체**를 몇 개 점에서 비교:

#### 7.3.1 포인트 샘플링 전략

부모 1개에 대해 샘플 포인트 $x_s$를 선택:
- child means들
- 부모 타원 경계(주축/부축 방향 ±kσ)
- 랜덤(부모 커널에서 샘플)
- child 타원들의 bbox에서 sparse grid

#### 7.3.2 alpha 비교식

- 부모: $a_p(x)=\alpha_p g_p(x)$
- 자식 합성(순서 무시한 union coverage):
$$
a_{\text{grp}}(x)=1-\prod_i (1-\alpha_i g_i(x))
$$
- loss:
$$
L_\alpha = \frac{1}{S}\sum_s \|a_p(x_s)-a_{\text{grp}}(x_s)\|^2
$$

장점:
- 알파 포화/비선형성을 정확히 반영.
- “image space contribution”의 기반인 coverage를 직접 맞춤.

단점:
- 샘플링/연산이 늘어남(하지만 S를 작게(예: 16~64)하면 충분히 실용적).

---

### 7.4 (중~고비용/렌더링 근거↑↑) Occlusion-aware premul matching

v3의 $\exp(-\beta z)$는 "앞에 있을수록 더 보인다" 정도만 반영합니다. 더 근거 있는 방향들:

#### 7.4.1 Depth-sorted hard approximation (정렬 + 누적 곱)

parent 내 children을 depth로 정렬한 뒤,

$$
\tilde{T}_i \approx \prod_{j<i}(1-\tilde{a}_j)
$$

같은 형태로 누적 투과율을 근사.  
여기서 $\tilde{a}_j$는 픽셀 의존성을 없애기 위해 "대표 alpha mass"를 쓰거나, 특정 샘플 포인트에서 평가한 $a_j(x_s)$를 평균낼 수 있습니다.

#### 7.4.2 SoftSort/NeuralSort 기반 soft ordering

정렬은 비미분 가능이므로 soft sorting을 써서 근사 ordering을 만들 수 있습니다.

장점: occlusion 구조를 더 직접 반영.  
단점: 구현 복잡/비용 증가.

#### 7.4.3 Tau-domain(광학두께) additive consistency

작은 알파에서는 $a\approx\tau$ (optical thickness)로 근사 가능하고, $\tau$는 더 "가산적"입니다.

- $\tau_{\text{grp}}(x) \approx \sum_i \tau_i(x)$
- $\alpha(x) \approx 1-e^{-\tau(x)}$

이 구조를 이용해 alpha consistency를 더 물리적으로 만들 수 있습니다.

---

### 7.5 (구조 제약/안정성) Containment의 더 ‘수학적’ 변형들

#### 7.5.1 Mahalanobis containment (k-시그마 포함)

부모 타원 안에 child mean이 들어가도록:

$$
d_M^2 = (\mu_c-\mu_p)^\top \Sigma_p^{-1}(\mu_c-\mu_p)
$$
$$
L_{\text{contain}}=\mathrm{ReLU}(d_M - k)
$$

장점:
- 좌표/스케일에 더 불변이고 해석이 명확(“kσ 안”).

#### 7.5.2 PSD ordering penalty (child가 parent보다 커지지 않게)

이상적으로는 $\Sigma_c \preceq \Sigma_p$를 원할 수 있음.
이를 근사하려면,
- $\lambda_{\max}(\Sigma_c-\Sigma_p)$를 계산해서 $\mathrm{ReLU}(\lambda_{\max})$ 페널티
- 또는 고유값/주축 길이 비교 페널티

---

### 7.6 (멀티뷰 근거↑) Multi-view consistency

현재 v2/v3는 “현재 step의 뷰 1개”에서 consistency를 계산합니다. 하지만 hierarchy 관계는 뷰-불변이어야 하므로:

- 각 step에서 1~K개 뷰를 추가 샘플(예: 2~3개)
- 또는 cycle마다 1회 multi-view batch로만 consistency를 주기적으로 적용

장점:
- 특정 뷰에서만 성립하는 “편향된” 2D consistency를 줄임.

단점:
- projection/렌더 비용 증가 (하지만 projection-only면 완화 가능)

---

### 7.7 (스케줄링/커리큘럼) 언제/얼마나 적용할까?

실전에서 consistency는 “초기”와 “후기”에 서로 다른 역할을 합니다.

- 초기: 구조를 잡고(containment/covar) gross alignment
- 후기: 색/opacity의 섬세한 일치, 과도한 regularization은 성능 저하 가능

스케줄 제안:
- consistency_lambda warmup/cosine decay
- depth_loss weight는 초기에만 높게
- containment는 초기 강하게, 후기 약하게

---

### 7.8 (효율/캐싱/부분집합) 계산을 어디까지 할까?

#### 7.8.1 Visible-only children set

현재도 projection/raster info 기반으로 “visible”만 고려하는 흐름이 있습니다. 이를 더 적극적으로:
- 부모별로 “현재 뷰에서 기여 가능성이 있는 자식 subset”만

#### 7.8.2 Parent-local microbatch

전체 parent를 한 번에 하지 않고,
- 랜덤 parent subset을 선택해 consistency를 주는 방식(regularizer로서 충분한 경우 많음)

---

## 8. 이 repo 기준 “추천 로드맵” (현실적인 우선순위)

### 8.1 가장 추천 (저비용/근거↑)

1) v3의 color를 “depth-softmax premul” 대신(혹은 추가로)  
   **integrated premul mass matching**으로 정합성 강화  
2) covar/mean은 “moment matching”을 유지하되, covar loss metric을 KL/whitened/Log-Euclidean 중 하나로 개선(선택)

### 8.2 다음 추천 (중비용/효과↑)

3) alpha-field 샘플링 매칭(부모당 소수 포인트) 추가  
   - falloff를 대체하거나, falloff는 유지하고 보조로 사용

### 8.3 가장 faithful (고비용)

4) 국소 render-to-render consistency (patch/low-res)

---

## 9. 구현 스케치 (코드에 붙일 때의 형태)

### 9.1 Integrated premul mass matching (가장 쉬운 추가)

- 각 gaussian의 $M_i = \alpha_i \cdot \sqrt{\det\Sigma_i}$ (상수 $2\pi$는 생략 가능)
- 각 gaussian의 $P_i = M_i \cdot c_i$
- parent와 children 합을 맞추는 MSE/L1/Huber

추가 포인트:
- v3에서는 covar/opacity weight가 no_grad이므로, 이 term도 동일 정책으로 맞출지 결정
- direction detach 규칙(Down: children detach, Up: parent detach)은 유지

### 9.2 Alpha-field sampling

- 부모당 샘플 $x_s$를 16~64개 생성
- $a_p(x_s)$, $a_{\text{grp}}(x_s)$ 계산
- MSE

가속:
- 샘플 포인트를 parent마다 동일한 템플릿(정규좌표)로 만들고 affine transform으로 생성
- child kernel 평가를 벡터화

---

## 10. 실험/평가 플랜(권장)

### 10.1 기본 비교

- PSNR/SSIM/LPIPS
- hierarchy loss 분해 로그(inside/oversize/covar/opacity/color/depth)
- level별 안정성(특히 level increase 직후)

### 10.2 ablation

- v2 vs v3 (현 상태)
- v3 + integrated premul mass
- v3 + alpha-field sampling
- covar metric 변경(KL vs 기존 relative Frobenius)

### 10.3 failure mode 체크

- 자식이 부모 밖으로 튀는지(containment)
- child가 과도하게 커지는지(oversize/PSD penalty)
- depth consistency가 과한지(가시성/디테일 손상)

---

## Appendix A. "왜 opacity×area가 자연스러운가" (짧은 유도)

$$
g(x)=\exp\left(-\frac12 (x-\mu)^\top \Sigma^{-1} (x-\mu)\right)
\Rightarrow
\int g(x)\,dx = 2\pi \sqrt{\det\Sigma}
$$

따라서 $a(x)=\alpha g(x)$이면
$$
\int a(x)\,dx = \alpha \int g(x)\,dx \propto \alpha\sqrt{\det\Sigma}
$$

즉 "이미지에 얼마나 많이 깔리는가(coverage 총량)"를 요약하려면 $\alpha\sqrt{\det\Sigma}$가 1순위 후보입니다.

