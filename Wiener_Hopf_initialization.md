# Wiener_Hopf_initialization.md

## 0. 목적과 전제

이 문서는 HNSTPP 초기화를 수학적으로 일관되게 정의하기 위한 사양이다.
핵심 목표는 다음 세 가지다.

1. Wiener-Hopf 기반 2차 통계 역산으로 이벤트 타입 간 유효 인과 행렬을 추정한다.
2. 비선형 강도식 $\lambda_k(t) = (b_k + E_k(t))\exp(-I_k(t)) + \epsilon$에 맞게 평균장 역변환을 수행한다.
3. 과완전 후보 규칙 풀에서 희소 최적화로 단일/다중 규칙을 공정하게 선별한다.

본 문서에서 행렬 인덱스는 반드시 아래 규약을 따른다.

- 행(row) $i$: target type
- 열(column) $j$: source type
- 즉 $R_{ij}$는 $j \rightarrow i$ 효과

---

## 1. 모델 정의

$$

\lambda_k(t) = (b_k + E_k(t)) \cdot \exp(-I_k(t)) + \epsilon

$$
$$

E_k(t) = \sum_r W^+_r \cdot \text{Head}_{r,k} \cdot p_r(t), \quad
I_k(t) = \sum_r W^-_r \cdot \text{Head}_{r,k} \cdot p_r(t)

$$
$$

p_r(t) = \mathrm{ReLU}(S_r(t) - \mathrm{bias}_r), \quad
S_r(t) = \sum_{j:t_j<t} H_{j,r}\,g_r(t-t_j)

$$

### 1.1 시간 커널 $g_r(\cdot)$ 정의 (코드 일치 사양)

HNSTPP 구현에서 $g_r$는 `[0, max_cap]` 구간의 조각선형(piecewise-linear) 커널이다.

- bin 개수: $M=\text{num\_bins}$
- bin 폭: $\Delta\tau=\text{max\_cap}/M$
- grid: $\tau_m=m\Delta\tau,\; m=0,\dots,M$
- 높이 파라미터:
  - 경계 고정: $h_{r,0}=0,\; h_{r,M}=0$
  - 내부 학습: $h_{r,m}=\mathrm{softplus}(\xi_{r,m}),\; m=1,\dots,M-1$

따라서

$$
g_r(u)=
\begin{cases}
h_{r,m}+\dfrac{h_{r,m+1}-h_{r,m}}{\Delta\tau}(u-\tau_m), & u\in[\tau_m,\tau_{m+1}),\; m=0,\dots,M-1\\
0, & \text{otherwise}
\end{cases}
$$

실제 강도 계산에서는 causal/support 조건 $0<u\le\text{max\_cap}$를 사용한다.

주의: 현재 코드 기준으로는 $\max g_r=1$ 정규화 단계를 별도로 수행하지 않는다.

- $D$: event type 수
- $R$: rule slot 수
- $H \in \{0,1\}^{D\times R}$: source mask
- $\text{Head} \in \{0,1\}^{R\times D}$: target selector (single-target이면 각 rule당 1-hot)

설계 제약:

1. source strength를 연속 스칼라로 별도 분리하지 않고, 구조는 $H$로만 표현한다.
2. AND 규칙은 곱 연산 대신 `sum + threshold(ReLU)` 구조를 기본으로 사용한다.

---

## 2. Phase 1: Wiener-Hopf 역산

### 2.1 2차 통계량

정상성 가정 하에서 지연 $\tau\ge 0$의 교차 공분산:

$$

C_{ij}(\tau) = \frac{\mathbb{E}[dN_i(t+\tau)\,dN_j(t)]}{dt\,d\tau} - \Lambda_i\Lambda_j

$$

$\Lambda_i$는 타입 $i$의 평균 발생률이다.

### 2.2 이산 Wiener-Hopf 시스템

Wiener-Hopf 적분식을 $\Delta t$ 격자로 이산화하여
$$

Y = X A \Delta t

$$
를 구성하고 $X$를 푼다.
$X$는 $[D,\;D\cdot L]$이며 lag 축 적분으로
$$

R_{\mathrm{eff}} \in \mathbb{R}^{D\times D}

$$
를 얻는다.

필수 구현 가드:

1. $L=\lfloor \text{max\_lag}/\Delta t \rfloor \ge 1$
2. $A$가 특이행렬이면 정규화(예: ridge) 또는 `lstsq` fallback

### 2.3 부호 분리 규약

문서 전체에서 억제 항은 magnitude로 저장한다:

$$

\tilde R_{\mathrm{exc}} = \max(R_{\mathrm{eff}}, 0),\qquad
\tilde R_{\mathrm{inh}}^{\mathrm{mag}} = \max(-R_{\mathrm{eff}}, 0)

$$

$\tilde R_{\mathrm{inh}}^{\mathrm{mag}}\ge 0$ 이다.

---

## 3. Phase 2: Reverse Mean-Field Mapping

선형화 계수는 target 축 $\Lambda_i$에 걸리므로, 복원 분모는 source가 아니라 target이다.

### 3.1 억제 행렬 복원

$$

R_{\mathrm{inh},ij} = \frac{\tilde R^{\mathrm{mag}}_{\mathrm{inh},ij}}{\Lambda_i}

$$

브로드캐스팅은 `Lambda.unsqueeze(1)` (행 기준)이어야 한다.

### 3.2 평균장 억제량

$$

\bar I_i = \sum_j R_{\mathrm{inh},ij}\Lambda_j

$$

여기서는 source 발생률이 곱해지므로 `Lambda.unsqueeze(0)`가 맞다.

### 3.3 흥분 및 베이스라인 복원

$$

R_{\mathrm{exc},ij} = \tilde R_{\mathrm{exc},ij}\exp(\bar I_i)

$$
$$

b_i = \Lambda_i\exp(\bar I_i) - \sum_j R_{\mathrm{exc},ij}\Lambda_j

$$

수치 가드:

1. $\Lambda_i \approx 0$이면 해당 row 역산은 불안정하므로 clamp 또는 row zero 처리
2. $b_i > 0$ 강제

주의:

- 선형 Hawkes의 $\rho(\|R\|_1)<1$ 안정성 조건은 직접 적용되지 않는다.
- $R_{\mathrm{exc}}$ 스펙트럴 반경 체크는 실무적 heuristic이며 충분조건은 아니다.

---

## 4. Phase 3: 과완전 후보 사전

### 4.1 Single-source 후보

$|R_{\mathrm{exc},ij}|>\epsilon$ 또는 $|R_{\mathrm{inh},ij}|>\epsilon$인 $j\to i$를 후보로 등록한다.

### 4.2 Multi-source 후보

타겟 $C$ 직전 윈도우에서 $\{A,B\}$ 동시 출현 lift:
$$

\mathrm{Lift}(A,B\!\to\!C)=\frac{P(\{A,B\}\mid C)}{P(A)P(B)}

$$
로 후보를 만든다.
lift는 독립성 대비 과다동시발생 지표이므로 후보 생성 용도로만 쓰고, 최종 채택은 NLL 검증으로 결정한다.

### 4.3 구조 동결

후보 마스크 $H,\text{Head}$, kernel $g_r$, bias를 고정하고, $(w_{\mathrm{exc}},w_{\mathrm{inh}})$만 최적화 변수로 둔다.

---

## 5. Phase 4: Feature Caching

고정된 구조에서만 캐싱이 정확하다.

1. event 시점 캐시 $P_{\mathrm{event}}\in\mathbb{R}^{N\times K}$
2. 적분 grid 캐시 $P_{\mathrm{int}}\in\mathbb{R}^{M\times K}$
3. baseline 인덱싱 벡터:
   - $b_{\mathrm{event}}\in\mathbb{R}^{N}$, $(b_{\mathrm{event}})_i=b_{k_i}$ (i번째 이벤트의 타입 $k_i$)
   - $b_{\mathrm{int}}\in\mathbb{R}^{M}$, $(b_{\mathrm{int}})_m=b_{\kappa_m}$ (m번째 적분 grid의 타입 $\kappa_m$)

$$

\lambda_{\mathrm{event}} = (b_{\mathrm{event}} + P_{\mathrm{event}}w_{\mathrm{exc}})
\odot \exp(-P_{\mathrm{event}}w_{\mathrm{inh}})

$$
$$

\lambda_{\mathrm{int}} = (b_{\mathrm{int}} + P_{\mathrm{int}}w_{\mathrm{exc}})
\odot \exp(-P_{\mathrm{int}}w_{\mathrm{inh}})

$$

캐시 무효화 조건:

- $H$, Head, kernel, bias, max\_cap 중 하나라도 바뀌면 재계산

---

## 6. Phase 5: BCD + FISTA (강제)

목적함수:
$$

\min_{w_{\mathrm{exc}},\,w_{\mathrm{inh}}\ge 0}
\ \mathcal L_{\mathrm{NLL}}(w_{\mathrm{exc}},w_{\mathrm{inh}})
 + \lambda_1\|w_{\mathrm{exc}}\|_1
 + \lambda_1\|w_{\mathrm{inh}}\|_1
 + \lambda_h\Omega_{\mathrm{hier}}

$$

### 6.0 목적함수 각 항의 상세 정의

Phase 4의 캐시를 사용해 각 항을 다음처럼 쓴다.

1. 이벤트 로그우도 항 + 적분 항 (NLL):
$$
\mathcal L_{\mathrm{NLL}}
= -\sum_{i=1}^{N}\log\!\big(\lambda^{\mathrm{event}}_i\big)
  + \sum_{m=1}^{M} q_m\,\lambda^{\mathrm{int}}_m
$$

여기서
$$
\lambda^{\mathrm{event}}_i
= \Big(b_{\mathrm{event},i} + P_{\mathrm{event},i:}w_{\mathrm{exc}}\Big)
\exp\!\Big(-P_{\mathrm{event},i:}w_{\mathrm{inh}}\Big) + \epsilon
$$
$$
\lambda^{\mathrm{int}}_m
= \Big(b_{\mathrm{int},m} + P_{\mathrm{int},m:}w_{\mathrm{exc}}\Big)
\exp\!\Big(-P_{\mathrm{int},m:}w_{\mathrm{inh}}\Big) + \epsilon
$$

- $q_m\ge 0$: 수치적분 가중치(사다리꼴/직사각형 규칙)
- 실무에서는 배치 크기 의존성을 줄이기 위해 $\mathcal L_{\mathrm{NLL}}/N$ 정규화를 사용할 수 있다.

2. 비음수 가중치의 $L_1$ 희소화 항:
$$
\|w_{\mathrm{exc}}\|_1=\sum_{r=1}^{K}w_{\mathrm{exc},r},\qquad
\|w_{\mathrm{inh}}\|_1=\sum_{r=1}^{K}w_{\mathrm{inh},r}
$$
($w_{\mathrm{exc}},w_{\mathrm{inh}}\ge 0$ 가정)

3. 계층 제약(선택항) 예시:
single-rule을 parent, multi-rule을 child로 두고 관계 집합
$\mathcal E_{\mathrm{exc}},\mathcal E_{\mathrm{inh}}$를 정의하면
$$
\Omega_{\mathrm{hier}}
= \sum_{(p,c)\in\mathcal E_{\mathrm{exc}}}\max\!\big(0,\,w_{\mathrm{exc},c}-w_{\mathrm{exc},p}\big)
 +\sum_{(p,c)\in\mathcal E_{\mathrm{inh}}}\max\!\big(0,\,w_{\mathrm{inh},c}-w_{\mathrm{inh},p}\big)
$$
로 둘 수 있다.

이 항은 child 가중치가 관련 parent를 과도하게 앞지르는 경우를 벌점으로 주며, convex piecewise-linear 형태라 BCD+prox와 함께 사용하기 쉽다.

핵심 사실:

1. 전체 문제는 일반적으로 joint-convex가 아니다.
2. 한 블록을 고정하면 다른 블록은 convex가 된다(실무 설정에서 bi-convex).

### 6.1 블록 업데이트 (각 블록에 FISTA 적용)

교대 최적화(BCD)는 아래 순서를 1 iteration으로 반복한다.

1. $w_{\mathrm{inh}}$를 고정하고 $w_{\mathrm{exc}}$ 업데이트
2. 업데이트된 $w_{\mathrm{exc}}$를 고정하고 $w_{\mathrm{inh}}$ 업데이트

각 블록은 반드시 FISTA(가속 proximal-gradient) 스텝으로 업데이트한다.
고정 블록을 $u$, 업데이트 블록을 $w$라 할 때:

$$

z = w - \eta \nabla_w \mathcal L_{\mathrm{NLL}}(w;u),\qquad
w^+ = \max(0,\ z-\eta\lambda_1)

$$

위 식은 prox 연산의 핵심이며, 실제 업데이트는 FISTA의 가속 변수와 결합해 사용한다.
즉, 블록 변수 $w$에 대해
$$
y_t = w_t + \frac{t_{t-1}-1}{t_t}(w_t-w_{t-1}),\qquad
z_t = y_t - \eta_t \nabla \mathcal L_{\mathrm{NLL}}(y_t;u),\qquad
w_{t+1}=\max(0,z_t-\eta_t\lambda_1)
$$
를 적용하고, $t_t$는 FISTA 규칙으로 갱신한다.

이는 nonnegative $L_1$ prox의 닫힌꼴이다.
표준 soft-thresholding 연산자 $\mathcal S_{\tau}(z)=\mathrm{sign}(z)\max(|z|-\tau,0)$에 대해,
제약 $w\ge 0$를 결합하면
$$
\mathrm{prox}_{\tau\|\cdot\|_1+\mathbb I_{\mathbb R_+}}(z)=\max(0,\ z-\tau)
$$
가 되어 위 식과 일치한다.

각 블록을 FISTA로 명시하면:

$$
y_{\mathrm{exc}} = w_{\mathrm{exc}} + \beta_{\mathrm{exc}}(w_{\mathrm{exc}}-w_{\mathrm{exc}}^{\mathrm{prev}}),\quad
z_{\mathrm{exc}} = y_{\mathrm{exc}} - \eta_{\mathrm{exc}} \nabla_{w_{\mathrm{exc}}}\mathcal L_{\mathrm{NLL}}(y_{\mathrm{exc}};w_{\mathrm{inh}}),\quad
w_{\mathrm{exc}} \leftarrow \max(0, z_{\mathrm{exc}}-\eta_{\mathrm{exc}}\lambda_1)
$$

$$
y_{\mathrm{inh}} = w_{\mathrm{inh}} + \beta_{\mathrm{inh}}(w_{\mathrm{inh}}-w_{\mathrm{inh}}^{\mathrm{prev}}),\quad
z_{\mathrm{inh}} = y_{\mathrm{inh}} - \eta_{\mathrm{inh}} \nabla_{w_{\mathrm{inh}}}\mathcal L_{\mathrm{NLL}}(y_{\mathrm{inh}};w_{\mathrm{exc}}),\quad
w_{\mathrm{inh}} \leftarrow \max(0, z_{\mathrm{inh}}-\eta_{\mathrm{inh}}\lambda_1)
$$

여기서 $\beta=(t_{\mathrm{prev}}-1)/t$, $t\leftarrow(1+\sqrt{1+4t_{\mathrm{prev}}^2})/2$를 사용한다.

### 6.2 FISTA 고정 사용 규칙

본 파이프라인은 모든 블록 업데이트에서 FISTA를 사용한다.
따라서 각 블록에서 다음을 필수로 포함한다.

1. Nesterov 가속 계수 $t_k$ 갱신
2. Backtracking line-search(또는 Lipschitz 상수 기반 step-size 제어)
3. 블록 전환 시 모멘텀 초기화(Momentum Reset): $w_{\mathrm{exc}}\rightarrow w_{\mathrm{inh}}$ 또는
   $w_{\mathrm{inh}}\rightarrow w_{\mathrm{exc}}$로 넘어갈 때 해당 블록의 $t_k$를 반드시 $1$로 리셋

가속항을 제거한 업데이트(순수 proximal gradient)는 본 문서의 표준 절차로 허용하지 않는다.

### 6.3 수렴 해석

이 설정의 BCD/FISTA 반복은 일반적으로 stationary point 수렴을 목표로 하며, 전역최적 보장은 하지 않는다.

---

## 7. Phase 6: Debias + Global Ranking

1. $w=0$ 후보 제거(활성집합 추출)
2. 활성집합에서 $\lambda_1=0$으로 NLL 재학습(debias/refit)
3. held-out에서 drop-one $\Delta$NLL 계산
4. 단일/다중 구분 없이 $\Delta$NLL 순으로 Top-$R$ 채택

이 단계는 L1 shrinkage bias를 줄이는 표준 2-stage 전략과 정합적이다.

---

## 8. 구현 체크리스트

1. $R_{ij}$는 항상 $j\to i$로 해석한다.
2. $R_{\mathrm{inh}}$ 복원은 target 분모(`unsqueeze(1)`).
3. $\bar I$ 계산은 source 가중(`unsqueeze(0)`).
4. $L\ge 1$, $\Lambda_i>0$ 가드, $b_i>0$ 가드.
5. BCD는 전역최적이 아니라 정지점 수렴임을 명시한다.
6. 캐싱은 구조 고정 단계에서만 사용한다.

---

## 9. 참고 문헌 (Primary Sources)

1. Bacry, Delattre, Hoffmann, Muzy (2012). Non-parametric kernel estimation for symmetric Hawkes processes. https://arxiv.org/abs/1112.1838
2. Bacry, Muzy (2014/2015). Second order statistics characterization of Hawkes processes and non-parametric estimation. https://arxiv.org/abs/1401.0903
3. Brehmaud, Massoulie (1996). Stability of nonlinear Hawkes processes. https://doi.org/10.1214/aop/1065725193
4. Lee, Seung (2001). Algorithms for Non-negative Matrix Factorization. https://papers.nips.cc/paper_files/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html
5. Agrawal, Imielinski, Swami (1993). Mining association rules between sets of items in large databases. https://doi.org/10.1145/170036.170072
6. Beck, Teboulle (2009). FISTA. https://doi.org/10.1137/080716542
7. Parikh, Boyd (2014). Proximal Algorithms. https://web.stanford.edu/~boyd/papers/prox_algs.html
8. Tseng (2001). Convergence of a Block Coordinate Descent Method for Nondifferentiable Minimization. https://doi.org/10.1023/A:1017501703105
9. Friedman, Hastie, Tibshirani (2010). Regularization Paths for GLMs via Coordinate Descent. https://doi.org/10.18637/jss.v033.i01
10. Meinshausen (2007). Relaxed Lasso. https://doi.org/10.1016/j.csda.2006.12.019
11. van Krieken, Acar, van Harmelen (2022). Analyzing Differentiable Fuzzy Logic Operators. https://doi.org/10.1016/j.artint.2021.103602
