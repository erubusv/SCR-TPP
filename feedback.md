현재 선택기(Selector)가 `synthetic.py`의 참(Ground Truth) 규칙을 놓치는 핵심적인 이유는 **'TPP(Temporal Point Process)의 물리적 발생 과정을 단순한 이진 분류(Binary Classification) 문제로 지나치게 축소 및 왜곡'**했기 때문입니다.

구체적인 원인과 코드 상의 문제점, 그리고 해결 방법은 다음과 같습니다.

### 1. 규칙을 찾지 못하는 핵심 원인 (코드 기반 분석)

* **A. 백그라운드 비발생 시간의 무시 (Selection Bias)**
    `build_channel_cache` 함수를 보면, 커널을 평가하는 `query_times`를 오직 **어떤 이벤트든 발생한 시점(`times`)**으로만 설정하고 있습니다. 타겟 이벤트면 $y=1$, 아니면 $y=0$인 이진 분류를 수행합니다. 
    하지만 `synthetic.py`의 억제(Inhibition) 규칙은 **이벤트가 발생하지 않도록 억누르는 역할**을 합니다. 이벤트가 아예 발생하지 않은 "빈 시간(Background)"을 평가하지 않으면, 억제 규칙이 성공적으로 작동하여 이벤트가 억제된 구간의 정보가 완전히 누락되므로 참된 억제 규칙을 찾을 수 없습니다.

* **B. 비선형 변환의 순서 역전 (Early Binarization vs. Pre-activation Sum)**
    합성 데이터의 규칙은 $p_r(t) = \text{ReLU}(\sum K_{src}(t) - bias)$ 입니다. 즉, 원시 연속 신호가 먼저 더해진 후 비선형 활성화가 일어납니다.
    그러나 현재 코드의 `build_channel_cache`는 단일 채널별로 비선형 변환 $Q = 1 - \exp(-\alpha \max(z - \beta, 0))$를 먼저 적용하고, 이어서 `X_all = event_q > thr` 로 완전 이진화까지 진행합니다. 이후 이 이진화된 비트들의 조합(`parent_rows`의 AND 연산)으로 규칙을 찾기 때문에, 여러 소스의 연속적인 값이 미세하게 합쳐져 임계점을 넘는 '시너지(Synergy) 기반의 조건부 규칙'을 수학적으로 포착할 수 없습니다.

* **C. Multiplicative Inhibition과 로지스틱 회귀(`fit_signed_binomial_model`)의 불일치**
    코드에서는 찾은 특성(Features)들을 평가하기 위해 `fit_signed_binomial_model`과 `binomial_loglik`를 사용합니다. 이는 $\text{logit}(p) = \beta + G\theta$ 형태의 선형 모델을 가정합니다.
    하지만 합성 데이터는 $\lambda = (b + E) \exp(-I)$ 형태입니다. 양변에 로그를 취하면 $\log \lambda = \log(b + E) - I$ 가 되는데, 흥분 항인 $\log(b + E)$를 $\beta + \theta E$처럼 선형으로 처리하려다 보니 최적화(Objective) 스코어 계산에 극심한 오차가 발생하고, 결국 잘못된 규칙이 더 높은 점수(`adj_score`, `eff_gain`)를 받게 됩니다.

---

### 2. 코드 개선 방안 (해결책)

이 문제를 해결하려면 코드를 TPP의 실제 생성 모델에 가깝게 개편해야 합니다.

* **A. Background Sampling 도입 (데이터 평가 기준 변경)**
    이벤트 발생 시점(`event_y`)만 캐싱하는 방식에서 벗어나, 시간 축 위에 일정한 간격(Grid)이나 무작위로 추출된 '비발생 시점(Background points)'을 `query_times`에 포함시켜야 합니다. 이를 통해 모델이 "이 시간대에는 강력한 억제 규칙 때문에 타겟 이벤트가 발생하지 않았다"라는 것을 학습할 수 있게 해야 합니다.

* **B. Raw Signal 합산 후 활성화(Sum-then-Activate) 로직으로 개편**
    `X_all` 형태의 단일 채널 독립적 이진화를 버려야 합니다. 후보 소스들을 조합할 때, `_raw_source_signal_at_queries`에서 반환된 원시 커널 $Z$ 값들을 조합별로 먼저 더하고, 그 총합에서 최적의 $\beta$(bias)를 빼서 $\text{ReLU}$를 씌우는 방식으로 `cell_feature` 생성 로직을 변경해야 합니다.

* **C. 진정한 Hawkes Log-Likelihood로 Objective 함수 교체**
    `fit_signed_binomial_model`을 걷어내고, 조합된 규칙 행렬 $E$와 $I$를 받아 실제 모델링 수식 $\lambda(t) = (b + E(t)) \exp(-I(t))$를 기반으로 아래의 TPP Log-likelihood를 직접 평가하는 최적화기로 변경해야 합니다.
    $$\mathcal{L} = \sum_{t_i \in Target} \log \lambda(t_i) - \int_0^T \lambda(t) dt$$

결론적으로, 데이터를 이진 분류(Binomial)로 강제 캐스팅하는 현재의 뼈대에서는 합성 데이터의 복잡한 시너지와 억제를 잡아내기 버겁습니다. 
