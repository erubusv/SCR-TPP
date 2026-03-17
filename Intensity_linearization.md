
### Phase 1. 정상 상태(Steady-State) 가정 및 평균 강도 정의

목표 타겟 사건 $k$의 비선형 강도 함수(Intensity function)는 다음과 같습니다.


$$\lambda_k(t) = (b_k + E_k(t)) \exp(-I_k(t))$$

시스템이 충분한 시간이 흘러 안정적인 정상 상태(Steady-state)에 도달했다고 가정합니다. 이때 각 사건 $j$의 발생 빈도는 평균 발생률(Mean event rate) $\Lambda_j$로 수렴합니다.
과거 사건들이 현재 타겟 $k$에 미치는 누적 흥분 $E_k(t)$와 억제 $I_k(t)$의 기댓값은, 개별 커널의 총 적분 가중치($R$)와 평균 발생률($\Lambda$)의 선형 결합으로 정의됩니다.

* **누적 흥분 기댓값:** $\bar{E}_k = \sum_j \Lambda_j [\mathbf{R}_{exc}]_{k,j}$
* **누적 억제 기댓값:** $\bar{I}_k = \sum_j \Lambda_j [\mathbf{R}_{inh}]_{k,j}$

따라서 정상 상태에서의 0차 적률, 즉 **평균 강도(Mean Intensity)** $\Lambda_k$는 다음과 같은 닫힌 꼴(Closed-form)의 항등식으로 정의됩니다.


$$\Lambda_k = (b_k + \bar{E}_k) \exp(-\bar{I}_k)$$

---

### Phase 2. 평균장 선형화 (Mean-Field Linearization) 및 유효 커널 도출

위너-호프(Wiener-Hopf) 방정식과 같은 2차 통계량 도구를 사용하려면 시스템이 반드시 선형(Linear)이어야 합니다. 따라서 비선형 강도 함수 $\lambda_k(t)$를 평균 기댓값 $(\bar{E}_k, \bar{I}_k)$ 근처에서 1차 테일러 전개(Taylor Expansion)합니다.

$$\lambda_k(t) \approx \lambda_k(\bar{E}_k, \bar{I}_k) + \left. \frac{\partial \lambda_k}{\partial E_k} \right|_{\text{mean}} (E_k(t) - \bar{E}_k) + \left. \frac{\partial \lambda_k}{\partial I_k} \right|_{\text{mean}} (I_k(t) - \bar{I}_k)$$

**1. 편미분 계수 증명:**

* $E_k$에 대한 편미분:

$$\frac{\partial \lambda_k}{\partial E_k} = \exp(-\bar{I}_k)$$


* $I_k$에 대한 편미분 (합성함수 미분 및 Phase 1의 $\Lambda_k$ 정의 대입):

$$\frac{\partial \lambda_k}{\partial I_k} = -(b_k + \bar{E}_k) \exp(-\bar{I}_k) = -\Lambda_k$$



**2. 선형 유효 모델(Effective Linear Model)로의 재조립:**
구해진 미분 계수를 전개식에 대입하고, 시변(Time-varying) 항과 상수(Constant) 항을 분리합니다.


$$\lambda_k(t) \approx \underbrace{\left[ \Lambda_k - \exp(-\bar{I}_k)\bar{E}_k + \Lambda_k \bar{I}_k \right]}_{\text{유효 기저 강도 } \tilde{b}_k} + \exp(-\bar{I}_k) E_k(t) - \Lambda_k I_k(t)$$

이때 $E_k(t) = \int \Phi_{exc}(t-s) dN(s)$ 이므로, 밖에 있는 스칼라 곱셈 계수를 적분 기호 안의 커널 $\Phi$로 밀어 넣으면, 완벽한 선형 호크스(Linear Hawkes) 형태의 **'유효 커널(Effective Kernel)'**이 탄생합니다.

* **유효 흥분 커널:** $\tilde{\Phi}_{exc}(t) = \exp(-\bar{I}_k) \Phi_{exc}(t)$
* **유효 억제 커널:** $\tilde{\Phi}_{inh}(t) = -\Lambda_k \Phi_{inh}(t)$

*(비판적 고찰: 이 테일러 전개 과정에서 버려진 2차 이상의 분산/고차항들이 바로 수학적 '근사 오차'를 만듭니다. 우리는 딥러닝의 NLL 최적화를 통해 이 오차를 사후 교정하게 됩니다.)*

---

### Phase 3. 거시적 통계량 기반의 인과적 뼈대 역산 (Method of Moments)

이제 모델이 선형이 되었으므로, 경험적 데이터의 2차 교차 공분산(Cross-covariance) 행렬 $C(\tau)$와 유효 커널 행렬 $\tilde{\mathbf{\Phi}}(\tau)$를 연결하는 절대 법칙, **위너-호프 적분 방정식(Wiener-Hopf Integral Equation)**을 적용할 수 있습니다. (Bacry & Muzy, 2014)

$$C(\tau) = \tilde{\mathbf{\Phi}}(\tau) \mathbf{\Sigma} + \int_0^\infty \tilde{\mathbf{\Phi}}(u) C(\tau-u) du \quad (\text{for } \tau > 0)$$

이 방정식을 이산화(Discretization)된 블록-토플리츠(Block-Toeplitz) 선형 연립방정식으로 풀어내어 모든 시간에 대한 $\tilde{\mathbf{\Phi}}(t)$를 구합니다. 이 유효 커널 행렬을 시간에 대해 $0$부터 $\infty$까지 모두 적분하면, 타겟 사건들을 촉발하는 뼈대인 **선형 유효 가중치 행렬 $\tilde{\mathbf{R}}$** 이 도출됩니다.


$$\tilde{\mathbf{R}} = \int_0^\infty \tilde{\mathbf{\Phi}}(t) dt$$


(이 행렬의 양수 성분은 $\tilde{\mathbf{R}}*{exc}$로, 음수 성분은 $\tilde{\mathbf{R}}*{inh}$로 분리합니다.)

---

### Phase 4. 비선형 파라미터 대수적 복원 (Reverse Mapping)

Phase 3에서 구한 선형 가상 세계의 가중치($\tilde{\mathbf{R}}$)를, Phase 2의 유효 커널 정의식을 역으로 이용하여 진짜 비선형 모델의 물리적 파라미터로 복원합니다. 커널을 적분한 총량이 곧 행렬 $\mathbf{R}$이므로, 복원 공식은 대수학적으로 다음과 같이 전개됩니다.

**1. 실제 억제 행렬 ($\mathbf{R}_{inh}$) 증명:**
유효 억제 커널 식의 양변을 적분합니다.


$$\tilde{\mathbf{R}}_{inh} = -\mathbf{\Lambda} \odot \mathbf{R}_{inh}$$

$$\therefore \mathbf{R}_{inh} = -\frac{\tilde{\mathbf{R}}_{inh}}{\mathbf{\Lambda}}$$


(타겟 $k$에 대한 브로드캐스팅 나눗셈 적용)

**2. 실제 흥분 행렬 ($\mathbf{R}_{exc}$) 증명:**
유효 흥분 커널 식의 양변을 적분합니다.


$$\tilde{\mathbf{R}}_{exc} = \exp(-\bar{\mathbf{I}}) \odot \mathbf{R}_{exc}$$


여기에 $\bar{I}_k = \sum_j \Lambda_j [\mathbf{R}_{inh}]_{k,j}$ 를 대입하고 역수를 곱해 정리합니다.


$$\therefore \mathbf{R}_{exc} = \tilde{\mathbf{R}}_{exc} \odot \exp(\mathbf{R}_{inh} \mathbf{\Lambda}^T)$$

**3. 기저 강도 벡터 ($\mathbf{b}$) 증명:**
Phase 1의 정상 상태 평균 강도 식을 $b_k$에 대해 정리합니다.


$$\mathbf{\Lambda} = (\mathbf{b} + \mathbf{R}_{exc}\mathbf{\Lambda}^T) \odot \exp(-\mathbf{R}_{inh}\mathbf{\Lambda}^T)$$


양변에 역수를 곱하고 항을 이동시킵니다.


$$\mathbf{b} + \mathbf{R}_{exc}\mathbf{\Lambda}^T = \mathbf{\Lambda} \odot \exp(\mathbf{R}_{inh}\mathbf{\Lambda}^T)$$

$$\therefore \mathbf{b} = \mathbf{\Lambda} \odot \exp(\mathbf{R}_{inh}\mathbf{\Lambda}^T) - \mathbf{R}_{exc}\mathbf{\Lambda}^T$$
