# CEMS Digital Twin v5 기술 문서

**100MW급 AI 데이터센터 신재생 마이크로그리드 Digital Twin**

> 버전: DT5 | 작성일: 2026-03-02 | 테스트: 92/92 통과

---

## 목차

1. [개요](#1-개요)
2. [모듈별 상세](#2-모듈별-상세)
3. [데이터소스](#3-데이터소스)
4. [경제성 분석 프레임워크](#4-경제성-분석-프레임워크)
5. [시연 시나리오 가이드](#5-시연-시나리오-가이드)
6. [Expansion 모듈](#6-expansion-모듈)
7. [Hyperscaler 전략](#7-hyperscaler-전략)
8. [검증 결과](#8-검증-결과)

---

## 1. 개요

### 1.1 시스템 목적

CEMS(Campus Energy Management System) Digital Twin은 100MW급 AI 데이터센터(AIDC)에 최적화된 신재생 마이크로그리드의 설계, 운영, 투자 의사결정을 지원하는 시뮬레이션 플랫폼이다. 1ms(전력전자 응답) ~ 1년(경제성 분석) 스케일의 다중 시간 해상도를 지원하며, 13개 핵심 모듈과 3개 확장 모듈로 구성된다.

### 1.2 시스템 아키텍처

```
                        ┌─────────────────────────────────┐
                        │        M10. Weather (TMY)        │
                        │   GHI, Temp, Wind, Humidity      │
                        └──────────────┬──────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │   M1. PV Array   │   │   M3. AIDC Load  │   │   M8. Grid       │
   │ c-Si/Tandem/     │   │ GPU×N, PUE,      │   │ SMP, REC, K-ETS  │
   │ Triple/Infinite  │   │ Workload Mix     │   │ 보호계전, FR/VR  │
   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
            │                      │                       │
            ▼                      ▼                       ▼
   ┌───────────────────────────────────────────────────────────────┐
   │                    M4. DC Bus (380V)                          │
   │  PV→Bus  Bus→AIDC  Bus↔HESS  Bus↔H₂  Bus↔Grid               │
   │  SiC/GaN 컨버터, 순시 전력 균형, 변환 손실 추적              │
   └──────────┬──────────────┬──────────────┬─────────────────────┘
              │              │              │
              ▼              ▼              ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │ M2. HESS     │  │ M5. H₂       │  │ M6. AI-EMS   │
   │ 6-Layer:     │  │ PEM Elec.    │  │ 3-Tier:      │
   │ SC→Li→Na→    │  │ + Storage    │  │ T1: 실시간   │
   │ RFB→CAES→H₂  │  │ + PEM FC     │  │ T2: 예측     │
   └──────────────┘  │ + Solar Bat. │  │ T3: 전략     │
                     └──────────────┘  └──────┬───────┘
                                              │
              ┌───────────────────────────────┼──────────────────┐
              │                               │                  │
              ▼                               ▼                  ▼
   ┌──────────────┐              ┌──────────────┐   ┌──────────────┐
   │ M7. Carbon   │              │ M9. Economics │   │ M11. Policy  │
   │ Scope 1/2/3  │              │ NPV/IRR/LCOE │   │ K-ETS, CBAM  │
   │ K-ETS, CBAM  │              │ MC 10,000    │   │ RE100        │
   └──────────────┘              └──────┬───────┘   └──────────────┘
                                        │
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
             ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
             │ M12. Industry│  │ M13. Invest  │  │ Expansion    │
             │ CSP 분석     │  │ Go/No-Go     │  │ Stress/Data/ │
             │ BYOG, Scale  │  │ MC Dashboard │  │ Unified      │
             └──────────────┘  └──────────────┘  └──────────────┘
```

### 1.3 시뮬레이션 설정

| 파라미터 | 값 | 용도 |
|---------|-----|------|
| 시간 해상도 (최소) | 1 ms | 전력전자 응답 (HESS Supercap) |
| 시간 해상도 (제어) | 1 s | HESS 충방전 제어 |
| 시간 해상도 (EMS) | 1 min / 15 min | Tier 1/2 디스패치 |
| 시간 해상도 (경제) | 1 hr | 시계열 시뮬레이션, SMP |
| 시뮬레이션 기간 | 8,760 hr (1년) | 연간 에너지 수지 |
| 공간 모델 | 집중정수(lumped) | — |

---

## 2. 모듈별 상세

### 2.1 M1. PV 발전 모듈

#### 목적
4가지 PV 기술(c-Si, 탠덤, 3접합, 무한접합)의 출력을 기상 조건에 따라 시뮬레이션한다. NOCT 셀 온도 모델, IEC 61215 기반 온도계수, 연간 열화를 반영하며, 능동 제어(MPPT 최적화) 옵션을 포함한다.

#### 지배방정식

**셀 온도 (NOCT 모델)**

$$T_{cell} = T_{amb} + \frac{NOCT - 20}{800} \cdot G \cdot f_{wind}$$

여기서 $f_{wind} = 1 - 0.04 \cdot \max(0, v_{wind} - 2)$이며, $v_{wind}$는 풍속(m/s), $G$는 전천일사량(W/m²)이다.

**온도 의존 효율 (IEC 61215 / IEC 61853)**

$$\eta(T) = \eta_{STC} \left[1 + \beta_{rel} \cdot (T_{cell} - 25)\right]$$

- $\eta_{STC}$: STC(25°C, 1000 W/m², AM1.5G) 기준 효율 (%)
- $\beta_{rel}$: **상대** 온도계수 (1/°C). config의 `beta`는 %/°C 단위이므로 $\beta_{rel} = \beta / 100$
  - 예) c-Si: $\beta = -0.35$ %/°C → $\beta_{rel} = -0.0035$ /°C
  - 절대 온도계수: $\beta_{abs} = \eta_{STC} \times \beta_{rel} = 24.4 \times 0.0035 = 0.0854$ %p/°C
- Ref: De Soto et al., Solar Energy 80(1), 2006; IEC 61215:2021; IEC 61853-1:2011

**열화 모델 (지수 감소)**

$$\eta(t) = \eta_0 \cdot (1 - \delta/100)^t$$

여기서 $\delta$는 연간 상대 열화율(%/yr), $t$는 운영 연수이다.

**출력 전력**

$$P_{PV} = \frac{\eta}{100} \cdot A_{total} \cdot G \cdot 10^{-6} \quad [\text{MW}]$$

능동 제어 시 MPPT 최적화(+3~7%), 부분 음영 저감(+2~5%), DC Bus 전압 안정화(+1~3%)를 적용하며 총 5~15% 출력 향상을 모델링한다.

#### 핵심 파라미터

| 파라미터 | c-Si | Tandem | Triple | Infinite | 출처 |
|---------|------|--------|--------|----------|------|
| $\eta_{STC}$ (%) | 24.4 | 34.85 | 39.5 | 68.7 | STC 측정 기준 |
| $\beta$ (%/°C) | -0.35 | -0.25 | -0.20 | -0.15 | IEC 61215 |
| NOCT (°C) | 45 | 43 | 42 | 40 | IEC 61215 |
| $\delta$ (%/yr) | 0.5 | 0.8 | 1.0 | 0.5 | 제조사 보증 기준 |
| $V_{OC}$ (V) | 0.72 | 2.15 | 3.20 | 5.0 (가정) | — |
| $J_{SC}$ (mA/cm²) | 40.5 | 19.5 | 13.5 | 25.0 (계산) | — |
| FF | 0.83 | 0.82 | 0.80 | 0.85 | — |
| 면적 (ha/100MW) | 93 | 55 | 48 | 28 | — |

**Solar Battery (2030+ Emerging, TRL 2-3)**

Water-soluble polymer가 태양광을 포집하여 화학 에너지로 저장한 뒤 on-demand로 H₂를 방출하는 신기술이다.

$$\text{STH} = \eta_{capture} \times \eta_{H_2} \times (1 - 0.005 \cdot d) \times (1 - 0.02 \cdot t)$$

| 파라미터 | 값 | 출처 |
|---------|-----|------|
| $\eta_{capture}$ | 0.80 | Nature Communications, DOI:10.1038/s41467-026-68342-2 |
| $\eta_{H_2}$ | 0.72 | 상동 |
| STH 효율 (이론) | 57.6% | $0.80 \times 0.72$ (이상적 조건 하 이론값) |
| 저장 손실 | 0.5%/day | 상동 |
| 연간 열화 | 2%/yr | 상동 |

> ⚠️ STH 57.6%는 2단계 프로세스(광→화학에너지 포집 + 온디맨드 H₂ 방출)의 각 단계 효율 곱이며, 기존 PEC/PV-electrolysis 경로의 STH(~20%)와 직접 비교 불가. 보수적 시나리오(30~40%)도 분석에 포함한다.

#### 가정 및 한계
- 집중정수 모델로 모듈 간 전압/전류 미스매치 미반영
- 부분 음영 효과는 확률적 모델링 (결정론적 음영 해석 미포함)
- Infinite PV(무한접합)는 Shockley-Queisser 이론 기반 가정값
- 풍속 냉각 효과: $f_{wind}$가 27 m/s 이상에서 0 이하가 될 수 있으므로 $\max(T_{amb}, T_{cell})$ 하한 적용

---

### 2.2 M2. HESS (Hybrid Energy Storage System)

#### 목적
6-layer 하이브리드 에너지 저장 시스템의 주파수 기반 부하 분리, SOC 밸런싱, 열화 모델링을 수행한다. μs~계절 시간 상수를 6개 저장 기술로 분담한다.

#### 6-Layer 구조

| Layer | 기술 | 용량 | 정격출력 | 응답시간 | η (충/방) | 시간상수 | CAPEX ($/kWh) |
|-------|------|------|---------|---------|-----------|---------|---------------|
| L1 | Supercapacitor | 50 kWh | 10 MW | 1 μs | 98/98% | μs~s | 10,000 |
| L2 | Li-ion BESS | 2,000 MWh | 200 MW | 100 ms | 95/95% | s~hr | 200 |
| L3 | Na-ion BESS | 1,000 MWh | 100 MW | 200 ms | 92/92% | hr~12hr | 80 |
| L4 | Vanadium RFB | 750 MWh | 50 MW | 1 s | 85/85% | hr~day | 300 |
| L5 | CAES | 1,000 MWh | 100 MW | 30 s | 75/75% | day~week | 100 |
| L6 | H₂ Storage | 5,000 MWh | 50 MW | 5 min | 40/40% | day~season | 20 |

**총 저장 용량: 9,800 MWh (≈ 10 GWh)**

**Na-ion BESS (Layer 3)** — Ref: Nature Reviews Materials (2025), doi:10.1038/s41578-025-00857-4
- Hard carbon 양극, layered oxide 음극 (CATL Naxtra 아키텍처)
- 광온도 범위: -40°C ~ +60°C (Li-ion 대비 핵심 장점)
- 사이클 수명: ~10,000 cycles (Li-ion 대비 2배)
- CAPEX: $80/kWh (Li-ion의 40%, 2026 목표치, 보수적 범위 $80~$120/kWh)

#### 지배방정식

**주파수 기반 전력 배분**

각 레이어 $i$의 시간 상수 범위 $[\tau_{min,i}, \tau_{max,i}]$에 대응하는 주파수 범위 $[f_{min,i}, f_{max,i}]$에서:

$$f_{min,i} = 1/\tau_{max,i}, \quad f_{max,i} = 1/\tau_{min,i}$$

요청 주파수 $f$가 레이어 $i$의 범위에 속하면 해당 레이어에 우선 배분한다. 배분 우선순위: L1(Supercap) → L2(Li-ion) → L3(Na-ion) → L4(RFB) → L5(CAES) → L6(H₂).

**SOC 업데이트**

충전 시:
$$SOC_{new} = SOC + \frac{P \cdot \Delta t \cdot \eta_{charge}}{3600 \cdot C_{eff}} \cdot (1 - \sigma \cdot \Delta t / 3600)$$

방전 시:
$$SOC_{new} = SOC - \frac{P \cdot \Delta t}{3600 \cdot C_{eff} \cdot \eta_{discharge}} \cdot (1 - \sigma \cdot \Delta t / 3600)$$

여기서 $C_{eff} = C_{rated} \cdot D_f$는 열화 반영 유효 용량, $\sigma$는 자기방전율(/hr), $D_f$는 열화 계수이다.

**열화 모델**

$$D_f = \max\left(0.5, \min(D_{cycle}, D_{temp})\right)$$

- $D_{cycle} = 1 - k_{cycle} \cdot N_{cycle}$ (사이클 열화)
- $D_{temp} = 1 - k_{temp} \cdot \max(0, (T - 25)/10)$ (Arrhenius 온도 열화)

**시스템 효율**

시스템 전체 Round-Trip 효율은 가중평균이 아닌 **최효율 레이어 선택 운전** 기준:

$$\eta_{RT,sys} = \max_i \sqrt{\eta_{charge,i} \cdot \eta_{discharge,i}}$$

#### SOC 목표 범위

| Layer | SOC 목표 | 용도 |
|-------|---------|------|
| Supercap | 40~60% | 즉시 응답 여유 확보 |
| Li-ion | 20~80% | 일중 변동 대응 |
| Na-ion | 20~80% | 중주기 저비용 운전 |
| RFB | 30~70% | 장주기 대응 |
| CAES | 40~60% | 주간 저장 |
| H₂ | 10~90% | 계절 저장 (넓은 범위) |

#### 에너지 보존 검증

`operate_hess()` 반환값에 에너지 보존 검증 블록을 포함한다:

$$E_{requested} = E_{delivered} + E_{unallocated} + E_{loss}$$

극단 주파수($f < 1/\tau_{max,H_2}$ 또는 $f > 1/\tau_{min,SC}$)에서 미배분 전력은 `_unallocated_kw` 필드로 명시적 추적한다.

#### 가정 및 한계
- 각 레이어 독립 운전 가정 (레이어 간 직접 에너지 전달 미반영)
- 자기방전율은 상수 가정 (SOC/온도 의존성 미반영)
- Na-ion CAPEX $80/kWh은 2026 목표치이며 실증 수준은 $100~120/kWh
- BESS 2 GWh 설계 근거: worst-case 72시간 독립 운전 + AIDC 99.99% 가용성

---

### 2.3 M3. AIDC 부하 모듈

#### 목적
GPU 기반 AI 데이터센터의 시간별·분별 전력 수요 프로파일을 확률적으로 생성한다. 워크로드 믹스(LLM 추론, AI 훈련, MoE), PUE 티어, GPU burst 패턴을 반영한다.

#### GPU 및 PUE 사양

| GPU | 전력 (W) | 메모리 (GB) | FP16 (TFLOPS) |
|-----|---------|-------------|---------------|
| H100 SXM | 700 | 80 | 1,979 |
| B200 (Blackwell) | 1,000 | 192 | 2,500 (추정) |
| 차세대 (2027+) | 1,200 | 256 | 3,000 (추정) |

| PUE Tier | PUE | 냉각 방식 |
|----------|-----|----------|
| Tier 1 | 1.40 | 공냉 |
| Tier 2 | 1.20 | 하이브리드 |
| Tier 3 | 1.07 | 단상 액침 |
| Tier 4 | 1.03 | 이상 액침 |

#### 부하 모델

$$P_{total} = P_{GPU} \cdot (1 + r_{IT,add}) \cdot PUE$$

여기서 $r_{IT,add} = 0.12$ (CPU/Memory/Network/Storage 부하, GPU 대비 12%)이다.

**워크로드별 활용률 패턴**

| 워크로드 | Base 활용률 | Peak 활용률 | Burst 빈도 (/hr) | 시간 패턴 |
|---------|-----------|-----------|-----------------|----------|
| LLM 추론 | 55% | 98% | 10 | 사용자 트래픽 (오후 피크) |
| AI 훈련 | 85% | 100% | 2 | 야간 배치 (checkpoint spike) |
| MoE | 40% | 95% | 20 | 불규칙 Expert 활성화 |

Ref: Google TPU Report, Meta LLaMA 3 GPU failure analysis, MLPerf benchmark

#### 가정 및 한계
- GPU 전력은 정격 기준 (DVFS에 의한 동적 전력 조절 미반영)
- PUE는 상수 가정 (외기온도 의존 PUE 변동 미반영)
- GPU 장애 확률: Meta 기준 3시간당 1회 → 시간당 5% 확률로 모델링

---

### 2.4 M4. DC Bus 전력 분배

#### 목적
380V DC Bus를 중심으로 PV, HESS, H₂, 그리드, AIDC 간 순시 전력 균형을 관리한다. SiC/GaN 전력 변환기의 효율을 반영하고, 잉여/부족 전력의 우선순위 기반 배분을 수행한다.

#### 컨버터 효율

| 경로 | SiC (기본) | GaN (고효율) |
|-----|-----------|-------------|
| PV → DC Bus | 98.5% | 99.5% |
| DC Bus → BESS | 97.5% | 99.0% |
| DC Bus → Supercap | 99.0% | 99.5% |
| DC Bus → 전해조 | 97.0% | 98.5% |
| 연료전지 → DC Bus | 97.0% | 98.5% |
| DC Bus → AIDC | 96.0% | 98.0% |
| 그리드 (양방향) | 97.0% | 98.5% |

#### 전력 배분 우선순위

**잉여 시 (PV > AIDC)**:
1. BESS 충전 (SOC < 90%)
2. 수전해 (장주기 저장)
3. 계통 판매
4. 출력 제한 (curtailment)

**부족 시 (PV < AIDC)**:
1. BESS 방전 (SOC > 20%)
2. 연료전지 (H₂ 저장 활용)
3. 계통 구매
4. 부하 차단 (비상, warning 발생)

#### 에너지 균형

$$\sum P_{supply} \cdot \eta_{supply} = \sum P_{demand} / \eta_{demand}$$

$$P_{balance} = P_{PV} \cdot \eta_{PV \to Bus} + P_{BESS,dis} \cdot \eta_{BESS} + P_{FC} \cdot \eta_{FC} + P_{grid,imp} \cdot \eta_{grid} - P_{AIDC}/\eta_{AIDC} - P_{BESS,chg}/\eta_{BESS} - P_{elec}/\eta_{elec} - P_{grid,exp}/\eta_{grid}$$

$P_{balance} \approx 0$이 에너지 보존 조건이다.

#### 가정 및 한계
- DC Bus 전압 안정성은 Tier 1 AI-EMS에서 별도 관리
- 컨버터 효율은 부분 부하에서도 상수 가정 (실제로는 부하율 의존)
- 스위칭 손실의 주파수 의존성 미반영

---

### 2.5 M5. H₂ 시스템 (Power-to-Gas-to-Power)

#### 목적
PEM 전해조 + 압축 H₂ 저장 + PEM 연료전지로 구성된 장주기 에너지 저장 및 변환 시스템을 모델링한다.

#### 시스템 구성

| 구성요소 | 사양 | 효율 | 운전 온도 | 시동 시간 |
|---------|------|------|----------|----------|
| PEM 전해조 | 50 MW | 65% (LHV) | 80°C | 15 min |
| PEM 연료전지 | 50 MW | 55% (LHV) | 80°C | 5 min |
| H₂ 저장 | 150 ton, 350 bar | 95% (압축 효율) | — | — |

Ref: IRENA Green Hydrogen Cost Reduction 2024 (전해조), DOE Hydrogen Program Record 2024 (연료전지)

#### Round-Trip 효율

$$\eta_{RT,elec} = \eta_{elec} \times \eta_{FC} = 0.65 \times 0.55 = 0.3575$$

config 값 37.5%는 BOP(Balance of Plant) 보정 포함. IEA Global Hydrogen Review 2023 기준.

CHP(열병합) 모드: $\eta_{RT,CHP} = 0.825$ (폐열 80~85% 회수)

#### 전기화학 모델

**Nernst 전압**:

$$E(T, p) = E_0 + \frac{\Delta S}{2F}(T - 298.15) + \frac{RT}{2F}\ln(p)$$

여기서 $E_0 = 1.229$ V, $F = 96485.33$ C/mol, $R = 8.314$ J/(mol·K)이다.

**SOEC 효율 계산**:

$$\eta_{elec} = \eta_{Faraday} \cdot \eta_{voltage} \cdot f_{thermal} \cdot D_f$$

- $\eta_{Faraday} = 1.0 - 0.05 \cdot (J/J_{max})^2$
- $\eta_{voltage} = E_{Nernst} / E_{actual}$, $E_{actual} = E_{Nernst} \cdot (1.2 + 0.3 \cdot r_{load})$
- $f_{thermal} = \min(1.15, 1.0 + 0.0008 \cdot (T - 25))$
- 상한: $\eta_{elec} \leq \eta_{nominal}$ (config 기반)

**H₂ 생산량**:

$$m_{H_2} = \frac{P \cdot \Delta t \cdot \eta_{elec}}{HHV_{H_2}}$$

여기서 $HHV_{H_2} = 39.39$ kWh/kg, $LHV_{H_2} = 33.33$ kWh/kg이다.

#### H₂ 저장

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| 용량 | 150,000 kg (150 ton) | 압축 수소 |
| 압력 | 350 bar | — |
| 누출율 | 0.1%/day | — |
| 압축 에너지 | 3.0 kWh/kg | — |

#### BNEF 2025 국가별 LCOH ($/kg H₂)

| 국가 | LCOH | 국가 | LCOH |
|------|------|------|------|
| 중국 | 3.2 | 미국(텍사스) | 6.5 |
| 사우디 | 3.9 | 독일 | 8.0 |
| 인도 | 5.0 | **한국** | **8.5** |
| 스페인 | 5.9 | 일본 | 10.2 |

Ref: BloombergNEF 2025 Hydrogen Levelized Cost Report

#### 가정 및 한계
- PEM 기준 모델링 (SOEC/SOFC 고온 시스템은 별도 시나리오)
- 스택 열화: 전해조 0.5%/1000h, 연료전지 0.3%/1000h (선형)
- 열사이클 열화: 1000회당 0.1% (전해조), 0.05% (연료전지)
- H₂ 저장: compressed만 구현, metal hydride는 구조만 설계

---

### 2.6 M6. AI-EMS (3-Tier 에너지 관리)

#### 목적
3계층(실시간/예측/전략) 에너지 관리 시스템으로 PV-HESS-H₂-Grid 자원의 최적 디스패치를 수행한다.

#### 3-Tier 제어 구조

```
┌─────────────────────────────────────────────────────────────┐
│ Tier 3: 전략 최적화 (1 hr ~ 24 hr)                         │
│  - 24시간 최적 운전 스케줄                                  │
│  - 경제적 최적화 (SMP 시간대별 거래)                         │
│  - 유지보수 스케줄                                          │
├─────────────────────────────────────────────────────────────┤
│ Tier 2: 예측 제어 (15 min)                                  │
│  - PV/부하 4시간 예측                                       │
│  - LP 기반 최적 디스패치 (scipy.optimize.linprog)           │
│  - Merit Order: PV→AIDC > HESS > H₂ FC > Grid              │
├─────────────────────────────────────────────────────────────┤
│ Tier 1: 실시간 제어 (1 ms)                                  │
│  - MPPT 추적 (99% 추적 속도)                                │
│  - DC Bus 전압 안정화 (380V ±2%)                            │
│  - HESS 충방전 실시간 배분                                  │
└─────────────────────────────────────────────────────────────┘
```

#### Tier 2 LP 디스패치 정식화

**결정 변수** ($n = 8$):

$x_0$: PV→AIDC, $x_1$: PV→HESS, $x_2$: PV→Grid, $x_3$: H₂ 전해, $x_4$: HESS→AIDC, $x_5$: H₂ FC, $x_6$: Grid→AIDC, $x_7$: Curtailment

**목적 함수** (비용 최소화):

$$\min \; c^T x$$

여기서 비용 벡터:

| 변수 | 비용 계수 | 의미 |
|-----|----------|------|
| $x_0$ (PV→AIDC) | $-P_{grid}$ | 자가소비 (그리드 회피 = 최대 절감) |
| $x_1$ (PV→HESS) | $-P_{grid} \cdot \max(0, 1.2 - SOC_{HESS})$ | SOC 낮을수록 충전 가치 ↑ |
| $x_2$ (PV→Grid) | $-P_{sell}$ | 매전 수입 ($P_{sell} = 0.85 \cdot P_{grid}$) |
| $x_3$ (H₂ 전해) | $-P_{grid} \cdot \max(0, 0.95 - L_{H_2})$ | H₂ 저장 낮을수록 ↑ |
| $x_4$ (HESS→AIDC) | $0.25 \cdot P_{grid}$ | 열화 + 기회비용 |
| $x_5$ (H₂ FC) | $0.50 \cdot P_{grid}$ | 왕복 효율 ~35-45% 반영 |
| $x_6$ (Grid→AIDC) | $P_{grid}$ | 최후 수단 (최고 비용) |
| $x_7$ (Curtailment) | 1,000 | 낭비 패널티 |

**등식 제약**:

PV 균형: $x_0 + x_1 + x_2 + x_3 + x_7 = P_{PV}$

부하 균형: $x_0 + x_4 + x_5 + x_6 = P_{load}$

**부등식 제약**: SOC/저장 수준에 따른 상한 적용

#### 에너지 보존 검증

$$P_{PV,out} = x_0 + x_1 + x_2 + x_3 + x_7 \approx P_{PV,in}$$

오차 허용: $|P_{PV,out} - P_{PV,in}| \leq 0.01 \cdot \max(1, P_{PV,in})$

#### Tier 1 제어 파라미터

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| MPPT 추적 속도 | 99% | — |
| DC Bus 목표 전압 | 380 V | — |
| 전압 허용 범위 | ±2% | 372.4~387.6 V |
| HESS SOC 목표 | SC: 50%, BESS: 50% | — |
| Tier 2 주기 | 15 min | 예측 갱신 |
| Tier 3 주기 | 1 hr | 전략 갱신 |

#### 가정 및 한계
- Tier 2 LP는 확정적 최적화 (확률적 MPC 미구현)
- PV 예측: Persistence 모델 + 감쇠 (LSTM/Transformer 기반 예측은 향후 과제)
- PINN(Physics-Informed Neural Network) 접근은 아키텍처 설계 완료, 학습 데이터 수집 중

---

### 2.7 M7. 탄소 회계

#### 목적
GHG Protocol 기반 Scope 1/2/3 탄소 배출을 추적하고, K-ETS 탄소 거래 및 EU CBAM 비용을 산정한다.

#### Scope 경계 정의

**Scope 1 (직접 배출)**:
- PEM 전해조: 전기분해이므로 **직접 배출 0**
- 비상 디젤 발전기: $E_{S1} = V_{diesel} \times 2.68 / 1000$ (tCO₂), 여기서 $V_{diesel}$은 리터 단위 디젤 소비량

**Scope 2 (전력 간접 배출)**:

$$E_{S2} = Q_{grid} \times EF_{grid}$$

여기서 $Q_{grid}$는 계통 전력 구매량(MWh), $EF_{grid} = 0.4594$ tCO₂/MWh (2024 한국 기준)이다.

**Scope 3 (공급망 배출)**:

$$E_{S3} = \frac{C_{PV} \times EF_{PV} + C_{BESS} \times EF_{BESS} + C_{H_2} \times EF_{H_2}}{T_{life}}$$

| 설비 | 배출계수 | 단위 |
|------|---------|------|
| PV | 40 tCO₂/MW | 제조 기준 |
| BESS | 65 tCO₂/MWh | 제조 기준 |
| H₂ | 30 tCO₂/MW | 제조 기준 |

**탄소 회피량**:

$$E_{avoided} = Q_{self} \times EF_{grid}$$

여기서 $Q_{self}$는 재생에너지 자가소비량(MWh)이다.

**순배출**:

$$E_{net} = E_{S1} + E_{S2} + E_{S3} - E_{avoided}$$

#### K-ETS 및 CBAM

| 파라미터 | 값 | 출처 |
|---------|-----|------|
| 그리드 배출계수 | 0.4594 tCO₂/MWh | 전력거래소 2024 |
| K-ETS 가격 | 25,000 ₩/tCO₂ | K-ETS 할당거래소 2024 |
| CBAM 가격 | 80 €/tCO₂ | EU-ETS (ICE ECX) |
| EUR/KRW | 1,450 | — |
| 탄소크레딧 가격 | 25,000 ₩/tCO₂ | — |

**CBAM 차등 비용**:

$$C_{CBAM} = E_{export} \times \max(0, P_{EU} \times R_{EUR/KRW} - P_{KETS})$$

#### 가정 및 한계
- Scope 3: 설비 제조만 포함, 운송/폐기/재활용 미반영
- 그리드 배출계수: 연간 평균 사용 (시간대별 한계 배출계수 미적용)
- K-ETS/CBAM 가격은 시나리오 분석으로 변동성 반영

---

### 2.8 M8. 그리드 인터페이스

#### 목적
한전 계통과의 양방향 전력 거래, 보호 계전, 보조서비스(주파수·전압 응답), 경제적 급전을 관리한다.

#### 보호 계전 설정

| 보호 항목 | 기준값 | 동작 |
|---------|-------|------|
| 과전압 | 1.1 p.u. (110%) | Trip |
| 저전압 | 0.9 p.u. (90%) | Trip |
| 과주파수 | 50.5 Hz | Trip |
| 저주파수 | 49.5 Hz | Trip |
| 최소 역률 | 0.95 | Warning |
| 재연결 지연 | 300 s (5분) | — |

#### 전력조류 계산 (PCC)

$$|I| = \frac{|S|}{|V|}, \quad \Delta V = |I| \cdot Z_{grid}, \quad V_{PCC} = V_{grid} - \Delta V$$

$$P_{loss} = |I|^2 \cdot Z_{grid} \cdot S_{base} \times 0.1$$

#### 주파수 응답 (Droop 제어)

$$\Delta P = -K \cdot \Delta f, \quad K = \frac{P_{rated}}{\delta \cdot f_0}$$

여기서 $\delta$는 Droop 특성(%, 기본 5%), $f_0 = 50$ Hz, 데드밴드 ±0.02 Hz이다.

#### SMP 시간대별 가격 패턴

기준 SMP: 80,000 ₩/MWh

| 계절 | 특성 | 피크 시간 | 피크 배율 |
|------|------|----------|----------|
| 여름 | 에어컨 피크 | 14~16시 | ×1.45~1.50 |
| 겨울 | 이중 피크 (난방) | 8~9시, 18~19시 | ×1.30~1.45 |
| 봄/가을 | 완만 | 11~13시 | ×1.20~1.25 |

경부하(심야 2~5시): ×0.60~0.65

#### 그리드 요금 구조

| 항목 | 값 | 비고 |
|------|-----|------|
| SMP 기준 | 80,000 ₩/MWh | — |
| REC 가격 | 25,000 ₩/MWh | — |
| REC 가중치 (태양광) | 1.2 | — |
| 탄소 가격 | 22,500 ₩/tCO₂ | K-ETS |
| 계통이용요금 | 1,000,000 ₩/MW/월 | — |
| 송전손실률 | 5% | — |

---

### 2.9 M9. 경제 최적화

#### 목적
CAPEX/OPEX 모델, NPV/IRR/LCOE/Payback 계산, 학습곡선, Monte Carlo 민감도 분석, 토네이도 차트를 통해 투자 의사결정을 지원한다.

**과장 금지 원칙**: 모든 결과는 범위+신뢰구간을 필수로 제시한다.

상세는 [제4장 경제성 분석 프레임워크](#4-경제성-분석-프레임워크)에서 다룬다.

---

### 2.10 M10. 기상 모듈

#### 목적
한국 중부(서울, 37.5°N, 127.0°E) 기준 합성 TMY(Typical Meteorological Year) 데이터를 생성한다.

#### 기상 파라미터

| 파라미터 | 값 | 출처 |
|---------|-----|------|
| 위도 | 37.5°N | 서울 |
| 경도 | 127.0°E | 서울 |
| 시간대 | KST (UTC+9) | — |
| 연간 GHI | 1,350 kWh/m² | KMA 기상청 TMY3 |
| 연평균 피크 일조 | 3.5 시간 | — |

#### 월별 패턴

| 월 | GHI 정규화 | 평균 기온 (°C) |
|---|-----------|---------------|
| 1 | 0.45 | -2 |
| 2 | 0.55 | 1 |
| 3 | 0.75 | 7 |
| 4 | 0.90 | 14 |
| 5 | 1.00 | 20 |
| 6 | 0.95 | 25 |
| 7 | 0.85 | 27 |
| 8 | 0.90 | 28 |
| 9 | 0.80 | 23 |
| 10 | 0.70 | 16 |
| 11 | 0.55 | 8 |
| 12 | 0.40 | 1 |

7월 장마: 구름 확률 70%로 설정.

---

### 2.11 M11. 정책 시뮬레이터

#### 목적
K-ETS 탄소가격, REC 시장, EU CBAM, RE100, 전력수급기본계획, Ratepayer Protection 정책이 경제성에 미치는 영향을 분석한다.

#### 정책 시나리오

| 정책 | 현행 | 중도 | 공격적 |
|------|------|------|--------|
| K-ETS (₩/tCO₂) | 25,000 | 50,000 | 100,000 |
| REC 가격 (₩/MWh) | 25,000 | 35,000 | 50,000 |
| CBAM (€/tCO₂) | 80 | 120 | 200 |

**Ratepayer Protection Pledge** (2026 White House Hyperscaler Pledge):
- AIDC 자체 전력 확보 → 일반 소비자 요금 전가 방지
- 자가발전 요구: 80%
- LCOE 영향: ×1.10
- 그리드 가격 영향: -5%

---

### 2.12 M12. 산업 상용화 모델

#### 목적
한국 CSP별 맞춤 분석, BYOG(Bring Your Own Grid) 시나리오, 규모의 경제(power law) 스케일링을 수행한다.

#### 한국 CSP 프로파일

| CSP | 용량 (MW) | PUE | 수출 비중 | RE100 | CAPEX 배율 |
|-----|----------|-----|----------|-------|-----------|
| 삼성 평택 | 500 | 1.15 | 60% | ✓ | 1.00 |
| SK 이천 | 300 | 1.12 | 70% | ✓ | 1.05 |
| 네이버 세종 | 200 | 1.08 | 20% | ✓ | 0.95 |
| 카카오 안산 | 100 | 1.10 | 10% | ✓ | 1.10 |

#### 스케일링 모델

CAPEX 스케일링 (규모의 경제):

$$C(S) = C_{100} \cdot \left(\frac{S}{100}\right)^{0.85}$$

여기서 $S$는 목표 용량(MW), $C_{100}$은 100MW 기준 CAPEX이다. 지수 0.85는 대규모 설비의 단위비용 감소를 반영한다.

---

### 2.13 M13. 투자 의사결정 대시보드

#### 목적
What-if 분석, MC 시뮬레이션, Go/No-Go 매트릭스, 보조금 민감도 분석을 통합한 투자 의사결정 도구이다.

#### Go/No-Go 기준

| 기준 | 임계값 | 판정 |
|------|-------|------|
| IRR | ≥ 5% | Pass/Fail |
| NPV | ≥ 0 | Pass/Fail |
| Payback | ≤ 15년 | Pass/Fail |
| P(NPV>0) | ≥ 50% | Pass/Fail |

- 4/4 Pass → **GO**
- 3/4 Pass → **CONDITIONAL GO** (조건부 승인)
- ≤ 2/4 Pass → **NO-GO** (투자 보류)

---

## 3. 데이터소스

### 3.1 기상 데이터

| DB | 설명 | 국가 | 용도 |
|----|------|------|------|
| KMA 기상청 TMY3 | 한국 표준기상연도 | 🇰🇷 | GHI, 기온 기준 |
| NREL NSRDB | 미국 태양자원 DB | 🇺🇸 | 벤치마크 |
| CMA TRY | 중국 표준기상연도 | 🇨🇳 | 벤치마크 |
| JMA AMeDAS | 일본 자동기상관측 | 🇯🇵 | 벤치마크 |
| DWD TRY 2024 | 독일 표준기상연도 | 🇩🇪 | 벤치마크 |

### 3.2 에너지 가격

| DB | 설명 | 갱신 주기 |
|----|------|----------|
| KEPCO 전기요금표 | 한국 산업용(갑) II | 연간 |
| KPX 전력시장 | SMP, 보조서비스 | 실시간 |
| EIA Commercial | 미국 상업용 전력가격 | 연간 |
| NDRC Industrial | 중국 산업용 요금 | 연간 |
| METI Industrial | 일본 산업용 요금 | 연간 |
| Destatis Industrial | 독일 산업용 요금 | 연간 |

### 3.3 탄소 시장

| DB | 설명 | 갱신 주기 |
|----|------|----------|
| K-ETS 할당거래소 | 한국 배출권 가격 | 일간 |
| EU-ETS (ICE ECX) | EU 배출권 가격 | 일간 |
| Shanghai EEX | 중국 탄소시장 | 주간 |
| Ember Climate API | 전세계 실시간 탄소강도 | 분기 |

### 3.4 기술 비용

| DB | 설명 | 갱신 주기 |
|----|------|----------|
| NREL ATB 2024 | 미국 연간기술기준선 | 연간 |
| IRENA RENEWCOST | 전세계 재생에너지 비용 | 연간 |
| Fraunhofer ISE LCOE | 독일 LCOE 연구 | 연간 |
| BNEF H₂ LCOH | 국가별 수소 균등화비용 | 연간 |
| Lazard LCOS 2024 | 저장장치 균등화비용 | 연간 |
| METI Cost Verification | 일본 비용검증위원회 | 연간 |
| CPIA Annual Report | 중국 PV산업연례보고 | 연간 |

### 3.5 정책 및 제도

| DB | 설명 |
|----|------|
| 전력거래소 배출계수 | 한국 그리드 탄소강도 |
| EPA eGRID 2024 | 미국 그리드 배출계수 |
| MEE China Grid EF | 중국 그리드 배출계수 |
| MOE Japan Grid EF | 일본 그리드 배출계수 |
| UBA Germany Grid EF | 독일 그리드 배출계수 |
| GX Surcharge | 일본 탄소가격 (추정) |

### 3.6 벤치마크 자동 업데이트 API

| API | URL | 갱신 주기 | 제공 필드 |
|-----|-----|----------|----------|
| Ember Carbon Intensity | ember-climate.org/v1/carbon-intensity | 분기 | 탄소강도 |
| EU-ETS Price | ember-climate.org/v1/carbon-price | 월간 | 탄소가격 |
| NREL ATB | atb.nrel.gov/electricity | 연간 | LCOE, CF |
| IRENA RENEWCOST | irena.org/Data | 연간 | LCOE |

마지막 업데이트: 2026-02-22

---

## 4. 경제성 분석 프레임워크

### 4.1 CAPEX 구조

**에너지 인프라 (경제성 분석 대상)**:

| 항목 | CAPEX (억원) | 비고 |
|------|-------------|------|
| PV 100MW | 1,500 | — |
| BESS 2GWh | 4,000 | Li-ion + Na-ion |
| Supercap | 500 | — |
| H₂ System | 3,000 | 전해조 + 저장 + 연료전지 |
| DC Bus | 500 | SiC/GaN 변환기 |
| Grid Interface | 200 | — |
| AI-EMS | 300 | — |
| **소계** | **10,000** | **에너지 인프라** |

AIDC 시설/인프라: 12,500억원 (BAU 동일, 경제성 분석에서 제외)

R&D: 400억원 (분리)

> ※ AIDC 시설비를 제외하는 근거: BAU(기존 그리드 의존 AIDC) 대비 **추가 투자(incremental CAPEX)**로 분석. AIDC 시설은 어차피 건설하므로 에너지 인프라 10,000억원이 순수 추가 투자이다.

### 4.2 OPEX

$$OPEX_{annual} = (C_{maint} + C_{labor} + C_{ins}) \times (1 + r_{inf})^t$$

| 항목 | 값 | 비고 |
|------|-----|------|
| 유지보수 | CAPEX × 0.8% | 에너지 설비 기준 |
| 인건비 | 50억원/년 | — |
| 보험 | CAPEX × 0.2% | — |
| 인플레이션 | 2%/년 | — |

### 4.3 수익 모델

**기본 수익**:

| 항목 | 단가 | 산출식 |
|------|------|--------|
| 전력 자급 절감 | 80,000 ₩/MWh | 자가소비 × 단가 |
| 잉여 판매 (SMP) | 70,000 ₩/MWh | 잉여 × 단가 |
| REC | 25,000 ₩/MWh × 1.2 | PV 발전 × 단가 × 가중치 |
| 탄소크레딧 | 25,000 ₩/tCO₂ | 회피량 × 단가 |

**추가 수익 (630억원/년)**:

| 항목 | 금액 (억원/년) | 산출근거 | 출처 |
|------|--------------|---------|------|
| 수요요금 절감 | 280 | 피크 20MW 절감 × 기본요금 14,000₩/kW/월 × 12 = 33.6억 + TOU 차익 246.4억 (경부하 55₩/kWh, 최대부하 110₩/kWh, 일 8h 피크시프트, 365일) | KEPCO 산업용(갑) 2024 |
| 그리드 안정성 | 220 | 정전회피 65억 + FR 서비스 40억 + RE100 프리미엄 15억 + 용량시장/혼잡완화 100억 | EPRI Value of DER 2023, KPX 보조서비스 2024 |
| BESS 차익거래 | 130 | 에너지 아비트라지 42.7억 (2GWh × 65₩/kWh × 365d × η90%) + 보조서비스 50억 + 피크셰이빙 37.3억 | Lazard LCOS 2024, KPX 2024 |
| **합계** | **630** | | |

### 4.4 재무 지표

$$NPV = -CAPEX + \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t}$$

$$IRR: \sum_{t=0}^{T} \frac{CF_t}{(1+IRR)^t} = 0 \quad \text{(Bisection method)}$$

$$LCOE = \frac{\sum_{t=0}^{T} (CAPEX_t + OPEX_t)/(1+r)^t}{\sum_{t=1}^{T} E_t/(1+r)^t} \quad [₩/\text{MWh}]$$

$$Payback = t^* \text{ such that } \sum_{t=1}^{t^*} CF_t \geq CAPEX$$

| 파라미터 | 값 |
|---------|-----|
| 할인율 | 5% |
| 프로젝트 수명 | 20년 |
| 인플레이션 | 2%/년 |

### 4.5 학습곡선

| 기술 | 연간 비용 감소율 | 20년 후 비용 비율 |
|------|----------------|-----------------|
| PV | -7%/yr | 23% |
| BESS | -10%/yr | 12% |
| H₂ | -8%/yr | 19% |

### 4.6 Monte Carlo 시뮬레이션 (10,000회)

**변동 변수**:

| 변수 | 표준편차 (%) | 분포 | 비고 |
|------|------------|------|------|
| PV 효율 | 5 | 정규 | Clipping: 50~150% |
| 전력 가격 | 15 | 정규 | Clipping: 50~200% |
| 탄소 가격 | 20 | 정규 | Clipping: 30~300% |
| 할인율 | 10 | 정규 | Clipping: 50~200% |
| 부하 변동 | 10 | 정규 | Clipping: 70~130% |

> ⚠️ 정규분포 가정의 한계: 전력/탄소 가격은 fat-tail(log-normal 또는 jump-diffusion)이 현실적. 현재 정규분포+clipping은 극단 시나리오를 과소평가할 수 있다. 보수적 가정으로 간주한다.

### 4.7 토네이도 민감도 분석

각 변수를 low/high로 변경 후 `run_base_case()` 재실행하여 실제 IRR 변화를 측정한다.

| 변수 | 하한 배율 | 상한 배율 |
|------|----------|----------|
| 전력가격 | 0.8 | 1.3 |
| 탄소가격 | 0.5 | 2.0 |
| PV효율 | 0.85 | 1.15 |
| 할인율 | 0.7 | 1.3 |
| CAPEX | 0.8 | 1.2 |
| 부하변동 | 0.9 | 1.1 |

---

## 5. 시연 시나리오 가이드

### 5.1 프리셋 A: 기본 100MW AIDC

| 파라미터 | 설정 |
|---------|------|
| PV 타입 | c-Si |
| GPU | H100 |
| PUE | Tier 3 (1.07) |
| 용량 | 100 MW |

**기대 결과**: 연간 PV 발전 ~150 GWh, 자급률 ~20%, Base Case IRR ~4.5%

**조작법**: 사이드바 "📋 시연 시나리오" → Preset A 선택

### 5.2 프리셋 B: CSP 비교

| 파라미터 | 설정 |
|---------|------|
| PV 타입 | Tandem |
| GPU | B200 |
| PUE | Tier 4 (1.03) |
| 용량 | 150 MW |

**기대 결과**: Tandem PV 고효율 + B200 고전력 시나리오, CSP별 경제성 차이 확인

### 5.3 프리셋 C: 정책 시나리오

| 파라미터 | 설정 |
|---------|------|
| 탄소가격 | 80,000 ₩/tCO₂ |
| 전력단가 | 120,000 ₩/MWh |
| REC 가중치 | 2.0 |

**기대 결과**: 복합 시나리오 IRR 12~15%, Go/No-Go → GO

### 5.4 프리셋 D: Solar Battery 2030+

| 파라미터 | 설정 |
|---------|------|
| PV 타입 | Infinite |
| GPU | next_gen |
| 할인율 | 4% |

**기대 결과**: 미래 기술 시나리오 확인, Solar Battery H₂ 생산 경로 비교

---

## 6. Expansion 모듈

### 6.1 Stress Engine

극한 시나리오(3일 연속 흐림, 폭염+피크부하, GPU 대량 장애 등)에서의 시스템 회복력을 검증하는 스트레스 테스트 프레임워크.

### 6.2 Data Survival

데이터 무결성 및 시뮬레이션 상태의 장기 보존. 체크포인트 기반 상태 저장 및 재현성 보장.

### 6.3 Unified Analytics

13개 모듈의 결과를 통합 대시보드로 시각화. 크로스 모듈 KPI(자급률, 탄소중립률, LCOE 등) 실시간 추적.

---

## 7. Hyperscaler 전략

### 7.1 글로벌 CSP 에너지 전략 비교

| CSP | 전략명 | 에너지 믹스 | 전략 유형 | 특징 |
|-----|--------|-----------|----------|------|
| Google | Co-located Renewables | Solar 50%, Wind 30%, Grid 20% | BYPASS_QUEUE | AES deal, 현장 재생에너지 + 장기 PPA |
| Amazon | Dedicated Gas Generation | Gas 65%, Battery 15%, Grid 20% | DEDICATED_GEN | NIPSCO GenCo, 전용 가스발전 2.6GW |
| Meta | Behind-the-Meter Gas | Gas 70%, Grid 30% | MULTI_SITE | 모듈형 가스터빈, 다중 사이트 |
| Microsoft | Grid Partnership | Grid 60%, Wind 25%, Solar 15% | PAY_AND_BUILD | MISO 송전망 직접 투자, 7.9GW 계약 |
| Samsung SDS | 한국형 그리드 의존 | Grid 85%, Solar 10%, ESS 5% | GRID_DEPENDENT | 수원/화성 데이터센터 |
| Naver | 친환경 DC | Grid 60%, Fuel Cell 25%, Solar 15% | HYBRID | 세종 데이터센터, PPA + 연료전지 |

### 7.2 전략별 경제성 비교

| CSP | LCOE (₩/kWh) | 탄소 (tCO₂/MWh) | 그리드 의존도 |
|-----|-------------|-----------------|-------------|
| Google | 64.0 | 0.092 | 20% |
| Amazon | 97.5 | 0.259 | 20% |
| Meta | 87.0 | 0.259 | 30% |
| Microsoft | 69.5 | 0.276 | 60% |
| Samsung SDS | 79.0 | 0.391 | 85% |
| Naver | 74.5 | 0.276 | 60% |

---

## 8. 검증 결과

### 8.1 테스트 현황

**92/92 테스트 통과** (2026-03-02 기준)

13개 모듈 + Expansion + Solar Battery 전체 테스트 스위트.

### 8.2 Advisory Board 리뷰 주요 포인트

**수정 완료 (Must Fix 4건)**:

| 항목 | 내용 | 상태 |
|------|------|------|
| MF-1 | PV 온도계수 주석/의미론 명확화 (상대 vs 절대 구분) | ✅ |
| MF-2 | HESS 미배분 전력 에너지 보존 (`_unallocated_kw` 추적) | ✅ |
| MF-3 | H₂ RT 효율 config↔모델 정합 (PEM 기준 통일, 37.5%) | ✅ |
| MF-4 | 경제성 추가수익 630억 산출근거 문서화 | ✅ |

**주요 강점** (Arbiter 평가):

1. 6-Layer HESS 주파수 분리 제어 — 학술적 독창성
2. Monte Carlo + 토네이도 — "과장하지 않고 불확실성을 정량화"
3. CSP별 맞춤 분석 — 실명 프로파일로 현실감
4. 92/92 테스트 통과 — 포괄적 검증
5. 국제 벤치마크 5개국 비교 — 글로벌 컨텍스트

**종합 평가**: B+ → A- (수정 후)

### 8.3 국제 벤치마크

| 국가 | PV 타입 | 일사량 (kWh/m²/yr) | 전력가격 ($/MWh) | 탄소강도 (gCO₂/kWh) | 탄소가격 ($/ton) |
|------|--------|-------------------|-----------------|-------------------|-----------------|
| 🇰🇷 한국 (본 DT) | Tandem | 1,340 | 90 | 415 | 20 |
| 🇺🇸 미국 | c-Si Bifacial | 1,800 | 65 | 370 | 0 |
| 🇨🇳 중국 | c-Si | 1,500 | 55 | 555 | 10 |
| 🇯🇵 일본 | c-Si + Perovskite | 1,200 | 150 | 450 | 5 |
| 🇩🇪 독일 | c-Si + Agri-PV | 1,050 | 180 | 350 | 55 |

---

*CEMS Digital Twin v5 Technical Document (Korean)*
*Generated: 2026-03-02*
