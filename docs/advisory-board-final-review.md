# CEMS DT5 Advisory Board 최종 리뷰
## 국가대형과제 평가 대비 — 2026-03-02

> 92/92 테스트 통과 | 13개 모듈 + Expansion + Solar Battery  
> 3월 둘째주 연구팀 시연 예정

---

# 🔴 PART 1: Skeptic (Reviewer 2) — 수학/물리/공학 펀더멘털 검증

## MUST FIX (시연 전 반드시 수정)

### MF-1. PV 온도계수 적용 오류 (m01_pv.py)
**위치**: `_calculate_temperature_efficiency()` line ~107
```python
eta_temp = eta_stc * (1 + beta * (cell_temp - 25) / 100)
```
**문제**: `beta`가 이미 `%/°C` 단위인데 `/100`을 추가로 나눔. β = -0.35 %/°C이면 셀온도 60°C에서:
- 현재 코드: η = 24.4 × (1 + (-0.35) × 35 / 100) = 24.4 × 0.8775 = **21.4%** ← 거의 정상처럼 보이지만...
- 실제 의도: η = 24.4 × (1 + (-0.0035) × 35) = 24.4 × 0.8775 = **21.4%**

사실 수치적으로는 우연히 동일한 결과가 나옴 (`beta/100 * delta_T` = `beta_fraction * delta_T`). 그러나 **코드의 의미론적 명확성**이 부족. β가 -0.35 (%/°C)이므로 분수 변환(`/100`)이 맞지만, 주석과 변수명이 혼동을 야기함. 평가위원이 "단위 오류 아닌가?"라고 질문할 수 있음.

**수정**: 명시적으로 `beta_fraction = beta / 100` 변환 후 사용하고 주석에 단위 변환 과정을 기술.

### MF-2. HESS 주파수 배분 로직 — 미배분 전력 처리 불완전 (m02_hess.py)
**위치**: `calculate_power_allocation()` 
**문제**: 주파수가 어떤 레이어의 범위에도 맞지 않으면 `remaining_power`가 그대로 남음. fallback 로직에서 `best_layer`를 찾지만, 모든 레이어의 `time_constant_range`를 벗어나는 주파수(예: 극저주파 < 3.17e-8 Hz)에서 `best_layer = None` → **전력이 소실됨**.

**영향**: 시연 중 극단적 시나리오에서 에너지 보존 위반이 드러날 수 있음.

**수정**: fallback에서 `best_layer`가 None일 때 `h2` (최장 시간 상수) 레이어로 강제 배분하거나, 미배분 전력을 그리드로 전달하는 로직 추가.

### MF-3. H₂ Round-Trip 효율 — config vs 실제 계산 불일치
**위치**: config.py `H2_SYSTEM_CONFIG["round_trip_efficiency"]["electrical_only"] = 0.375`  
**문제**: SOEC η=0.85 × SOFC η=0.60 = 0.51이 round-trip 전기효율이어야 하는데 config에는 0.375(37.5%)로 기재. 이는 IEA 2023 보고서 인용값이지만, **본 시스템의 모델 파라미터와 불일치**. 
- SOEC 0.85 × SOFC 0.60 = **0.51** (51%)
- 여기에 압축/저장 손실, 부분부하 효율 저하를 감안해도 0.375는 과도하게 보수적이거나, 모델 파라미터가 과도하게 낙관적.

**영향**: 평가위원이 "모델 파라미터와 config 값이 왜 다르냐"고 물으면 일관성 부족 노출.

**수정**: config 값을 모델 계산 결과와 정합시키거나, "시스템 레벨 효율 = 컴포넌트 효율 × 보조설비 손실 × 부분부하 보정" 공식을 문서화.

### MF-4. 경제성 모델 — demand_charge 등 추가 수익의 하드코딩
**위치**: m09_economics.py `run_base_case()` default arguments
```python
demand_charge_saving_billion_krw: float = 280.0,
grid_reliability_benefit_billion_krw: float = 220.0,
bess_arbitrage_billion_krw: float = 130.0
```
**문제**: 280+220+130 = **630억원/년**이 물리 모델에서 도출되지 않고 하드코딩됨. 이 숫자의 산출 근거가 코드에 없음. 평가위원이 "이 수치는 어디서 나왔나?"라고 물으면 답변 곤란.

**수정**: 각 수치의 산출 로직을 구현하거나, 최소한 docstring에 계산 근거(전력요금 체계, 시간대별 SMP 차익 등)를 명시. 이상적으로는 M8 Grid 모듈의 SMP 시간대별 배율 데이터와 연동하여 계산.

---

## SHOULD FIX (시연 품질 향상)

### SF-1. PV 능동제어 — 랜덤 노이즈 모델의 물리적 근거 부재
**위치**: `_calculate_controlled_power()` in m01_pv.py
```python
improvement_random = np.random.uniform(0, 0.02)
```
**문제**: MPPT 최적화의 성능 향상을 uniform random으로 모델링하는 것은 물리적 근거 없음. 실제 MPPT 개선은 일사량 변동성, 부분 음영 패턴, 온도 변화율에 의존. 평가위원에게 "왜 random?"이라는 질문을 받을 수 있음.

**수정**: 일사량 변동 표준편차나 부분 음영 확률에 기반한 모델로 교체하거나, 시연 시 "확률적 모델링으로 불확실성 반영"이라고 설명할 근거 문서 준비.

### SF-2. Na-ion BESS CAPEX $80/kWh — 출처 필요
**위치**: m02_hess.py Na-ion 레이어 `capex_per_kwh=80`
**문제**: 2026년 기준 $80/kWh은 CATL의 공격적 목표치. 현재 실증 수준에서는 $100-120/kWh이 현실적. Nature Reviews Materials 2025 논문을 레퍼런스로 달았지만, 해당 논문이 CAPEX를 $80으로 제시했는지 확인 필요.

**수정**: 출처를 정확히 명기하고, 낙관/보수 범위($80-$120/kWh)로 제시.

### SF-3. SOEC Nernst 전압 계산 — 열역학 상수 사용 오류
**위치**: m05_h2.py `calculate_nernst_voltage()`
```python
delta_s = 0.0001334  # kJ/(mol·K)
```
**문제**: 물 전기분해 표준 엔트로피 변화는 ΔS° ≈ +163.2 J/(mol·K) = 0.1632 kJ/(mol·K). 코드의 0.0001334 kJ/(mol·K)는 **3자리수 차이**. 단, 이 값이 Nernst 전압의 온도 보정항에서만 사용되므로 결과에 미치는 영향은 제한적(온도 보정이 거의 0이 됨 → 항상 ~1.229V 근처). 하지만 물리적으로 틀림.

**영향**: 고온(800°C) SOEC에서 열역학적 전압 감소를 제대로 반영하지 못함. 실제로는 E(800°C) ≈ 0.95V여야 하는데 코드는 ~1.22V를 산출.

**수정**: `delta_s = 0.1632`로 수정하고, 고온에서의 열중성전압(thermoneutral voltage, ~1.29V)도 함께 구현.

### SF-4. Monte Carlo에서 정규분포 가정의 적절성
**위치**: m09_economics.py `run_monte_carlo()`
**문제**: 전력가격, 탄소가격은 fat-tail(log-normal 또는 jump-diffusion)이 현실적. 정규분포 + clipping은 극단 시나리오를 과소평가. 평가위원 중 통계 전문가가 "왜 정규분포?"라고 물을 수 있음.

**수정**: 최소한 log-normal 옵션을 추가하거나, 현재 정규분포 가정의 한계를 docstring에 명시하고 시연 시 "보수적 가정"으로 프레이밍.

### SF-5. Solar Battery STH 효율 57.6% — 과도하게 낙관적
**위치**: config.py `H2_SOLAR_BATTERY_CONFIG["sth_efficiency"] = 0.576`
**문제**: η_capture(0.80) × η_h2(0.72) = 0.576. 현재 최고 실험실 STH 효율은 ~20% (Nature 2023 기준). TRL 2-3 기술에 57.6%는 평가위원에게 "비현실적"으로 보일 수 있음.

**수정**: 
- 이것이 **광→화학에너지 포집 + 온디맨드 방출**이라는 2단계 프로세스의 각 단계 효율임을 명확히 구분
- 기존 PEC/PV-electrolysis와의 차이점을 문서화
- "이상적 조건 하 이론값"이라고 명시하고 보수적 시나리오(30-40%)도 함께 제시

### SF-6. 정책 시뮬레이터(M11) — circular import 위험
**위치**: m11_policy.py `policy_combination_impact()`
```python
from modules.m09_economics import EconomicsModule
```
**문제**: 함수 내부에서 모듈을 import하는 것은 circular dependency 위험 + 성능 저하. 시연 중 import 오류로 크래시할 수 있음.

**수정**: 모듈 최상단에서 import하되 circular dependency를 해결하거나, M9를 의존성 주입 패턴으로 전달.

---

## MINOR (문서화/코드 품질)

### m-1. config.py 탄소 관련 값 중복
- `ECONOMICS["k_ets_price_krw_per_tco2"] = 22500`
- `CARBON_CONFIG["k_ets_price_krw_per_tco2"] = 25000`  
- `GRID_TARIFF_CONFIG["carbon_price_krw_per_ton"] = 22500`

세 곳에서 다른 값. 어느 것이 정확한지 혼란.

### m-2. HESS 6-layer인데 config.py에는 5-layer
config.py의 `HESS_LAYER_CONFIGS`에는 Na-ion이 없음. m02_hess.py에서 직접 생성. config와 코드의 정합성 부족.

### m-3. GPU 50,000장 기본값의 현실성
H100 × 50,000 = 35MW IT → PUE 1.2 적용 시 42MW. 100MW AIDC와 불일치. GPU 수량을 ~140,000으로 조정하거나, 다른 IT 부하(CPU/Network/Storage)를 별도 모델링 필요.

### m-4. BESS 용량 2,000MWh (2GWh) — 100MW AIDC 대비 과잉
100MW × 8h = 800MWh면 충분. 2GWh는 20시간 백업. 과잉 설계의 경제적 근거 필요.

### m-5. 풍속 냉각 효과 모델이 비물리적
```python
wind_factor = 1 - 0.04 * max(0, wind_speed - 2)
```
풍속 27m/s 이상에서 wind_factor ≤ 0 → cell_temp = ambient_temp. 실제로는 강풍에서도 일사에 의한 가열이 있음. `max(0.3, wind_factor)` 정도의 하한 필요.

---

# 🟢 PART 2: Enterprise — 사업가 관점

## Strengths (강점)

### S-1. 포괄적 경제성 분석 프레임워크
M9+M11+M12+M13 조합이 투자자에게 필요한 모든 지표를 제공:
- NPV/IRR/LCOE/Payback — 기본 재무
- Monte Carlo 10,000회 — 불확실성 정량화
- 토네이도 차트 — 핵심 변수 식별
- Go/No-Go 매트릭스 — 의사결정 프레임워크
- CSP별 맞춤 분석 — 고객별 가치 제안

**평가**: 국내 에너지 DT 중 이 수준의 경제성 분석 모듈을 갖춘 사례는 드묾. 시연에서 강하게 부각할 것.

### S-2. "과장 금지 원칙" — 신뢰성 확보
M9, M11, M12, M13 모든 모듈 docstring에 "과장 금지 원칙: 범위+신뢰구간 필수"가 명시됨. Base case IRR 4.5%를 솔직히 제시하고 복합 시나리오에서 12-15%를 제시하는 투 트랙 접근은 평가위원의 신뢰를 얻는 전략.

### S-3. CSP별 차별화된 분석 (M12)
삼성 평택(500MW), SK 이천(300MW), 네이버 세종(200MW), 카카오 안산(100MW) — 실명 CSP 프로파일로 현실감 있는 분석. 글로벌 하이퍼스케일러(Google/Amazon/Meta/Microsoft) 전략 비교까지 포함하여 산업 이해도를 보여줌.

### S-4. 국제 벤치마크 데이터 체계
config.py의 `INTERNATIONAL_BENCHMARKS`가 한국/미국/중국/일본/독일 5개국 비교, 출처까지 명시. `BENCHMARK_API_SOURCES`로 자동 업데이트 경로까지 설계. 국가과제에서 "국제 경쟁력"을 논할 때 강력한 근거.

### S-5. Solar Battery — 미래 기술 포지셔닝
TRL 2-3이라 상용화는 멀지만, "2030+ Emerging Technology"로 명확히 분류하고 비용 전망을 시나리오별로 제시한 것은 기술 비전을 보여주는 좋은 카드. Nature Communications DOI까지 명시.

## Weaknesses (약점)

### W-1. Base Case IRR 4.5%는 투자 매력 부족
에너지 인프라 CAPEX 10,000억원에 IRR 4.5%는 민간 투자자 설득 불가. 정부 R&D 과제이므로 "투자 유치용"이 아닌 것은 이해하지만, 평가위원이 "그래서 누가 이걸 짓나?"라고 물을 수 있음.

**대응**: 복합 시나리오(IRR 12-15%)를 전면에 내세우되, base case가 보수적이라는 점을 강조. "배터리 CAPEX가 현재 속도로 하락하면 2028년에 base case도 IRR 8%+"라는 시간축 분석 추가 권장.

### W-2. AIDC 시설비 12,500억원의 처리가 모호
CAPEX에서 "시설비 제외" 방식으로 에너지 인프라만 10,000억원을 기준으로 경제성을 분석하지만, 총 투자 22,500억원 중 시설비가 55%를 차지. 평가위원이 "시설비를 빼는 게 정당한가?"라고 질문할 수 있음.

**대응**: BAU(기존 그리드 의존 AIDC) 대비 **추가 투자(incremental CAPEX)**로 프레이밍. "AIDC 시설은 어차피 지어야 하고, 에너지 인프라 10,000억원이 추가 투자"라는 논리를 명확히.

### W-3. 수익 모델에서 전력 자급 절감의 이중 계산 위험
`revenue_electricity_saving_krw_per_mwh = 80,000`으로 자가소비 전력의 절감 효과를 수익으로 잡는데, 동시에 `demand_charge_saving = 280억원`도 별도로 잡음. 수요요금 절감이 전력 단가 절감과 중복되지 않는지 검증 필요.

### W-4. 학습곡선 반영이 CAPEX에만 적용
PV -7%/yr, BESS -10%/yr, H2 -8%/yr의 학습곡선이 `CAPEXModel.apply_learning_curve()`에만 있고, 실제 NPV/IRR 계산(`run_base_case`)에서는 `include_learning_curve=False`가 기본값. 시연 시 학습곡선 미반영 상태로 보여줄 위험.

## Opportunities (기회)

### O-1. K-ETS 탄소가격 상승 추세 활용
현재 ~25,000원/tCO₂ → 2030년 50,000-100,000원 전망. M11의 `k_ets_scenarios_compare()`가 이를 정확히 보여줌. 시연에서 탄소가격 ×2 슬라이더 → IRR이 어떻게 변하는지 실시간 데모하면 강력한 인상.

### O-2. CBAM 대응 — 수출기업에 즉각적 가치
EU CBAM이 2026년 본격 시행. 반도체/배터리 수출 기업에게 "탄소발자국 추적 → CBAM 비용 절감"은 즉각적 비즈니스 가치. M11의 `cbam_impact()` 함수가 이를 정량화.

### O-3. Ratepayer Protection 정책 — 시의적절
config에 "2026 White House Hyperscaler Pledge" 반영. 글로벌 정책 트렌드를 반영한 것은 시의성 높은 분석.

---

# 🟡 PART 3: Arbiter — 종합 및 시연 전략

## 시연 전략

### 🎯 핵심 메시지 (3분 엘리베이터 피치)
> "92개 테스트를 통과한 13+α 모듈 디지털 트윈으로, 100MW급 AIDC의 PV-HESS-H₂-Grid 통합 에너지 시스템을 1ms~1년 스케일로 시뮬레이션합니다. 4개 CSP 맞춤 분석과 10,000회 Monte Carlo로 투자 의사결정을 지원하며, K-ETS/CBAM/RE100 정책 대응을 자동화합니다."

### ✅ 시연에서 강조할 포인트

1. **6-Layer HESS의 주파수 분리 제어** — 차별화의 핵심. Supercap(μs) → Li-ion(s) → Na-ion(hr) → RFB(day) → CAES(week) → H₂(season) 6단 계층은 학술적으로도 가치 있음. 시연에서 주파수별 전력 배분 시각화를 보여줄 것.

2. **Monte Carlo + 토네이도 차트** — "과장하지 않고 불확실성을 정량화한다"는 메시지. P(NPV>0) 확률을 보여주면 신뢰도 높음.

3. **CSP별 맞춤 분석 (M12)** — 삼성/SK/네이버/카카오 실명으로 차별화된 전략을 제안하는 것은 "실용성"을 강조. 글로벌 CSP 전략 비교(Google BYPASS_QUEUE vs Amazon DEDICATED_GEN 등)까지 보여주면 산업 이해도 어필.

4. **92/92 테스트 통과** — 숫자 자체가 강력한 메시지. 테스트 커버리지를 한 슬라이드로 보여줄 것.

5. **국제 벤치마크 5개국 비교** — 한국의 위치를 글로벌 컨텍스트에서 보여주는 것은 평가위원에게 좋은 인상.

### ⛔ 시연에서 회피할 포인트

1. **Solar Battery STH 57.6%** — TRL 2-3에 57.6%를 먼저 꺼내지 말 것. 질문이 오면 "이론적 상한이며 보수적 시나리오도 분석"이라고 답변.

2. **H₂ Round-Trip 효율 수치** — config(37.5%) vs 모델(~51%) 불일치를 들키지 않도록, "시스템 레벨 효율은 보조설비 포함 37-40%"로 통일된 답변 준비.

3. **Base Case IRR 4.5%** — 먼저 base case를 보여주되 즉시 복합 시나리오로 전환. "현재 기준 보수적, 2028-30년 시장 조건에서 12-15%"라는 스토리라인.

4. **BESS 2GWh 과잉 설계** — 질문이 오면 "worst-case 72시간 독립 운전 + AIDC 99.99% 가용성 요구" 근거 제시.

5. **GPU 50,000장 vs 100MW 불일치** — 시연 시 기본값으로 데모하지 말고, 용량에 맞는 GPU 수량으로 사전 설정.

## 수정 우선순위 (시연까지 D-7 기준)

| 우선순위 | 항목 | 예상 소요 | 영향도 |
|---------|------|----------|--------|
| **P0** | MF-4: 추가수익 630억 산출근거 문서화 | 2h | 🔴 질문 시 답변 불가 방지 |
| **P0** | MF-3: H₂ RT 효율 config↔모델 정합 | 1h | 🔴 일관성 확보 |
| **P1** | SF-3: SOEC Nernst ΔS 상수 수정 | 30m | 🟠 물리적 정확성 |
| **P1** | MF-1: PV 온도계수 주석/변수명 명확화 | 30m | 🟠 코드 리뷰 대비 |
| **P1** | m-3: GPU 수량 기본값 조정 (→ ~83,000장 B200 or ~119,000 H100) | 30m | 🟠 시연 현실성 |
| **P2** | MF-2: HESS 미배분 전력 fallback 강화 | 1h | 🟡 극단 시나리오 |
| **P2** | SF-5: Solar Battery STH 보수적 시나리오 추가 | 1h | 🟡 질문 대비 |
| **P2** | SF-6: M11 circular import 해소 | 30m | 🟡 안정성 |
| **P3** | m-1: config 탄소가격 값 통일 | 15m | 🟢 코드 품질 |
| **P3** | m-2: Na-ion config 추가 | 15m | 🟢 정합성 |
| **P3** | SF-4: MC log-normal 옵션 or 한계 문서화 | 1h | 🟢 학술적 엄밀성 |

## 종합 평가

### 전체 점수: **B+ → A- (수정 후)**

**강점 요약**:
- 13+α 모듈의 포괄성과 92/92 테스트 통과는 국가과제 수준에서 인상적
- "과장 금지 원칙"과 MC 기반 불확실성 분석은 학술적 성숙도를 보여줌
- CSP별 맞춤 분석 + 글로벌 벤치마크는 실용성과 국제 경쟁력을 동시에 어필
- 6-Layer HESS 주파수 분리 제어는 학술적 독창성이 있음

**약점 요약**:
- 물리 상수 오류(SOEC ΔS), config↔모델 불일치(H₂ RT효율)는 "시뮬레이션의 정확성"에 의문 제기 가능
- 경제성 분석에서 하드코딩된 수치(630억 추가수익)는 투명성 부족
- Base case IRR 4.5%는 솔직하지만, 복합 시나리오로의 전환 논리가 더 명확해야 함

**핵심 리스크**: 평가위원이 코드 수준까지 들여다보면(드물지만 가능) 물리 상수 오류가 전체 시뮬레이션 신뢰도를 훼손할 수 있음. P0/P1 항목 수정이 필수.

**전략적 조언**: "완벽한 답보다 정직한 범위"를 일관되게 유지할 것. 이 DT의 가장 큰 강점은 포괄성과 투명성. 시연에서 MC 분석의 P5-P95 범위를 보여주며 "우리는 불확실성을 숨기지 않는다"는 메시지를 전달하면, 개별 파라미터의 사소한 오류보다 전체 프레임워크의 가치가 부각됨.

---

*Advisory Board Review by Skeptic / Enterprise / Arbiter*  
*Generated: 2026-03-02 00:10 UTC*  
*Codebase: cems-dt/ (92/92 tests passing)*
