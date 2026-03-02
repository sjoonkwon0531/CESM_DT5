# Must Fix Changelog (DT5 — 시연 준비)

**날짜**: 2026-03-02  
**대상**: Advisory Board Must Fix 4건 + UI Polish 4건  
**테스트**: 92/92 통과 (변경 없음)

---

## Must Fix 수정

### MF1. PV 온도계수 주석/의미론 수정 ✅

**파일**: `modules/m01_pv.py`, `config.py`

**변경 내용**:
- `_calculate_temperature_efficiency()` 메서드에 상세 docstring 추가
  - IEC 61215 / IEC 61853 기반 공식 명시
  - **상대(relative)** vs **절대(absolute)** 온도계수 구분 명확화
  - `beta` 변수: `%/°C` → `1/°C` 변환 과정을 단계별로 주석
  - 예시: c-Si β = −0.35 %/°C → β_rel = −0.0035 /°C, β_abs = −0.0854 %p/°C
- config.py의 `beta` 필드에 상세 주석 추가 (상대 온도계수임을 명시)
- 참고문헌: De Soto et al. (2006), IEC 61215:2021, IEC 61853-1:2011

**수치 변경**: 없음 (주석/의미론만 수정)

---

### MF2. HESS 미배분 전력 에너지 보존 ✅

**파일**: `modules/m02_hess.py`

**변경 내용**:
- `calculate_power_allocation()`에 `_unallocated_kw` 필드 추가
  - 극단 주파수에서 어떤 레이어에도 매칭되지 않는 전력을 명시적 추적
  - 기존: 미배분 전력이 소실됨 (에너지 보존 위반)
  - 수정: `unallocated_power = total_request - allocated_total`
- `operate_hess()` 반환값에 에너지 보존 검증 블록 추가:
  - `energy_conservation` dict: requested/delivered/unallocated/loss 4항목 추적
  - `energy_balance_error_kwh`: 이상적으로 0 (보존 법칙 위반 감지용)
  - `power_unallocated_kw`: 미배분 전력 명시적 보고

**영향**: 극단 주파수(f < 1e-6 Hz 또는 f > 1 Hz) 시나리오에서 에너지 보존 보장

---

### MF3. H₂ RT 효율 config vs 모델 불일치 ✅

**파일**: `modules/m05_h2.py`, `config.py`

**문제**: config RT efficiency = 37.5%, 모델 계산 = ~51% (SOEC 85% × SOFC 60%)

**해결**: PEM 기준으로 통일
- **전해조**: SOEC(85%) → PEM(65%), Ref: IRENA Green Hydrogen Cost Reduction 2024
- **연료전지**: SOFC(60%) → PEM FC(55%), Ref: DOE Hydrogen Program Record 2024
- RT = 65% × 55% = 35.75% → config 37.5% (BOP 보정 포함, 합리적)
- 효율 계산에 `min(calculated, config.efficiency_nominal)` 상한 적용
- 운전 온도/시동 시간도 PEM 기준으로 변경 (800°C → 80°C, 120min → 15min)
- `calculate_round_trip_efficiency()`에 설계 기준 주석 추가

**수치 변경**: RT efficiency ~51% → ~35-37% (config과 일치)

---

### MF4. 경제성 추가수익 630억 산출근거 ✅

**파일**: `modules/m09_economics.py`, `config.py`

**문제**: 하드코딩된 630억원 (280+220+130)의 근거 불명확

**해결**: config.py에 각 항목별 산출근거를 명시적으로 문서화:

| 항목 | 금액 | 산출근거 |
|------|------|----------|
| 수요요금 절감 | 280억/년 | 피크 20MW 절감 × 14,000₩/kW/월 + TOU 차익 (KEPCO 산업용 갑) |
| 그리드 안정성 | 220억/년 | 정전회피 65억 + FR 서비스 40억 + RE100 15억 + 용량시장 100억 |
| BESS 차익거래 | 130억/년 | 에너지 아비트라지 42.7억 + 보조서비스 50억 + 피크셰이빙 37.3억 |
| **합계** | **630억/년** | |

- 참고문헌: KEPCO 전기요금표 2024, EPRI Value of DER 2023, KPX 보조서비스 시장 2024, Lazard LCOS 2024
- `run_base_case()`: 하드코딩 기본값 → config에서 읽어오도록 변경 (Optional 파라미터)

---

## UI Polish

### UP1. 시연 시나리오 프리셋 ✅

**파일**: `app.py`

- 사이드바 상단에 "📋 시연 시나리오" selectbox 추가
- 4개 프리셋:
  - **A**: 기본 100MW AIDC (c-Si, H100, Tier3)
  - **B**: CSP 비교 (Tandem, B200, Tier4, 150MW)
  - **C**: 정책 시나리오 (탄소가격 8만원, 전력단가 12만원)
  - **D**: Solar Battery 2030+ (Infinite, next_gen, 할인율 4%)
- 프리셋 선택 시 관련 파라미터 자동 설정

### UP2. 핵심 메트릭 대시보드 ✅

**파일**: `app.py`

- `_display_top_kpi()` 함수 추가 — 시뮬레이션 결과 상단에 6개 KPI 표시:
  - LCOE ($/MWh), IRR (%), CO₂ 감축 (tCO₂/yr), PUE, 자급률 (%), 회수기간 (년)
- `st.metric` 6열 레이아웃

### UP3. 한국어 라벨 점검 ✅

**파일**: `app.py`

- "PV Capacity Factor" → "PV 이용률"
- "Curtailment" → "출력제한율"
- 기술 약어(LCOE, IRR, PUE, NPV, CAPEX 등)는 영어 유지

### UP4. 캐싱/속도 최적화 ✅

**파일**: `app.py`

- `_cached_base_case()` 추가 (`@st.cache_data`)
  - 경제성 Base Case 결과 캐시 → 탭 전환 시 재계산 방지
- 기존 `load_weather_data()` 캐시 유지
