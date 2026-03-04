"""
CEMS-DT Configuration Module
전역 설정 및 상수 정의
"""
from typing import Dict, Any
import numpy as np

# =============================================================================
# 시뮬레이션 설정
# =============================================================================
SIMULATION_CONFIG = {
    "time_resolution_ms": 1,      # 전력전자용 최소 해상도
    "time_resolution_s": 1,       # HESS 제어용
    "time_resolution_min": 1,     # EMS 제어용  
    "time_resolution_hr": 1,      # 경제성 계산용
    "simulation_hours": 8760,     # 1년 = 8760 시간
    "spatial_resolution": "lumped"  # 집중정수 모델
}

# =============================================================================
# M1. PV 모듈 상수
# =============================================================================
PV_TYPES = {
    "c-Si": {
        "name": "c-Si (단결정 실리콘)",
        "eta_stc": 24.4,      # % STC 효율 (STC: 25°C, 1000 W/m², AM1.5G)
        "beta": -0.35,        # 상대 온도계수 (%/°C) → 1°C 상승 시 효율이 STC 대비 0.35% 상대 감소
                               # β_rel = beta/100 = -0.0035 /°C (IEC 61215)
                               # β_abs = η_STC × β_rel = -0.0854 %p/°C
        "noct": 45,          # °C NOCT (Nominal Operating Cell Temperature, IEC 61215)
        "delta": 0.5,        # %/yr 열화율 (연간 상대 열화)
        "v_oc": 0.72,        # V 개방전압
        "j_sc": 40.5,        # mA/cm² 단락전류밀도
        "ff": 0.83,          # Fill Factor
        "area_per_100mw": 93  # ha 100MW당 필요면적
    },
    "tandem": {
        "name": "탠덤 (페로브스카이트/Si)",
        "eta_stc": 34.85,     # % STC 효율
        "beta": -0.25,        # 상대 온도계수 (%/°C) — 탠덤은 Si 단독 대비 온도 민감도 낮음
        "noct": 43,
        "delta": 0.8,
        "v_oc": 2.15,
        "j_sc": 19.5,
        "ff": 0.82,
        "area_per_100mw": 55
    },
    "triple": {
        "name": "3접합 (III-V 다접합)",
        "eta_stc": 39.5,
        "beta": -0.20,
        "noct": 42,
        "delta": 1.0,
        "v_oc": 3.20,
        "j_sc": 13.5,
        "ff": 0.80,
        "area_per_100mw": 48
    },
    "infinite": {
        "name": "무한접합 (이상적)",
        "eta_stc": 68.7,
        "beta": -0.15,
        "noct": 40,
        "delta": 0.5,
        "v_oc": 5.0,  # 가정값
        "j_sc": 25.0,  # 계산값
        "ff": 0.85,
        "area_per_100mw": 28
    }
}

# =============================================================================
# M3. AIDC 부하 모듈 상수
# =============================================================================
GPU_TYPES = {
    "H100": {
        "name": "NVIDIA H100 SXM",
        "power_w": 700,
        "memory_gb": 80,
        "fp16_tflops": 1979
    },
    "B200": {
        "name": "NVIDIA B200 (Blackwell)",
        "power_w": 1000,
        "memory_gb": 192,
        "fp16_tflops": 2500  # 추정값
    },
    "next_gen": {
        "name": "차세대 GPU (2027+)",
        "power_w": 1200,
        "memory_gb": 256,
        "fp16_tflops": 3000  # 추정값
    },
    "B200_Ultra": {
        "name": "NVIDIA Blackwell Ultra",
        "power_w": 1200,
        "memory_gb": 288,
        "fp16_tflops": 3500  # 추정값
    }
}

PUE_TIERS = {
    "tier1": {"name": "Tier 1 (공냉)", "pue": 1.4},
    "tier2": {"name": "Tier 2 (하이브리드)", "pue": 1.2}, 
    "tier3": {"name": "Tier 3 (단상액침)", "pue": 1.07},
    "tier4": {"name": "Tier 4 (이상액침)", "pue": 1.03}
}

WORKLOAD_TYPES = {
    "llm": {
        "name": "LLM 추론",
        "base_utilization": 0.55,       # 실제 AIDC 평균 GPU 활용률 50-70% (Google TPU report)
        "peak_utilization": 0.98,       # API surge 시 거의 full
        "burst_frequency": 10           # per hour
    },
    "training": {
        "name": "AI 훈련",
        "base_utilization": 0.85,       # Training은 sustained high (Meta LLaMA 3: 85-95%)
        "peak_utilization": 1.0,        # Checkpoint spike 시 100%
        "burst_frequency": 2            # checkpoint + gradient sync
    },
    "moe": {
        "name": "MoE (Mixture of Experts)",
        "base_utilization": 0.40,       # Expert 20-30% 동시 활성 → base 높임
        "peak_utilization": 0.95,       # 다수 expert 동시 활성화 시
        "burst_frequency": 20
    }
}

# =============================================================================
# M4. DC Bus 모듈 상수
# =============================================================================
CONVERTER_EFFICIENCY = {
    "default": {
        "pv_to_dcbus": 0.985,      # SiC 기본
        "dcbus_to_bess": 0.975,
        "dcbus_to_supercap": 0.990,
        "dcbus_to_electrolyzer": 0.970,
        "fc_to_dcbus": 0.970,
        "dcbus_to_aidc": 0.960,
        "grid_bidirectional": 0.970
    },
    "advanced": {
        "pv_to_dcbus": 0.995,      # GaN 고효율
        "dcbus_to_bess": 0.990,
        "dcbus_to_supercap": 0.995,
        "dcbus_to_electrolyzer": 0.985,
        "fc_to_dcbus": 0.985,
        "dcbus_to_aidc": 0.980,
        "grid_bidirectional": 0.985
    }
}

# =============================================================================
# M10. 기상 모듈 상수 (한국 중부 기준)
# =============================================================================
WEATHER_PARAMS = {
    "latitude": 37.5,  # 서울 위도
    "longitude": 127.0,  # 서울 경도
    "timezone": 9,  # KST
    "annual_ghi_kwh_per_m2": 1350,  # 연간 전천일사량
    "peak_sun_hours": 3.5  # 연평균 피크 일조시간
}

# 월별 일사량 패턴 (정규화, 최대값 = 1.0)
MONTHLY_GHI_PATTERN = {
    1: 0.45, 2: 0.55, 3: 0.75, 4: 0.90, 5: 1.00, 6: 0.95,
    7: 0.85, 8: 0.90, 9: 0.80, 10: 0.70, 11: 0.55, 12: 0.40
}

# 월별 평균 온도 (°C)
MONTHLY_TEMP_PATTERN = {
    1: -2, 2: 1, 3: 7, 4: 14, 5: 20, 6: 25,
    7: 27, 8: 28, 9: 23, 10: 16, 11: 8, 12: 1
}

# =============================================================================
# M2. HESS 모듈 상수
# =============================================================================
HESS_LAYER_CONFIGS = {
    "supercap": {
        "name": "Supercapacitor",
        "capacity_kwh": 50,
        "power_rating_kw": 10000,
        "response_time_ms": 0.001,
        "efficiency_charge": 0.98,
        "efficiency_discharge": 0.98,
        "time_constant_range": (0.001, 1.0)  # μs ~ s
    },
    "bess": {
        "name": "Li-ion BESS", 
        "capacity_kwh": 2000000,  # 2,000 MWh
        "power_rating_kw": 200000,  # 200 MW
        "response_time_ms": 100,
        "efficiency_charge": 0.95,
        "efficiency_discharge": 0.95,
        "time_constant_range": (1.0, 3600.0)  # s ~ hr
    },
    "rfb": {
        "name": "Vanadium RFB",
        "capacity_kwh": 750000,  # 750 MWh
        "power_rating_kw": 50000,  # 50 MW
        "response_time_ms": 1000,
        "efficiency_charge": 0.85,
        "efficiency_discharge": 0.85,
        "time_constant_range": (3600.0, 86400.0)  # hr ~ day
    },
    "caes": {
        "name": "CAES",
        "capacity_kwh": 1000000,  # 1,000 MWh
        "power_rating_kw": 100000,  # 100 MW
        "response_time_ms": 30000,
        "efficiency_charge": 0.75,
        "efficiency_discharge": 0.75,
        "time_constant_range": (86400.0, 604800.0)  # day ~ week
    },
    "h2": {
        "name": "H2 Storage",
        "capacity_kwh": 5000000,  # 5,000 MWh
        "power_rating_kw": 50000,  # 50 MW
        "response_time_ms": 300000,
        "efficiency_charge": 0.40,
        "efficiency_discharge": 0.40,
        "time_constant_range": (604800.0, 31536000.0)  # week ~ seasonal
    }
}

# =============================================================================
# M5. H₂ 시스템 상수  
# =============================================================================
H2_SYSTEM_CONFIG = {
    # H₂ Round-Trip 효율 설계 기준:
    #   PEM 전해조 η_elec ≈ 65% (LHV 기준, IRENA 2024)
    #   PEM 연료전지 η_FC ≈ 55% (LHV 기준, DOE 2024)
    #   RT_efficiency = η_elec × η_FC ≈ 35.75% → config 37.5% (BOP 포함 보정)
    # Note: SOEC/SOFC (85%×60%=51%)는 고온 시스템 기준이며,
    #       본 모델은 보수적 PEM 기준 37.5%를 적용
    "soec": {
        "rated_power_kw": 50000,  # 50 MW 전해조
        "efficiency_nominal": 0.65,  # PEM 전해조 공칭 효율 (LHV 기준)
                                      # Ref: IRENA Green Hydrogen Cost Reduction 2024
        "operating_temp_celsius": 80,  # PEM 운전 온도
        "startup_time_min": 15,        # PEM은 저온 → 빠른 시동
        "degradation_rate_per_1000h": 0.5
    },
    "sofc": {
        "rated_power_kw": 50000,  # 50 MW 연료전지
        "efficiency_nominal": 0.55,  # PEM 연료전지 공칭 효율 (LHV 기준)
                                      # Ref: DOE Hydrogen Program Record 2024
        "operating_temp_celsius": 80,  # PEM 운전 온도
        "startup_time_min": 5,         # PEM FC 빠른 시동
        "degradation_rate_per_1000h": 0.3
    },
    "storage": {
        "capacity_kg": 150000,  # 150 ton H₂
        "storage_type": "compressed",  # "compressed" or "metal_hydride"
        "pressure_bar": 350,
        "leakage_rate_per_day": 0.001
    },
    "round_trip_efficiency": {
        "electrical_only": 0.375,  # 37.5% (IEA 2023)
        "chp_mode": 0.825          # 82.5% (열 회수 포함)
    }
}

# Solar Battery H₂ 생산 설정 (2030+ Emerging Technology)
# Ref: Nature Communications (Ulm/Jena, DOI:10.1038/s41467-026-68342-2)
H2_SOLAR_BATTERY_CONFIG = {
    "eta_capture": 0.80,          # 광 포집 효율 80%
    "eta_h2": 0.72,               # H₂ 방출 효율 72%
    "sth_efficiency": 0.576,      # Solar-to-Hydrogen ≈ 57.6%
    "storage_days_default": 3,    # 기본 저장 일수
    "storage_loss_per_day": 0.005,  # 일당 저장 손실 0.5%
    "degradation_rate_per_year": 0.02,  # 연 2% 열화
    "trl": 2.5,                   # Technology Readiness Level
    "technology_class": "2030+ Emerging Technology",
    "reference": "Nature Communications, DOI:10.1038/s41467-026-68342-2",
    # 비용 전망 ($/kg H₂) — 시나리오별
    "cost_projections_usd_per_kg": {
        "2030": {"optimistic": 4.0, "base": 6.0, "conservative": 9.0},
        "2035": {"optimistic": 2.5, "base": 4.0, "conservative": 6.0},
        "2040": {"optimistic": 1.5, "base": 2.5, "conservative": 4.0},
    },
}

# H₂ 물리 상수
H2_CONSTANTS = {
    "hhv_kwh_per_kg": 39.39,    # Higher Heating Value
    "lhv_kwh_per_kg": 33.33,    # Lower Heating Value
    "faraday_constant": 96485.33212,  # C/mol
    "gas_constant": 8.314462618,      # J/(mol·K)
    "nernst_voltage_stc": 1.229       # V @ 25°C, 1 bar
}

# =============================================================================
# M8. 그리드 인터페이스 상수
# =============================================================================
GRID_TARIFF_CONFIG = {
    "smp_base_krw_per_mwh": 80000,        # 기준 SMP 가격 (₩/MWh)
    "rec_price_krw_per_mwh": 25000,       # REC 가격 (₩/MWh)
    "rec_multiplier": 1.2,                # REC 가중치 (태양광)
    "carbon_price_krw_per_ton": 22500,    # K-ETS 탄소가격 (₩/tCO₂)
    "grid_access_fee_krw_per_mw_month": 1000000,  # 계통이용요금 (₩/MW/월)
    "transmission_loss_factor": 0.05      # 송전손실률 (5%)
}

# 한국 전력시장 시간대별 SMP 배율
SMP_HOURLY_MULTIPLIERS = {
    "summer": {  # 여름 (에어컨 피크)
        "day": [0.7, 0.65, 0.6, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
                1.35, 1.4, 1.45, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.75]
    },
    "winter": {  # 겨울 (난방 피크)  
        "day": [0.75, 0.7, 0.65, 0.6, 0.65, 0.75, 0.9, 1.1, 1.3, 1.35, 1.2, 1.15,
                1.1, 1.05, 1.0, 1.05, 1.1, 1.2, 1.4, 1.45, 1.3, 1.1, 0.95, 0.8]
    },
    "spring_autumn": {  # 봄/가을 (완만한 패턴)
        "day": [0.75, 0.7, 0.65, 0.65, 0.7, 0.75, 0.85, 0.95, 1.05, 1.15, 1.2, 1.25,
                1.25, 1.2, 1.15, 1.1, 1.05, 1.1, 1.15, 1.2, 1.1, 1.0, 0.9, 0.8]
    }
}

# 그리드 보호 설정
GRID_PROTECTION_SETTINGS = {
    "voltage_high_pu": 1.1,      # 과전압 트립 (110%)
    "voltage_low_pu": 0.9,       # 저전압 트립 (90%)
    "frequency_high_hz": 50.5,   # 과주파수 트립
    "frequency_low_hz": 49.5,    # 저주파수 트립
    "power_factor_min": 0.95,    # 최소 역률
    "reconnect_delay_s": 300     # 재연결 지연시간 (5분)
}

# =============================================================================
# 경제성 모델 상수
# =============================================================================
ECONOMICS = {
    "discount_rate": 0.05,  # 할인율 5%
    "project_lifetime": 20,  # 프로젝트 수명 20년
    "korean_grid_emission_factor": 0.4168,  # tCO2/MWh (2024년 기준)
    "k_ets_price_krw_per_tco2": 22500,  # K-ETS 배출권 가격
    "rec_price_krw_per_mwh": 25000,  # REC 가격
}

# =============================================================================
# UI 설정
# =============================================================================
UI_CONFIG = {
    "page_title": "CEMS Digital Twin",
    "page_icon": "⚡",
    "layout": "wide",
    "sidebar_width": 350,
    "chart_height": 400
}

# 색상 팔레트 (GDI 스타일 - 절제된 테마)
COLOR_PALETTE = {
    "carbon": "#34d399",  # 연한 그린 (탄소)
    "scope1": "#f97316",  # 오렌지 (Scope 1)
    "scope2": "#00d4ff",  # 밝은 시안 (Scope 2)
    "scope3": "#a78bfa",  # 라벤더 (Scope 3)
    "economics": "#fbbf24", # 골드 (경제)
    "pv": "#ffdd00",      # 골드 (태양광)
    "aidc": "#ff4060",    # 연한 빨강 (부하)
    "bess": "#00d4ff",    # 밝은 시안 (배터리)
    "grid": "#a78bfa",    # 라벤더 (계통)
    "h2": "#34d399",      # 연한 그린 (수소)
    "surplus": "#34d399", # 연한 그린 (잉여)
    "deficit": "#ff4060"  # 연한 빨강 (부족)
}

# =============================================================================
# M6. AI-EMS 제어 설정
# =============================================================================
AI_EMS_CONFIG = {
    # Tier 1: 실시간 제어
    "tier1_interval_ms": 1,
    "mppt_tracking_speed": 0.99,       # MPPT 추적 속도
    "dc_bus_voltage_target_v": 380,
    "dc_bus_voltage_tolerance_pct": 2,  # ±2%
    "hess_soc_target": {"supercap": 0.5, "bess": 0.5},

    # Tier 2: 예측 제어
    "tier2_interval_min": 15,
    "pv_forecast_horizon_hr": 4,
    "load_forecast_horizon_hr": 4,
    "dispatch_optimization": True,

    # Tier 3: 전략 최적화
    "tier3_interval_hr": 1,
    "economic_optimization": True,
    "grid_trading_enabled": True,
    "maintenance_scheduling": True,

    # Grid Flexibility (Demand Response)
    # Ref: National Grid × Emerald AI × NVIDIA 런던 실증 (2025.12)
    #   96× Blackwell Ultra GPU, 5일간 200+ 실시간 그리드 이벤트
    #   최대 40% 부하 감축, 워크로드 중단 없이 달성
    #   긴급 30% 감축 ~30초, 지속 감축 최대 10시간
    "grid_flex_enabled": True,
    "grid_flex_max_curtailment_pct": 40,   # 최대 감축률 (%) — National Grid 실증 검증
    "grid_flex_emergency_pct": 30,         # 긴급 감축률 (%) — 30초 내 달성
    "grid_flex_emergency_response_s": 30,  # 긴급 응답 시간 (초)
    "grid_flex_sustained_hours": 10,       # 지속 감축 최대 시간
    "grid_flex_ramp_rate_pct_per_min": 5,  # 분당 감축 속도 (%/min)
    "grid_flex_workload_priority": {
        "training": 3,    # 최우선 보호 (checkpoint 손실 위험)
        "llm": 2,         # 중간 (배치 크기 축소로 대응)
        "moe": 1,         # 유연 (expert 비활성화로 빠른 감축)
    },
}

# =============================================================================
# M7. 탄소 회계 설정
# =============================================================================
CARBON_CONFIG = {
    # 한국 전력 배출계수 (tCO₂/MWh)
    "grid_emission_factor_tco2_per_mwh": 0.4594,   # 2024 기준
    "grid_emission_factor_year": 2024,

    # K-ETS 탄소 가격 (₩/tCO₂)
    "k_ets_price_krw_per_tco2": 25000,

    # CBAM (EU Carbon Border Adjustment Mechanism)
    "cbam_price_eur_per_tco2": 80,
    "eur_to_krw": 1450,

    # Scope 3 기본 배출계수 (tCO₂/MW 설비)
    "scope3_pv_manufacturing_tco2_per_mw": 40,
    "scope3_bess_manufacturing_tco2_per_mwh": 65,
    "scope3_h2_manufacturing_tco2_per_mw": 30,

    # 탄소크레딧 (REC과 별도)
    "carbon_credit_price_krw_per_tco2": 25000,
}

# =============================================================================
# M9. 경제 최적화 설정
# =============================================================================
ECONOMICS_CONFIG = {
    # 프로젝트 기본
    "project_lifetime_years": 20,
    "discount_rate": 0.05,
    "inflation_rate": 0.02,

    # CAPEX (억원) — 100MW AIDC 기준
    "capex_pv_billion_krw": 1500,           # PV 100MW
    "capex_bess_billion_krw": 4000,         # BESS 2GWh
    "capex_supercap_billion_krw": 500,      # Supercap
    "capex_h2_billion_krw": 3000,           # H₂ system (SOEC+SOFC+저장)
    "capex_dcbus_billion_krw": 500,         # DC Bus + 변환기
    "capex_grid_billion_krw": 200,          # 계통 연계
    "capex_aiems_billion_krw": 300,         # AI-EMS 시스템
    "capex_facility_billion_krw": 12500,    # AIDC 건축/인프라
    "capex_total_infra_billion_krw": 22500, # 인프라 총계
    "capex_rd_billion_krw": 400,            # R&D 과제비 (분리)

    # OPEX (억원/년)
    "opex_maintenance_pct_of_capex": 0.008,  # 에너지 설비 CAPEX 대비 0.8%
    "opex_maintenance_capex_base_billion_krw": 10000,  # 유지보수 대상 설비 CAPEX (에너지 인프라)
    "opex_electricity_krw_per_mwh": 80000,   # 전력 단가
    "opex_labor_billion_krw_per_year": 50,   # 인건비
    "opex_insurance_pct": 0.002,             # 보험 0.2%

    # 수익 모델
    "revenue_electricity_saving_krw_per_mwh": 80000,  # 자가소비 절감
    "revenue_surplus_sale_krw_per_mwh": 70000,        # 잉여 판매 (SMP)
    "revenue_rec_krw_per_mwh": 25000,                 # REC
    "revenue_rec_multiplier": 1.2,                     # 태양광 가중치
    "revenue_carbon_credit_krw_per_tco2": 25000,

    # 학습곡선 (연간 비용 감소율)
    "learning_curve_pv_pct_per_yr": -7,
    "learning_curve_bess_pct_per_yr": -10,
    "learning_curve_h2_pct_per_yr": -8,

    # ─── 추가수익 산출근거 (기존 하드코딩 630억원 대체) ───
    # 총 추가수익 ≈ 630억원/년 = 수요요금절감 + 그리드안정성 + BESS차익거래
    #
    # (1) 수요요금 절감 (피크 시프팅): ~280억원/년
    #     - 100MW AIDC의 피크 수요 → 계약전력 기반 기본요금 절감
    #     - 피크 20MW 절감 × 기본요금 14,000원/kW/월 × 12개월 = 33.6억
    #     - + 시간대별 전력요금 차익 (경부하/중부하/최대부하 차등)
    #     - 연간 환산: ~280억원 (보수적, 한전 산업용 갑 기준)
    #     Ref: KEPCO 전기요금표 2024, 한전 산업용(갑) II 기본요금
    "additional_revenue_demand_charge_billion_krw": 280.0,
    "additional_revenue_demand_charge_basis": (
        "피크 20MW 절감 × 기본요금 14,000₩/kW/월 × 12 = 33.6억 + "
        "TOU 차익 246.4억 (경부하 55₩/kWh, 최대부하 110₩/kWh, 일 8h 피크시프트, 365일)"
    ),
    #
    # (2) 그리드 안정성/신뢰성 가치: ~220억원/년
    #     - UPS 기능에 의한 정전 회피 가치: 100MW × 정전비용 $50/kWh × 연 10회 × 1h = ~65억
    #     - 주파수 조정(FR) 서비스 수익: 50MW × 40,000₩/MW/h × 2,000h = 40억
    #     - RE100 인증 프리미엄 (자발적 시장): 150GWh × 10₩/kWh = 15억
    #     - 계통 혼잡 완화 수익 + 용량시장 수익: ~100억
    #     Ref: EPRI Value of DER Study 2023, KPX 보조서비스 시장 2024
    "additional_revenue_grid_reliability_billion_krw": 220.0,
    "additional_revenue_grid_reliability_basis": (
        "정전회피 65억 + FR 서비스 40억 + RE100 프리미엄 15억 + 용량시장/혼잡완화 100억"
    ),
    #
    # (3) BESS 차익거래 (에너지 아비트라지): ~130억원/년
    #     - 2GWh BESS, 일 1사이클, 충방전 가격차 65₩/kWh (경부하↔최대부하)
    #     - 2,000MWh × 65₩/kWh × 365일 × 효율0.9 = ~42.7억 (순수 아비트라지)
    #     - + 보조서비스(주파수, 전압) 수익: ~50억
    #     - + 피크 셰이빙 추가 절감: ~37.3억
    #     Ref: Lazard LCOS 2024, KPX 전력시장 가격 데이터 2024
    "additional_revenue_bess_arbitrage_billion_krw": 130.0,
    "additional_revenue_bess_arbitrage_basis": (
        "에너지 아비트라지 42.7억 + 보조서비스 50억 + 피크셰이빙 37.3억"
    ),
    # 총계: 280 + 220 + 130 = 630억원/년

    # Monte Carlo
    "mc_iterations": 10000,
    "mc_variables": {
        "pv_efficiency_std_pct": 5,
        "electricity_price_std_pct": 15,
        "carbon_price_std_pct": 20,
        "discount_rate_std_pct": 10,
        "load_variation_std_pct": 10,
    },
}
# =============================================================================
# 국제 벤치마크 데이터
# 출처: NREL ATB 2024, IRENA RENEWCOST 2024, Fraunhofer ISE, METI, SERC
# 최종 업데이트: 2026-02-22
# =============================================================================
INTERNATIONAL_BENCHMARKS = {
    'KR': {
        'country': 'Korea', 'flag': '🇰🇷', 'label': '🇰🇷 한국 (본 DT)',
        'capacity_mw': 100, 'pv_type': 'Tandem Perovskite-Si',
        'storage': 'HESS (Supercap+BESS) + H₂',
        'grid_type': 'Island + Grid-tied hybrid',
        'irradiance_kwh_m2_yr': 1340,
        'elec_price_usd_mwh': 90,
        'carbon_intensity_gco2_kwh': 415,
        'carbon_price_usd_ton': 20,
        'pv_lcoe_usd_mwh': None,
        'capacity_factor': None,
        'self_sufficiency': None,
        'notes': '100MW급 AIDC 전용, AI-EMS 3-tier 최적화',
        'sources': {
            'irradiance': 'KMA 기상청 TMY3',
            'elec_price': 'KEPCO 산업용 2024',
            'carbon_intensity': '전력거래소 2024',
            'carbon_price': 'K-ETS 할당거래소 2024',
        }
    },
    'US': {
        'country': 'USA', 'flag': '🇺🇸', 'label': '🇺🇸 미국 (NREL)',
        'capacity_mw': 100, 'pv_type': 'c-Si Bifacial',
        'storage': 'Li-ion BESS (4h)',
        'grid_type': 'Grid-tied + DR',
        'irradiance_kwh_m2_yr': 1800,
        'elec_price_usd_mwh': 65,
        'carbon_intensity_gco2_kwh': 370,
        'carbon_price_usd_ton': 0,
        'pv_lcoe_usd_mwh': 28,
        'capacity_factor': 0.26,
        'self_sufficiency': 0.45,
        'notes': 'NREL ATB 2024, Southwest US, utility-scale PV+BESS',
        'sources': {
            'irradiance': 'NREL NSRDB TMY3',
            'elec_price': 'EIA Commercial Avg 2024',
            'carbon_intensity': 'EPA eGRID 2024',
            'carbon_price': 'N/A (no federal price)',
            'lcoe': 'NREL ATB 2024',
        }
    },
    'CN': {
        'country': 'China', 'flag': '🇨🇳', 'label': '🇨🇳 중국 (SERC)',
        'capacity_mw': 100, 'pv_type': 'c-Si (LONGi/JA Solar)',
        'storage': 'LFP BESS (2h mandatory)',
        'grid_type': 'Grid-tied (mandatory storage)',
        'irradiance_kwh_m2_yr': 1500,
        'elec_price_usd_mwh': 55,
        'carbon_intensity_gco2_kwh': 555,
        'carbon_price_usd_ton': 10,
        'pv_lcoe_usd_mwh': 22,
        'capacity_factor': 0.18,
        'self_sufficiency': 0.35,
        'notes': '중국 SERC 기준, 서북부 대규모 PV 기지, 2h 저장 의무',
        'sources': {
            'irradiance': 'CMA Typical Meteorological Year',
            'elec_price': 'NDRC Industrial Tariff 2024',
            'carbon_intensity': 'MEE China Grid EF 2024',
            'carbon_price': 'Shanghai Environment Energy Exchange',
            'lcoe': 'CPIA Annual Report 2024',
        }
    },
    'JP': {
        'country': 'Japan', 'flag': '🇯🇵', 'label': '🇯🇵 일본 (METI)',
        'capacity_mw': 50, 'pv_type': 'c-Si + Perovskite pilot',
        'storage': 'Li-ion + 레독스플로우',
        'grid_type': 'Island-capable (방재)',
        'irradiance_kwh_m2_yr': 1200,
        'elec_price_usd_mwh': 150,
        'carbon_intensity_gco2_kwh': 450,
        'carbon_price_usd_ton': 5,
        'pv_lcoe_usd_mwh': 75,
        'capacity_factor': 0.15,
        'self_sufficiency': 0.30,
        'notes': 'METI 2024, 분산형 마이크로그리드, 방재 겸용 설계',
        'sources': {
            'irradiance': 'JMA AMeDAS',
            'elec_price': 'METI Industrial Tariff 2024',
            'carbon_intensity': 'MOE Japan Grid EF 2024',
            'carbon_price': 'GX Surcharge (est. ¥750/ton)',
            'lcoe': 'METI Cost Verification Committee 2024',
        }
    },
    'DE': {
        'country': 'Germany', 'flag': '🇩🇪', 'label': '🇩🇪 독일 (Fraunhofer)',
        'capacity_mw': 80, 'pv_type': 'c-Si Bifacial + Agri-PV',
        'storage': 'Li-ion + Green H₂',
        'grid_type': 'Grid-tied (Energiewende)',
        'irradiance_kwh_m2_yr': 1050,
        'elec_price_usd_mwh': 180,
        'carbon_intensity_gco2_kwh': 350,
        'carbon_price_usd_ton': 55,
        'pv_lcoe_usd_mwh': 45,
        'capacity_factor': 0.12,
        'self_sufficiency': 0.38,
        'notes': 'Fraunhofer ISE 2024, Agri-PV + Green H₂ 시범',
        'sources': {
            'irradiance': 'DWD TRY 2024',
            'elec_price': 'Destatis Industrial 2024',
            'carbon_intensity': 'UBA Germany Grid EF 2024',
            'carbon_price': 'EU-ETS (ICE ECX)',
            'lcoe': 'Fraunhofer ISE LCOE Study 2024',
        }
    },
}

# 벤치마크 자동 업데이트 API 소스
BENCHMARK_API_SOURCES = {
    'ember_carbon_intensity': {
        'url': 'https://api.ember-climate.org/v1/carbon-intensity/latest',
        'description': 'Ember Climate — 전세계 실시간 탄소강도',
        'update_freq': 'quarterly',
        'fields': ['carbon_intensity_gco2_kwh'],
    },
    'eu_ets_price': {
        'url': 'https://api.ember-climate.org/v1/carbon-price/eu-ets',
        'description': 'EU-ETS 탄소배출권 가격',
        'update_freq': 'monthly',
        'fields': ['carbon_price_usd_ton'],
    },
    'nrel_atb': {
        'url': 'https://atb.nrel.gov/electricity/data',
        'description': 'NREL Annual Technology Baseline',
        'update_freq': 'annual',
        'fields': ['pv_lcoe_usd_mwh', 'capacity_factor'],
    },
    'irena_renewcost': {
        'url': 'https://www.irena.org/Data/View-data-by-topic/Costs/Global-LCOE-and-Auction-values',
        'description': 'IRENA Renewable Cost Database',
        'update_freq': 'annual',
        'fields': ['pv_lcoe_usd_mwh'],
    },
}

BENCHMARK_LAST_UPDATED = '2026-02-22'

# =============================================================================
# Ember Useful Energy Framework (2026-02-17)
# Ref: "Reframing Energy for the Age of Electricity"
#   Authors: Daan Walter, Kingsmill Bond, Sam Butler-Sloss, Antoine Issac, Michael Liebreich
#   Data: IEA WEB 2023, IIASA PFU methodology, Nick Eyre (Oxford) Work/Heat framework
# =============================================================================
EMBER_USEFUL_ENERGY = {
    # ── 글로벌 에너지 잔고 (2023, EJ) ──
    "primary_energy_ej": {
        "electro": 31,       # 태양광 + 풍력 + 수력
        "thermal": 560,      # 화석연료 + 원자력 + 바이오매스
        "total": 591,
    },
    "final_energy_ej": {
        "electrons": 91,     # 전력
        "molecules": 296,    # 화석연료/바이오매스 직접 소비
        "total": 387,
    },
    "useful_energy_ej": {
        "work": 90,          # 운동 에너지 (운송, 기계, 컴퓨팅)
        "heat": 119,         # 열 에너지 (난방, 공정열)
        "total": 209,
    },
    "non_energy_use_ej": 42,    # 석유화학 원료 등
    "system_loss_ej": 380,      # 전체 1차 에너지의 ~2/3

    # ── 4-Battle 효율 매트릭스 (%) ──
    # Primary → Final
    "efficiency_primary_to_final": {
        "electro_to_electrons": 0.92,   # Battle 1: 태양광/풍력 → 전력
        "thermal_to_electrons": 0.29,   # Battle 1: 화석/원자력 → 전력 (카르노 한계)
        "thermal_to_molecules": 0.85,   # Battle 4: 화석연료 → 정제연료
        "electro_to_molecules": 0.70,   # Battle 4: 그린수소 등 (<70%)
    },
    # Final → Useful
    "efficiency_final_to_useful": {
        "electrons_to_work": 0.68,      # Battle 2: 전력 → 유효 일 (모터 90%+)
        "molecules_to_work": 0.29,      # Battle 2: 분자 연소 → 일 (카르노 한계)
        "electrons_to_heat": 0.91,      # Battle 3: 전력 → 유효 열 (히트펌프 300-400%)
        "molecules_to_heat": 0.64,      # Battle 3: 분자 연소 → 열
    },

    # ── 1인당 에너지 (GJ/capita) ──
    "per_capita_gj": {
        "primary_energy": 75,
        "useful_energy": 26,
        "shell_projection": 100,  # Shell의 고발전 시나리오 (과대추정)
    },

    # ── 시장 전환 지표 ──
    "market_transition": {
        "electro_share_generation_2023_pct": 30,      # 누적 전력 생산 점유율
        "electro_share_generation_growth_2025_pct": 96,  # 2025년 9개월 성장분 점유율
        "electrons_share_useful_work_2023_pct": 53,    # 유효 일 중 전력 비중
        "electrons_share_useful_work_growth_pct": 80,  # 2019-23 유효 일 성장분
        "electrons_share_useful_heat_2023_pct": 16,    # 유효 열 중 전력 비중
        "electrons_share_useful_heat_growth_pct": 25,  # 2019-23 유효 열 성장분
    },

    # ── AIDC 마이크로그리드 적용 ──
    # PV(Electro) → DC Bus(Electrons) → AIDC(Work) 경로 효율
    # 전체 경로: 0.92 × 0.68 = 0.626 (62.6%)
    # vs Thermal 경로: 0.29 × 0.29 = 0.084 (8.4%)
    # → Electro-Electron-Work 경로가 Thermal 대비 7.4배 효율적
    "aidc_pathway_efficiency": {
        "electro_electron_work": 0.92 * 0.68,   # 62.6%
        "thermal_electron_work": 0.29 * 0.68,   # 19.7%
        "thermal_molecule_work": 0.85 * 0.29,   # 24.7%
        "electro_molecule_work": 0.70 * 0.29,   # 20.3%
    },
}
