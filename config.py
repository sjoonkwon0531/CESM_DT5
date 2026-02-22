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
        "eta_stc": 24.4,      # % STC 효율
        "beta": -0.35,        # %/°C 온도 계수
        "noct": 45,          # °C NOCT 온도
        "delta": 0.5,        # %/yr 열화율
        "v_oc": 0.72,        # V 개방전압
        "j_sc": 40.5,        # mA/cm² 단락전류밀도
        "ff": 0.83,          # Fill Factor
        "area_per_100mw": 93  # ha 100MW당 필요면적
    },
    "tandem": {
        "name": "탠덤 (페로브스카이트/Si)",
        "eta_stc": 34.85,
        "beta": -0.25,
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
    "soec": {
        "rated_power_kw": 50000,  # 50 MW SOEC
        "efficiency_nominal": 0.85,
        "operating_temp_celsius": 800,
        "startup_time_min": 120,
        "degradation_rate_per_1000h": 0.5
    },
    "sofc": {
        "rated_power_kw": 50000,  # 50 MW SOFC
        "efficiency_nominal": 0.60, 
        "operating_temp_celsius": 800,
        "startup_time_min": 60,
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
