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

# 색상 팔레트 (Plotly 차트용)
COLOR_PALETTE = {
    "pv": "#FFD700",      # 골드 (태양광)
    "aidc": "#DC143C",    # 빨강 (부하)
    "bess": "#32CD32",    # 라임그린 (배터리)
    "grid": "#4169E1",    # 로얄블루 (계통)
    "h2": "#00CED1",      # 다크터쿼이즈 (수소)
    "surplus": "#90EE90", # 연한 녹색 (잉여)
    "deficit": "#FFB6C1"  # 연한 빨강 (부족)
}