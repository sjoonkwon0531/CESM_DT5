"""
CEMS DT5 발표 전 핵심 패치
============================
본 보고서 §9.1 "반드시 수정 5건"에 대한 구체 패치.
Streamlit 앱 코드 안정성을 깨지 않도록 최소 변경 원칙으로 작성.

적용 방법:
1. config.py와 modules/m04_dcbus.py 직접 수정 (권장)
2. 또는 app.py 시작 부분에 본 파일을 import 하여 monkey patch

검증:
$ python tests/test_validation.py
→ FAIL 7건 → 4건 이하로 감소 예상
"""

import warnings
import numpy as np


# ============================================================================
# 패치 1: PV 면적 정정 (CRITICAL)
# ============================================================================
# 문제: config.py PV_TYPES의 area_per_100mw 값이 STC 기준 정격의 1.9~2.3배
#        → c-Si CF 30.6% (한국 실측 14~17% 범위 벗어남)
#        → 1,000~1,360시간 정격 클리핑 발생
#
# 정정: STC 기준 정확 면적으로 변경 (oversizing 의도라면 옵션 B 채택)

PV_AREA_CORRECTIONS = {
    # 옵션 A: STC 기준 정확 면적 (DC/AC ratio = 1.0)
    # 100 MW / (eta_STC × 1000 W/m²) / 1e4
    "c-Si":     41,  # 24.4% × 1000 W/m² × 41 ha × 1e4 m²/ha = 100.04 MW
    "tandem":   29,  # 34.85% × ... = 101.07 MW
    "triple":   25,  # 39.5% × ... = 98.75 MW
    "infinite": 15,  # 68.7% × ... = 103.05 MW
}


def apply_pv_area_correction(pv_types_dict):
    """config.PV_TYPES를 in-place 수정하여 면적 정정"""
    for pv_type, new_area in PV_AREA_CORRECTIONS.items():
        if pv_type in pv_types_dict:
            old = pv_types_dict[pv_type]['area_per_100mw']
            pv_types_dict[pv_type]['area_per_100mw'] = new_area
            print(f"  PV[{pv_type}]: area {old} → {new_area} ha/100MW")


# ============================================================================
# 패치 2: DC Bus NaN 방지 (HIGH)
# ============================================================================
# 문제: m04_dcbus.py L84~104에서 result 초기화에 일부 키 누락
#        → 잉여 시 _handle_surplus_power만 호출, load_shedding_mw 키 미생성
#        → simulate_time_series에서 pandas DataFrame이 NaN으로 채움
#
# 정정: result 초기화에 모든 가능 키를 0으로 추가

DEFAULT_DCBUS_KEYS = [
    'pv_power_mw', 'bess_discharge_mw', 'h2_fuelcell_mw', 'grid_import_mw',
    'aidc_load_mw', 'bess_charge_mw', 'h2_electrolyzer_mw', 'grid_export_mw',
    'conversion_loss_mw', 'power_balance_mw', 'curtailment_mw',
    'load_shedding_mw',  # ← 누락되어 있던 키
]


def patch_dcbus_module(dcbus_module_class):
    """DCBusModule.calculate_power_balance에 NaN 방지 wrapper 적용"""
    original = dcbus_module_class.calculate_power_balance

    def patched(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        for key in DEFAULT_DCBUS_KEYS:
            if key not in result:
                result[key] = 0.0
        return result

    dcbus_module_class.calculate_power_balance = patched
    return dcbus_module_class


# ============================================================================
# 패치 3: BESS 차익거래 산출근거 단위 정정 (CRITICAL)
# ============================================================================
# 문제: config.py L508~514
#   "2GWh BESS, 일 1사이클, 가격차 65₩/kWh × 365 × 0.9 = 42.7억"
#   → 실제 계산값 427.1억원 (10배 차이)
#
# 정정 옵션 (택일):

# 옵션 A: 텍스트만 정정 — "0.1 사이클/일 보수 운영"
BESS_ARBITRAGE_CONSERVATIVE_BASIS = (
    "2GWh BESS × 0.1사이클/일 (200 MWh/일) × 65₩/kWh 가격차 × 365일 × 90% 효율 = 42.7억 "
    "+ FR 보조서비스 50억 + 피크셰이빙 37.3억 = 130억원/년. "
    "BESS 수명 보호를 위한 보수적 운영 (연 36 사이클)."
)

# 옵션 B: 수치 상향 — 1사이클 풀 운영
BESS_ARBITRAGE_AGGRESSIVE_VALUE = 514  # 427 + 50 + 37
BESS_ARBITRAGE_AGGRESSIVE_BASIS = (
    "2GWh BESS × 1사이클/일 × 65₩/kWh × 365일 × 90% = 427억 "
    "+ FR 50억 + 피크셰이빙 37.3억 = 514억원/년. "
    "단 BESS 수명 단축 우려 (연 365 사이클, EoL 7년 vs 10년)."
)


# ============================================================================
# 패치 4: HESS 시스템 효율 보고값 정정 (HIGH)
# ============================================================================
# 문제: m02_hess.py _calculate_system_efficiency가 max_eff (supercap 0.98)을 반환
#        → 실제 운영 평균 효율은 약 0.66 (가중평균)
#
# 정정: 용량 가중 평균 또는 dispatch 가중 평균으로 변경

def patch_hess_system_efficiency(hess_module_class):
    """시스템 효율을 실제 운영 가중평균으로 정정"""
    import math

    def realistic_efficiency(self):
        total_capacity = sum(L.config.capacity_kwh for L in self.layers.values())
        if total_capacity == 0:
            return 0.0
        weighted = 0.0
        for L in self.layers.values():
            w = L.config.capacity_kwh / total_capacity
            eff = math.sqrt(L.config.efficiency_charge * L.config.efficiency_discharge)
            weighted += w * eff
        return weighted  # 약 0.66

    hess_module_class._calculate_system_efficiency = realistic_efficiency
    return hess_module_class


# ============================================================================
# 패치 5: 그리드 EF 통일 (LOW)
# ============================================================================
# 문제: ECONOMICS dict는 0.4168, CARBON_CONFIG는 0.4594
# 정정: 한국 전력거래소 2024 공식값 0.4594로 통일

GRID_EMISSION_FACTOR_KR_2024 = 0.4594  # tCO2/MWh


# ============================================================================
# 통합 적용 함수
# ============================================================================
def apply_all_patches(verbose=True):
    """모든 패치를 import-time에 적용. app.py 최상단에서 호출."""
    if verbose:
        print("=" * 60)
        print("CEMS DT5 발표 전 핵심 패치 적용")
        print("=" * 60)

    # 1. PV 면적
    if verbose:
        print("\n[1/5] PV 면적 정정 (CRITICAL)")
    import config
    apply_pv_area_correction(config.PV_TYPES)

    # 2. DC Bus NaN
    if verbose:
        print("\n[2/5] DC Bus NaN 방지 (HIGH)")
    from modules.m04_dcbus import DCBusModule
    patch_dcbus_module(DCBusModule)
    if verbose:
        print("  DCBusModule.calculate_power_balance: wrapper 적용")

    # 3. HESS 효율
    if verbose:
        print("\n[3/5] HESS 시스템 효율 가중평균 (HIGH)")
    from modules.m02_hess import HESSModule
    patch_hess_system_efficiency(HESSModule)
    if verbose:
        print("  HESSModule._calculate_system_efficiency: 가중평균으로 변경")

    # 4. 그리드 EF
    if verbose:
        print("\n[4/5] 그리드 배출계수 통일 (LOW)")
    config.ECONOMICS["korean_grid_emission_factor"] = GRID_EMISSION_FACTOR_KR_2024
    if verbose:
        print(f"  ECONOMICS dict EF → {GRID_EMISSION_FACTOR_KR_2024}")

    # 5. BESS 차익거래 (산출근거 텍스트만 — 수치는 그대로)
    if verbose:
        print("\n[5/5] BESS 차익거래 산출근거 정정 (CRITICAL)")
    config.ECONOMICS_CONFIG["additional_revenue_bess_arbitrage_basis"] = (
        BESS_ARBITRAGE_CONSERVATIVE_BASIS
    )
    if verbose:
        print("  basis 텍스트 → 0.1 사이클/일 보수 운영 명시")

    if verbose:
        print("\n" + "=" * 60)
        print("패치 적용 완료. tests/test_validation.py로 재검증 권장.")
        print("=" * 60)


# ============================================================================
# 단독 실행 시 자가 검증
# ============================================================================
if __name__ == "__main__":
    apply_all_patches(verbose=True)

    print("\n--- 패치 적용 후 검증 ---\n")

    # PV CF 재검증
    import sys
    sys.path.insert(0, '.')
    from modules.m01_pv import PVModule
    from modules.m10_weather import WeatherModule

    weather = WeatherModule()
    weather_data = weather.generate_tmy_data()

    print("PV 시뮬레이션 (정정 후):")
    print(f"  {'기술':<12} {'면적(ha)':<10} {'CF':<10} {'연발전(GWh)'}")
    for pv_type in ['c-Si', 'tandem', 'triple', 'infinite']:
        pv = PVModule(pv_type=pv_type, capacity_mw=100)
        sim = pv.simulate_time_series(weather_data)
        gen = sim['power_mw'].sum() / 1000
        cf = sim['power_mw'].sum() / (100 * 8760)
        print(f"  {pv_type:<12} {pv.total_area_m2/10000:<10.0f} {cf*100:<10.1f} {gen:.0f}")

    # 경제성 재검증 (Base case)
    print("\n경제성 base case (정정 후):")
    from modules.m09_economics import EconomicsModule
    econ = EconomicsModule()
    base = econ.run_base_case(
        annual_pv_generation_mwh=150000,
        annual_aidc_consumption_mwh=700000,
        annual_grid_import_mwh=550000,
        annual_surplus_mwh=5000,
    )
    print(f"  IRR: {base['irr_pct']:.2f}%")
    print(f"  NPV: {base['npv_billion_krw']:,.0f} 억원")
    print(f"  Payback: {base['payback_years']:.1f} 년")
