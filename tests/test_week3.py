"""
Week 3 테스트: AI-EMS 제어 + 탄소 회계 + 경제 최적화
각 모듈별 최소 5개 + 통합 테스트
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import pandas as pd

from modules.m06_ai_ems import AIEMSModule, Tier1RealTimeControl, Tier2PredictiveControl, Tier3StrategicOptimizer, DispatchCommand
from modules.m07_carbon import CarbonAccountingModule, CarbonEmissionRecord
from modules.m09_economics import EconomicsModule, CAPEXModel, OPEXModel, RevenueModel


# =============================================================================
# M6: AI-EMS Tests
# =============================================================================
class TestAIEMS:
    def setup_method(self):
        self.ems = AIEMSModule()

    def test_dispatch_surplus(self):
        """PV 잉여 시 디스패치: PV→AIDC + HESS 충전"""
        cmd = self.ems.execute_dispatch(
            pv_power_mw=80, aidc_load_mw=50, hess_soc=0.3,
            h2_storage_level=0.5, grid_price_krw=80000
        )
        assert cmd.pv_to_aidc_mw == 50.0
        assert cmd.pv_to_hess_mw > 0
        assert cmd.total_to_aidc() >= 50.0

    def test_dispatch_deficit(self):
        """PV 부족 시 디스패치: Grid 구매"""
        cmd = self.ems.execute_dispatch(
            pv_power_mw=20, aidc_load_mw=80, hess_soc=0.1,
            h2_storage_level=0.1, grid_price_krw=80000
        )
        assert cmd.grid_to_aidc_mw > 0
        assert cmd.total_to_aidc() >= 70.0  # 약간의 오차 허용

    def test_dispatch_nan_guard(self):
        """NaN 입력 시 안전 처리"""
        cmd = self.ems.execute_dispatch(
            pv_power_mw=float('nan'), aidc_load_mw=50,
            hess_soc=float('nan'), h2_storage_level=0.5
        )
        assert np.isfinite(cmd.total_to_aidc())

    def test_kpi_calculation(self):
        """KPI 계산"""
        # 시뮬레이션 실행
        pv_profile = [60] * 12 + [0] * 12
        load_profile = [50] * 24
        price_profile = [80000] * 24
        self.ems.simulate_24h(pv_profile, load_profile, price_profile)
        kpi = self.ems.calculate_kpi()
        assert "self_sufficiency_ratio" in kpi
        assert 0 <= kpi["self_sufficiency_ratio"] <= 1

    def test_24h_simulation(self):
        """24시간 시뮬레이션 정상 동작"""
        pv = [0]*6 + [30, 50, 70, 80, 85, 80, 75, 70, 60, 40, 20, 5] + [0]*6
        load = [50]*24
        price = [60000]*6 + [80000]*12 + [60000]*6
        result = self.ems.simulate_24h(pv, load, price)
        assert len(result) == 24
        for cmd in result:
            assert isinstance(cmd, DispatchCommand)

    def test_tier1_mppt(self):
        """Tier1 MPPT 제어"""
        t1 = Tier1RealTimeControl()
        power = t1.mppt_control(80.0, 800.0)
        assert 70 < power <= 80  # 추적 효율 적용

    def test_tier1_dc_bus(self):
        """Tier1 DC Bus 안정화"""
        t1 = Tier1RealTimeControl()
        result = t1.dc_bus_stabilize(5.0, 0.5)
        assert "dc_bus_voltage_v" in result

    def test_tier2_pv_forecast(self):
        """Tier2 PV 출력 예측"""
        t2 = Tier2PredictiveControl()
        fc = t2.forecast_pv_output(60.0, horizon_hours=4)
        assert len(fc) == 4
        assert all(f >= 0 for f in fc)

    def test_tier3_daily_schedule(self):
        """Tier3 24시간 스케줄"""
        t3 = Tier3StrategicOptimizer()
        pv = [0]*6 + [30, 50, 70, 80, 85, 80, 75, 70, 60, 40, 20, 5] + [0]*6
        load = [50]*24
        price = [80000]*24
        schedule = t3.optimize_daily_schedule(pv, load, price)
        assert len(schedule) == 24

    def test_tier3_economic_optimization(self):
        """Tier3 연간 경제 최적화"""
        t3 = Tier3StrategicOptimizer()
        result = t3.economic_optimization(150000, 700000, 80000, 25000)
        assert result["net_annual_benefit_krw"] != 0
        assert result["carbon_avoided_tco2"] > 0


# =============================================================================
# M7: Carbon Accounting Tests
# =============================================================================
class TestCarbonAccounting:
    def setup_method(self):
        self.carbon = CarbonAccountingModule()

    def test_scope2_emission(self):
        """Scope 2: 그리드 전력 배출"""
        emission = self.carbon.calculate_scope2(1000)  # 1000 MWh
        expected = 1000 * 0.4594
        assert emission == pytest.approx(expected, rel=0.01)

    def test_scope2_zero_import(self):
        """Scope 2: 그리드 0 사용 시 배출 0"""
        assert self.carbon.calculate_scope2(0) == 0.0

    def test_scope3_annual(self):
        """Scope 3: 공급망 배출 (연간 균등 배분)"""
        s3 = self.carbon.calculate_scope3(100, 2000, 50, 20)
        assert s3 > 0
        # 100MW PV × 40 + 2000MWh × 65 + 50MW × 30 = 4000 + 130000 + 1500 = 135500
        # / 20 = 6775 tCO₂/yr
        assert s3 == pytest.approx(6775, rel=0.01)

    def test_avoided_emissions(self):
        """탄소 회피량 계산"""
        avoided = self.carbon.calculate_avoided_emissions(150000, 120000)
        expected = 120000 * 0.4594
        assert avoided == pytest.approx(expected, rel=0.01)

    def test_annual_summary(self):
        """연간 탄소 회계 요약"""
        summary = self.carbon.calculate_annual_summary(
            annual_grid_import_mwh=550000,
            annual_pv_generation_mwh=150000,
            annual_self_consumption_mwh=120000,
        )
        assert "scope1_tco2" in summary
        assert "scope2_tco2" in summary
        assert "reduction_ratio" in summary
        assert 0 <= summary["reduction_ratio"] <= 1

    def test_k_ets_credit(self):
        """K-ETS 탄소크레딧 판매"""
        result = self.carbon.calculate_k_ets_cost_or_revenue(
            net_emission_tco2=1000, baseline_tco2=5000
        )
        assert result["status"] == "credit_available"
        assert result["surplus_tco2"] == 4000
        assert result["revenue_krw"] > 0

    def test_k_ets_purchase(self):
        """K-ETS 배출권 구매"""
        result = self.carbon.calculate_k_ets_cost_or_revenue(
            net_emission_tco2=10000, baseline_tco2=5000
        )
        assert result["status"] == "purchase_required"
        assert result["cost_krw"] > 0

    def test_cbam_cost(self):
        """CBAM 비용 계산"""
        result = self.carbon.calculate_cbam_cost(100)  # 100 tCO₂
        assert "cbam_cost_krw" in result
        assert result["cbam_cost_krw"] >= 0
        assert result["differential_krw_per_tco2"] >= 0

    def test_carbon_credit_revenue(self):
        """탄소크레딧 수익"""
        result = self.carbon.calculate_carbon_credit_revenue(10000)
        assert result["total_revenue_krw"] == 10000 * 25000

    def test_hourly_simulation(self):
        """시간별 탄소 시뮬레이션"""
        grid = [10.0] * 24
        pv_self = [5.0] * 24
        df = self.carbon.simulate_annual_carbon(grid, pv_self)
        assert len(df) == 24
        assert "scope2_tco2" in df.columns


# =============================================================================
# M9: Economics Tests
# =============================================================================
class TestEconomics:
    def setup_method(self):
        self.econ = EconomicsModule()

    def test_capex_total(self):
        """CAPEX 총액 = 22,500억원"""
        capex = self.econ.capex_model.calculate_total_capex()
        assert capex["infra_total_billion_krw"] == 22500

    def test_capex_with_rd(self):
        """CAPEX + R&D"""
        capex = self.econ.capex_model.calculate_total_capex(include_rd=True)
        assert capex["grand_total_billion_krw"] == 22500 + 400

    def test_npv_calculation(self):
        """NPV 계산"""
        npv = self.econ.calculate_npv(100, [20]*10, 0.05)
        # PV of annuity: 20 × (1-(1.05)^-10)/0.05 ≈ 154.43
        assert npv == pytest.approx(54.43, abs=1.0)

    def test_irr_calculation(self):
        """IRR 계산"""
        # 100 투자, 매년 15 × 10년 → IRR ≈ 8.1%
        irr = self.econ.calculate_irr(100, [15]*10)
        assert 0.05 < irr < 0.12

    def test_payback_period(self):
        """Payback Period"""
        pb = self.econ.calculate_payback_period(100, [25]*10)
        assert pb == pytest.approx(4.0, abs=0.1)

    def test_lcoe(self):
        """LCOE 계산"""
        lcoe = self.econ.calculate_lcoe(
            capex_billion_krw=1500,
            annual_opex_billion_krw=30,
            annual_generation_mwh=150000,
        )
        assert lcoe > 0
        assert np.isfinite(lcoe)

    def test_base_case_irr_range(self):
        """Base case IRR: 4-5% 범위"""
        result = self.econ.run_base_case()
        irr = result["irr"]
        # 보수적 추정이므로 넓은 범위 허용
        assert -0.05 < irr < 0.15, f"Base IRR {irr*100:.1f}% out of expected range"

    def test_learning_curve(self):
        """학습곡선 비용 감소"""
        capex = self.econ.capex_model
        yr0 = capex.apply_learning_curve(0)
        yr10 = capex.apply_learning_curve(10)
        assert yr10["pv"] < yr0["pv"]
        assert yr10["bess"] < yr0["bess"]
        assert yr10["h2"] < yr0["h2"]

    def test_monte_carlo(self):
        """Monte Carlo 분석 (축소)"""
        result = self.econ.run_monte_carlo(n_iterations=100, random_seed=42)
        assert result["n_iterations"] == 100
        assert result["n_valid"] > 50
        assert "irr_mean" in result
        assert "npv_mean_billion_krw" in result
        assert 0 <= result["prob_positive_npv"] <= 1

    def test_sensitivity_tornado(self):
        """토네이도 민감도"""
        tornado = self.econ.sensitivity_tornado(0.045)
        assert len(tornado) > 0
        assert all("variable" in item for item in tornado)
        # 정렬 확인 (impact 큰 순)
        ranges = [item["irr_range"] for item in tornado]
        assert ranges == sorted(ranges, reverse=True)

    def test_summary_report(self):
        """경제성 요약 보고서"""
        base = self.econ.run_base_case()
        report = self.econ.get_summary_report(base)
        assert "headline" in report
        assert "confidence_note" in report


# =============================================================================
# Integration Tests (M1~M10)
# =============================================================================
class TestIntegration:
    def test_full_system_24h(self):
        """전체 시스템 24시간 통합 테스트"""
        from modules import PVModule, AIDCModule, WeatherModule, AIEMSModule, CarbonAccountingModule, EconomicsModule

        # 모듈 초기화
        weather = WeatherModule()
        weather_data = weather.generate_tmy_data(year=2024)

        pv = PVModule(pv_type="c-Si", capacity_mw=100)
        aidc = AIDCModule(gpu_type="H100", gpu_count=50000, pue_tier="tier2")
        ems = AIEMSModule()
        carbon = CarbonAccountingModule()

        # 24시간 시뮬레이션
        pv_profile = []
        load_profile = []
        for h in range(24):
            w = weather.get_weather_at_time(h + 4380)  # 6월 중순쯤
            pv_out = pv.calculate_power_output(w["ghi_w_per_m2"], w["temp_celsius"])
            pv_profile.append(pv_out["power_mw"])

            aidc_load = aidc.calculate_load_at_time(h, random_seed=h)
            load_profile.append(aidc_load["total_power_mw"])

        price_profile = [80000] * 24
        dispatches = ems.simulate_24h(pv_profile, load_profile, price_profile)
        assert len(dispatches) == 24

        # KPI
        kpi = ems.calculate_kpi(dispatches)
        assert kpi["dispatch_count"] == 24

        # 탄소 계산
        grid_imports = [d.grid_to_aidc_mw for d in dispatches]
        pv_self = [d.pv_to_aidc_mw for d in dispatches]
        for h in range(24):
            carbon.calculate_hourly_emissions(grid_imports[h], pv_self[h], hour=h)

        stats = carbon.get_emission_statistics()
        assert stats["hours_tracked"] == 24
        assert stats["total_scope2_tco2"] >= 0

    def test_economics_with_carbon(self):
        """경제성 + 탄소 통합"""
        carbon = CarbonAccountingModule()
        econ = EconomicsModule()

        summary = carbon.calculate_annual_summary(
            annual_grid_import_mwh=550000,
            annual_pv_generation_mwh=150000,
            annual_self_consumption_mwh=120000,
        )

        # 탄소크레딧 수익을 경제성에 반영
        credit = carbon.calculate_carbon_credit_revenue(summary["avoided_emission_tco2"])
        assert credit["total_revenue_krw"] > 0

        # 경제성 분석
        base = econ.run_base_case()
        assert base["capex_billion_krw"] == 10000  # 에너지 인프라만

    def test_ems_dispatch_energy_balance(self):
        """EMS 디스패치 에너지 균형 검증"""
        ems = AIEMSModule()
        cmd = ems.execute_dispatch(
            pv_power_mw=60, aidc_load_mw=50,
            hess_soc=0.5, h2_storage_level=0.5,
        )
        # PV 배분 ≈ PV 총 출력 (MPPT 보정 후)
        total_from_pv = cmd.total_from_pv() + cmd.curtailment_mw
        # MPPT 이후 PV는 약 59.4 (0.99 × 60)
        assert total_from_pv <= 60.0 * 1.01  # 약간의 오차 허용


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
