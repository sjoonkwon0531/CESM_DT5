"""
Week 4 테스트: 정책 시뮬레이터, 산업 상용화, 투자 의사결정
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.m11_policy import PolicySimulator
from modules.m12_industry import IndustryModel, CSP_PROFILES
from modules.m13_investment import InvestmentDashboard


# ═══════════════════════════════════════════════════════
# M11: 정책 시뮬레이터
# ═══════════════════════════════════════════════════════

class TestPolicySimulator:

    def setup_method(self):
        self.sim = PolicySimulator()

    def test_k_ets_base_scenario(self):
        """K-ETS 현행 가격 시나리오"""
        result = self.sim.k_ets_scenario(carbon_price_krw=25_000)
        assert result["carbon_price_krw_per_tco2"] == 25_000
        assert result["annual_revenue_billion_krw"] > 0
        assert result["npv_billion_krw"] > 0

    def test_k_ets_price_sensitivity(self):
        """탄소가격 상승 → 수익 증가"""
        low = self.sim.k_ets_scenario(carbon_price_krw=25_000)
        high = self.sim.k_ets_scenario(carbon_price_krw=100_000)
        assert high["annual_revenue_billion_krw"] > low["annual_revenue_billion_krw"]
        assert high["annual_revenue_billion_krw"] == pytest.approx(
            low["annual_revenue_billion_krw"] * 4, rel=0.01)

    def test_k_ets_scenarios_compare(self):
        """K-ETS 시나리오 비교"""
        results = self.sim.k_ets_scenarios_compare()
        assert len(results) == 3
        # 가격 순서대로 수익 증가
        assert results[0]["annual_revenue_billion_krw"] < results[2]["annual_revenue_billion_krw"]

    def test_rec_revenue(self):
        """REC 수익 계산"""
        result = self.sim.rec_revenue()
        assert result["pv_rec_count"] > 0
        assert result["ess_rec_count"] > 0
        assert result["annual_revenue_billion_krw"] > 0

    def test_cbam_impact(self):
        """CBAM 영향 분석"""
        result = self.sim.cbam_impact()
        assert result["cbam_savings_billion_krw"] > 0
        assert result["cbam_cost_with_cems_billion_krw"] < result["cbam_cost_without_cems_billion_krw"]

    def test_re100_not_achieved(self):
        """RE100 미달성 케이스"""
        result = self.sim.re100_achievement(
            total_load_mwh=700_000, pv_generation_mwh=150_000)
        assert result["achievement_pct"] < 100
        assert result["status"] == "미달"
        assert result["gap_mwh"] > 0

    def test_re100_achieved(self):
        """RE100 달성 케이스"""
        result = self.sim.re100_achievement(
            total_load_mwh=100_000, pv_generation_mwh=110_000)
        assert result["achievement_pct"] == 100.0
        assert result["status"] == "달성"

    def test_power_plan_renewable_increase(self):
        """신재생 비율 증가 → 그리드 배출계수 감소"""
        r30 = self.sim.power_plan_scenario(renewable_ratio=0.30)
        r50 = self.sim.power_plan_scenario(renewable_ratio=0.50)
        assert r50["grid_emission_factor_tco2_per_mwh"] < r30["grid_emission_factor_tco2_per_mwh"]

    def test_policy_combination(self):
        """정책 조합 경제성"""
        result = self.sim.policy_combination_impact(
            carbon_price_krw=50_000, rec_price_krw=30_000, subsidy_pct=0.1)
        assert result["irr_pct"] is not None
        assert result["irr_pct"] > result["base_irr_pct"]

    def test_policy_heatmap(self):
        """정책 히트맵 데이터"""
        hm = self.sim.policy_heatmap_data(
            carbon_prices=[25_000, 50_000],
            rec_prices=[25_000, 50_000])
        assert len(hm["irr_matrix"]) == 2
        assert len(hm["irr_matrix"][0]) == 2

    def test_irr_in_range(self):
        """정책 조합 IRR 범위 검증 (과장 금지)"""
        # Base case IRR ~4.5%, 최적 조합도 20% 미만
        result = self.sim.policy_combination_impact(
            carbon_price_krw=100_000, rec_price_krw=50_000, subsidy_pct=0.3)
        if result["irr_pct"] is not None:
            assert result["irr_pct"] < 30, "IRR이 비현실적으로 높음"


# ═══════════════════════════════════════════════════════
# M12: 산업 상용화
# ═══════════════════════════════════════════════════════

class TestIndustryModel:

    def setup_method(self):
        self.model = IndustryModel()

    def test_all_csp_keys(self):
        """모든 CSP 프로필 존재"""
        for key in ["samsung_pyeongtaek", "sk_icheon", "naver_sejong", "kakao_ansan"]:
            assert key in CSP_PROFILES

    def test_csp_analysis_samsung(self):
        """삼성 평택 분석"""
        result = self.model.csp_analysis("samsung_pyeongtaek")
        assert result["power_demand_mw"] == 500
        assert result["energy_capex_billion_krw"] > 0
        assert result["irr_pct"] is not None

    def test_csp_scaling_consistency(self):
        """CSP 스케일링 정합성: 큰 CSP → 큰 CAPEX"""
        samsung = self.model.csp_analysis("samsung_pyeongtaek")  # 500MW
        kakao = self.model.csp_analysis("kakao_ansan")  # 100MW
        assert samsung["energy_capex_billion_krw"] > kakao["energy_capex_billion_krw"]
        assert samsung["annual_co2_reduction_ton"] > kakao["annual_co2_reduction_ton"]

    def test_all_csp_comparison(self):
        """전체 CSP 비교"""
        results = self.model.all_csp_comparison()
        assert len(results) == 4

    def test_csp_with_subsidy(self):
        """보조금 적용 시 CAPEX 감소"""
        no_sub = self.model.csp_analysis("naver_sejong", subsidy_pct=0.0)
        with_sub = self.model.csp_analysis("naver_sejong", subsidy_pct=0.2)
        assert with_sub["energy_capex_billion_krw"] < no_sub["energy_capex_billion_krw"]

    def test_byog_scenario(self):
        """BYOG 시나리오"""
        result = self.model.byog_scenario("samsung_pyeongtaek", own_grid_pct=0.5)
        assert result["own_mw"] == 250.0
        assert result["grid_mw"] == 250.0
        assert "savings_pct" in result

    def test_scaling_analysis(self):
        """스케일링 모델"""
        results = self.model.scaling_analysis(target_capacities_mw=[50, 100, 200, 500])
        assert len(results) == 4
        # 규모의 경제: 큰 용량 → 낮은 단위비용 (cost_factor)
        assert results[-1]["cost_factor"] < results[0]["cost_factor"]

    def test_invalid_csp_raises(self):
        """잘못된 CSP 키"""
        with pytest.raises(ValueError):
            self.model.csp_analysis("nonexistent")

    def test_co2_reduction_positive(self):
        """모든 CSP CO₂ 감축량 양수"""
        for key in CSP_PROFILES:
            result = self.model.csp_analysis(key)
            assert result["annual_co2_reduction_ton"] > 0


# ═══════════════════════════════════════════════════════
# M13: 투자 의사결정
# ═══════════════════════════════════════════════════════

class TestInvestmentDashboard:

    def setup_method(self):
        self.dash = InvestmentDashboard()

    def test_whatif_base(self):
        """What-if 기본"""
        result = self.dash.whatif_analysis()
        assert result["irr_pct"] is not None
        assert result["npv_billion_krw"] is not None
        assert result["payback_years"] > 0

    def test_whatif_capex_increase(self):
        """CAPEX 증가 → IRR 감소"""
        base = self.dash.whatif_analysis(capex_variation=0.0)
        high = self.dash.whatif_analysis(capex_variation=0.3)
        if base["irr_pct"] and high["irr_pct"]:
            assert high["irr_pct"] < base["irr_pct"]

    def test_monte_carlo_statistics(self):
        """MC 시뮬레이션 통계 검증"""
        result = self.dash.monte_carlo(n_iterations=1000)
        assert result["n_valid"] > 900  # 대부분 유효
        # IRR은 합리적 범위
        assert -10 < result["irr_mean_pct"] < 20
        assert result["irr_p5_pct"] < result["irr_p95_pct"]
        # NPV 분포
        assert result["npv_p5_billion_krw"] < result["npv_p95_billion_krw"]

    def test_monte_carlo_distribution_length(self):
        """MC 분포 데이터 길이"""
        result = self.dash.monte_carlo(n_iterations=500)
        assert len(result["irr_distribution"]) > 400
        assert len(result["npv_distribution"]) == 500

    def test_scenario_comparison(self):
        """시나리오 비교"""
        results = self.dash.scenario_comparison()
        assert len(results) == 3
        # 최적 시나리오 IRR > Base
        base_irr = results[0]["irr_pct"]
        optimal_irr = results[2]["irr_pct"]
        if base_irr and optimal_irr:
            assert optimal_irr > base_irr

    def test_go_nogo_go(self):
        """Go 판정"""
        result = self.dash.go_nogo_decision(
            irr_pct=8.0, npv_billion=500, payback_years=12,
            prob_positive_npv_pct=70)
        assert result["decision"] == "GO"
        assert result["color"] == "green"

    def test_go_nogo_nogo(self):
        """No-Go 판정"""
        result = self.dash.go_nogo_decision(
            irr_pct=2.0, npv_billion=-500, payback_years=25,
            prob_positive_npv_pct=30)
        assert result["decision"] == "NO-GO"
        assert result["color"] == "red"

    def test_go_nogo_conditional(self):
        """Conditional Go 판정"""
        result = self.dash.go_nogo_decision(
            irr_pct=6.0, npv_billion=100, payback_years=14,
            prob_positive_npv_pct=40)  # probability만 미달
        assert result["decision"] == "CONDITIONAL GO"

    def test_subsidy_sensitivity(self):
        """보조금 민감도"""
        results = self.dash.subsidy_sensitivity()
        assert len(results) == 4
        # 보조금 증가 → IRR 증가
        if results[0]["irr_pct"] and results[-1]["irr_pct"]:
            assert results[-1]["irr_pct"] > results[0]["irr_pct"]

    def test_subsidy_capex_decrease(self):
        """보조금 → CAPEX 감소"""
        results = self.dash.subsidy_sensitivity()
        assert results[-1]["effective_capex_billion_krw"] < results[0]["effective_capex_billion_krw"]


# ═══════════════════════════════════════════════════════
# 통합 테스트
# ═══════════════════════════════════════════════════════

class TestIntegration:

    def test_policy_to_investment_flow(self):
        """정책 → 투자 의사결정 파이프라인"""
        pol = PolicySimulator()
        inv = InvestmentDashboard()

        combo = pol.policy_combination_impact(
            carbon_price_krw=50_000, subsidy_pct=0.1)
        
        decision = inv.go_nogo_decision(
            irr_pct=combo["irr_pct"] or 0,
            npv_billion=combo["npv_billion_krw"],
            payback_years=15,
            prob_positive_npv_pct=50)
        
        assert decision["decision"] in ["GO", "CONDITIONAL GO", "NO-GO"]

    def test_industry_to_investment_flow(self):
        """산업 모델 → 투자 의사결정"""
        ind = IndustryModel()
        inv = InvestmentDashboard()

        csp = ind.csp_analysis("naver_sejong")
        mc = inv.monte_carlo(
            n_iterations=500,
            capex_billion=csp["energy_capex_billion_krw"],
            annual_revenue_billion=csp["annual_revenue_billion_krw"])
        
        assert mc["n_valid"] > 400
        assert mc["irr_mean_pct"] is not None
