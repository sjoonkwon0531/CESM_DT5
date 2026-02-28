"""
M11. 정책 시뮬레이터 (Policy Simulator Module)
K-ETS, REC, CBAM, RE100, 전력수급계획 시나리오 분석
과장 금지 원칙: 범위+신뢰구간 필수
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config import ECONOMICS_CONFIG, CARBON_CONFIG


def _to_list(v):
    """numpy array → list 변환 (Plotly 호환)"""
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _nan_guard(v, default=0.0):
    """NaN/Inf 가드"""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return default
    return v


# ─────────────────────────────────────────────────────────────────
# 정책 시나리오 기본값
# ─────────────────────────────────────────────────────────────────
POLICY_SCENARIOS = {
    "k_ets": {
        "current": 25_000,      # ₩/tCO₂ (2025 현행)
        "moderate": 50_000,
        "aggressive": 100_000,
    },
    "rec": {
        "smp_base_krw_per_mwh": 80_000,
        "rec_spot_krw_per_mwh": 25_000,
        "rec_weight_solar": 1.0,
        "rec_weight_ess_charge": 5.0,   # ESS 충전 REC 가중치
    },
    "cbam": {
        "eu_price_eur_per_tco2": 80,
        "eur_to_krw": 1_450,
        "export_ratio": 0.3,           # 수출 비중 (AIDC 서비스)
    },
    "re100": {
        "target_pct": 100,
        "interim_2030_pct": 60,
        "interim_2040_pct": 90,
    },
    "ratepayer_protection": {
        "name": "Ratepayer Protection Pledge",
        "description": "AIDC 자체 전력 확보 → 일반 소비자 요금 전가 방지",
        "grid_cost_pass_through": 0.0,
        "self_generation_requirement": 0.80,
        "policy_context": "2026 White House Hyperscaler Pledge",
        "impact_on_lcoe": 1.10,
        "impact_on_grid_price": -0.05,
    },
    "power_plan": {
        "renewable_30": 0.30,
        "renewable_40": 0.40,
        "renewable_50": 0.50,
    },
}


class PolicySimulator:
    """정책 시뮬레이터: 다양한 정책 변수가 경제성에 미치는 영향 분석"""

    def __init__(self, config: Optional[Dict] = None, carbon_config: Optional[Dict] = None):
        self.config = config or ECONOMICS_CONFIG.copy()
        self.carbon_config = carbon_config or CARBON_CONFIG.copy()

    # ─────────────────────────────────────────────
    # K-ETS 탄소가격 시나리오
    # ─────────────────────────────────────────────
    def k_ets_scenario(self,
                       carbon_price_krw: float = 25_000,
                       annual_avoided_tco2: float = 30_000,
                       project_years: int = 20,
                       discount_rate: float = 0.05) -> Dict:
        """
        K-ETS 탄소가격 변화에 따른 경제성 영향

        Args:
            carbon_price_krw: 탄소가격 (₩/tCO₂)
            annual_avoided_tco2: 연간 회피 배출량 (tCO₂)
            project_years: 프로젝트 기간
            discount_rate: 할인율

        Returns:
            탄소크레딧 수익 분석
        """
        annual_revenue = carbon_price_krw * annual_avoided_tco2
        annual_revenue_billion = annual_revenue / 1e8  # 억원

        # NPV of carbon credit stream
        cashflows = [annual_revenue_billion] * project_years
        npv = sum(cf / (1 + discount_rate) ** (t + 1)
                  for t, cf in enumerate(cashflows))

        return {
            "carbon_price_krw_per_tco2": carbon_price_krw,
            "annual_avoided_tco2": annual_avoided_tco2,
            "annual_revenue_billion_krw": round(_nan_guard(annual_revenue_billion), 2),
            "npv_billion_krw": round(_nan_guard(npv), 2),
            "total_revenue_20yr_billion_krw": round(
                _nan_guard(annual_revenue_billion * project_years), 2),
        }

    def k_ets_scenarios_compare(self,
                                 prices: Optional[List[float]] = None,
                                 **kwargs) -> List[Dict]:
        """K-ETS 가격 시나리오 비교"""
        if prices is None:
            prices = [25_000, 50_000, 100_000]
        return [self.k_ets_scenario(carbon_price_krw=p, **kwargs) for p in prices]

    # ─────────────────────────────────────────────
    # REC 시장
    # ─────────────────────────────────────────────
    def rec_revenue(self,
                    annual_generation_mwh: float = 150_000,
                    rec_price_krw: float = 25_000,
                    rec_weight: float = 1.0,
                    ess_charge_mwh: float = 50_000,
                    ess_rec_weight: float = 5.0) -> Dict:
        """
        REC 수익 분석

        Args:
            annual_generation_mwh: 연간 PV 발전량
            rec_price_krw: REC 현물가 (₩/MWh)
            rec_weight: 태양광 REC 가중치
            ess_charge_mwh: ESS 충전량 (MWh)
            ess_rec_weight: ESS 충전 REC 가중치

        Returns:
            REC 수익 분석
        """
        pv_rec_count = annual_generation_mwh * rec_weight / 1000  # 1 REC = 1 MWh
        ess_rec_count = ess_charge_mwh * ess_rec_weight / 1000
        total_rec = pv_rec_count + ess_rec_count
        revenue = total_rec * rec_price_krw * 1000  # REC 단가는 ₩/REC
        revenue_billion = revenue / 1e8

        return {
            "pv_rec_count": round(pv_rec_count, 1),
            "ess_rec_count": round(ess_rec_count, 1),
            "total_rec": round(total_rec, 1),
            "rec_price_krw": rec_price_krw,
            "annual_revenue_billion_krw": round(_nan_guard(revenue_billion), 2),
        }

    # ─────────────────────────────────────────────
    # CBAM (탄소국경조정)
    # ─────────────────────────────────────────────
    def cbam_impact(self,
                    annual_revenue_billion_krw: float = 1000,
                    export_ratio: float = 0.3,
                    eu_carbon_price_eur: float = 80,
                    eur_to_krw: float = 1_450,
                    emission_intensity_tco2_per_billion: float = 50,
                    avoided_ratio: float = 0.7) -> Dict:
        """
        CBAM 영향 분석

        Args:
            annual_revenue_billion_krw: 연간 매출 (억원)
            export_ratio: EU 수출 비중
            eu_carbon_price_eur: EU 탄소가격 (€/tCO₂)
            eur_to_krw: 환율
            emission_intensity_tco2_per_billion: 매출 억원당 배출량 (tCO₂)
            avoided_ratio: CEMS로 회피 가능 비율

        Returns:
            CBAM 회피 비용 분석
        """
        export_revenue = annual_revenue_billion_krw * export_ratio
        export_emissions = export_revenue * emission_intensity_tco2_per_billion
        cbam_cost_without = export_emissions * eu_carbon_price_eur * eur_to_krw / 1e8  # 억원
        cbam_cost_with = cbam_cost_without * (1 - avoided_ratio)
        savings = cbam_cost_without - cbam_cost_with

        return {
            "export_revenue_billion_krw": round(export_revenue, 2),
            "export_emissions_tco2": round(export_emissions, 1),
            "cbam_cost_without_cems_billion_krw": round(_nan_guard(cbam_cost_without), 2),
            "cbam_cost_with_cems_billion_krw": round(_nan_guard(cbam_cost_with), 2),
            "cbam_savings_billion_krw": round(_nan_guard(savings), 2),
            "eu_carbon_price_eur": eu_carbon_price_eur,
        }

    # ─────────────────────────────────────────────
    # RE100 달성률
    # ─────────────────────────────────────────────
    def re100_achievement(self,
                          total_load_mwh: float = 700_000,
                          pv_generation_mwh: float = 150_000,
                          rec_purchased_mwh: float = 0,
                          ppa_mwh: float = 0) -> Dict:
        """
        RE100 달성률 시뮬레이션

        Args:
            total_load_mwh: 총 전력 소비량
            pv_generation_mwh: PV 자가발전량
            rec_purchased_mwh: REC 구매량
            ppa_mwh: PPA 조달량

        Returns:
            RE100 달성률 분석
        """
        renewable_total = pv_generation_mwh + rec_purchased_mwh + ppa_mwh
        achievement_pct = min(100.0, (renewable_total / total_load_mwh) * 100) \
            if total_load_mwh > 0 else 0.0

        gap_mwh = max(0, total_load_mwh - renewable_total)

        return {
            "total_load_mwh": total_load_mwh,
            "pv_self_mwh": pv_generation_mwh,
            "rec_purchased_mwh": rec_purchased_mwh,
            "ppa_mwh": ppa_mwh,
            "renewable_total_mwh": renewable_total,
            "achievement_pct": round(_nan_guard(achievement_pct), 1),
            "gap_mwh": round(gap_mwh, 1),
            "status": "달성" if achievement_pct >= 100 else "미달",
        }

    # ─────────────────────────────────────────────
    # 전력수급기본계획 시나리오
    # ─────────────────────────────────────────────
    def power_plan_scenario(self,
                            renewable_ratio: float = 0.30,
                            grid_emission_base: float = 0.4594) -> Dict:
        """
        전력수급기본계획 신재생 비율에 따른 그리드 배출계수 변화

        Args:
            renewable_ratio: 신재생 비율 (0~1)
            grid_emission_base: 현재 그리드 배출계수 (tCO₂/MWh)

        Returns:
            시나리오별 배출계수 및 영향
        """
        # 신재생 비율 증가 → 그리드 배출계수 감소 (선형 근사)
        # 현재 ~10% 신재생 → 0.4594 tCO₂/MWh
        # fossil 비중 비례로 배출계수 감소
        current_renewable = 0.10
        fossil_current = 1 - current_renewable
        fossil_new = 1 - renewable_ratio
        new_emission = grid_emission_base * (fossil_new / fossil_current) \
            if fossil_current > 0 else 0

        # CEMS 관점: 그리드 배출계수 감소 → 자가발전의 탄소 회피 가치 감소
        annual_grid_import_mwh = 550_000  # 그리드 수입 전력
        avoided_current = annual_grid_import_mwh * grid_emission_base / 1000  # 천 tCO₂가 아닌 tCO₂
        avoided_new = annual_grid_import_mwh * new_emission / 1000

        return {
            "renewable_ratio_pct": round(renewable_ratio * 100, 1),
            "grid_emission_factor_tco2_per_mwh": round(_nan_guard(new_emission), 4),
            "emission_reduction_pct": round(
                _nan_guard((1 - new_emission / grid_emission_base) * 100), 1)
                if grid_emission_base > 0 else 0,
            "grid_import_avoided_tco2": round(_nan_guard(avoided_current - avoided_new), 1),
            "note": "신재생 비율 ↑ → 그리드 배출계수 ↓ → 자가발전 탄소회피 가치 ↓",
        }

    # ─────────────────────────────────────────────
    # 정책 조합별 경제성 영향
    # ─────────────────────────────────────────────
    def policy_combination_impact(self,
                                   carbon_price_krw: float = 25_000,
                                   rec_price_krw: float = 25_000,
                                   cbam_eur: float = 80,
                                   subsidy_pct: float = 0.0,
                                   renewable_ratio: float = 0.30) -> Dict:
        """
        정책 조합이 NPV/IRR에 미치는 영향 계산

        Args:
            carbon_price_krw: K-ETS 탄소가격
            rec_price_krw: REC 가격
            cbam_eur: EU CBAM 가격
            subsidy_pct: 정부 보조금 비율 (0~1)
            renewable_ratio: 신재생 비율

        Returns:
            정책 조합별 경제성 변화
        """
        from modules.m09_economics import EconomicsModule

        # 정책에 따른 추가 수익/비용 절감 (연간, 억원)
        k_ets = self.k_ets_scenario(carbon_price_krw=carbon_price_krw)
        rec = self.rec_revenue(rec_price_krw=rec_price_krw)
        cbam = self.cbam_impact(eu_carbon_price_eur=cbam_eur)

        additional_annual_billion = (
            k_ets["annual_revenue_billion_krw"] +
            rec["annual_revenue_billion_krw"] +
            cbam["cbam_savings_billion_krw"]
        )

        # 보조금에 의한 CAPEX 감소
        base_capex = 10_000  # 에너지 인프라 CAPEX (억원)
        effective_capex = base_capex * (1 - subsidy_pct)

        # IRR 재계산 (간이 모델: 20년 균등 캐시플로)
        econ = EconomicsModule()
        base = econ.run_base_case()
        base_annual_cf = base.get("annual_revenue_yr1", 500)  # 억원

        total_annual = base_annual_cf + additional_annual_billion
        cashflows = [-effective_capex] + [total_annual] * 20

        # IRR 계산
        irr = _calculate_irr(cashflows)
        npv = sum(cf / (1.05 ** t) for t, cf in enumerate(cashflows))

        return {
            "carbon_price_krw": carbon_price_krw,
            "rec_price_krw": rec_price_krw,
            "cbam_eur": cbam_eur,
            "subsidy_pct": subsidy_pct,
            "renewable_ratio": renewable_ratio,
            "additional_annual_revenue_billion_krw": round(
                _nan_guard(additional_annual_billion), 2),
            "effective_capex_billion_krw": round(effective_capex, 2),
            "irr_pct": round(_nan_guard(irr * 100), 2) if irr is not None else None,
            "npv_billion_krw": round(_nan_guard(npv), 2),
            "base_irr_pct": round(_nan_guard(base["irr_pct"]), 2),
        }

    def simulate_policy_impact(self,
                              scenario_key: str = "ratepayer_protection",
                              base_lcoe_krw: float = 80.0,
                              dc_capacity_mw: float = 100.0) -> Dict:
        """
        정책 시나리오의 경제성 영향 시뮬레이션

        Args:
            scenario_key: 정책 시나리오 키
            base_lcoe_krw: 기준 LCOE (₩/kWh)
            dc_capacity_mw: 데이터센터 용량 (MW)

        Returns:
            정책 영향 분석 결과
        """
        if scenario_key not in POLICY_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_key}")

        scenario = POLICY_SCENARIOS[scenario_key]

        if scenario_key == "ratepayer_protection":
            self_gen_req = scenario["self_generation_requirement"]
            lcoe_mult = scenario["impact_on_lcoe"]
            grid_price_impact = scenario["impact_on_grid_price"]

            self_gen_mw = dc_capacity_mw * self_gen_req
            grid_mw = dc_capacity_mw * (1 - self_gen_req)

            adjusted_lcoe = base_lcoe_krw * lcoe_mult
            annual_hours = 8760
            self_gen_cost = self_gen_mw * annual_hours * adjusted_lcoe / 1e6  # 백만원
            grid_cost = grid_mw * annual_hours * base_lcoe_krw * (1 + grid_price_impact) / 1e6

            return {
                "scenario": scenario["name"],
                "description": scenario["description"],
                "policy_context": scenario["policy_context"],
                "self_generation_mw": round(self_gen_mw, 1),
                "grid_mw": round(grid_mw, 1),
                "adjusted_lcoe_krw_per_kwh": round(adjusted_lcoe, 1),
                "annual_self_gen_cost_million_krw": round(_nan_guard(self_gen_cost), 0),
                "annual_grid_cost_million_krw": round(_nan_guard(grid_cost), 0),
                "total_annual_cost_million_krw": round(_nan_guard(self_gen_cost + grid_cost), 0),
                "grid_price_impact_pct": round(grid_price_impact * 100, 1),
                "consumer_protection": "소비자 요금 전가 0%",
            }

        # 기본 fallback
        return {
            "scenario": scenario_key,
            "note": "해당 시나리오의 상세 시뮬레이션은 개별 함수 참조",
        }

    def policy_heatmap_data(self,
                            carbon_prices: Optional[List[float]] = None,
                            rec_prices: Optional[List[float]] = None) -> Dict:
        """
        정책 조합 히트맵 데이터 생성 (carbon_price × rec_price → IRR)
        """
        if carbon_prices is None:
            carbon_prices = [25_000, 50_000, 75_000, 100_000]
        if rec_prices is None:
            rec_prices = [15_000, 25_000, 35_000, 50_000]

        irr_matrix = []
        npv_matrix = []

        for cp in carbon_prices:
            irr_row = []
            npv_row = []
            for rp in rec_prices:
                result = self.policy_combination_impact(
                    carbon_price_krw=cp, rec_price_krw=rp)
                irr_row.append(_nan_guard(result["irr_pct"], 0))
                npv_row.append(_nan_guard(result["npv_billion_krw"], 0))
            irr_matrix.append(irr_row)
            npv_matrix.append(npv_row)

        return {
            "carbon_prices": _to_list(carbon_prices),
            "rec_prices": _to_list(rec_prices),
            "irr_matrix": irr_matrix,
            "npv_matrix": npv_matrix,
        }


def _calculate_irr(cashflows: List[float], tol: float = 1e-6,
                   max_iter: int = 200) -> Optional[float]:
    """Newton-Raphson IRR 계산"""
    if not cashflows or len(cashflows) < 2:
        return None
    rate = 0.05
    for _ in range(max_iter):
        npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))
        d_npv = sum(-t * cf / (1 + rate) ** (t + 1)
                    for t, cf in enumerate(cashflows))
        if abs(d_npv) < 1e-12:
            break
        new_rate = rate - npv / d_npv
        if abs(new_rate - rate) < tol:
            return new_rate
        rate = new_rate
        # Guard against divergence
        if abs(rate) > 10:
            return None
    return rate if abs(sum(cf / (1 + rate) ** t
                          for t, cf in enumerate(cashflows))) < 1.0 else None
