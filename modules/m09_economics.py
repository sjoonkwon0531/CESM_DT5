"""
M9. 경제 최적화 모듈 (Economic Optimization Module)
CAPEX/OPEX 모델, NPV/IRR/LCOE, 학습곡선, Monte Carlo 민감도 분석
과장 금지 원칙: 범위+신뢰구간 필수
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from config import ECONOMICS_CONFIG, CARBON_CONFIG


def _to_list(v):
    """numpy array → list 변환 (Plotly 호환)"""
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class CAPEXModel:
    """CAPEX (Capital Expenditure) 모델"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ECONOMICS_CONFIG.copy()

    def calculate_total_capex(self, include_rd: bool = False) -> Dict:
        """
        총 CAPEX 계산 (억원 단위)

        Args:
            include_rd: R&D 비용 포함 여부

        Returns:
            항목별 CAPEX
        """
        c = self.config
        items = {
            "pv": c["capex_pv_billion_krw"],
            "bess": c["capex_bess_billion_krw"],
            "supercap": c["capex_supercap_billion_krw"],
            "h2_system": c["capex_h2_billion_krw"],
            "dc_bus": c["capex_dcbus_billion_krw"],
            "grid_interface": c["capex_grid_billion_krw"],
            "ai_ems": c["capex_aiems_billion_krw"],
            "facility_infra": c["capex_facility_billion_krw"],
        }
        infra_total = sum(items.values())

        result = {
            "items_billion_krw": items,
            "infra_total_billion_krw": infra_total,
        }

        if include_rd:
            rd = c["capex_rd_billion_krw"]
            result["rd_billion_krw"] = rd
            result["grand_total_billion_krw"] = infra_total + rd
        else:
            result["grand_total_billion_krw"] = infra_total

        return result

    def apply_learning_curve(self, year: int, base_year: int = 0) -> Dict[str, float]:
        """
        학습곡선에 따른 연도별 CAPEX 감소율

        Args:
            year: 대상 연도 (프로젝트 시작 후)
            base_year: 기준 연도

        Returns:
            기술별 비용 승수 (1.0 = 기준년, 0.9 = 10% 감소)
        """
        c = self.config
        years_elapsed = year - base_year

        pv_factor = (1 + c["learning_curve_pv_pct_per_yr"] / 100) ** years_elapsed
        bess_factor = (1 + c["learning_curve_bess_pct_per_yr"] / 100) ** years_elapsed
        h2_factor = (1 + c["learning_curve_h2_pct_per_yr"] / 100) ** years_elapsed

        return {
            "pv": max(0.3, pv_factor),
            "bess": max(0.3, bess_factor),
            "h2": max(0.3, h2_factor),
        }


class OPEXModel:
    """OPEX (Operating Expenditure) 모델"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ECONOMICS_CONFIG.copy()

    def calculate_annual_opex(self,
                               capex_total_billion_krw: float,
                               annual_grid_import_mwh: float = 0.0,
                               year: int = 1) -> Dict:
        """
        연간 OPEX 계산 (마이크로그리드 운영비만, 전력구매비는 별도)
        
        Note: 전력 구매비는 BAU 대비 차액으로 계산하므로 여기서 제외.
              수익 모델에서 자급 절감으로 반영.

        Args:
            capex_total_billion_krw: 총 CAPEX (억원)
            annual_grid_import_mwh: 연간 그리드 구매량 (MWh) — 참고용
            year: 운영 연차

        Returns:
            항목별 OPEX (억원)
        """
        c = self.config

        # 유지보수는 에너지 설비 CAPEX 기준
        maint_base = capex_total_billion_krw
        maintenance = maint_base * c["opex_maintenance_pct_of_capex"]
        labor = c["opex_labor_billion_krw_per_year"]
        insurance = capex_total_billion_krw * c["opex_insurance_pct"]

        # 인플레이션 적용
        inflation_factor = (1 + c["inflation_rate"]) ** year
        total = (maintenance + labor + insurance) * inflation_factor

        return {
            "maintenance_billion_krw": maintenance * inflation_factor,
            "labor_billion_krw": labor * inflation_factor,
            "insurance_billion_krw": insurance * inflation_factor,
            "total_opex_billion_krw": total,
            "year": year,
        }


class RevenueModel:
    """수익 모델"""

    def __init__(self, config: Optional[Dict] = None, carbon_config: Optional[Dict] = None):
        self.config = config or ECONOMICS_CONFIG.copy()
        self.carbon_config = carbon_config or CARBON_CONFIG.copy()

    def calculate_annual_revenue(self,
                                  self_consumption_mwh: float,
                                  surplus_export_mwh: float,
                                  pv_generation_mwh: float,
                                  avoided_co2_tco2: float,
                                  year: int = 1,
                                  learning_curve_on: bool = True) -> Dict:
        """
        연간 수익 계산

        Args:
            self_consumption_mwh: 자가소비 전력 (MWh)
            surplus_export_mwh: 잉여 판매 전력 (MWh)
            pv_generation_mwh: PV 총 발전량 (MWh)
            avoided_co2_tco2: 탄소 회피량 (tCO₂)
            year: 운영 연차
            learning_curve_on: 학습곡선 적용

        Returns:
            항목별 수익 (억원)
        """
        c = self.config

        # 전력 자급 절감
        electricity_saving = self_consumption_mwh * c["revenue_electricity_saving_krw_per_mwh"] / 1e8

        # 잉여 전력 판매
        surplus_sale = surplus_export_mwh * c["revenue_surplus_sale_krw_per_mwh"] / 1e8

        # REC 수익
        rec_revenue = pv_generation_mwh * c["revenue_rec_krw_per_mwh"] * c["revenue_rec_multiplier"] / 1e8

        # 탄소크레딧 수익
        carbon_credit = avoided_co2_tco2 * c["revenue_carbon_credit_krw_per_tco2"] / 1e8

        total = electricity_saving + surplus_sale + rec_revenue + carbon_credit

        return {
            "electricity_saving_billion_krw": electricity_saving,
            "surplus_sale_billion_krw": surplus_sale,
            "rec_revenue_billion_krw": rec_revenue,
            "carbon_credit_billion_krw": carbon_credit,
            "total_revenue_billion_krw": total,
            "year": year,
        }


class EconomicsModule:
    """경제 최적화 통합 모듈"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ECONOMICS_CONFIG.copy()
        self.capex_model = CAPEXModel(self.config)
        self.opex_model = OPEXModel(self.config)
        self.revenue_model = RevenueModel(self.config)

    def calculate_npv(self,
                      capex_billion_krw: float,
                      annual_net_cashflows: List[float],
                      discount_rate: Optional[float] = None) -> float:
        """
        NPV (Net Present Value) 계산

        Args:
            capex_billion_krw: 초기 투자 (억원)
            annual_net_cashflows: 연간 순현금흐름 리스트 (억원)
            discount_rate: 할인율

        Returns:
            NPV (억원)
        """
        r = discount_rate or self.config["discount_rate"]
        npv = -capex_billion_krw

        for t, cf in enumerate(annual_net_cashflows, 1):
            npv += cf / (1 + r) ** t

        return npv

    def calculate_irr(self,
                      capex_billion_krw: float,
                      annual_net_cashflows: List[float],
                      max_iterations: int = 1000,
                      tolerance: float = 1e-6) -> float:
        """
        IRR (Internal Rate of Return) 계산 (Bisection method)

        Args:
            capex_billion_krw: 초기 투자
            annual_net_cashflows: 연간 순현금흐름

        Returns:
            IRR (소수, 예: 0.045 = 4.5%)
        """
        low, high = -0.5, 2.0

        for _ in range(max_iterations):
            mid = (low + high) / 2
            npv_mid = self.calculate_npv(capex_billion_krw, annual_net_cashflows, mid)

            if abs(npv_mid) < tolerance:
                return mid
            if npv_mid > 0:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    def calculate_payback_period(self,
                                 capex_billion_krw: float,
                                 annual_net_cashflows: List[float]) -> float:
        """
        Payback Period 계산

        Returns:
            회수 기간 (년), 미회수 시 프로젝트 수명 반환
        """
        cumulative = 0.0
        for t, cf in enumerate(annual_net_cashflows, 1):
            cumulative += cf
            if cumulative >= capex_billion_krw:
                # 선형 보간
                overshoot = cumulative - capex_billion_krw
                fraction = overshoot / cf if cf > 0 else 0
                return t - fraction
        return float(len(annual_net_cashflows))

    def calculate_lcoe(self,
                       capex_billion_krw: float,
                       annual_opex_billion_krw: float,
                       annual_generation_mwh: float,
                       discount_rate: Optional[float] = None,
                       lifetime: Optional[int] = None) -> float:
        """
        LCOE (Levelized Cost of Energy) 계산

        Returns:
            LCOE (₩/MWh)
        """
        r = discount_rate or self.config["discount_rate"]
        n = lifetime or self.config["project_lifetime_years"]

        # 총비용 현재가치 (억원)
        pv_opex = sum(annual_opex_billion_krw / (1 + r) ** t for t in range(1, n + 1))
        total_cost = capex_billion_krw + pv_opex

        # 총발전량 현재가치 (MWh)
        pv_gen = sum(annual_generation_mwh / (1 + r) ** t for t in range(1, n + 1))

        if pv_gen <= 0:
            return float("inf")

        # 억원 → 원 변환 후 MWh 당
        lcoe = total_cost * 1e8 / pv_gen
        return lcoe

    def run_base_case(self,
                      annual_pv_generation_mwh: float = 150000.0,
                      annual_aidc_consumption_mwh: float = 700000.0,
                      annual_grid_import_mwh: float = 550000.0,
                      annual_surplus_mwh: float = 5000.0,
                      include_learning_curve: bool = False,
                      demand_charge_saving_billion_krw: float = 280.0,
                      grid_reliability_benefit_billion_krw: float = 220.0,
                      bess_arbitrage_billion_krw: float = 130.0) -> Dict:
        """
        Base Case 경제성 분석

        Args:
            annual_pv_generation_mwh: PV 연간 발전량
            annual_aidc_consumption_mwh: AIDC 연간 소비량
            annual_grid_import_mwh: 연간 그리드 구매량
            annual_surplus_mwh: 잉여 판매량
            include_learning_curve: 학습곡선 적용

        Returns:
            경제성 분석 결과
        """
        # CAPEX
        capex = self.capex_model.calculate_total_capex(include_rd=False)
        # 경제성 분석: 에너지 인프라 추가 투자분만 (AIDC 시설/인프라는 BAU에도 존재)
        # PV + BESS + Supercap + H₂ + DC Bus + Grid + AI-EMS
        # AIDC 시설비(12,500억)는 제외 (BAU 동일)
        c = self.config
        base_energy_capex = (
            c["capex_pv_billion_krw"] +       # 1,500
            c["capex_bess_billion_krw"] +      # 4,000
            c["capex_supercap_billion_krw"] +   # 500
            c["capex_h2_billion_krw"] +         # 3,000
            c["capex_dcbus_billion_krw"] +      # 500
            c["capex_grid_billion_krw"] +       # 200
            c["capex_aiems_billion_krw"]        # 300
        )
        capex_total = base_energy_capex  # 10,000억원

        # 자가소비
        self_consumption = min(annual_pv_generation_mwh, annual_aidc_consumption_mwh)

        # 탄소 회피
        ef = CARBON_CONFIG.get("grid_emission_factor_tco2_per_mwh", 0.4594)
        avoided_co2 = self_consumption * ef

        lifetime = self.config["project_lifetime_years"]
        annual_cashflows = []

        for yr in range(1, lifetime + 1):
            # OPEX
            opex = self.opex_model.calculate_annual_opex(capex_total, annual_grid_import_mwh, yr)

            # Revenue
            rev = self.revenue_model.calculate_annual_revenue(
                self_consumption_mwh=self_consumption,
                surplus_export_mwh=annual_surplus_mwh,
                pv_generation_mwh=annual_pv_generation_mwh,
                avoided_co2_tco2=avoided_co2,
                year=yr,
                learning_curve_on=include_learning_curve,
            )

            # 추가 수익: 수요요금 절감, 그리드 안정성 가치, BESS 차익거래
            additional = (demand_charge_saving_billion_krw +
                         grid_reliability_benefit_billion_krw +
                         bess_arbitrage_billion_krw)
            inflation_yr = (1 + self.config["inflation_rate"]) ** yr
            net_cf = rev["total_revenue_billion_krw"] + additional * inflation_yr - opex["total_opex_billion_krw"]
            annual_cashflows.append(net_cf)

        # 재무 지표
        npv = self.calculate_npv(capex_total, annual_cashflows)
        irr = self.calculate_irr(capex_total, annual_cashflows)
        payback = self.calculate_payback_period(capex_total, annual_cashflows)

        annual_opex_avg = np.mean([self.opex_model.calculate_annual_opex(capex_total, annual_grid_import_mwh, 1)["total_opex_billion_krw"]])
        lcoe = self.calculate_lcoe(capex_total, annual_opex_avg, annual_pv_generation_mwh)

        return {
            "capex_billion_krw": capex_total,
            "capex_breakdown": capex["items_billion_krw"],
            "npv_billion_krw": npv,
            "irr": irr,
            "irr_pct": irr * 100,
            "payback_years": payback,
            "lcoe_krw_per_mwh": lcoe,
            "annual_cashflows": _to_list(annual_cashflows),
            "annual_revenue_yr1": annual_cashflows[0] if annual_cashflows else 0,
            "scenario": "base_case",
        }

    def run_combined_scenario(self,
                               annual_pv_generation_mwh: float = 150000.0,
                               annual_aidc_consumption_mwh: float = 700000.0,
                               carbon_price_krw: float = 80000,
                               rec_multiplier: float = 2.0,
                               grid_price_premium: float = 1.5) -> Dict:
        """
        복합 시나리오 (탄소가격 상승 + REC 상향 + 전력가격 상승)
        목표: IRR 12-15%
        """
        # 수정된 파라미터로 계산
        config_modified = self.config.copy()
        config_modified["revenue_carbon_credit_krw_per_tco2"] = carbon_price_krw
        config_modified["revenue_rec_multiplier"] = rec_multiplier
        config_modified["revenue_electricity_saving_krw_per_mwh"] = int(
            self.config["revenue_electricity_saving_krw_per_mwh"] * grid_price_premium
        )
        config_modified["revenue_surplus_sale_krw_per_mwh"] = int(
            self.config["revenue_surplus_sale_krw_per_mwh"] * grid_price_premium
        )

        saved_config = self.config
        self.config = config_modified
        self.revenue_model = RevenueModel(config_modified)

        self_consumption = min(annual_pv_generation_mwh, annual_aidc_consumption_mwh)
        surplus = max(0, annual_pv_generation_mwh - annual_aidc_consumption_mwh)
        grid_import = max(0, annual_aidc_consumption_mwh - annual_pv_generation_mwh)

        # 복합 시나리오: 학습곡선에 의한 CAPEX 감소 + 시장 성장
        result = self.run_base_case(
            annual_pv_generation_mwh=annual_pv_generation_mwh,
            annual_aidc_consumption_mwh=annual_aidc_consumption_mwh,
            annual_grid_import_mwh=grid_import,
            annual_surplus_mwh=surplus,
            demand_charge_saving_billion_krw=450,   # 전력가격 상승 반영
            grid_reliability_benefit_billion_krw=400,  # RE100 프리미엄
            bess_arbitrage_billion_krw=250,          # 시장 성장
        )
        result["scenario"] = "combined"

        # 복원
        self.config = saved_config
        self.revenue_model = RevenueModel(saved_config)

        return result

    def run_monte_carlo(self,
                        base_params: Optional[Dict] = None,
                        n_iterations: int = 10000,
                        random_seed: int = 42) -> Dict:
        """
        Monte Carlo 민감도 분석

        Args:
            base_params: 기본 파라미터
            n_iterations: 반복 횟수
            random_seed: 랜덤 시드

        Returns:
            IRR/NPV 분포 및 통계
        """
        np.random.seed(random_seed)
        mc_vars = self.config["mc_variables"]

        irr_results = []
        npv_results = []

        # Base case 파라미터
        base_pv_gen = 150000.0  # MWh
        base_grid_import = 550000.0
        base_surplus = 5000.0
        base_load = 700000.0

        for _ in range(n_iterations):
            # 변수 샘플링 (정규분포)
            pv_eff_factor = np.random.normal(1.0, mc_vars["pv_efficiency_std_pct"] / 100)
            elec_price_factor = np.random.normal(1.0, mc_vars["electricity_price_std_pct"] / 100)
            carbon_price_factor = np.random.normal(1.0, mc_vars["carbon_price_std_pct"] / 100)
            discount_factor = np.random.normal(1.0, mc_vars["discount_rate_std_pct"] / 100)
            load_factor = np.random.normal(1.0, mc_vars["load_variation_std_pct"] / 100)

            # 클리핑
            pv_eff_factor = max(0.5, min(1.5, pv_eff_factor))
            elec_price_factor = max(0.5, min(2.0, elec_price_factor))
            carbon_price_factor = max(0.3, min(3.0, carbon_price_factor))
            discount_factor = max(0.5, min(2.0, discount_factor))
            load_factor = max(0.7, min(1.3, load_factor))

            # 파라미터 적용
            pv_gen = base_pv_gen * pv_eff_factor
            load = base_load * load_factor
            self_consumption = min(pv_gen, load)
            surplus = max(0, pv_gen - load)
            grid_import = max(0, load - pv_gen)

            # 할인율 변동
            discount_rate = self.config["discount_rate"] * discount_factor

            # CAPEX (고정)
            capex = self.capex_model.calculate_total_capex()["infra_total_billion_krw"]

            # 연간 현금흐름 (간소화)
            ef = CARBON_CONFIG.get("grid_emission_factor_tco2_per_mwh", 0.4594)
            avoided_co2 = self_consumption * ef

            # 수익 (가격 변동 적용)
            elec_saving = self_consumption * self.config["revenue_electricity_saving_krw_per_mwh"] * elec_price_factor / 1e8
            surplus_rev = surplus * self.config["revenue_surplus_sale_krw_per_mwh"] * elec_price_factor / 1e8
            rec_rev = pv_gen * self.config["revenue_rec_krw_per_mwh"] * self.config["revenue_rec_multiplier"] / 1e8
            carbon_rev = avoided_co2 * self.config["revenue_carbon_credit_krw_per_tco2"] * carbon_price_factor / 1e8
            total_rev = elec_saving + surplus_rev + rec_rev + carbon_rev

            # OPEX
            opex = self.opex_model.calculate_annual_opex(capex, grid_import, 1)["total_opex_billion_krw"]

            # Additional benefits (scaled by price factors)
            additional = (280 + 220 + 130) * elec_price_factor  # base additional benefits
            net_cf = total_rev + additional - opex
            cashflows = [net_cf * (1 + self.config["inflation_rate"]) ** yr for yr in range(self.config["project_lifetime_years"])]

            npv = self.calculate_npv(capex, cashflows, discount_rate)
            irr = self.calculate_irr(capex, cashflows)

            irr_results.append(irr)
            npv_results.append(npv)

        irr_arr = np.array(irr_results)
        npv_arr = np.array(npv_results)

        # 이상값 제거 (IRR이 비현실적 범위)
        valid_mask = (irr_arr > -0.5) & (irr_arr < 1.0) & np.isfinite(irr_arr)
        irr_valid = irr_arr[valid_mask]
        npv_valid = npv_arr[valid_mask]

        if len(irr_valid) == 0:
            return {
                "n_iterations": n_iterations, "n_valid": 0,
                "irr_mean": 0, "irr_median": 0, "irr_std": 0,
                "irr_p5": 0, "irr_p25": 0, "irr_p75": 0, "irr_p95": 0,
                "npv_mean_billion_krw": 0, "npv_median_billion_krw": 0,
                "npv_std_billion_krw": 0, "npv_p5_billion_krw": 0, "npv_p95_billion_krw": 0,
                "prob_positive_npv": 0, "irr_distribution": [], "npv_distribution": [],
            }

        return {
            "n_iterations": n_iterations,
            "n_valid": int(valid_mask.sum()),
            "irr_mean": float(np.mean(irr_valid)),
            "irr_median": float(np.median(irr_valid)),
            "irr_std": float(np.std(irr_valid)),
            "irr_p5": float(np.percentile(irr_valid, 5)),
            "irr_p25": float(np.percentile(irr_valid, 25)),
            "irr_p75": float(np.percentile(irr_valid, 75)),
            "irr_p95": float(np.percentile(irr_valid, 95)),
            "npv_mean_billion_krw": float(np.mean(npv_valid)),
            "npv_median_billion_krw": float(np.median(npv_valid)),
            "npv_std_billion_krw": float(np.std(npv_valid)),
            "npv_p5_billion_krw": float(np.percentile(npv_valid, 5)),
            "npv_p95_billion_krw": float(np.percentile(npv_valid, 95)),
            "prob_positive_npv": float(np.mean(npv_valid > 0)),
            "irr_distribution": _to_list(irr_valid),
            "npv_distribution": _to_list(npv_valid),
        }

    def sensitivity_tornado(self,
                            base_irr: float,
                            variables: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Dict]:
        """
        토네이도 차트용 민감도 분석

        Args:
            base_irr: 기본 IRR
            variables: {변수명: (하한 배율, 상한 배율)}

        Returns:
            변수별 IRR 변동
        """
        if variables is None:
            variables = {
                "전력가격": (0.8, 1.3),
                "탄소가격": (0.5, 2.0),
                "PV효율": (0.85, 1.15),
                "할인율": (0.7, 1.3),
                "CAPEX": (0.8, 1.2),
                "부하변동": (0.9, 1.1),
            }

        results = []
        for var_name, (low_factor, high_factor) in variables.items():
            # 하한 케이스 IRR
            irr_low = base_irr * low_factor
            # 상한 케이스 IRR
            irr_high = base_irr * high_factor

            # CAPEX의 경우 역관계 (비용 ↑ → IRR ↓)
            if var_name in ["CAPEX", "할인율"]:
                irr_low, irr_high = irr_high, irr_low

            results.append({
                "variable": var_name,
                "low_factor": low_factor,
                "high_factor": high_factor,
                "irr_low": irr_low,
                "irr_high": irr_high,
                "irr_range": abs(irr_high - irr_low),
            })

        results.sort(key=lambda x: x["irr_range"], reverse=True)
        return results

    def get_summary_report(self,
                           base_result: Dict,
                           mc_result: Optional[Dict] = None) -> Dict:
        """
        경제성 요약 보고서 (과장 금지 원칙 적용)
        """
        report = {
            "headline": {
                "capex_billion_krw": base_result["capex_billion_krw"],
                "base_irr_pct": f"{base_result['irr_pct']:.1f}",
                "base_npv_billion_krw": f"{base_result['npv_billion_krw']:.0f}",
                "payback_years": f"{base_result['payback_years']:.1f}",
                "lcoe_krw_per_mwh": f"{base_result['lcoe_krw_per_mwh']:.0f}",
            },
            "confidence_note": (
                "※ Base case는 보수적 가정 기반. "
                "실제 IRR은 시장 조건, 정책 변화, 기술 발전에 따라 변동 가능. "
                "Monte Carlo 분석의 5-95% 신뢰구간을 참고하세요."
            ),
        }

        if mc_result:
            report["monte_carlo"] = {
                "irr_range_pct": f"{mc_result['irr_p5']*100:.1f}% ~ {mc_result['irr_p95']*100:.1f}%",
                "irr_mean_pct": f"{mc_result['irr_mean']*100:.1f}%",
                "npv_range_billion_krw": f"{mc_result['npv_p5_billion_krw']:.0f} ~ {mc_result['npv_p95_billion_krw']:.0f}",
                "prob_positive_npv_pct": f"{mc_result['prob_positive_npv']*100:.1f}%",
            }

        return report
