"""
M13. 투자 의사결정 대시보드 (Investment Decision Module)
NPV/IRR What-if, MC 시뮬레이션, Go/No-Go 매트릭스
과장 금지 원칙: 범위+신뢰구간 필수
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from config import ECONOMICS_CONFIG


def _to_list(v):
    """numpy array → list 변환"""
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _nan_guard(v, default=0.0):
    """NaN/Inf 가드"""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return default
    return v


class InvestmentDashboard:
    """투자 의사결정 분석 모듈"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ECONOMICS_CONFIG.copy()

    # ─────────────────────────────────────────────
    # NPV/IRR What-if 분석
    # ─────────────────────────────────────────────
    def whatif_analysis(self,
                        capex_billion: float = 10_000,
                        annual_revenue_billion: float = 500,
                        discount_rate: float = 0.05,
                        project_years: int = 20,
                        capex_variation: float = 0.0,
                        revenue_variation: float = 0.0) -> Dict:
        """
        What-if 분석: 슬라이더로 변수 조정

        Args:
            capex_billion: 에너지 인프라 CAPEX (억원)
            annual_revenue_billion: 연간 수익 (억원)
            discount_rate: 할인율
            project_years: 프로젝트 기간
            capex_variation: CAPEX 변동률 (-0.3 ~ +0.3)
            revenue_variation: 수익 변동률 (-0.3 ~ +0.3)

        Returns:
            What-if 분석 결과
        """
        adj_capex = capex_billion * (1 + capex_variation)
        adj_revenue = annual_revenue_billion * (1 + revenue_variation)

        cashflows = [-adj_capex] + [adj_revenue] * project_years
        npv = sum(cf / (1 + discount_rate) ** t
                  for t, cf in enumerate(cashflows))
        irr = self._calculate_irr(cashflows)

        payback = adj_capex / adj_revenue if adj_revenue > 0 else float('inf')

        return {
            "capex_billion_krw": round(_nan_guard(adj_capex), 1),
            "annual_revenue_billion_krw": round(_nan_guard(adj_revenue), 1),
            "discount_rate": discount_rate,
            "npv_billion_krw": round(_nan_guard(npv), 1),
            "irr_pct": round(_nan_guard(irr * 100), 2) if irr is not None else None,
            "payback_years": round(_nan_guard(payback), 1),
            "capex_variation_pct": round(capex_variation * 100, 1),
            "revenue_variation_pct": round(revenue_variation * 100, 1),
        }

    # ─────────────────────────────────────────────
    # MC 시뮬레이션 (10,000회)
    # ─────────────────────────────────────────────
    def monte_carlo(self,
                    n_iterations: int = 10_000,
                    capex_billion: float = 10_000,
                    annual_revenue_billion: float = 500,
                    capex_std_pct: float = 15,
                    revenue_std_pct: float = 20,
                    discount_rate: float = 0.05,
                    project_years: int = 20,
                    random_seed: int = 42) -> Dict:
        """
        Monte Carlo 시뮬레이션

        Args:
            n_iterations: 시뮬레이션 횟수
            capex_billion: 기준 CAPEX (억원)
            annual_revenue_billion: 기준 연간 수익 (억원)
            capex_std_pct: CAPEX 표준편차 (%)
            revenue_std_pct: 수익 표준편차 (%)
            discount_rate: 할인율
            project_years: 프로젝트 기간
            random_seed: 랜덤 시드

        Returns:
            MC 시뮬레이션 결과 (분포 + 통계)
        """
        rng = np.random.default_rng(random_seed)

        capex_samples = rng.normal(capex_billion,
                                   capex_billion * capex_std_pct / 100,
                                   n_iterations)
        capex_samples = np.maximum(capex_samples, capex_billion * 0.5)  # 하한

        revenue_samples = rng.normal(annual_revenue_billion,
                                      annual_revenue_billion * revenue_std_pct / 100,
                                      n_iterations)
        revenue_samples = np.maximum(revenue_samples, 0)

        irr_values = []
        npv_values = []

        for i in range(n_iterations):
            cf = [-capex_samples[i]] + [revenue_samples[i]] * project_years
            npv = sum(c / (1 + discount_rate) ** t for t, c in enumerate(cf))
            npv_values.append(npv)

            irr = self._calculate_irr(cf)
            irr_values.append(irr if irr is not None else np.nan)

        irr_arr = np.array(irr_values)
        npv_arr = np.array(npv_values)

        # NaN 제거
        valid_irr = irr_arr[~np.isnan(irr_arr)]
        valid_npv = npv_arr[~np.isnan(npv_arr)]

        return {
            "n_iterations": n_iterations,
            "n_valid": len(valid_irr),
            # IRR 통계
            "irr_mean_pct": round(_nan_guard(float(np.mean(valid_irr)) * 100), 2),
            "irr_median_pct": round(_nan_guard(float(np.median(valid_irr)) * 100), 2),
            "irr_std_pct": round(_nan_guard(float(np.std(valid_irr)) * 100), 2),
            "irr_p5_pct": round(_nan_guard(float(np.percentile(valid_irr, 5)) * 100), 2),
            "irr_p25_pct": round(_nan_guard(float(np.percentile(valid_irr, 25)) * 100), 2),
            "irr_p75_pct": round(_nan_guard(float(np.percentile(valid_irr, 75)) * 100), 2),
            "irr_p95_pct": round(_nan_guard(float(np.percentile(valid_irr, 95)) * 100), 2),
            # NPV 통계
            "npv_mean_billion_krw": round(_nan_guard(float(np.mean(valid_npv))), 1),
            "npv_median_billion_krw": round(_nan_guard(float(np.median(valid_npv))), 1),
            "npv_p5_billion_krw": round(_nan_guard(float(np.percentile(valid_npv, 5))), 1),
            "npv_p95_billion_krw": round(_nan_guard(float(np.percentile(valid_npv, 95))), 1),
            "prob_positive_npv_pct": round(
                float(np.sum(valid_npv > 0) / len(valid_npv) * 100), 1),
            # 분포 (히스토그램용)
            "irr_distribution": _to_list(valid_irr * 100),  # % 단위
            "npv_distribution": _to_list(valid_npv),
        }

    # ─────────────────────────────────────────────
    # 시나리오 비교 테이블
    # ─────────────────────────────────────────────
    def scenario_comparison(self) -> List[Dict]:
        """
        Base vs 복합 vs 최적 시나리오 비교

        Returns:
            시나리오별 경제성 비교
        """
        scenarios = [
            {
                "name": "Base (현행 정책)",
                "capex_billion": 10_000,
                "annual_revenue_billion": 500,
                "carbon_price_krw": 25_000,
                "subsidy_pct": 0,
            },
            {
                "name": "복합 (정책 강화)",
                "capex_billion": 10_000,
                "annual_revenue_billion": 700,  # 탄소+REC 수익 증가
                "carbon_price_krw": 50_000,
                "subsidy_pct": 0.1,
            },
            {
                "name": "최적 (보조금+정책)",
                "capex_billion": 10_000,
                "annual_revenue_billion": 900,
                "carbon_price_krw": 100_000,
                "subsidy_pct": 0.2,
            },
        ]

        results = []
        for s in scenarios:
            effective_capex = s["capex_billion"] * (1 - s["subsidy_pct"])
            cashflows = [-effective_capex] + [s["annual_revenue_billion"]] * 20
            irr = self._calculate_irr(cashflows)
            npv = sum(cf / 1.05 ** t for t, cf in enumerate(cashflows))
            payback = effective_capex / s["annual_revenue_billion"] \
                if s["annual_revenue_billion"] > 0 else float('inf')

            results.append({
                "scenario": s["name"],
                "capex_billion_krw": round(effective_capex, 1),
                "annual_revenue_billion_krw": s["annual_revenue_billion"],
                "irr_pct": round(_nan_guard(irr * 100), 2) if irr else None,
                "npv_billion_krw": round(_nan_guard(npv), 1),
                "payback_years": round(_nan_guard(payback), 1),
                "carbon_price_krw": s["carbon_price_krw"],
                "subsidy_pct": s["subsidy_pct"],
            })

        return results

    # ─────────────────────────────────────────────
    # Go/No-Go 의사결정 매트릭스
    # ─────────────────────────────────────────────
    def go_nogo_decision(self,
                         irr_pct: float = 4.5,
                         npv_billion: float = -500,
                         payback_years: float = 18,
                         prob_positive_npv_pct: float = 45,
                         min_irr_threshold: float = 5.0,
                         max_payback_threshold: float = 15,
                         min_prob_npv_threshold: float = 50) -> Dict:
        """
        투자 의사결정 매트릭스

        Args:
            irr_pct: 예상 IRR (%)
            npv_billion: NPV (억원)
            payback_years: 회수 기간 (년)
            prob_positive_npv_pct: NPV > 0 확률 (%)
            min_irr_threshold: 최소 IRR 기준 (%)
            max_payback_threshold: 최대 회수 기간 기준 (년)
            min_prob_npv_threshold: 최소 NPV>0 확률 기준 (%)

        Returns:
            Go/No-Go 판정
        """
        criteria = {
            "irr": {
                "value": irr_pct,
                "threshold": min_irr_threshold,
                "pass": irr_pct >= min_irr_threshold,
                "label": f"IRR {irr_pct:.1f}% {'≥' if irr_pct >= min_irr_threshold else '<'} {min_irr_threshold}%",
            },
            "npv": {
                "value": npv_billion,
                "threshold": 0,
                "pass": npv_billion >= 0,
                "label": f"NPV {npv_billion:.0f}억 {'≥' if npv_billion >= 0 else '<'} 0",
            },
            "payback": {
                "value": payback_years,
                "threshold": max_payback_threshold,
                "pass": payback_years <= max_payback_threshold,
                "label": f"Payback {payback_years:.1f}년 {'≤' if payback_years <= max_payback_threshold else '>'} {max_payback_threshold}년",
            },
            "probability": {
                "value": prob_positive_npv_pct,
                "threshold": min_prob_npv_threshold,
                "pass": prob_positive_npv_pct >= min_prob_npv_threshold,
                "label": f"P(NPV>0) {prob_positive_npv_pct:.0f}% {'≥' if prob_positive_npv_pct >= min_prob_npv_threshold else '<'} {min_prob_npv_threshold}%",
            },
        }

        passed = sum(1 for c in criteria.values() if c["pass"])
        total = len(criteria)

        if passed == total:
            decision = "GO"
            color = "green"
            recommendation = "모든 기준 충족. 투자 진행 권장."
        elif passed >= total - 1:
            decision = "CONDITIONAL GO"
            color = "yellow"
            failed = [k for k, v in criteria.items() if not v["pass"]]
            recommendation = f"조건부 승인. 미충족 기준: {', '.join(failed)}. 추가 검토 필요."
        else:
            decision = "NO-GO"
            color = "red"
            recommendation = f"투자 보류 권장. {total - passed}/{total} 기준 미충족."

        return {
            "decision": decision,
            "color": color,
            "recommendation": recommendation,
            "criteria": criteria,
            "passed": passed,
            "total": total,
            "score_pct": round(passed / total * 100, 1),
        }

    # ─────────────────────────────────────────────
    # 보조금 민감도
    # ─────────────────────────────────────────────
    def subsidy_sensitivity(self,
                            capex_billion: float = 10_000,
                            annual_revenue_billion: float = 500,
                            subsidy_rates: Optional[List[float]] = None) -> List[Dict]:
        """
        보조금 비율별 경제성 변화

        Args:
            capex_billion: 기준 CAPEX
            annual_revenue_billion: 연간 수익
            subsidy_rates: 보조금 비율 목록

        Returns:
            보조금별 IRR/NPV
        """
        if subsidy_rates is None:
            subsidy_rates = [0.0, 0.10, 0.20, 0.30]

        results = []
        for rate in subsidy_rates:
            effective_capex = capex_billion * (1 - rate)
            cashflows = [-effective_capex] + [annual_revenue_billion] * 20
            irr = self._calculate_irr(cashflows)
            npv = sum(cf / 1.05 ** t for t, cf in enumerate(cashflows))
            payback = effective_capex / annual_revenue_billion \
                if annual_revenue_billion > 0 else float('inf')

            results.append({
                "subsidy_pct": round(rate * 100, 0),
                "effective_capex_billion_krw": round(_nan_guard(effective_capex), 1),
                "irr_pct": round(_nan_guard(irr * 100), 2) if irr else None,
                "npv_billion_krw": round(_nan_guard(npv), 1),
                "payback_years": round(_nan_guard(payback), 1),
            })

        return results

    @staticmethod
    def _calculate_irr(cashflows: List[float], tol: float = 1e-6,
                       max_iter: int = 200) -> Optional[float]:
        """Newton-Raphson IRR"""
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
            if abs(rate) > 10:
                return None
        return rate if abs(sum(cf / (1 + rate) ** t
                              for t, cf in enumerate(cashflows))) < 1.0 else None
