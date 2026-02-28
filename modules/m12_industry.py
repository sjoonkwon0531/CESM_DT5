"""
M12. 산업 상용화 모델 (Industry Commercialization Module)
CSP별 맞춤 분석, BYOG 시나리오, 스케일링 모델
과장 금지 원칙: 범위+신뢰구간 필수
"""
import numpy as np
from typing import Dict, List, Optional
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


# ─────────────────────────────────────────────────────────────────
# CSP 프로필 정의
# ─────────────────────────────────────────────────────────────────
CSP_PROFILES = {
    "samsung_pyeongtaek": {
        "name": "삼성 평택",
        "description": "반도체 팹 + AIDC",
        "power_demand_mw": 500,
        "pue": 1.15,
        "workload_mix": {"training": 0.3, "inference": 0.5, "hpc": 0.2},
        "export_ratio": 0.6,        # 반도체 수출 비중
        "re100_committed": True,
        "land_available_ha": 200,
        "grid_capacity_mw": 400,
        "capex_multiplier": 1.0,     # 기준
    },
    "sk_icheon": {
        "name": "SK 이천",
        "description": "HBM 생산 + AIDC",
        "power_demand_mw": 300,
        "pue": 1.12,
        "workload_mix": {"training": 0.4, "inference": 0.4, "hpc": 0.2},
        "export_ratio": 0.7,
        "re100_committed": True,
        "land_available_ha": 120,
        "grid_capacity_mw": 250,
        "capex_multiplier": 1.05,
    },
    "naver_sejong": {
        "name": "네이버 세종",
        "description": "하이퍼스케일 DC",
        "power_demand_mw": 200,
        "pue": 1.08,
        "workload_mix": {"training": 0.2, "inference": 0.6, "hpc": 0.2},
        "export_ratio": 0.2,
        "re100_committed": True,
        "land_available_ha": 80,
        "grid_capacity_mw": 180,
        "capex_multiplier": 0.95,
    },
    "kakao_ansan": {
        "name": "카카오 안산",
        "description": "중규모 DC",
        "power_demand_mw": 100,
        "pue": 1.10,
        "workload_mix": {"training": 0.1, "inference": 0.7, "hpc": 0.2},
        "export_ratio": 0.1,
        "re100_committed": True,
        "land_available_ha": 40,
        "grid_capacity_mw": 80,
        "capex_multiplier": 1.10,    # 소규모 → 단위비용 ↑
    },
}

# 100MW 기준 CAPEX (억원) — config에서 가져오되, 스케일링용 기준값
BASE_CAPEX_100MW = {
    "pv": 1_500,
    "bess": 4_000,
    "supercap": 500,
    "h2": 3_000,
    "dcbus": 500,
    "grid": 200,
    "aiems": 300,
    "total_energy": 10_000,  # 에너지 인프라
    "facility": 12_500,       # AIDC 시설
    "total": 22_500,
}


# ─────────────────────────────────────────────────────────────────
# 글로벌 하이퍼스케일러 에너지 전략 프리셋
# ─────────────────────────────────────────────────────────────────
CSP_ENERGY_STRATEGIES = {
    "Google": {
        "name": "Co-located Renewables",
        "description": "현장 재생에너지 설비 공동 배치, 장기 PPA",
        "energy_mix": {"solar": 0.50, "wind": 0.30, "grid": 0.20},
        "ppa_years": 20,
        "strategy": "BYPASS_QUEUE",
        "capex_premium": 1.15,
        "example": "AES deal, Wilbarger Co. TX, Solar+Wind on-site"
    },
    "Amazon": {
        "name": "Dedicated Gas Generation",
        "description": "전용 가스 발전 설비 + 저장",
        "energy_mix": {"natural_gas": 0.65, "battery_storage": 0.15, "grid": 0.20},
        "gas_capacity_gw": 2.6,
        "storage_mw": 400,
        "investment_b": 7.0,
        "strategy": "DEDICATED_GEN",
        "example": "NIPSCO GenCo, Indiana"
    },
    "Meta": {
        "name": "Behind-the-Meter Gas",
        "description": "미터기 뒤편 직접 가스 발전, 다중 사이트",
        "energy_mix": {"natural_gas": 0.70, "grid": 0.30},
        "sites": ["Louisiana 2.25GW", "El Paso $473M", "Ohio 400MW"],
        "strategy": "MULTI_SITE",
        "example": "366MW modular gas turbines for 1GW DC"
    },
    "Microsoft": {
        "name": "Grid Partnership",
        "description": "그리드 인프라 직접 투자, 송전망 증설 비용 부담",
        "energy_mix": {"grid": 0.60, "wind": 0.25, "solar": 0.15},
        "contracted_gw": 7.9,
        "grid_org": "MISO",
        "strategy": "PAY_AND_BUILD",
        "example": "Black Hills Energy, Wyoming"
    },
    # 한국 CSP 프리셋
    "Samsung_SDS": {
        "name": "한국형 그리드 의존",
        "description": "한전 전력 + 자체 태양광 일부",
        "energy_mix": {"grid": 0.85, "solar": 0.10, "ess": 0.05},
        "strategy": "GRID_DEPENDENT",
        "example": "수원/화성 데이터센터"
    },
    "Naver": {
        "name": "친환경 DC (세종)",
        "description": "PPA + 연료전지 하이브리드",
        "energy_mix": {"grid": 0.60, "fuel_cell": 0.25, "solar": 0.15},
        "strategy": "HYBRID",
        "example": "세종 각 데이터센터"
    }
}

# 에너지원별 탄소 배출계수 (tCO₂/MWh)
_EMISSION_FACTORS = {
    "solar": 0.0, "wind": 0.0, "battery_storage": 0.0, "ess": 0.0,
    "grid": 0.4594, "natural_gas": 0.37, "fuel_cell": 0.0,
}

# 에너지원별 LCOE 추정 (₩/kWh)
_LCOE_FACTORS = {
    "solar": 60, "wind": 70, "battery_storage": 120, "ess": 120,
    "grid": 80, "natural_gas": 90, "fuel_cell": 150,
}


def get_csp_strategy(csp_name: str) -> Dict:
    """CSP 이름으로 에너지 전략 프리셋 반환"""
    if csp_name not in CSP_ENERGY_STRATEGIES:
        raise ValueError(f"Unknown CSP: {csp_name}. "
                         f"Available: {list(CSP_ENERGY_STRATEGIES.keys())}")
    return CSP_ENERGY_STRATEGIES[csp_name]


def compare_csp_strategies() -> List[Dict]:
    """전체 CSP 전략 비교 (LCOE, 탄소, 그리드 의존도)"""
    results = []
    for csp_name, strategy in CSP_ENERGY_STRATEGIES.items():
        mix = strategy["energy_mix"]

        # 가중평균 LCOE
        lcoe = sum(mix.get(src, 0) * _LCOE_FACTORS.get(src, 80)
                   for src in mix)

        # 가중평균 탄소 배출 (tCO₂/MWh)
        carbon = sum(mix.get(src, 0) * _EMISSION_FACTORS.get(src, 0.4)
                     for src in mix)

        # 그리드 의존도
        grid_dep = mix.get("grid", 0)

        results.append({
            "csp": csp_name,
            "strategy_name": strategy["name"],
            "description": strategy["description"],
            "energy_mix": mix,
            "lcoe_krw_per_kwh": round(lcoe, 1),
            "carbon_tco2_per_mwh": round(carbon, 4),
            "grid_dependency_pct": round(grid_dep * 100, 1),
            "strategy_type": strategy.get("strategy", "N/A"),
            "example": strategy.get("example", ""),
        })
    return results


class IndustryModel:
    """산업 상용화 분석 모델"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ECONOMICS_CONFIG.copy()
        self.csp_profiles = CSP_PROFILES.copy()

    def csp_analysis(self, csp_key: str,
                     subsidy_pct: float = 0.0,
                     carbon_price_krw: float = 25_000) -> Dict:
        """
        CSP별 맞춤 경제성 분석

        Args:
            csp_key: CSP 키 (samsung_pyeongtaek 등)
            subsidy_pct: 보조금 비율 (0~1)
            carbon_price_krw: 탄소가격

        Returns:
            CSP별 경제성 분석 결과
        """
        if csp_key not in self.csp_profiles:
            raise ValueError(f"Unknown CSP: {csp_key}. "
                             f"Available: {list(self.csp_profiles.keys())}")

        profile = self.csp_profiles[csp_key]
        demand_mw = profile["power_demand_mw"]

        # 스케일링: 100MW 기준 → 해당 용량
        scale = demand_mw / 100.0
        scaled = self._scale_capex(scale, profile["capex_multiplier"])

        # 보조금 적용 (에너지 인프라에만)
        energy_capex = scaled["energy_capex_billion"] * (1 - subsidy_pct)
        total_capex = energy_capex + scaled["facility_capex_billion"]

        # 연간 수익 (스케일 비례 + PUE 효과)
        base_annual_revenue = 500  # 억원 (100MW 기준, base case 근사)
        pue_factor = 1.4 / profile["pue"]  # PUE 개선 효과
        annual_revenue = base_annual_revenue * scale * pue_factor * 0.85  # 0.85 = 보수적 조정

        # 탄소 회피 수익
        annual_pv_gen_mwh = 150_000 * scale
        grid_ef = 0.4594  # tCO₂/MWh
        avoided_tco2 = annual_pv_gen_mwh * grid_ef
        carbon_revenue = avoided_tco2 * carbon_price_krw / 1e8  # 억원

        total_annual = annual_revenue + carbon_revenue

        # ROI, Payback
        roi_pct = (total_annual / energy_capex * 100) if energy_capex > 0 else 0
        payback_years = energy_capex / total_annual if total_annual > 0 else float('inf')

        # IRR (간이)
        cashflows = [-energy_capex] + [total_annual] * 20
        irr = self._calculate_irr(cashflows)

        # CO₂ 감축
        annual_co2_reduction_ton = avoided_tco2
        lifetime_co2_reduction_kton = annual_co2_reduction_ton * 20 / 1000

        return {
            "csp_key": csp_key,
            "csp_name": profile["name"],
            "description": profile["description"],
            "power_demand_mw": demand_mw,
            "pue": profile["pue"],
            "energy_capex_billion_krw": round(_nan_guard(energy_capex), 1),
            "facility_capex_billion_krw": round(scaled["facility_capex_billion"], 1),
            "total_capex_billion_krw": round(_nan_guard(total_capex), 1),
            "annual_revenue_billion_krw": round(_nan_guard(total_annual), 1),
            "annual_energy_revenue_billion_krw": round(_nan_guard(annual_revenue), 1),
            "annual_carbon_revenue_billion_krw": round(_nan_guard(carbon_revenue), 1),
            "roi_pct": round(_nan_guard(roi_pct), 2),
            "payback_years": round(_nan_guard(payback_years), 1),
            "irr_pct": round(_nan_guard(irr * 100), 2) if irr is not None else None,
            "annual_co2_reduction_ton": round(annual_co2_reduction_ton, 0),
            "lifetime_co2_reduction_kton": round(lifetime_co2_reduction_kton, 1),
            "subsidy_pct": subsidy_pct,
            "carbon_price_krw": carbon_price_krw,
            "re100_committed": profile["re100_committed"],
        }

    def all_csp_comparison(self,
                           subsidy_pct: float = 0.0,
                           carbon_price_krw: float = 25_000) -> List[Dict]:
        """전체 CSP 비교 분석"""
        return [self.csp_analysis(k, subsidy_pct, carbon_price_krw)
                for k in self.csp_profiles]

    # ─────────────────────────────────────────────
    # BYOG (Bring Your Own Grid) 시나리오
    # ─────────────────────────────────────────────
    def byog_scenario(self, csp_key: str,
                      own_grid_pct: float = 0.5) -> Dict:
        """
        BYOG 시나리오: 자체 그리드 비율에 따른 경제성

        Args:
            csp_key: CSP 키
            own_grid_pct: 자체 전력 조달 비율 (0~1)

        Returns:
            BYOG 분석 결과
        """
        profile = self.csp_profiles[csp_key]
        demand_mw = profile["power_demand_mw"]

        # 자체 그리드: PV + BESS + H₂로 조달하는 비율
        own_mw = demand_mw * own_grid_pct
        grid_mw = demand_mw * (1 - own_grid_pct)

        # 자체 조달 비용 (LCOE 136.5 ₩/kWh)
        own_lcoe_krw_per_kwh = 136.5
        own_annual_cost = own_mw * 8760 * 0.5 * own_lcoe_krw_per_kwh * 1000 / 1e8  # 억원
        # 0.5 = capacity factor 근사

        # 그리드 전력 비용 (80 ₩/kWh = 80,000 ₩/MWh)
        grid_cost_krw_per_kwh = 80
        grid_annual_cost = grid_mw * 8760 * 0.9 * grid_cost_krw_per_kwh * 1000 / 1e8

        total_cost = own_annual_cost + grid_annual_cost

        # 100% 그리드 대비 비용
        full_grid_cost = demand_mw * 8760 * 0.9 * grid_cost_krw_per_kwh * 1000 / 1e8

        return {
            "csp_key": csp_key,
            "csp_name": profile["name"],
            "own_grid_pct": round(own_grid_pct * 100, 1),
            "own_mw": round(own_mw, 1),
            "grid_mw": round(grid_mw, 1),
            "own_annual_cost_billion_krw": round(_nan_guard(own_annual_cost), 1),
            "grid_annual_cost_billion_krw": round(_nan_guard(grid_annual_cost), 1),
            "total_annual_cost_billion_krw": round(_nan_guard(total_cost), 1),
            "full_grid_cost_billion_krw": round(_nan_guard(full_grid_cost), 1),
            "savings_pct": round(
                _nan_guard((1 - total_cost / full_grid_cost) * 100), 1)
                if full_grid_cost > 0 else 0,
        }

    # ─────────────────────────────────────────────
    # 스케일링 모델
    # ─────────────────────────────────────────────
    def scaling_analysis(self,
                         target_capacities_mw: Optional[List[float]] = None,
                         subsidy_pct: float = 0.0) -> List[Dict]:
        """
        스케일링 분석: 다양한 용량에서의 경제성

        Args:
            target_capacities_mw: 분석 대상 용량 목록 (MW)
            subsidy_pct: 보조금 비율

        Returns:
            용량별 경제성 분석
        """
        if target_capacities_mw is None:
            target_capacities_mw = [50, 100, 200, 500]

        results = []
        for cap in target_capacities_mw:
            scale = cap / 100.0

            # 규모의 경제: 대규모 → 단위비용 감소 (0.85 power law)
            cost_factor = scale ** 0.85 / scale  # <1 for scale>1
            scaled = self._scale_capex(scale, cost_factor)

            energy_capex = scaled["energy_capex_billion"] * (1 - subsidy_pct)

            # 연간 수익 (선형 스케일링)
            annual_revenue = 500 * scale * 0.85  # 억원

            irr = self._calculate_irr(
                [-energy_capex] + [annual_revenue] * 20)

            results.append({
                "capacity_mw": cap,
                "energy_capex_billion_krw": round(_nan_guard(energy_capex), 1),
                "annual_revenue_billion_krw": round(_nan_guard(annual_revenue), 1),
                "irr_pct": round(_nan_guard(irr * 100), 2) if irr is not None else None,
                "payback_years": round(
                    _nan_guard(energy_capex / annual_revenue), 1)
                    if annual_revenue > 0 else None,
                "cost_factor": round(cost_factor, 3),
                "note": f"규모의 경제 적용 (0.85 power law)",
            })

        return results

    # ─────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────
    def _scale_capex(self, scale: float, multiplier: float = 1.0) -> Dict:
        """CAPEX 스케일링"""
        energy = BASE_CAPEX_100MW["total_energy"] * scale * multiplier
        facility = BASE_CAPEX_100MW["facility"] * scale
        return {
            "energy_capex_billion": energy,
            "facility_capex_billion": facility,
            "total_capex_billion": energy + facility,
        }

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
