"""
M7. 탄소 회계 모듈 (Carbon Accounting Module)
Scope 1/2/3 탄소 배출 추적, K-ETS 연동, CBAM 시나리오
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config import CARBON_CONFIG


def _to_list(v):
    """numpy array → list 변환 (Plotly 호환)"""
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


@dataclass
class CarbonEmissionRecord:
    """탄소 배출 기록"""
    scope1_tco2: float = 0.0
    scope2_tco2: float = 0.0
    scope3_tco2: float = 0.0
    avoided_tco2: float = 0.0
    net_tco2: float = 0.0
    timestamp_hour: int = 0

    @property
    def total_tco2(self) -> float:
        return self.scope1_tco2 + self.scope2_tco2 + self.scope3_tco2

    def to_dict(self) -> Dict:
        return {
            "scope1_tco2": self.scope1_tco2,
            "scope2_tco2": self.scope2_tco2,
            "scope3_tco2": self.scope3_tco2,
            "total_tco2": self.total_tco2,
            "avoided_tco2": self.avoided_tco2,
            "net_tco2": self.net_tco2,
            "timestamp_hour": self.timestamp_hour,
        }


class CarbonAccountingModule:
    """탄소 회계 통합 모듈"""

    def __init__(self,
                 grid_emission_factor: Optional[float] = None,
                 k_ets_price: Optional[float] = None,
                 cbam_price_eur: Optional[float] = None):
        """
        탄소 회계 모듈 초기화

        Args:
            grid_emission_factor: 전력 배출계수 (tCO₂/MWh)
            k_ets_price: K-ETS 가격 (₩/tCO₂)
            cbam_price_eur: CBAM 가격 (€/tCO₂)
        """
        self.config = CARBON_CONFIG.copy()
        if grid_emission_factor is not None:
            self.config["grid_emission_factor_tco2_per_mwh"] = grid_emission_factor
        if k_ets_price is not None:
            self.config["k_ets_price_krw_per_tco2"] = k_ets_price
        if cbam_price_eur is not None:
            self.config["cbam_price_eur_per_tco2"] = cbam_price_eur

        self.emission_history: List[CarbonEmissionRecord] = []

    # =========================================================================
    # Scope 1: 직접 배출
    # =========================================================================
    def calculate_scope1(self,
                         h2_production_kg: float = 0.0,
                         diesel_backup_liters: float = 0.0) -> float:
        """
        Scope 1: 직접 배출 계산

        Args:
            h2_production_kg: H₂ 생산량 (kg) — SOEC는 무배출이나 보조연료 사용 시
            diesel_backup_liters: 비상 디젤 발전기 사용량 (L)

        Returns:
            tCO₂ 배출량
        """
        # SOEC 수전해는 전기분해이므로 직접 배출 없음
        # 비상 디젤 발전기: 2.68 kgCO₂/L
        diesel_emission = diesel_backup_liters * 2.68 / 1000  # tCO₂

        # H₂ 생산 시 보조 가열 (천연가스 사용 시, 일반적으로 전기 가열이지만 보수적으로)
        # SOEC는 전기 가열이므로 0, 하지만 SMR 방식이면 ~10 kgCO₂/kgH₂
        # 여기서는 SOEC (전기) 가정 → Scope 1 = 0
        h2_direct_emission = 0.0

        return diesel_emission + h2_direct_emission

    # =========================================================================
    # Scope 2: 간접 배출 (전력 사용)
    # =========================================================================
    def calculate_scope2(self, grid_import_mwh: float) -> float:
        """
        Scope 2: 전력 사용 간접 배출

        Args:
            grid_import_mwh: 계통 전력 구매량 (MWh)

        Returns:
            tCO₂ 배출량
        """
        if not np.isfinite(grid_import_mwh) or grid_import_mwh < 0:
            return 0.0

        ef = self.config["grid_emission_factor_tco2_per_mwh"]
        return grid_import_mwh * ef

    # =========================================================================
    # Scope 3: 공급망 배출
    # =========================================================================
    def calculate_scope3(self,
                         pv_capacity_mw: float = 100.0,
                         bess_capacity_mwh: float = 2000.0,
                         h2_capacity_mw: float = 50.0,
                         project_lifetime_years: int = 20) -> float:
        """
        Scope 3: 공급망 배출 (설비 제조, 운송 등)

        Args:
            pv_capacity_mw: PV 설비 용량 (MW)
            bess_capacity_mwh: BESS 용량 (MWh)
            h2_capacity_mw: H₂ 시스템 용량 (MW)
            project_lifetime_years: 수명

        Returns:
            연간 tCO₂ 배출량 (생애주기 균등 배분)
        """
        pv_mfg = pv_capacity_mw * self.config["scope3_pv_manufacturing_tco2_per_mw"]
        bess_mfg = bess_capacity_mwh * self.config["scope3_bess_manufacturing_tco2_per_mwh"]
        h2_mfg = h2_capacity_mw * self.config["scope3_h2_manufacturing_tco2_per_mw"]

        total_lifecycle = pv_mfg + bess_mfg + h2_mfg
        annual_scope3 = total_lifecycle / max(1, project_lifetime_years)

        return annual_scope3

    # =========================================================================
    # 탄소 회피량 (Avoided Emissions)
    # =========================================================================
    def calculate_avoided_emissions(self,
                                    renewable_generation_mwh: float,
                                    self_consumption_mwh: float) -> float:
        """
        회피된 탄소 배출량 (재생에너지 자가소비 + 그리드 대체)

        Args:
            renewable_generation_mwh: 재생에너지 발전량 (MWh)
            self_consumption_mwh: 재생에너지 자가소비량 (MWh)

        Returns:
            tCO₂ 회피량
        """
        ef = self.config["grid_emission_factor_tco2_per_mwh"]
        # 자가소비 → 그리드 전력 대체
        avoided = self_consumption_mwh * ef
        return max(0.0, avoided)

    # =========================================================================
    # 시간별 탄소 회계
    # =========================================================================
    def calculate_hourly_emissions(self,
                                   grid_import_mwh: float,
                                   pv_self_consumption_mwh: float,
                                   diesel_liters: float = 0.0,
                                   hour: int = 0) -> CarbonEmissionRecord:
        """시간별 탄소 배출 계산"""
        scope1 = self.calculate_scope1(diesel_backup_liters=diesel_liters)
        scope2 = self.calculate_scope2(grid_import_mwh)
        scope3 = self.calculate_scope3() / 8760  # 연간 → 시간당
        avoided = self.calculate_avoided_emissions(pv_self_consumption_mwh, pv_self_consumption_mwh)

        record = CarbonEmissionRecord(
            scope1_tco2=scope1,
            scope2_tco2=scope2,
            scope3_tco2=scope3,
            avoided_tco2=avoided,
            net_tco2=scope1 + scope2 + scope3 - avoided,
            timestamp_hour=hour,
        )
        self.emission_history.append(record)
        return record

    # =========================================================================
    # 연간 탄소 회계 요약
    # =========================================================================
    def calculate_annual_summary(self,
                                 annual_grid_import_mwh: float,
                                 annual_pv_generation_mwh: float,
                                 annual_self_consumption_mwh: float,
                                 pv_capacity_mw: float = 100.0,
                                 bess_capacity_mwh: float = 2000.0,
                                 h2_capacity_mw: float = 50.0) -> Dict:
        """연간 탄소 회계 요약"""
        scope1 = self.calculate_scope1()  # SOEC 기반이므로 거의 0
        scope2 = self.calculate_scope2(annual_grid_import_mwh)
        scope3 = self.calculate_scope3(pv_capacity_mw, bess_capacity_mwh, h2_capacity_mw)
        avoided = self.calculate_avoided_emissions(annual_pv_generation_mwh, annual_self_consumption_mwh)

        total = scope1 + scope2 + scope3
        net = total - avoided

        # BAU (Business As Usual): 전량 그리드 구매 시
        bau_total_mwh = annual_grid_import_mwh + annual_self_consumption_mwh
        bau_emissions = bau_total_mwh * self.config["grid_emission_factor_tco2_per_mwh"]

        # 감축량
        reduction = bau_emissions - net
        reduction_ratio = reduction / bau_emissions if bau_emissions > 0 else 0

        # 탄소중립 달성률
        neutrality_ratio = avoided / total if total > 0 else 1.0

        return {
            "scope1_tco2": scope1,
            "scope2_tco2": scope2,
            "scope3_tco2": scope3,
            "total_emission_tco2": total,
            "avoided_emission_tco2": avoided,
            "net_emission_tco2": net,
            "bau_emission_tco2": bau_emissions,
            "reduction_tco2": reduction,
            "reduction_ratio": float(np.clip(reduction_ratio, 0, 1)),
            "carbon_neutrality_ratio": float(np.clip(neutrality_ratio, 0, 1)),
        }

    # =========================================================================
    # K-ETS 탄소 거래
    # =========================================================================
    def calculate_k_ets_cost_or_revenue(self,
                                         net_emission_tco2: float,
                                         baseline_tco2: float = 0.0) -> Dict:
        """
        K-ETS 비용/수익 계산

        Args:
            net_emission_tco2: 순 배출량
            baseline_tco2: 배출 할당량 (무상할당)

        Returns:
            K-ETS 비용 또는 수익
        """
        price = self.config["k_ets_price_krw_per_tco2"]
        excess = net_emission_tco2 - baseline_tco2

        if excess > 0:
            # 배출권 구매 필요
            cost = excess * price
            return {
                "status": "purchase_required",
                "excess_tco2": excess,
                "cost_krw": cost,
                "revenue_krw": 0,
            }
        else:
            # 탄소크레딧 판매 가능
            surplus = -excess
            revenue = surplus * self.config["carbon_credit_price_krw_per_tco2"]
            return {
                "status": "credit_available",
                "surplus_tco2": surplus,
                "cost_krw": 0,
                "revenue_krw": revenue,
            }

    # =========================================================================
    # CBAM (EU Carbon Border Adjustment Mechanism)
    # =========================================================================
    def calculate_cbam_cost(self,
                            exported_product_tco2: float,
                            eu_carbon_price_eur: Optional[float] = None) -> Dict:
        """
        CBAM 비용 계산 (EU 수출 시)

        Args:
            exported_product_tco2: 수출 제품 내재 탄소 (tCO₂)
            eu_carbon_price_eur: EU ETS 가격 (€/tCO₂)

        Returns:
            CBAM 비용 분석
        """
        if eu_carbon_price_eur is None:
            eu_carbon_price_eur = self.config["cbam_price_eur_per_tco2"]

        eur_to_krw = self.config["eur_to_krw"]
        k_ets_price = self.config["k_ets_price_krw_per_tco2"]

        # CBAM 비용 = (EU ETS 가격 - 기지불 탄소가격) × 내재 탄소량
        eu_price_krw = eu_carbon_price_eur * eur_to_krw
        cbam_differential = max(0, eu_price_krw - k_ets_price)
        cbam_cost_krw = exported_product_tco2 * cbam_differential

        return {
            "exported_tco2": exported_product_tco2,
            "eu_ets_price_eur": eu_carbon_price_eur,
            "eu_ets_price_krw": eu_price_krw,
            "k_ets_price_krw": k_ets_price,
            "differential_krw_per_tco2": cbam_differential,
            "cbam_cost_krw": cbam_cost_krw,
            "cbam_cost_eur": cbam_cost_krw / eur_to_krw,
        }

    # =========================================================================
    # 탄소크레딧 수익
    # =========================================================================
    def calculate_carbon_credit_revenue(self,
                                         avoided_tco2: float,
                                         credit_price_krw: Optional[float] = None) -> Dict:
        """
        탄소크레딧 수익 계산

        Args:
            avoided_tco2: 회피 배출량 (tCO₂)
            credit_price_krw: 크레딧 단가 (₩/tCO₂)

        Returns:
            탄소크레딧 수익
        """
        if credit_price_krw is None:
            credit_price_krw = self.config["carbon_credit_price_krw_per_tco2"]

        revenue = avoided_tco2 * credit_price_krw

        return {
            "avoided_tco2": avoided_tco2,
            "credit_price_krw_per_tco2": credit_price_krw,
            "total_revenue_krw": revenue,
            "total_revenue_billion_krw": revenue / 1e8,
        }

    # =========================================================================
    # 시계열 시뮬레이션
    # =========================================================================
    def simulate_annual_carbon(self,
                                grid_import_hourly_mwh: List[float],
                                pv_self_consumption_hourly_mwh: List[float]) -> pd.DataFrame:
        """
        연간 시간별 탄소 배출 시뮬레이션

        Args:
            grid_import_hourly_mwh: 시간별 그리드 구매량
            pv_self_consumption_hourly_mwh: 시간별 PV 자가소비량

        Returns:
            시간별 탄소 배출 DataFrame
        """
        records = []
        n = min(len(grid_import_hourly_mwh), len(pv_self_consumption_hourly_mwh))

        for h in range(n):
            grid_mwh = grid_import_hourly_mwh[h] if np.isfinite(grid_import_hourly_mwh[h]) else 0.0
            pv_mwh = pv_self_consumption_hourly_mwh[h] if np.isfinite(pv_self_consumption_hourly_mwh[h]) else 0.0

            record = self.calculate_hourly_emissions(
                grid_import_mwh=grid_mwh,
                pv_self_consumption_mwh=pv_mwh,
                hour=h,
            )
            records.append(record.to_dict())

        return pd.DataFrame(records)

    # =========================================================================
    # 통계
    # =========================================================================
    def get_emission_statistics(self) -> Dict:
        """배출 이력 통계"""
        if not self.emission_history:
            return {}

        total_scope1 = sum(r.scope1_tco2 for r in self.emission_history)
        total_scope2 = sum(r.scope2_tco2 for r in self.emission_history)
        total_scope3 = sum(r.scope3_tco2 for r in self.emission_history)
        total_avoided = sum(r.avoided_tco2 for r in self.emission_history)
        total_net = sum(r.net_tco2 for r in self.emission_history)

        return {
            "total_scope1_tco2": total_scope1,
            "total_scope2_tco2": total_scope2,
            "total_scope3_tco2": total_scope3,
            "total_emission_tco2": total_scope1 + total_scope2 + total_scope3,
            "total_avoided_tco2": total_avoided,
            "total_net_tco2": total_net,
            "hours_tracked": len(self.emission_history),
        }
