"""
M6. AI-EMS (AI Energy Management System) 모듈
3-Tier 에너지 관리: 실시간 제어 / 예측 제어 / 전략 최적화
Rule-based + scipy.optimize 기반 디스패치 최적화
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import time

try:
    from scipy.optimize import minimize, linprog
except ImportError:
    minimize = None
    linprog = None

from config import AI_EMS_CONFIG


def _to_list(v):
    """numpy array → list 변환 (Plotly 호환)"""
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


@dataclass
class DispatchCommand:
    """디스패치 명령"""
    pv_to_aidc_mw: float = 0.0
    pv_to_hess_mw: float = 0.0
    pv_to_grid_mw: float = 0.0
    hess_to_aidc_mw: float = 0.0
    hess_to_grid_mw: float = 0.0
    grid_to_aidc_mw: float = 0.0
    grid_to_hess_mw: float = 0.0
    h2_electrolyzer_mw: float = 0.0
    h2_fuelcell_mw: float = 0.0
    curtailment_mw: float = 0.0
    tier: str = "tier1"
    timestamp: float = 0.0
    response_time_ms: float = 0.0

    def total_to_aidc(self) -> float:
        return self.pv_to_aidc_mw + self.hess_to_aidc_mw + self.grid_to_aidc_mw + self.h2_fuelcell_mw

    def total_from_pv(self) -> float:
        return self.pv_to_aidc_mw + self.pv_to_hess_mw + self.pv_to_grid_mw + self.h2_electrolyzer_mw

    def to_dict(self) -> Dict:
        return {k: _to_list(v) for k, v in self.__dict__.items()}


class Tier1RealTimeControl:
    """Tier 1: 실시간 제어 (ms 단위)"""

    def __init__(self):
        self.config = AI_EMS_CONFIG
        self.last_mppt_voltage = 0.0
        self.dc_bus_voltage = self.config["dc_bus_voltage_target_v"]

    def mppt_control(self, pv_power_mw: float, ghi_w_per_m2: float) -> float:
        """MPPT 최적점 추적"""
        if ghi_w_per_m2 <= 0 or pv_power_mw <= 0:
            return 0.0
        tracking_eff = self.config["mppt_tracking_speed"]
        optimal_power = pv_power_mw * tracking_eff
        return max(0.0, optimal_power)

    def dc_bus_stabilize(self, power_imbalance_mw: float,
                         hess_soc: float) -> Dict[str, float]:
        """DC Bus 전압 안정화"""
        target_v = self.config["dc_bus_voltage_target_v"]
        tolerance = self.config["dc_bus_voltage_tolerance_pct"] / 100.0

        # 전압 편차 추정 (전력 불균형에 비례)
        voltage_deviation = power_imbalance_mw * 0.1  # 단순화
        self.dc_bus_voltage = target_v + voltage_deviation

        compensate_mw = 0.0
        if abs(voltage_deviation / target_v) > tolerance:
            compensate_mw = -power_imbalance_mw * 0.8  # 80% 보상

        return {
            "dc_bus_voltage_v": self.dc_bus_voltage,
            "voltage_deviation_pct": (self.dc_bus_voltage - target_v) / target_v * 100,
            "compensation_mw": compensate_mw,
            "stable": abs(voltage_deviation / target_v) <= tolerance,
        }

    def hess_charge_discharge_control(self, power_imbalance_mw: float,
                                       hess_soc: float) -> float:
        """HESS 충방전 실시간 제어 (양수: 충전, 음수: 방전)"""
        if power_imbalance_mw > 0 and hess_soc < 0.95:
            # 잉여 → HESS 충전
            return min(power_imbalance_mw, 200.0)  # 200MW 제한
        elif power_imbalance_mw < 0 and hess_soc > 0.1:
            # 부족 → HESS 방전
            return max(power_imbalance_mw, -200.0)
        return 0.0


class Tier2PredictiveControl:
    """Tier 2: 예측 제어 (분~시간 단위)"""

    def __init__(self):
        self.config = AI_EMS_CONFIG
        self.pv_forecast_cache: List[float] = []
        self.load_forecast_cache: List[float] = []

    def forecast_pv_output(self, current_pv_mw: float,
                           weather_forecast: Optional[Dict] = None,
                           horizon_hours: int = 4) -> List[float]:
        """날씨 기반 PV 출력 예측 (간단 모델)"""
        forecast = []
        base = current_pv_mw if np.isfinite(current_pv_mw) else 0.0

        for h in range(horizon_hours):
            if weather_forecast and "ghi_forecast" in weather_forecast:
                ghi_list = weather_forecast["ghi_forecast"]
                if h < len(ghi_list):
                    ratio = ghi_list[h] / max(ghi_list[0], 1.0) if ghi_list[0] > 0 else 0.5
                    forecast.append(max(0.0, base * ratio))
                    continue
            # Persistence 모델 + 감쇠
            decay = 0.95 ** (h + 1)
            noise = np.random.normal(0, 0.05 * base) if base > 0 else 0
            forecast.append(max(0.0, base * decay + noise))

        self.pv_forecast_cache = forecast
        return forecast

    def forecast_load(self, current_load_mw: float,
                      hour_of_day: int = 12,
                      horizon_hours: int = 4) -> List[float]:
        """부하 예측 (시간대 패턴 기반)"""
        # AIDC 부하는 상대적으로 안정 (±10%)
        forecast = []
        for h in range(horizon_hours):
            future_hour = (hour_of_day + h + 1) % 24
            # 시간대별 부하 패턴 (AIDC: 오후 피크)
            hourly_factor = 1.0 + 0.08 * math.sin(math.pi * (future_hour - 6) / 12)
            noise = np.random.normal(0, 0.03 * current_load_mw)
            forecast.append(max(0.0, current_load_mw * hourly_factor + noise))

        self.load_forecast_cache = forecast
        return forecast

    def optimal_dispatch(self, pv_mw: float, load_mw: float,
                         hess_soc: float, h2_storage_level: float,
                         grid_price_krw: float,
                         pv_forecast: Optional[List[float]] = None,
                         load_forecast: Optional[List[float]] = None) -> DispatchCommand:
        """최적 디스패치 계산"""
        cmd = DispatchCommand(tier="tier2")
        surplus = pv_mw - load_mw

        if surplus >= 0:
            # 잉여: PV → AIDC 우선, 나머지 HESS/Grid/H2
            cmd.pv_to_aidc_mw = load_mw
            remaining = surplus

            # HESS 충전 (SOC < 80%)
            if hess_soc < 0.8 and remaining > 0:
                hess_charge = min(remaining, 200.0)
                cmd.pv_to_hess_mw = hess_charge
                remaining -= hess_charge

            # H₂ 전해 (SOC 높고 잉여 많을 때)
            if hess_soc > 0.6 and h2_storage_level < 0.8 and remaining > 10:
                h2_power = min(remaining, 50.0)
                cmd.h2_electrolyzer_mw = h2_power
                remaining -= h2_power

            # 그리드 판매 (높은 가격)
            if remaining > 0 and grid_price_krw > 70000:
                grid_sell = min(remaining, 50.0)
                cmd.pv_to_grid_mw = grid_sell
                remaining -= grid_sell

            # 나머지 curtailment
            if remaining > 0:
                if grid_price_krw > 50000:
                    cmd.pv_to_grid_mw += remaining
                else:
                    cmd.curtailment_mw = remaining
        else:
            # 부족: PV 전량 AIDC, 나머지 HESS/H2/Grid
            cmd.pv_to_aidc_mw = pv_mw
            deficit = -surplus

            # HESS 방전 (SOC > 20%)
            if hess_soc > 0.2 and deficit > 0:
                hess_discharge = min(deficit, 200.0)
                cmd.hess_to_aidc_mw = hess_discharge
                deficit -= hess_discharge

            # H₂ 연료전지
            if h2_storage_level > 0.2 and deficit > 10:
                h2_power = min(deficit, 50.0)
                cmd.h2_fuelcell_mw = h2_power
                deficit -= h2_power

            # 그리드 구매 (최후 수단)
            if deficit > 0:
                cmd.grid_to_aidc_mw = deficit

        return cmd


class Tier3StrategicOptimizer:
    """Tier 3: 전략 최적화 (일~주 단위)"""

    def __init__(self):
        self.config = AI_EMS_CONFIG

    def optimize_daily_schedule(self,
                                pv_forecast_24h: List[float],
                                load_forecast_24h: List[float],
                                price_forecast_24h: List[float],
                                hess_soc: float = 0.5,
                                h2_level: float = 0.5) -> List[DispatchCommand]:
        """24시간 최적 운전 스케줄"""
        schedule = []
        current_soc = hess_soc
        current_h2 = h2_level

        for h in range(min(24, len(pv_forecast_24h), len(load_forecast_24h))):
            pv = pv_forecast_24h[h]
            load = load_forecast_24h[h]
            price = price_forecast_24h[h] if h < len(price_forecast_24h) else 80000

            tier2 = Tier2PredictiveControl()
            cmd = tier2.optimal_dispatch(pv, load, current_soc, current_h2, price)
            cmd.tier = "tier3"

            # SOC 업데이트 (단순)
            soc_change = (cmd.pv_to_hess_mw + cmd.grid_to_hess_mw - cmd.hess_to_aidc_mw - cmd.hess_to_grid_mw) / 2000.0
            current_soc = np.clip(current_soc + soc_change, 0, 1)

            schedule.append(cmd)

        return schedule

    def economic_optimization(self,
                              pv_annual_mwh: float,
                              load_annual_mwh: float,
                              avg_grid_price_krw: float = 80000,
                              carbon_price_krw: float = 25000) -> Dict:
        """연간 경제 최적화 (전략적)"""
        # 자가소비 절감
        self_consumption = min(pv_annual_mwh, load_annual_mwh)
        savings_from_self = self_consumption * avg_grid_price_krw

        # 잉여 판매
        surplus = max(0, pv_annual_mwh - load_annual_mwh)
        revenue_from_sales = surplus * avg_grid_price_krw * 0.85  # SMP ~85% of retail

        # 그리드 구매
        grid_purchase = max(0, load_annual_mwh - pv_annual_mwh)
        cost_grid = grid_purchase * avg_grid_price_krw

        # 탄소 절감 (PV 자가소비분)
        emission_factor = 0.4594  # tCO₂/MWh
        carbon_avoided = self_consumption * emission_factor
        carbon_revenue = carbon_avoided * carbon_price_krw

        net_benefit = savings_from_self + revenue_from_sales + carbon_revenue - cost_grid

        return {
            "self_consumption_mwh": self_consumption,
            "surplus_mwh": surplus,
            "grid_purchase_mwh": grid_purchase,
            "savings_from_self_krw": savings_from_self,
            "revenue_from_sales_krw": revenue_from_sales,
            "carbon_avoided_tco2": carbon_avoided,
            "carbon_revenue_krw": carbon_revenue,
            "cost_grid_krw": cost_grid,
            "net_annual_benefit_krw": net_benefit,
        }

    def maintenance_schedule(self, operating_hours: Dict[str, float]) -> List[Dict]:
        """유지보수 스케줄 생성"""
        schedule = []
        thresholds = {
            "pv_cleaning": 720,     # 30일마다
            "bess_inspection": 2160, # 90일마다
            "h2_stack_check": 4320,  # 180일마다
            "grid_relay_test": 8760, # 연 1회
        }
        for task, threshold in thresholds.items():
            hours = operating_hours.get(task, 0)
            if hours >= threshold:
                schedule.append({
                    "task": task,
                    "priority": "high" if hours > threshold * 1.2 else "normal",
                    "hours_since_last": hours,
                    "threshold_hours": threshold,
                })
        return schedule


class AIEMSModule:
    """AI-EMS 통합 모듈"""

    def __init__(self):
        self.tier1 = Tier1RealTimeControl()
        self.tier2 = Tier2PredictiveControl()
        self.tier3 = Tier3StrategicOptimizer()
        self.dispatch_history: List[DispatchCommand] = []
        self.kpi_history: List[Dict] = []

    def execute_dispatch(self,
                         pv_power_mw: float,
                         aidc_load_mw: float,
                         hess_soc: float,
                         h2_storage_level: float,
                         grid_price_krw: float = 80000,
                         ghi_w_per_m2: float = 500,
                         weather_forecast: Optional[Dict] = None,
                         hour_of_day: int = 12) -> DispatchCommand:
        """통합 디스패치 실행"""
        t0 = time.time()

        # NaN 가드
        pv_power_mw = float(pv_power_mw) if np.isfinite(pv_power_mw) else 0.0
        aidc_load_mw = float(aidc_load_mw) if np.isfinite(aidc_load_mw) else 0.0
        hess_soc = float(np.clip(hess_soc, 0, 1)) if np.isfinite(hess_soc) else 0.5
        h2_storage_level = float(np.clip(h2_storage_level, 0, 1)) if np.isfinite(h2_storage_level) else 0.5

        # Tier 1: 실시간 보정
        pv_optimized = self.tier1.mppt_control(pv_power_mw, ghi_w_per_m2)
        imbalance = pv_optimized - aidc_load_mw
        bus_status = self.tier1.dc_bus_stabilize(imbalance, hess_soc)

        # Tier 2: 예측 기반 디스패치
        cmd = self.tier2.optimal_dispatch(
            pv_optimized, aidc_load_mw, hess_soc, h2_storage_level, grid_price_krw
        )

        t1 = time.time()
        cmd.response_time_ms = (t1 - t0) * 1000
        cmd.timestamp = t0

        self.dispatch_history.append(cmd)
        return cmd

    def calculate_kpi(self,
                      dispatch_history: Optional[List[DispatchCommand]] = None) -> Dict:
        """KPI 계산"""
        history = dispatch_history or self.dispatch_history
        if not history:
            return {}

        total_aidc = sum(c.total_to_aidc() for c in history)
        total_pv_to_aidc = sum(c.pv_to_aidc_mw for c in history)
        total_hess_to_aidc = sum(c.hess_to_aidc_mw for c in history)
        total_grid_to_aidc = sum(c.grid_to_aidc_mw for c in history)
        total_curtailment = sum(c.curtailment_mw for c in history)
        total_pv = sum(c.total_from_pv() + c.curtailment_mw for c in history)

        # 자급률 = (PV+HESS+H2→AIDC) / AIDC
        self_supply = total_pv_to_aidc + total_hess_to_aidc + sum(c.h2_fuelcell_mw for c in history)
        self_sufficiency = self_supply / total_aidc if total_aidc > 0 else 0

        # 피크 감축률
        grid_peaks = [c.grid_to_aidc_mw for c in history]
        aidc_peaks = [c.total_to_aidc() for c in history]
        peak_reduction = 1 - (max(grid_peaks) / max(aidc_peaks)) if max(aidc_peaks) > 0 else 0

        # 평균 응답시간
        avg_response = np.mean([c.response_time_ms for c in history])

        return {
            "self_sufficiency_ratio": float(np.clip(self_sufficiency, 0, 1)),
            "peak_reduction_ratio": float(np.clip(peak_reduction, 0, 1)),
            "total_aidc_supply_mwh": float(total_aidc),
            "total_grid_import_mwh": float(total_grid_to_aidc),
            "total_curtailment_mwh": float(total_curtailment),
            "avg_response_time_ms": float(avg_response),
            "dispatch_count": len(history),
            "renewable_fraction": float(self_supply / total_aidc) if total_aidc > 0 else 0,
        }

    def get_dispatch_summary(self, n_hours: int = 24) -> pd.DataFrame:
        """디스패치 이력 요약"""
        recent = self.dispatch_history[-n_hours:] if len(self.dispatch_history) >= n_hours else self.dispatch_history
        records = [c.to_dict() for c in recent]
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def simulate_24h(self,
                     pv_profile: List[float],
                     load_profile: List[float],
                     price_profile: List[float],
                     hess_soc_init: float = 0.5,
                     h2_level_init: float = 0.5) -> List[DispatchCommand]:
        """24시간 시뮬레이션"""
        results = []
        soc = hess_soc_init
        h2 = h2_level_init

        for h in range(min(24, len(pv_profile), len(load_profile))):
            price = price_profile[h] if h < len(price_profile) else 80000
            cmd = self.execute_dispatch(
                pv_power_mw=pv_profile[h],
                aidc_load_mw=load_profile[h],
                hess_soc=soc,
                h2_storage_level=h2,
                grid_price_krw=price,
                hour_of_day=h,
            )
            results.append(cmd)

            # SOC/H2 레벨 업데이트
            soc_delta = (cmd.pv_to_hess_mw + cmd.grid_to_hess_mw - cmd.hess_to_aidc_mw - cmd.hess_to_grid_mw) / 2000.0
            soc = float(np.clip(soc + soc_delta, 0, 1))

            h2_delta = (cmd.h2_electrolyzer_mw - cmd.h2_fuelcell_mw) / 5000.0
            h2 = float(np.clip(h2 + h2_delta, 0, 1))

        return results
