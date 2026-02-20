"""
M8. Grid Interface 모듈
한전 계통 연계: 양방향 전력 거래, 계통 안정화 서비스, 보호 계전, 경제 급전
SMP, REC, K-ETS 가격 기반 최적 거래
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import datetime as dt

@dataclass  
class GridTariffConfig:
    """전력 요금 설정"""
    smp_base_krw_per_mwh: float  # 기준 SMP 가격
    rec_price_krw_per_mwh: float  # REC 가격
    rec_multiplier: float         # REC 가중치 (태양광 1.2배)
    carbon_price_krw_per_ton: float  # 탄소 가격 (K-ETS)
    grid_access_fee_krw_per_mw_month: float  # 계통 이용 요금
    transmission_loss_factor: float  # 송전 손실률

@dataclass
class ProtectionSettings:
    """보호 설정"""
    voltage_high_pu: float = 1.1   # 과전압 기준 (p.u.)
    voltage_low_pu: float = 0.9    # 저전압 기준
    frequency_high_hz: float = 50.5  # 과주파수 기준 (Hz)
    frequency_low_hz: float = 49.5   # 저주파수 기준
    power_factor_min: float = 0.95   # 최소 역률
    reconnect_delay_s: float = 300   # 재연결 지연 시간

class PowerFlowCalculator:
    """전력조류 계산기 (PCC 기준)"""
    
    def __init__(self, grid_voltage_kv: float = 22.9, base_mva: float = 100):
        self.grid_voltage_kv = grid_voltage_kv
        self.base_mva = base_mva
        self.base_impedance = (grid_voltage_kv**2) / base_mva
        
    def calculate_pcc_power_flow(self,
                               microgrid_power_mw: float,
                               microgrid_reactive_mvar: float,
                               grid_voltage_pu: float = 1.0,
                               grid_impedance_pu: float = 0.1) -> Dict:
        """
        PCC (Point of Common Coupling) 전력조류 계산
        
        Args:
            microgrid_power_mw: 마이크로그리드 유효전력 (양수: 소비, 음수: 공급)
            microgrid_reactive_mvar: 마이크로그리드 무효전력
            grid_voltage_pu: 계통 전압 (p.u.)
            grid_impedance_pu: 계통 임피던스 (p.u.)
            
        Returns:
            전력조류 해석 결과
        """
        # 기본값 설정
        if microgrid_power_mw == 0 and microgrid_reactive_mvar == 0:
            return {
                "pcc_voltage_pu": grid_voltage_pu,
                "power_flow_mw": 0,
                "reactive_flow_mvar": 0,
                "current_pu": 0,
                "power_factor": 1.0,
                "voltage_drop_pu": 0,
                "line_losses_mw": 0
            }
        
        # p.u. 변환
        p_pu = microgrid_power_mw / self.base_mva
        q_pu = microgrid_reactive_mvar / self.base_mva
        
        # 전류 크기 계산 |I| = |S|/|V|
        s_magnitude_pu = np.sqrt(p_pu**2 + q_pu**2)
        current_pu = s_magnitude_pu / grid_voltage_pu if grid_voltage_pu > 0 else 0
        
        # 전압 강하 계산 (단순화된 모델)
        voltage_drop_pu = current_pu * grid_impedance_pu
        pcc_voltage_pu = grid_voltage_pu - voltage_drop_pu
        
        # 역률 계산
        power_factor = abs(p_pu / s_magnitude_pu) if s_magnitude_pu > 0 else 1.0
        
        # 선로 손실 (I²R 손실)
        line_losses_mw = (current_pu**2) * grid_impedance_pu * self.base_mva * 0.1  # 단순화
        
        return {
            "pcc_voltage_pu": max(0, pcc_voltage_pu),
            "power_flow_mw": microgrid_power_mw,
            "reactive_flow_mvar": microgrid_reactive_mvar,
            "current_pu": current_pu,
            "power_factor": power_factor,
            "voltage_drop_pu": voltage_drop_pu,
            "line_losses_mw": line_losses_mw
        }

class ProtectionSystem:
    """보호 계전 시스템"""
    
    def __init__(self, settings: ProtectionSettings = None):
        self.settings = settings or ProtectionSettings()
        self.is_connected = True
        self.trip_history = []
        self.last_trip_time = None
        self.reconnect_attempts = 0
        
    def check_protection_limits(self,
                              voltage_pu: float,
                              frequency_hz: float, 
                              power_factor: float,
                              timestamp: dt.datetime = None) -> Dict:
        """보호 기준 확인"""
        if timestamp is None:
            timestamp = dt.datetime.now()
            
        violations = []
        trip_required = False
        
        # 전압 보호
        if voltage_pu > self.settings.voltage_high_pu:
            violations.append(f"과전압: {voltage_pu:.3f} p.u. > {self.settings.voltage_high_pu}")
            trip_required = True
        elif voltage_pu < self.settings.voltage_low_pu:
            violations.append(f"저전압: {voltage_pu:.3f} p.u. < {self.settings.voltage_low_pu}")  
            trip_required = True
            
        # 주파수 보호
        if frequency_hz > self.settings.frequency_high_hz:
            violations.append(f"과주파수: {frequency_hz:.2f} Hz > {self.settings.frequency_high_hz}")
            trip_required = True
        elif frequency_hz < self.settings.frequency_low_hz:
            violations.append(f"저주파수: {frequency_hz:.2f} Hz < {self.settings.frequency_low_hz}")
            trip_required = True
            
        # 역률 보호
        if power_factor < self.settings.power_factor_min:
            violations.append(f"역률 저하: {power_factor:.3f} < {self.settings.power_factor_min}")
            # 역률은 warning만, trip 하지 않음
            
        # 트립 실행
        if trip_required and self.is_connected:
            self.execute_trip(violations, timestamp)
            
        return {
            "connected": self.is_connected,
            "violations": violations,
            "trip_required": trip_required,
            "protection_status": "OK" if not violations else "VIOLATION"
        }
    
    def execute_trip(self, violations: List[str], timestamp: dt.datetime):
        """트립 실행"""
        self.is_connected = False
        self.last_trip_time = timestamp
        
        trip_record = {
            "timestamp": timestamp,
            "violations": violations,
            "reconnect_attempt": self.reconnect_attempts
        }
        self.trip_history.append(trip_record)
        
        print(f"⚠️  Grid Disconnected at {timestamp}: {'; '.join(violations)}")
    
    def attempt_reconnect(self, 
                         current_voltage_pu: float,
                         current_frequency_hz: float,
                         current_timestamp: dt.datetime = None) -> Dict:
        """재연결 시도"""
        if current_timestamp is None:
            current_timestamp = dt.datetime.now()
            
        if self.is_connected:
            return {"status": "already_connected", "connected": True}
            
        if self.last_trip_time is None:
            return {"status": "no_trip_history", "connected": False}
            
        # 지연 시간 확인
        time_since_trip = (current_timestamp - self.last_trip_time).total_seconds()
        if time_since_trip < self.settings.reconnect_delay_s:
            return {
                "status": "delay_not_met",
                "connected": False,
                "remaining_delay_s": self.settings.reconnect_delay_s - time_since_trip
            }
        
        # 현재 상태 확인
        current_check = self.check_protection_limits(
            current_voltage_pu, current_frequency_hz, 1.0, current_timestamp
        )
        
        if current_check["protection_status"] == "OK":
            self.is_connected = True
            self.reconnect_attempts += 1
            print(f"✅ Grid Reconnected at {current_timestamp} (Attempt #{self.reconnect_attempts})")
            return {"status": "reconnected", "connected": True}
        else:
            return {
                "status": "conditions_not_met", 
                "connected": False,
                "current_violations": current_check["violations"]
            }

class AncillaryServices:
    """보조서비스 (주파수 조정, 전압 조정)"""
    
    def __init__(self):
        self.frequency_response_enabled = True
        self.voltage_response_enabled = True
        self.services_revenue = 0.0  # 보조서비스 수익
        
    def calculate_frequency_response(self,
                                   frequency_hz: float,
                                   available_power_mw: float,
                                   droop_percent: float = 5.0) -> Dict:
        """
        주파수 응답 (FR: Frequency Response) 계산
        
        Args:
            frequency_hz: 현재 주파수
            available_power_mw: 사용 가능 전력 (양수: 방전 가능, 음수: 충전 가능)
            droop_percent: Droop 특성 (%)
            
        Returns:
            FR 응답 결과
        """
        if not self.frequency_response_enabled:
            return {"fr_power_mw": 0, "frequency_error_hz": 0, "service_active": False}
        
        nominal_frequency = 50.0  # Hz
        frequency_error = frequency_hz - nominal_frequency
        
        # Droop 제어: ΔP = -K × Δf
        # K = P_rated / (droop_percent/100 × f_nominal)
        droop_gain = available_power_mw / (droop_percent/100 * nominal_frequency)
        fr_power_mw = -droop_gain * frequency_error
        
        # 사용 가능 전력 범위 제한
        fr_power_mw = np.clip(fr_power_mw, -abs(available_power_mw), abs(available_power_mw))
        
        # 데드밴드 적용 (±0.02 Hz)
        if abs(frequency_error) < 0.02:
            fr_power_mw = 0
            
        return {
            "fr_power_mw": fr_power_mw,
            "frequency_error_hz": frequency_error,
            "droop_response": True if abs(fr_power_mw) > 0.1 else False,
            "service_active": True
        }
    
    def calculate_voltage_response(self,
                                 voltage_pu: float,
                                 available_reactive_mvar: float,
                                 droop_percent: float = 3.0) -> Dict:
        """
        전압 응답 (VR: Voltage Response) 계산
        
        Args:
            voltage_pu: 현재 전압 (p.u.)
            available_reactive_mvar: 사용 가능 무효전력
            droop_percent: 전압 Droop (%)
            
        Returns:
            VR 응답 결과
        """
        if not self.voltage_response_enabled:
            return {"vr_reactive_mvar": 0, "voltage_error_pu": 0, "service_active": False}
        
        nominal_voltage = 1.0  # p.u.
        voltage_error = voltage_pu - nominal_voltage
        
        # 무효전력 Droop 제어
        droop_gain = available_reactive_mvar / (droop_percent/100)
        vr_reactive_mvar = -droop_gain * voltage_error
        
        # 무효전력 범위 제한
        vr_reactive_mvar = np.clip(vr_reactive_mvar, -abs(available_reactive_mvar), abs(available_reactive_mvar))
        
        # 데드밴드 (±0.01 p.u.)
        if abs(voltage_error) < 0.01:
            vr_reactive_mvar = 0
            
        return {
            "vr_reactive_mvar": vr_reactive_mvar,
            "voltage_error_pu": voltage_error,
            "droop_response": True if abs(vr_reactive_mvar) > 0.1 else False,
            "service_active": True
        }

class EconomicDispatch:
    """경제적 급전 계산기"""
    
    def __init__(self, tariff_config: GridTariffConfig):
        self.tariff = tariff_config
        
    def get_hourly_smp(self, hour: int, season: str = "summer") -> float:
        """
        시간대별 SMP 가격 프로파일 (한국 전력시장 패턴)
        
        Args:
            hour: 시간 (0-23)
            season: 계절 ("summer", "winter", "spring", "autumn")
            
        Returns:
            SMP 가격 (₩/MWh)
        """
        base_price = self.tariff.smp_base_krw_per_mwh
        
        # 시간대별 가격 배율
        if season == "summer":
            # 여름: 오후 피크, 에어컨 수요
            hourly_multiplier = {
                0: 0.7, 1: 0.65, 2: 0.6, 3: 0.6, 4: 0.65, 5: 0.7,
                6: 0.8, 7: 0.9, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.3,
                12: 1.35, 13: 1.4, 14: 1.45, 15: 1.5, 16: 1.4, 17: 1.3,
                18: 1.2, 19: 1.1, 20: 1.0, 21: 0.9, 22: 0.8, 23: 0.75
            }
        elif season == "winter":
            # 겨울: 아침/저녁 이중 피크, 난방 수요
            hourly_multiplier = {
                0: 0.75, 1: 0.7, 2: 0.65, 3: 0.6, 4: 0.65, 5: 0.75,
                6: 0.9, 7: 1.1, 8: 1.3, 9: 1.35, 10: 1.2, 11: 1.15,
                12: 1.1, 13: 1.05, 14: 1.0, 15: 1.05, 16: 1.1, 17: 1.2,
                18: 1.4, 19: 1.45, 20: 1.3, 21: 1.1, 22: 0.95, 23: 0.8
            }
        else:  # spring, autumn
            # 봄/가을: 완만한 패턴
            hourly_multiplier = {
                0: 0.75, 1: 0.7, 2: 0.65, 3: 0.65, 4: 0.7, 5: 0.75,
                6: 0.85, 7: 0.95, 8: 1.05, 9: 1.15, 10: 1.2, 11: 1.25,
                12: 1.25, 13: 1.2, 14: 1.15, 15: 1.1, 16: 1.05, 17: 1.1,
                18: 1.15, 19: 1.2, 20: 1.1, 21: 1.0, 22: 0.9, 23: 0.8
            }
            
        return base_price * hourly_multiplier.get(hour, 1.0)
    
    def calculate_carbon_cost(self, power_mw: float, duration_hours: float = 1.0) -> float:
        """
        탄소 비용 계산 (K-ETS 연동)
        
        Args:
            power_mw: 전력 (양수: 그리드에서 구매, 음수: 그리드에 판매)
            duration_hours: 기간
            
        Returns:
            탄소 비용 (₩)
        """
        if power_mw <= 0:  # 판매 또는 중립
            return 0.0
            
        # 한국 계통 배출계수 (tCO₂/MWh)
        emission_factor = 0.4168
        
        # 탄소 배출량
        emissions_tco2 = power_mw * duration_hours * emission_factor
        
        # 탄소 비용
        carbon_cost = emissions_tco2 * self.tariff.carbon_price_krw_per_ton
        
        return carbon_cost
    
    def calculate_trading_revenue(self,
                                power_mw: float,
                                hour: int,
                                duration_hours: float = 1.0,
                                season: str = "summer",
                                is_renewable: bool = True) -> Dict:
        """
        전력 거래 수익 계산
        
        Args:
            power_mw: 거래 전력 (양수: 구매, 음수: 판매)
            hour: 시간대
            duration_hours: 거래 기간
            season: 계절
            is_renewable: 신재생에너지 여부 (REC 대상)
            
        Returns:
            거래 수익 분석
        """
        smp_price = self.get_hourly_smp(hour, season)
        energy_mwh = power_mw * duration_hours
        
        # 기본 전력 거래
        power_revenue = -energy_mwh * smp_price  # 음수: 판매 수익, 양수: 구매 비용
        
        # REC 수익 (신재생 판매시만)
        rec_revenue = 0.0
        if is_renewable and power_mw < 0:  # 신재생 판매
            rec_mwh = abs(energy_mwh) * self.tariff.rec_multiplier  # 태양광 1.2배
            rec_revenue = rec_mwh * self.tariff.rec_price_krw_per_mwh
            
        # 탄소 비용 (구매시만)
        carbon_cost = self.calculate_carbon_cost(power_mw, duration_hours)
        
        # 총 수익/비용
        total_revenue = power_revenue + rec_revenue - carbon_cost
        
        return {
            "power_mw": power_mw,
            "energy_mwh": energy_mwh,
            "smp_price_krw_per_mwh": smp_price,
            "power_revenue_krw": power_revenue,
            "rec_revenue_krw": rec_revenue,
            "carbon_cost_krw": carbon_cost,
            "total_revenue_krw": total_revenue,
            "unit_revenue_krw_per_mwh": total_revenue / abs(energy_mwh) if energy_mwh != 0 else 0
        }

class GridInterfaceModule:
    """그리드 인터페이스 통합 모듈"""
    
    def __init__(self,
                 connection_capacity_mw: float = 50.0,
                 tariff_config: GridTariffConfig = None,
                 protection_settings: ProtectionSettings = None):
        """
        그리드 인터페이스 초기화
        
        Args:
            connection_capacity_mw: 계통 연계 용량 (MW, 양방향)
            tariff_config: 요금 설정
            protection_settings: 보호 설정
        """
        self.connection_capacity_mw = connection_capacity_mw
        
        # 기본 요금 설정 (한국 기준)
        if tariff_config is None:
            tariff_config = GridTariffConfig(
                smp_base_krw_per_mwh=80000,    # 80,000 ₩/MWh
                rec_price_krw_per_mwh=25000,   # 25,000 ₩/MWh  
                rec_multiplier=1.2,            # 태양광 1.2배
                carbon_price_krw_per_ton=22500, # 22,500 ₩/tCO₂
                grid_access_fee_krw_per_mw_month=1000000,  # 100만 ₩/MW/month
                transmission_loss_factor=0.05   # 5% 송전손실
            )
        
        # 시스템 구성요소
        self.power_flow = PowerFlowCalculator()
        self.protection = ProtectionSystem(protection_settings)
        self.ancillary = AncillaryServices()
        self.dispatch = EconomicDispatch(tariff_config)
        
        # 운전 이력
        self.trading_history = []
        self.protection_events = []
        self.total_energy_imported_mwh = 0
        self.total_energy_exported_mwh = 0
        self.total_revenue_krw = 0
        
    def execute_grid_transaction(self,
                               requested_power_mw: float,
                               duration_hours: float = 1.0,
                               grid_voltage_pu: float = 1.0,
                               grid_frequency_hz: float = 50.0,
                               hour: int = 12,
                               season: str = "summer",
                               timestamp: dt.datetime = None) -> Dict:
        """
        그리드 거래 실행
        
        Args:
            requested_power_mw: 요청 전력 (양수: 그리드에서 구매, 음수: 그리드에 판매)
            duration_hours: 거래 기간
            grid_voltage_pu: 그리드 전압
            grid_frequency_hz: 그리드 주파수  
            hour: 시간대 (0-23)
            season: 계절
            timestamp: 타임스탬프
            
        Returns:
            거래 실행 결과
        """
        if timestamp is None:
            timestamp = dt.datetime.now()
            
        # 1. 보호 계전 확인
        protection_check = self.protection.check_protection_limits(
            grid_voltage_pu, grid_frequency_hz, 0.95, timestamp
        )
        
        if not protection_check["connected"]:
            return {
                "success": False,
                "reason": "grid_disconnected",
                "protection_status": protection_check,
                "power_delivered_mw": 0,
                "revenue_krw": 0
            }
        
        # 2. 용량 제한 확인
        actual_power_mw = np.clip(requested_power_mw, 
                                -self.connection_capacity_mw, 
                                self.connection_capacity_mw)
        
        if abs(actual_power_mw - requested_power_mw) > 0.1:
            capacity_limited = True
        else:
            capacity_limited = False
        
        # 3. 전력조류 계산
        # 무효전력 추정 (역률 0.95 가정)
        reactive_power_mvar = actual_power_mw * np.tan(np.arccos(0.95))
        
        power_flow_result = self.power_flow.calculate_pcc_power_flow(
            actual_power_mw, reactive_power_mvar, grid_voltage_pu
        )
        
        # 4. 송전 손실 적용
        if actual_power_mw > 0:  # 구매시
            transmission_losses_mw = actual_power_mw * self.dispatch.tariff.transmission_loss_factor
            net_power_mw = actual_power_mw - transmission_losses_mw
        else:  # 판매시
            net_power_mw = actual_power_mw
            transmission_losses_mw = abs(actual_power_mw) * self.dispatch.tariff.transmission_loss_factor
        
        # 5. 경제적 급전 계산
        revenue_result = self.dispatch.calculate_trading_revenue(
            net_power_mw, hour, duration_hours, season, is_renewable=True
        )
        
        # 6. 보조서비스 계산
        ancillary_result = {}
        if abs(actual_power_mw) < self.connection_capacity_mw * 0.8:  # 80% 이하 사용시 여유분으로 보조서비스
            available_capacity = self.connection_capacity_mw - abs(actual_power_mw)
            
            fr_result = self.ancillary.calculate_frequency_response(
                grid_frequency_hz, available_capacity
            )
            vr_result = self.ancillary.calculate_voltage_response(
                grid_voltage_pu, available_capacity * 0.3  # 무효전력 용량 추정
            )
            
            ancillary_result = {
                "frequency_response": fr_result,
                "voltage_response": vr_result,
                "ancillary_revenue_krw": (abs(fr_result["fr_power_mw"]) + abs(vr_result["vr_reactive_mvar"])) * 5000 * duration_hours  # 5,000₩/MW/h 가정
            }
        
        # 7. 이력 저장
        transaction_record = {
            "timestamp": timestamp,
            "requested_power_mw": requested_power_mw,
            "actual_power_mw": actual_power_mw,
            "net_power_mw": net_power_mw,
            "duration_hours": duration_hours,
            "revenue_krw": revenue_result["total_revenue_krw"],
            "smp_price": revenue_result["smp_price_krw_per_mwh"],
            "capacity_limited": capacity_limited,
            "transmission_losses_mw": transmission_losses_mw
        }
        self.trading_history.append(transaction_record)
        
        # 8. 통계 업데이트
        energy_mwh = net_power_mw * duration_hours
        if energy_mwh > 0:
            self.total_energy_imported_mwh += energy_mwh
        else:
            self.total_energy_exported_mwh += abs(energy_mwh)
        
        self.total_revenue_krw += revenue_result["total_revenue_krw"]
        if "ancillary_revenue_krw" in ancillary_result:
            self.total_revenue_krw += ancillary_result["ancillary_revenue_krw"]
        
        return {
            "success": True,
            "power_requested_mw": requested_power_mw,
            "power_delivered_mw": actual_power_mw,
            "net_power_after_losses_mw": net_power_mw,
            "duration_hours": duration_hours,
            "capacity_limited": capacity_limited,
            "transmission_losses_mw": transmission_losses_mw,
            "power_flow": power_flow_result,
            "revenue": revenue_result,
            "ancillary_services": ancillary_result,
            "protection_status": protection_check,
            "grid_conditions": {
                "voltage_pu": grid_voltage_pu,
                "frequency_hz": grid_frequency_hz,
                "power_factor": power_flow_result["power_factor"]
            }
        }
    
    def optimize_hourly_dispatch(self,
                               available_power_schedule: List[float],  # 24시간 사용 가능 전력
                               season: str = "summer") -> List[Dict]:
        """
        24시간 최적 급전 계획
        
        Args:
            available_power_schedule: 시간별 사용 가능 전력 (MW) [0-23시]
            season: 계절
            
        Returns:
            시간별 최적 거래 계획
        """
        optimal_schedule = []
        
        for hour in range(24):
            available_power = available_power_schedule[hour]
            smp_price = self.dispatch.get_hourly_smp(hour, season)
            
            # 간단한 최적화: 높은 가격대에 판매, 낮은 가격대에 구매 최소화
            price_threshold = self.dispatch.tariff.smp_base_krw_per_mwh  # 기준 가격
            
            if available_power < 0:  # 잉여 전력 있음
                # 판매 - 가격이 높을 때 더 많이 판매
                price_factor = smp_price / price_threshold
                optimal_power = available_power * min(1.0, price_factor)
            elif available_power > 0:  # 전력 부족
                # 구매 - 가격이 낮을 때만 구매, 높을 때는 최소화
                price_factor = price_threshold / smp_price
                optimal_power = available_power * min(1.0, price_factor)
            else:
                optimal_power = 0
            
            # 용량 제한
            optimal_power = np.clip(optimal_power, -self.connection_capacity_mw, self.connection_capacity_mw)
            
            # 수익 계산
            revenue_calc = self.dispatch.calculate_trading_revenue(
                optimal_power, hour, 1.0, season, True
            )
            
            optimal_schedule.append({
                "hour": hour,
                "available_power_mw": available_power,
                "optimal_power_mw": optimal_power,
                "smp_price_krw_per_mwh": smp_price,
                "expected_revenue_krw": revenue_calc["total_revenue_krw"],
                "utilization": abs(optimal_power) / self.connection_capacity_mw
            })
        
        return optimal_schedule
    
    def get_trading_statistics(self) -> Dict:
        """거래 통계 조회"""
        if not self.trading_history:
            return {"error": "No trading history available"}
        
        df = pd.DataFrame(self.trading_history)
        
        # 기본 통계
        total_transactions = len(df)
        avg_power_mw = df["actual_power_mw"].mean()
        total_energy_mwh = df["net_power_mw"].sum()  # duration 1시간 가정
        avg_smp_price = df["smp_price"].mean()
        
        # 수익 통계
        total_revenue = df["revenue_krw"].sum()
        avg_revenue_per_mwh = total_revenue / abs(total_energy_mwh) if total_energy_mwh != 0 else 0
        
        # 방향별 통계
        exports = df[df["net_power_mw"] < 0]
        imports = df[df["net_power_mw"] > 0]
        
        return {
            "total_transactions": total_transactions,
            "total_energy_imported_mwh": self.total_energy_imported_mwh,
            "total_energy_exported_mwh": self.total_energy_exported_mwh,
            "net_energy_balance_mwh": self.total_energy_imported_mwh - self.total_energy_exported_mwh,
            "total_revenue_krw": self.total_revenue_krw,
            "average_smp_price_krw_per_mwh": avg_smp_price,
            "average_revenue_per_mwh": avg_revenue_per_mwh,
            "export_transactions": len(exports),
            "import_transactions": len(imports),
            "capacity_utilization_avg": abs(avg_power_mw) / self.connection_capacity_mw,
            "transmission_losses_total_mw": df["transmission_losses_mw"].sum(),
            "protection_events": len(self.protection.trip_history)
        }
    
    def simulate_daily_operation(self,
                               power_profile_mw: List[float],  # 24시간 전력 프로파일
                               season: str = "summer",
                               base_voltage_pu: float = 1.0,
                               base_frequency_hz: float = 50.0) -> pd.DataFrame:
        """
        일일 운전 시뮬레이션
        
        Args:
            power_profile_mw: 시간별 전력 프로파일 (24시간)
            season: 계절
            base_voltage_pu: 기준 전압
            base_frequency_hz: 기준 주파수
            
        Returns:
            시간별 운전 결과 DataFrame
        """
        results = []
        
        for hour, power_mw in enumerate(power_profile_mw):
            # 전압/주파수 변동 시뮬레이션 (±2% 랜덤)
            voltage_variation = np.random.normal(0, 0.01)
            frequency_variation = np.random.normal(0, 0.1)
            
            grid_voltage = base_voltage_pu + voltage_variation
            grid_frequency = base_frequency_hz + frequency_variation
            
            # 거래 실행
            transaction_result = self.execute_grid_transaction(
                requested_power_mw=power_mw,
                duration_hours=1.0,
                grid_voltage_pu=grid_voltage,
                grid_frequency_hz=grid_frequency,
                hour=hour,
                season=season
            )
            
            # 결과 저장
            result_record = {
                "hour": hour,
                "requested_power_mw": power_mw,
                "actual_power_mw": transaction_result.get("power_delivered_mw", 0),
                "revenue_krw": transaction_result["revenue"]["total_revenue_krw"] if "revenue" in transaction_result else 0,
                "smp_price": transaction_result["revenue"]["smp_price_krw_per_mwh"] if "revenue" in transaction_result else 0,
                "grid_voltage_pu": grid_voltage,
                "grid_frequency_hz": grid_frequency,
                "connected": transaction_result.get("success", False),
                "capacity_limited": transaction_result.get("capacity_limited", False)
            }
            
            results.append(result_record)
        
        return pd.DataFrame(results)


# 테스트 코드
if __name__ == "__main__":
    # 그리드 인터페이스 생성
    grid = GridInterfaceModule(
        connection_capacity_mw=50,  # 50MW 연계 용량
    )
    
    print("🔌 Grid Interface Initialized")
    print(f"Connection Capacity: {grid.connection_capacity_mw} MW")
    
    # 단일 거래 테스트
    print("\n💱 Single Transaction Test")
    
    # 잉여 전력 판매 (오후 피크시간)
    export_result = grid.execute_grid_transaction(
        requested_power_mw=-30,  # 30MW 판매
        hour=14,  # 14시 (피크시간)
        season="summer"
    )
    
    print(f"Export Result: {export_result['success']}")
    if export_result["success"]:
        print(f"  Power Exported: {abs(export_result['power_delivered_mw']):.1f} MW")
        print(f"  Revenue: {export_result['revenue']['total_revenue_krw']:,.0f} ₩")
        print(f"  SMP Price: {export_result['revenue']['smp_price_krw_per_mwh']:,.0f} ₩/MWh")
        print(f"  REC Revenue: {export_result['revenue']['rec_revenue_krw']:,.0f} ₩")
    
    # 부족 전력 구매 (심야시간)
    import_result = grid.execute_grid_transaction(
        requested_power_mw=20,   # 20MW 구매
        hour=2,   # 2시 (심야)
        season="summer"
    )
    
    print(f"\nImport Result: {import_result['success']}")
    if import_result["success"]:
        print(f"  Power Imported: {import_result['power_delivered_mw']:.1f} MW")
        print(f"  Cost: {abs(import_result['revenue']['total_revenue_krw']):,.0f} ₩")
        print(f"  SMP Price: {import_result['revenue']['smp_price_krw_per_mwh']:,.0f} ₩/MWh")
        print(f"  Carbon Cost: {import_result['revenue']['carbon_cost_krw']:,.0f} ₩")
    
    # 24시간 최적 급전 계획
    print("\n📊 24-Hour Optimal Dispatch")
    
    # 가상의 사용 가능 전력 (잉여: 음수, 부족: 양수)
    available_schedule = [
        10, 8, 5, 3, 5, 8,           # 0-5시: 부족 (야간 부하)
        15, 20, 10, -5, -15, -25,    # 6-11시: 오전 PV 증가
        -35, -40, -35, -25, -15, -5, # 12-17시: PV 피크, 잉여 많음
        5, 15, 20, 15, 12, 10        # 18-23시: 저녁 부하 증가
    ]
    
    optimal_plan = grid.optimize_hourly_dispatch(available_schedule, "summer")
    
    total_optimal_revenue = sum(plan["expected_revenue_krw"] for plan in optimal_plan)
    peak_export_hour = max(optimal_plan, key=lambda x: abs(x["optimal_power_mw"]) if x["optimal_power_mw"] < 0 else 0)
    
    print(f"Total Expected Daily Revenue: {total_optimal_revenue:,.0f} ₩")
    print(f"Peak Export: {abs(peak_export_hour['optimal_power_mw']):.1f} MW at {peak_export_hour['hour']:02d}:00")
    print(f"Peak SMP Price: {max(plan['smp_price_krw_per_mwh'] for plan in optimal_plan):,.0f} ₩/MWh")
    
    # 거래 통계
    print("\n📈 Trading Statistics")
    stats = grid.get_trading_statistics()
    print(f"Total Transactions: {stats['total_transactions']}")
    print(f"Net Energy Balance: {stats['net_energy_balance_mwh']:+.1f} MWh")
    print(f"Total Revenue: {stats['total_revenue_krw']:,.0f} ₩")
    print(f"Average SMP Price: {stats['average_smp_price_krw_per_mwh']:,.0f} ₩/MWh")