"""
M5. H₂ System (Power-to-Gas-to-Power) 모듈
SOEC 수전해 + H₂ 저장 + SOFC 연료전지
고온 운전 (700-900°C) 및 CHP 모드 지원
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import math
from dataclasses import dataclass

@dataclass
class H2ComponentConfig:
    """H₂ 시스템 구성요소 설정"""
    name: str
    rated_power_kw: float
    efficiency_nominal: float
    operating_temp_celsius: float
    startup_time_min: float
    min_load_ratio: float
    max_load_ratio: float
    degradation_rate_per_1000h: float
    thermal_mass_kwh_per_k: float  # 열질량 (kWh/K)

class SOECElectrolyzer:
    """SOEC (Solid Oxide Electrolyzer) 수전해기"""
    
    def __init__(self, config: H2ComponentConfig):
        self.config = config
        self.current_temp = 25.0  # 초기 온도 (°C)
        self.is_online = False
        self.current_load_ratio = 0.0
        self.stack_voltage = 0.0
        self.operating_hours = 0.0
        self.thermal_cycles = 0
        self.degradation_factor = 1.0
        
        # 물리 상수
        self.faraday_constant = 96485.33212  # C/mol
        self.gas_constant = 8.314462618  # J/(mol·K)
        self.h2_hhv_kwh_per_kg = 39.39  # kWh/kg (Higher Heating Value)
        self.h2_lhv_kwh_per_kg = 33.33  # kWh/kg (Lower Heating Value)
        
    def calculate_nernst_voltage(self, temp_k: float, pressure_bar: float = 1.0) -> float:
        """Nernst 전압 계산"""
        # H₂O → H₂ + 1/2 O₂
        # E₀ = 1.229 V @ 25°C, 1 bar
        e0 = 1.229  # V
        
        # 온도 의존성
        delta_s = 0.0001334  # kJ/(mol·K) - 표준 엔트로피 변화
        delta_h = 285.83     # kJ/mol - 표준 엔탈피 변화
        
        e_temp = e0 + (delta_s / (2 * self.faraday_constant * 1000)) * (temp_k - 298.15)
        
        # 압력 의존성 (단순화)
        e_pressure = e_temp + (self.gas_constant * temp_k) / (2 * self.faraday_constant) * math.log(pressure_bar)
        
        return max(1.0, e_pressure)  # 최소 1V
    
    def calculate_efficiency(self, 
                           load_ratio: float, 
                           temp_celsius: float) -> Tuple[float, float]:
        """
        SOEC 효율 계산
        
        Returns:
            (electrical_efficiency, thermal_efficiency)
        """
        temp_k = temp_celsius + 273.15
        
        # Faraday 효율 (전류밀도 의존)
        current_density_ma_per_cm2 = load_ratio * 500  # 가정: 최대 500 mA/cm²
        faraday_eff = 1.0 - 0.05 * (current_density_ma_per_cm2 / 500)**2  # 고부하에서 효율 저하
        
        # 전압 효율
        nernst_v = self.calculate_nernst_voltage(temp_k)
        actual_v = nernst_v * (1.2 + 0.3 * load_ratio)  # 과전압 포함
        voltage_eff = nernst_v / actual_v
        
        # 열역학적 효율 (고온 운전 장점)
        thermal_factor = 1.0 + 0.0008 * (temp_celsius - 25)  # 고온에서 효율 향상
        thermal_factor = min(1.15, thermal_factor)  # 최대 15% 향상
        
        electrical_eff = faraday_eff * voltage_eff * thermal_factor * self.degradation_factor
        
        # 폐열 효율 (CHP 모드)
        waste_heat_ratio = (1 - electrical_eff) * 0.8  # 80%의 폐열 회수 가능
        
        return electrical_eff, waste_heat_ratio
    
    def startup_procedure(self, target_temp: float = 800.0, ambient_temp: float = 25.0) -> Dict:
        """시작 절차 (예열)"""
        if self.is_online:
            return {"status": "already_online", "startup_time_min": 0}
        
        # 예열 에너지 계산
        temp_rise = target_temp - ambient_temp
        heating_energy_kwh = self.config.thermal_mass_kwh_per_k * temp_rise
        
        # 시작 시간 (온도 상승률 기준)
        heating_rate_k_per_min = 5  # 5K/min 가정
        startup_time_min = temp_rise / heating_rate_k_per_min
        
        self.current_temp = target_temp
        self.is_online = True
        self.thermal_cycles += 1
        
        return {
            "status": "startup_complete",
            "startup_time_min": startup_time_min,
            "heating_energy_kwh": heating_energy_kwh,
            "final_temp": target_temp
        }
    
    def operate(self, 
                power_input_kw: float, 
                duration_hours: float = 1.0,
                water_temp_celsius: float = 25.0) -> Dict:
        """
        SOEC 운전
        
        Args:
            power_input_kw: 입력 전력 (kW)
            duration_hours: 운전 시간 (시간)
            water_temp_celsius: 급수 온도
            
        Returns:
            운전 결과
        """
        if not self.is_online:
            startup_result = self.startup_procedure()
            if startup_result["status"] != "startup_complete":
                return {"error": "startup_failed", "details": startup_result}
        
        # 부하율 계산
        max_power = self.config.rated_power_kw * self.degradation_factor
        load_ratio = min(self.config.max_load_ratio, 
                        max(self.config.min_load_ratio, power_input_kw / max_power))
        
        actual_power_kw = load_ratio * max_power
        
        # 효율 계산
        electrical_eff, thermal_eff = self.calculate_efficiency(load_ratio, self.current_temp)
        
        # H₂ 생산량 계산
        h2_energy_kwh = actual_power_kw * duration_hours * electrical_eff
        h2_production_kg = h2_energy_kwh / self.h2_hhv_kwh_per_kg
        
        # 폐열 생산량
        waste_heat_kwh = actual_power_kw * duration_hours * thermal_eff
        
        # 물 소비량
        water_consumption_kg = h2_production_kg * 9  # H₂O molecular weight ratio
        
        # 운전 시간 누적
        self.operating_hours += duration_hours
        self.current_load_ratio = load_ratio
        
        # 열화 업데이트
        self._update_degradation()
        
        return {
            "power_input_kw": actual_power_kw,
            "h2_production_kg": h2_production_kg,
            "h2_energy_content_kwh": h2_energy_kwh,
            "electrical_efficiency": electrical_eff,
            "waste_heat_kwh": waste_heat_kwh,
            "total_efficiency_chp": electrical_eff + thermal_eff,
            "water_consumption_kg": water_consumption_kg,
            "operating_temp": self.current_temp,
            "load_ratio": load_ratio,
            "degradation_factor": self.degradation_factor
        }
    
    def _update_degradation(self):
        """스택 열화 업데이트"""
        # 운전시간 기반 선형 열화
        time_degradation = 1 - (self.operating_hours / 1000) * self.config.degradation_rate_per_1000h
        
        # 열사이클 기반 열화
        cycle_degradation = 1 - self.thermal_cycles * 0.001  # 1000 사이클당 0.1% 열화
        
        self.degradation_factor = max(0.7, min(time_degradation, cycle_degradation))

class SOFCFuelCell:
    """SOFC (Solid Oxide Fuel Cell) 연료전지"""
    
    def __init__(self, config: H2ComponentConfig):
        self.config = config
        self.current_temp = 25.0
        self.is_online = False
        self.current_load_ratio = 0.0
        self.operating_hours = 0.0
        self.thermal_cycles = 0
        self.degradation_factor = 1.0
        
        # 물리 상수
        self.faraday_constant = 96485.33212  # C/mol
        self.gas_constant = 8.314462618  # J/(mol·K)
        self.h2_lhv_kwh_per_kg = 33.33  # kWh/kg (Lower Heating Value for fuel cells)
    
    def calculate_theoretical_voltage(self, temp_k: float) -> float:
        """이론적 전압 계산 (Nernst equation)"""
        # H₂ + 1/2 O₂ → H₂O
        e0 = 1.229  # V @ 25°C
        
        # 온도 계수 (고온에서 전압 감소)
        temp_coeff = -0.00085  # V/K
        e_temp = e0 + temp_coeff * (temp_k - 298.15)
        
        return max(0.8, e_temp)  # 최소 0.8V
    
    def calculate_efficiency(self, 
                           load_ratio: float, 
                           temp_celsius: float) -> Tuple[float, float]:
        """
        SOFC 효율 계산
        
        Returns:
            (electrical_efficiency, thermal_efficiency)
        """
        temp_k = temp_celsius + 273.15
        
        # 전압 효율
        theoretical_v = self.calculate_theoretical_voltage(temp_k)
        
        # 실제 전압 (부하에 따른 손실)
        voltage_drop = 0.1 + 0.2 * load_ratio  # 농도 분극 + 저항 손실
        actual_v = theoretical_v - voltage_drop
        
        voltage_eff = actual_v / theoretical_v
        
        # 연료 이용률
        fuel_utilization = 0.8 - 0.1 * (1 - load_ratio)  # 저부하에서 이용률 저하
        
        # 전기 효율
        electrical_eff = voltage_eff * fuel_utilization * self.degradation_factor
        
        # 고온 운전 장점 (열회수)
        thermal_eff = (1 - electrical_eff) * 0.85  # 85% 폐열 회수
        
        return electrical_eff, thermal_eff
    
    def operate(self, 
                h2_input_kg: float,
                duration_hours: float = 1.0,
                target_power_kw: Optional[float] = None) -> Dict:
        """
        SOFC 운전
        
        Args:
            h2_input_kg: H₂ 투입량 (kg)
            duration_hours: 운전 시간
            target_power_kw: 목표 출력 (None이면 최대 출력)
            
        Returns:
            운전 결과
        """
        if not self.is_online:
            startup_result = self.startup_procedure()
            if startup_result["status"] != "startup_complete":
                return {"error": "startup_failed", "details": startup_result}
        
        # H₂ 에너지 함량
        h2_energy_kwh = h2_input_kg * self.h2_lhv_kwh_per_kg
        theoretical_max_power_kw = h2_energy_kwh / duration_hours
        
        # 목표 출력 설정 (H₂ 에너지 제한 고려)
        max_power_kw = self.config.rated_power_kw * self.degradation_factor
        
        if target_power_kw is None:
            # H₂ 기준 최대 출력
            target_power_kw = min(theoretical_max_power_kw, max_power_kw)
        else:
            # 요청 출력을 H₂ 에너지 및 설비 용량으로 제한
            target_power_kw = min(target_power_kw, theoretical_max_power_kw, max_power_kw)
        
        # 부하율 계산
        load_ratio = min(self.config.max_load_ratio,
                        max(self.config.min_load_ratio, target_power_kw / max_power_kw))
        
        # 효율 계산 (부하율 기준)
        electrical_eff, thermal_eff = self.calculate_efficiency(load_ratio, self.current_temp)
        
        # 실제 H₂ 기준 출력 계산 (에너지 보존 법칙)
        max_electrical_power_from_h2 = h2_energy_kwh * electrical_eff / duration_hours
        actual_power_kw = min(target_power_kw, max_electrical_power_from_h2)
        
        # 실제 H₂ 소비량 (에너지 보존)
        h2_consumed_kg = actual_power_kw * duration_hours / (self.h2_lhv_kwh_per_kg * electrical_eff)
        
        # 폐열 생산량
        waste_heat_kwh = h2_consumed_kg * self.h2_lhv_kwh_per_kg * thermal_eff
        
        # 물 생성량
        water_produced_kg = h2_consumed_kg * 9  # H₂O molecular weight ratio
        
        # 운전 시간 누적
        self.operating_hours += duration_hours
        self.current_load_ratio = load_ratio
        
        # 열화 업데이트  
        self._update_degradation()
        
        return {
            "electrical_power_kw": actual_power_kw,
            "thermal_power_kw": waste_heat_kwh / duration_hours,
            "h2_consumed_kg": h2_consumed_kg,
            "h2_remaining_kg": h2_input_kg - h2_consumed_kg,
            "electrical_efficiency": electrical_eff,
            "thermal_efficiency": thermal_eff,
            "total_efficiency_chp": electrical_eff + thermal_eff,
            "water_produced_kg": water_produced_kg,
            "operating_temp": self.current_temp,
            "load_ratio": load_ratio,
            "degradation_factor": self.degradation_factor
        }
    
    def startup_procedure(self, target_temp: float = 800.0) -> Dict:
        """시작 절차"""
        if self.is_online:
            return {"status": "already_online", "startup_time_min": 0}
        
        # SOFC 시작 시간은 SOEC보다 짧음 (열부하 낮음)
        startup_time_min = self.config.startup_time_min
        
        self.current_temp = target_temp
        self.is_online = True
        self.thermal_cycles += 1
        
        return {
            "status": "startup_complete", 
            "startup_time_min": startup_time_min,
            "final_temp": target_temp
        }
    
    def _update_degradation(self):
        """스택 열화 업데이트"""
        time_degradation = 1 - (self.operating_hours / 1000) * self.config.degradation_rate_per_1000h
        cycle_degradation = 1 - self.thermal_cycles * 0.0005  # SOFC는 열화가 더 적음
        
        self.degradation_factor = max(0.7, min(time_degradation, cycle_degradation))

class H2StorageSystem:
    """H₂ 저장 시스템 (압축 또는 금속수소화물)"""
    
    def __init__(self, 
                 capacity_kg: float = 10000,
                 storage_type: str = "compressed",  # "compressed" or "metal_hydride"
                 pressure_bar: float = 350):
        self.capacity_kg = capacity_kg
        self.storage_type = storage_type
        self.pressure_bar = pressure_bar
        self.current_inventory_kg = capacity_kg * 0.5  # 초기 50% 저장
        self.temperature = 25.0  # °C
        
        # 저장 방식별 파라미터
        if storage_type == "compressed":
            self.storage_efficiency = 0.95  # 압축 손실
            self.leakage_rate_per_day = 0.001  # 0.1% per day
            self.energy_for_compression_kwh_per_kg = 3.0  # 압축 에너지
        else:  # metal_hydride
            self.storage_efficiency = 0.98
            self.leakage_rate_per_day = 0.0001  # 거의 누출 없음
            self.energy_for_compression_kwh_per_kg = 1.0  # 흡장/방출 에너지
    
    def store_h2(self, h2_kg: float) -> Dict:
        """H₂ 저장"""
        available_space = self.capacity_kg - self.current_inventory_kg
        actual_stored = min(h2_kg, available_space) * self.storage_efficiency
        
        compression_energy = actual_stored * self.energy_for_compression_kwh_per_kg
        self.current_inventory_kg += actual_stored
        
        return {
            "requested_kg": h2_kg,
            "stored_kg": actual_stored,
            "compression_energy_kwh": compression_energy,
            "storage_level": self.current_inventory_kg / self.capacity_kg,
            "remaining_capacity_kg": self.capacity_kg - self.current_inventory_kg
        }
    
    def retrieve_h2(self, h2_kg: float) -> Dict:
        """H₂ 인출"""
        actual_retrieved = min(h2_kg, self.current_inventory_kg)
        retrieval_energy = actual_retrieved * self.energy_for_compression_kwh_per_kg * 0.1  # 인출 에너지는 적음
        
        self.current_inventory_kg -= actual_retrieved
        
        return {
            "requested_kg": h2_kg,
            "retrieved_kg": actual_retrieved,
            "retrieval_energy_kwh": retrieval_energy,
            "storage_level": self.current_inventory_kg / self.capacity_kg,
            "remaining_inventory_kg": self.current_inventory_kg
        }
    
    def apply_leakage(self, duration_days: float) -> float:
        """누출 적용"""
        leakage_kg = self.current_inventory_kg * self.leakage_rate_per_day * duration_days
        self.current_inventory_kg = max(0, self.current_inventory_kg - leakage_kg)
        return leakage_kg

class H2SystemModule:
    """H₂ 시스템 통합 모듈"""
    
    def __init__(self,
                 soec_power_kw: float = 50000,  # 50 MW SOEC
                 sofc_power_kw: float = 50000,  # 50 MW SOFC  
                 storage_capacity_kg: float = 150000,  # 150 ton H₂ storage
                 storage_type: str = "compressed"):
        """
        H₂ 시스템 초기화
        
        Args:
            soec_power_kw: SOEC 수전해 용량 (kW)
            sofc_power_kw: SOFC 연료전지 용량 (kW)  
            storage_capacity_kg: H₂ 저장 용량 (kg)
            storage_type: 저장 방식 ("compressed" or "metal_hydride")
        """
        # SOEC 초기화
        soec_config = H2ComponentConfig(
            name="SOEC",
            rated_power_kw=soec_power_kw,
            efficiency_nominal=0.85,
            operating_temp_celsius=800,
            startup_time_min=120,  # 2시간 예열
            min_load_ratio=0.1,
            max_load_ratio=1.0,
            degradation_rate_per_1000h=0.5,  # 0.5%/1000h
            thermal_mass_kwh_per_k=50  # 큰 열질량
        )
        self.soec = SOECElectrolyzer(soec_config)
        
        # SOFC 초기화
        sofc_config = H2ComponentConfig(
            name="SOFC",
            rated_power_kw=sofc_power_kw,
            efficiency_nominal=0.60,
            operating_temp_celsius=800,
            startup_time_min=60,  # 1시간 예열
            min_load_ratio=0.1,
            max_load_ratio=1.0,
            degradation_rate_per_1000h=0.3,  # 0.3%/1000h
            thermal_mass_kwh_per_k=30
        )
        self.sofc = SOFCFuelCell(sofc_config)
        
        # H₂ 저장소 초기화
        self.storage = H2StorageSystem(
            capacity_kg=storage_capacity_kg,
            storage_type=storage_type
        )
        
        # 시스템 상태
        self.total_h2_produced_kg = 0
        self.total_h2_consumed_kg = 0
        self.total_electrical_energy_in_kwh = 0
        self.total_electrical_energy_out_kwh = 0
        self.total_thermal_energy_kwh = 0
    
    def power_to_gas(self, 
                     electrical_power_kw: float,
                     duration_hours: float = 1.0) -> Dict:
        """
        Power-to-Gas 운전 (전기 → H₂)
        
        Args:
            electrical_power_kw: 입력 전력 (kW)
            duration_hours: 운전 시간 (시간)
            
        Returns:
            P2G 운전 결과
        """
        if electrical_power_kw <= 0:
            return {"error": "Invalid power input", "power": electrical_power_kw}
        
        # SOEC 운전
        soec_result = self.soec.operate(electrical_power_kw, duration_hours)
        
        if "error" in soec_result:
            return soec_result
        
        # H₂ 저장
        storage_result = self.storage.store_h2(soec_result["h2_production_kg"])
        
        # 통계 업데이트
        self.total_h2_produced_kg += storage_result["stored_kg"]
        self.total_electrical_energy_in_kwh += soec_result["power_input_kw"] * duration_hours
        # 열에너지는 P2G에서는 추가하지 않음 (G2P에서만 유용한 열)
        
        return {
            "operation_mode": "power_to_gas",
            "electrical_input_kw": soec_result["power_input_kw"],
            "h2_produced_kg": soec_result["h2_production_kg"],
            "h2_stored_kg": storage_result["stored_kg"],
            "electrical_efficiency": soec_result["electrical_efficiency"],
            "waste_heat_kwh": soec_result["waste_heat_kwh"],
            "total_efficiency_chp": soec_result["total_efficiency_chp"],
            "storage_level": storage_result["storage_level"],
            "compression_energy_kwh": storage_result["compression_energy_kwh"],
            "net_electrical_consumption_kw": soec_result["power_input_kw"] + 
                                           storage_result["compression_energy_kwh"] / duration_hours
        }
    
    def gas_to_power(self, 
                     target_power_kw: float,
                     duration_hours: float = 1.0) -> Dict:
        """
        Gas-to-Power 운전 (H₂ → 전기)
        
        Args:
            target_power_kw: 목표 출력 (kW)
            duration_hours: 운전 시간 (시간)
            
        Returns:
            G2P 운전 결과
        """
        if target_power_kw <= 0:
            return {"error": "Invalid power target", "power": target_power_kw}
        
        # 최대 사용 가능한 H₂ 양 확인
        max_available_h2 = self.storage.current_inventory_kg
        
        # 요청된 전력으로부터 필요한 H₂ 추정 (초기 효율 가정)
        estimated_efficiency = 0.6  # 초기 추정
        required_h2_kg = target_power_kw * duration_hours / (self.sofc.h2_lhv_kwh_per_kg * estimated_efficiency)
        
        # 실제 사용할 H₂ 량 (재고 제한)
        actual_h2_to_use = min(required_h2_kg, max_available_h2)
        
        if actual_h2_to_use <= 0:
            return {"error": "No H2 available", "storage_level": self.storage.current_inventory_kg}
        
        # H₂ 인출
        retrieval_result = self.storage.retrieve_h2(actual_h2_to_use)
        
        if retrieval_result["retrieved_kg"] == 0:
            return {"error": "H2 retrieval failed", "storage_level": self.storage.current_inventory_kg}
        
        # SOFC 운전 (실제 인출된 H₂ 기준으로)
        sofc_result = self.sofc.operate(
            retrieval_result["retrieved_kg"], 
            duration_hours, 
            None  # target_power를 None으로 하여 H₂ 기준 최대 출력
        )
        
        if "error" in sofc_result:
            # H₂ 반환 (운전 실패시)
            self.storage.current_inventory_kg += retrieval_result["retrieved_kg"]
            return sofc_result
        
        # 통계 업데이트
        self.total_h2_consumed_kg += sofc_result["h2_consumed_kg"]
        self.total_electrical_energy_out_kwh += sofc_result["electrical_power_kw"] * duration_hours
        self.total_thermal_energy_kwh += sofc_result["thermal_power_kw"] * duration_hours
        
        return {
            "operation_mode": "gas_to_power",
            "electrical_output_kw": sofc_result["electrical_power_kw"],
            "thermal_output_kw": sofc_result["thermal_power_kw"], 
            "h2_consumed_kg": sofc_result["h2_consumed_kg"],
            "electrical_efficiency": sofc_result["electrical_efficiency"],
            "thermal_efficiency": sofc_result["thermal_efficiency"],
            "total_efficiency_chp": sofc_result["total_efficiency_chp"],
            "storage_level": self.storage.current_inventory_kg / self.storage.capacity_kg,
            "retrieval_energy_kwh": retrieval_result["retrieval_energy_kwh"],
            "net_electrical_output_kw": sofc_result["electrical_power_kw"] - 
                                       retrieval_result["retrieval_energy_kwh"] / duration_hours
        }
    
    def calculate_round_trip_efficiency(self) -> Dict:
        """Round-trip 효율 계산"""
        if self.total_electrical_energy_in_kwh == 0 or self.total_electrical_energy_out_kwh == 0:
            return {"error": "No complete round trip data"}
        
        # 순전기 효율 (IEA 2023 기준: 35-40%)
        electrical_efficiency = self.total_electrical_energy_out_kwh / self.total_electrical_energy_in_kwh
        
        # CHP 효율 (열 에너지 포함)
        total_energy_out = self.total_electrical_energy_out_kwh + self.total_thermal_energy_kwh  
        chp_efficiency = total_energy_out / self.total_electrical_energy_in_kwh
        
        return {
            "electrical_round_trip_efficiency": electrical_efficiency,
            "chp_round_trip_efficiency": chp_efficiency,
            "total_electrical_input_kwh": self.total_electrical_energy_in_kwh,
            "total_electrical_output_kwh": self.total_electrical_energy_out_kwh,
            "total_thermal_output_kwh": self.total_thermal_energy_kwh,
            "h2_inventory_kg": self.storage.current_inventory_kg,
            "h2_production_kg": self.total_h2_produced_kg,
            "h2_consumption_kg": self.total_h2_consumed_kg
        }
    
    def get_system_status(self) -> Dict:
        """시스템 전체 상태"""
        return {
            "soec": {
                "online": self.soec.is_online,
                "temperature": self.soec.current_temp,
                "load_ratio": self.soec.current_load_ratio,
                "degradation": self.soec.degradation_factor,
                "operating_hours": self.soec.operating_hours
            },
            "sofc": {
                "online": self.sofc.is_online,
                "temperature": self.sofc.current_temp,
                "load_ratio": self.sofc.current_load_ratio,
                "degradation": self.sofc.degradation_factor,
                "operating_hours": self.sofc.operating_hours
            },
            "storage": {
                "inventory_kg": self.storage.current_inventory_kg,
                "capacity_kg": self.storage.capacity_kg,
                "fill_level": self.storage.current_inventory_kg / self.storage.capacity_kg,
                "storage_type": self.storage.storage_type,
                "pressure_bar": self.storage.pressure_bar
            },
            "performance": {
                "total_h2_produced_kg": self.total_h2_produced_kg,
                "total_h2_consumed_kg": self.total_h2_consumed_kg,
                "net_h2_inventory_change_kg": self.total_h2_produced_kg - self.total_h2_consumed_kg,
                "electrical_energy_in_kwh": self.total_electrical_energy_in_kwh,
                "electrical_energy_out_kwh": self.total_electrical_energy_out_kwh,
                "thermal_energy_out_kwh": self.total_thermal_energy_kwh
            }
        }
    
    def simulate_daily_cycle(self, 
                           p2g_schedule: List[Tuple[float, float]],  # [(power_kw, hours), ...]
                           g2p_schedule: List[Tuple[float, float]]) -> pd.DataFrame:
        """
        일일 운전 주기 시뮬레이션
        
        Args:
            p2g_schedule: P2G 스케줄 [(전력, 시간), ...]
            g2p_schedule: G2P 스케줄 [(전력, 시간), ...]
            
        Returns:
            시간별 운전 결과 DataFrame
        """
        results = []
        current_hour = 0
        
        # P2G 운전
        for power_kw, duration_hours in p2g_schedule:
            if power_kw > 0:
                result = self.power_to_gas(power_kw, duration_hours)
                result["hour"] = current_hour
                result["duration_hours"] = duration_hours
                results.append(result)
            current_hour += duration_hours
        
        # G2P 운전  
        for power_kw, duration_hours in g2p_schedule:
            if power_kw > 0:
                result = self.gas_to_power(power_kw, duration_hours)
                result["hour"] = current_hour
                result["duration_hours"] = duration_hours
                results.append(result)
            current_hour += duration_hours
        
        return pd.DataFrame(results)


# 테스트 코드
if __name__ == "__main__":
    # H₂ 시스템 생성
    h2_system = H2SystemModule(
        soec_power_kw=50000,    # 50 MW
        sofc_power_kw=50000,    # 50 MW
        storage_capacity_kg=150000  # 150 ton
    )
    
    print("🔋 H₂ System Initialized")
    status = h2_system.get_system_status()
    print(f"Storage Capacity: {status['storage']['capacity_kg']:,.0f} kg")
    print(f"Initial H₂ Inventory: {status['storage']['inventory_kg']:,.0f} kg ({status['storage']['fill_level']:.1%})")
    
    # P2G 테스트 (전기 → H₂)
    print("\n⚡→🔋 Power-to-Gas Test")
    p2g_result = h2_system.power_to_gas(30000, 4)  # 30MW, 4시간
    print(f"Input: {p2g_result['electrical_input_kw']:,.0f} kW")
    print(f"H₂ Produced: {p2g_result['h2_produced_kg']:.1f} kg")
    print(f"H₂ Stored: {p2g_result['h2_stored_kg']:.1f} kg")
    print(f"Electrical Efficiency: {p2g_result['electrical_efficiency']:.1%}")
    print(f"CHP Efficiency: {p2g_result['total_efficiency_chp']:.1%}")
    
    # G2P 테스트 (H₂ → 전기)
    print("\n🔋→⚡ Gas-to-Power Test")
    g2p_result = h2_system.gas_to_power(25000, 6)  # 25MW, 6시간
    print(f"Electrical Output: {g2p_result['electrical_output_kw']:,.0f} kW")
    print(f"Thermal Output: {g2p_result['thermal_output_kw']:,.0f} kW")
    print(f"H₂ Consumed: {g2p_result['h2_consumed_kg']:.1f} kg")
    print(f"Electrical Efficiency: {g2p_result['electrical_efficiency']:.1%}")
    print(f"CHP Efficiency: {g2p_result['total_efficiency_chp']:.1%}")
    
    # Round-trip 효율
    print("\n🔄 Round-Trip Efficiency")
    rt_eff = h2_system.calculate_round_trip_efficiency()
    print(f"Electrical Round-Trip: {rt_eff['electrical_round_trip_efficiency']:.1%}")
    print(f"CHP Round-Trip: {rt_eff['chp_round_trip_efficiency']:.1%}")
    
    # 최종 상태
    final_status = h2_system.get_system_status()
    print(f"\n📊 Final Status")
    print(f"H₂ Inventory: {final_status['storage']['inventory_kg']:,.0f} kg ({final_status['storage']['fill_level']:.1%})")
    print(f"Net H₂ Change: {final_status['performance']['net_h2_inventory_change_kg']:+.1f} kg")