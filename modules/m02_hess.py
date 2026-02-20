"""
M2. HESS (Hybrid Energy Storage System) 모듈
5-layer 하이브리드 에너지 저장: Supercap + Li-ion BESS + RFB + CAES + H₂
주파수 기반 부하 분리 및 SOC 밸런싱 제어
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import math
from dataclasses import dataclass

@dataclass
class HESSLayerConfig:
    """HESS 레이어 설정"""
    name: str
    capacity_kwh: float
    power_rating_kw: float
    response_time_ms: float
    efficiency_charge: float
    efficiency_discharge: float
    self_discharge_rate_per_hr: float
    degradation_cycle_factor: float
    degradation_temp_factor: float
    operating_temp_range: Tuple[float, float]  # (min, max) °C
    capex_per_kwh: float  # $/kWh
    opex_per_kwh_year: float  # $/kWh/year
    time_constant_range: Tuple[float, float]  # (min, max) seconds

class HESSTechnologyLayer:
    """HESS 기술별 레이어 클래스"""
    
    def __init__(self, config: HESSLayerConfig):
        self.config = config
        self.current_soc = 0.5  # 초기 SOC 50%
        self.current_power = 0.0  # kW
        self.temperature = 25.0  # °C
        self.cycle_count = 0.0
        self.degradation_factor = 1.0
        self.energy_throughput_kwh = 0.0
        
    def calculate_available_power(self, 
                                operation: str,  # 'charge' or 'discharge'
                                duration_s: float = 1.0) -> float:
        """사용 가능한 전력 계산"""
        if operation == 'charge':
            # 충전 시: SOC 100% 제한, 전력 정격 제한
            soc_headroom = (1.0 - self.current_soc)
            max_energy_kwh = soc_headroom * self.config.capacity_kwh * self.degradation_factor
            max_power_from_energy = max_energy_kwh * 3600 / duration_s
            
            return min(
                self.config.power_rating_kw * self.degradation_factor,
                max_power_from_energy
            )
            
        elif operation == 'discharge':
            # 방전 시: SOC 0% 제한, 전력 정격 제한
            available_energy_kwh = self.current_soc * self.config.capacity_kwh * self.degradation_factor
            max_power_from_energy = available_energy_kwh * 3600 / duration_s
            
            return min(
                self.config.power_rating_kw * self.degradation_factor,
                max_power_from_energy
            )
        
        return 0.0
    
    def operate(self, 
                power_kw: float,
                duration_s: float = 1.0,
                temperature: float = 25.0) -> Dict[str, float]:
        """
        레이어 운전
        
        Args:
            power_kw: 요청 전력 (양수: 충전, 음수: 방전)
            duration_s: 운전 지속 시간 (초)
            temperature: 운전 온도 (°C)
            
        Returns:
            운전 결과 딕셔너리
        """
        self.temperature = temperature
        
        # 온도 범위 확인
        temp_penalty = 1.0
        if temperature < self.config.operating_temp_range[0]:
            temp_penalty = 0.8  # 저온에서 성능 저하
        elif temperature > self.config.operating_temp_range[1]:
            temp_penalty = 0.7  # 고온에서 성능 저하
        
        # 응답시간 지연 확인
        if duration_s * 1000 < self.config.response_time_ms:
            response_factor = duration_s * 1000 / self.config.response_time_ms
        else:
            response_factor = 1.0
        
        # 실제 운전 가능 전력 계산
        if power_kw > 0:  # 충전
            max_power = self.calculate_available_power('charge', duration_s)
            actual_power = min(power_kw, max_power) * temp_penalty * response_factor
            efficiency = self.config.efficiency_charge
        else:  # 방전
            max_power = self.calculate_available_power('discharge', duration_s)
            actual_power = max(power_kw, -max_power) * temp_penalty * response_factor
            efficiency = self.config.efficiency_discharge
        
        # 에너지 계산 (효율 적용)
        if actual_power > 0:  # 충전
            energy_stored_kwh = actual_power * duration_s / 3600 * efficiency
            self.current_soc = min(1.0, 
                self.current_soc + energy_stored_kwh / (self.config.capacity_kwh * self.degradation_factor))
        else:  # 방전
            energy_delivered_kwh = abs(actual_power) * duration_s / 3600
            energy_consumed_kwh = energy_delivered_kwh / efficiency
            self.current_soc = max(0.0, 
                self.current_soc - energy_consumed_kwh / (self.config.capacity_kwh * self.degradation_factor))
        
        # 자기방전 적용
        self_discharge_factor = self.config.self_discharge_rate_per_hr * duration_s / 3600
        self.current_soc *= (1 - self_discharge_factor)
        self.current_soc = max(0.0, self.current_soc)
        
        # 사이클 카운트 업데이트
        dod = abs(actual_power * duration_s / 3600) / (self.config.capacity_kwh * self.degradation_factor)
        self.cycle_count += dod / 2  # DOD 50% = 1 cycle
        self.energy_throughput_kwh += abs(actual_power * duration_s / 3600)
        
        # 열화 업데이트
        self._update_degradation()
        
        # 현재 전력 업데이트
        self.current_power = actual_power
        
        return {
            'requested_power_kw': power_kw,
            'actual_power_kw': actual_power,
            'efficiency': efficiency,
            'soc': self.current_soc,
            'energy_kwh': self.current_soc * self.config.capacity_kwh * self.degradation_factor,
            'cycle_count': self.cycle_count,
            'degradation_factor': self.degradation_factor,
            'temperature': temperature,
            'response_limited': response_factor < 1.0,
            'temp_limited': temp_penalty < 1.0
        }
    
    def _update_degradation(self):
        """열화 모델 업데이트"""
        # 사이클 열화
        cycle_degradation = 1 - self.config.degradation_cycle_factor * self.cycle_count
        
        # 온도 열화 (Arrhenius 모델)
        temp_stress = max(0, (self.temperature - 25) / 10)  # 25°C 기준
        temp_degradation = 1 - self.config.degradation_temp_factor * temp_stress
        
        self.degradation_factor = max(0.5, min(cycle_degradation, temp_degradation))


class HESSModule:
    """5-Layer HESS 통합 모듈"""
    
    def __init__(self):
        """HESS 모듈 초기화"""
        self.layers = self._initialize_layers()
        self.control_signals = {}
        self.frequency_filters = self._setup_frequency_filters()
        
    def _initialize_layers(self) -> Dict[str, HESSTechnologyLayer]:
        """5개 레이어 초기화"""
        layers = {}
        
        # Layer 1: Supercapacitor
        layers['supercap'] = HESSTechnologyLayer(HESSLayerConfig(
            name="Supercapacitor",
            capacity_kwh=50,  # 50 kWh
            power_rating_kw=10000,  # 10 MW (100-1,000C rate)
            response_time_ms=0.001,  # 1 μs
            efficiency_charge=0.98,
            efficiency_discharge=0.98,
            self_discharge_rate_per_hr=0.01,  # 1% per hour
            degradation_cycle_factor=1e-8,  # 매우 긴 수명 (1M+ cycles)
            degradation_temp_factor=1e-5,
            operating_temp_range=(-40, 85),
            capex_per_kwh=10000,  # $10,000/kWh
            opex_per_kwh_year=100,
            time_constant_range=(0.001, 1.0)  # μs ~ s
        ))
        
        # Layer 2: Li-ion BESS
        layers['bess'] = HESSTechnologyLayer(HESSLayerConfig(
            name="Li-ion BESS",
            capacity_kwh=2000000,  # 2,000 MWh
            power_rating_kw=200000,  # 200 MW (C/10 rate)
            response_time_ms=100,  # 100 ms
            efficiency_charge=0.95,
            efficiency_discharge=0.95,
            self_discharge_rate_per_hr=0.0005,  # 0.05% per hour
            degradation_cycle_factor=2e-5,  # 5,000 cycles @ 80% DOD
            degradation_temp_factor=3e-4,
            operating_temp_range=(0, 45),
            capex_per_kwh=200,  # $200/kWh
            opex_per_kwh_year=5,
            time_constant_range=(1.0, 3600.0)  # s ~ hr
        ))
        
        # Layer 3: Redox Flow Battery (RFB)
        layers['rfb'] = HESSTechnologyLayer(HESSLayerConfig(
            name="Vanadium RFB",
            capacity_kwh=750000,  # 750 MWh
            power_rating_kw=50000,  # 50 MW
            response_time_ms=1000,  # 1 s
            efficiency_charge=0.85,
            efficiency_discharge=0.85,
            self_discharge_rate_per_hr=0.0001,  # 0.01% per hour
            degradation_cycle_factor=5e-7,  # 20,000+ cycles
            degradation_temp_factor=1e-4,
            operating_temp_range=(15, 35),
            capex_per_kwh=300,  # $300/kWh
            opex_per_kwh_year=10,
            time_constant_range=(3600.0, 86400.0)  # hr ~ day
        ))
        
        # Layer 4: Compressed Air Energy Storage (CAES)
        layers['caes'] = HESSTechnologyLayer(HESSLayerConfig(
            name="CAES",
            capacity_kwh=1000000,  # 1,000 MWh
            power_rating_kw=100000,  # 100 MW
            response_time_ms=30000,  # 30 s
            efficiency_charge=0.75,  # Round-trip efficiency
            efficiency_discharge=0.75,
            self_discharge_rate_per_hr=0.00001,  # 거의 없음
            degradation_cycle_factor=1e-7,  # 매우 긴 수명
            degradation_temp_factor=5e-5,
            operating_temp_range=(-10, 50),
            capex_per_kwh=100,  # $100/kWh
            opex_per_kwh_year=2,
            time_constant_range=(86400.0, 604800.0)  # day ~ week
        ))
        
        # Layer 5: H₂ (연결은 M5와 연동)
        layers['h2'] = HESSTechnologyLayer(HESSLayerConfig(
            name="H2 Storage",
            capacity_kwh=5000000,  # 5,000 MWh (seasonal storage)
            power_rating_kw=50000,  # 50 MW
            response_time_ms=300000,  # 5 min
            efficiency_charge=0.40,  # Round-trip electrical efficiency
            efficiency_discharge=0.40,
            self_discharge_rate_per_hr=0.000001,  # 거의 없음
            degradation_cycle_factor=1e-6,
            degradation_temp_factor=2e-5,
            operating_temp_range=(-40, 80),
            capex_per_kwh=20,  # $20/kWh (저장 부분만)
            opex_per_kwh_year=1,
            time_constant_range=(86400.0, 31536000.0)  # day ~ seasonal (expanded range)
        ))
        
        return layers
    
    def _setup_frequency_filters(self) -> Dict[str, Dict]:
        """주파수 기반 필터 설정"""
        filters = {}
        
        for layer_name, layer in self.layers.items():
            tc_min, tc_max = layer.config.time_constant_range
            
            filters[layer_name] = {
                'frequency_range': (1/tc_max, 1/tc_min),  # Hz
                'time_constant_range': (tc_min, tc_max),    # seconds
                'weight': 1.0
            }
        
        return filters
    
    def calculate_power_allocation(self, 
                                 total_power_request_kw: float,
                                 duration_s: float = 1.0,
                                 frequency_hz: float = 0.001) -> Dict[str, float]:
        """
        주파수 기반 전력 배분 계산
        
        Args:
            total_power_request_kw: 총 전력 요청 (양수: 충전, 음수: 방전)
            duration_s: 지속 시간
            frequency_hz: 신호 주파수 (1/time_constant)
            
        Returns:
            레이어별 전력 배분
        """
        allocation = {}
        remaining_power = total_power_request_kw
        
        # 우선순위 기반 배분: Supercap → BESS → RFB → CAES → H₂
        priority_order = ['supercap', 'bess', 'rfb', 'caes', 'h2']
        
        for layer_name in priority_order:
            if abs(remaining_power) < 1.0:  # 1kW 미만은 무시
                allocation[layer_name] = 0.0
                continue
                
            layer = self.layers[layer_name]
            freq_range = self.frequency_filters[layer_name]['frequency_range']
            
            # 주파수 응답성 확인
            if freq_range[0] <= frequency_hz <= freq_range[1]:
                # 이 레이어가 해당 주파수에 대응 가능
                if remaining_power > 0:  # 충전
                    max_power = layer.calculate_available_power('charge', duration_s)
                    allocated_power = min(remaining_power, max_power)
                else:  # 방전
                    max_power = layer.calculate_available_power('discharge', duration_s)
                    allocated_power = max(remaining_power, -max_power)
                
                allocation[layer_name] = allocated_power
                remaining_power -= allocated_power
            else:
                allocation[layer_name] = 0.0
        
        # 미배분 전력 처리 (낮은 우선순위 레이어에 강제 배분)
        if abs(remaining_power) >= 1.0:
            # 가장 적합한 레이어 찾기 (시간 상수 기준)
            time_constant = 1 / max(frequency_hz, 1e-6)
            
            best_layer = None
            best_score = float('inf')
            
            for layer_name, layer in self.layers.items():
                tc_min, tc_max = layer.config.time_constant_range
                if tc_min <= time_constant <= tc_max:
                    # 시간 상수가 범위 내에 있음
                    score = min(abs(time_constant - tc_min), abs(time_constant - tc_max))
                    if score < best_score:
                        best_score = score
                        best_layer = layer_name
            
            if best_layer and best_layer in allocation:
                if remaining_power > 0:
                    max_additional = self.layers[best_layer].calculate_available_power('charge', duration_s)
                    additional_power = min(remaining_power, max_additional - allocation[best_layer])
                else:
                    max_additional = self.layers[best_layer].calculate_available_power('discharge', duration_s)
                    additional_power = max(remaining_power, -(max_additional + abs(allocation[best_layer])))
                
                allocation[best_layer] += additional_power
        
        return allocation
    
    def operate_hess(self, 
                     power_request_kw: float,
                     duration_s: float = 1.0,
                     frequency_hz: float = 0.001,
                     temperature: float = 25.0) -> Dict[str, any]:
        """
        HESS 통합 운전
        
        Args:
            power_request_kw: 전력 요청 (양수: 충전, 음수: 방전)
            duration_s: 지속 시간
            frequency_hz: 신호 주파수
            temperature: 환경 온도
            
        Returns:
            통합 운전 결과
        """
        # 1. 전력 배분 계산
        allocation = self.calculate_power_allocation(power_request_kw, duration_s, frequency_hz)
        
        # 2. 각 레이어 운전
        layer_results = {}
        total_actual_power = 0.0
        total_energy_stored = 0.0
        
        for layer_name, allocated_power in allocation.items():
            if abs(allocated_power) >= 0.1:  # 0.1kW 이상만 운전
                result = self.layers[layer_name].operate(allocated_power, duration_s, temperature)
                layer_results[layer_name] = result
                total_actual_power += result['actual_power_kw']
                
                if result['actual_power_kw'] > 0:  # 충전
                    total_energy_stored += result['actual_power_kw'] * duration_s / 3600
                else:  # 방전
                    total_energy_stored += result['actual_power_kw'] * duration_s / 3600  # 음수
        
        # 3. SOC 밸런싱 확인
        soc_balance = self._check_soc_balance()
        
        # 4. 통합 결과
        return {
            'power_request_kw': power_request_kw,
            'power_delivered_kw': total_actual_power,
            'energy_change_kwh': total_energy_stored,
            'power_allocation': allocation,
            'layer_results': layer_results,
            'soc_balance': soc_balance,
            'total_capacity_kwh': sum(layer.config.capacity_kwh * layer.degradation_factor 
                                    for layer in self.layers.values()),
            'average_soc': np.mean([layer.current_soc for layer in self.layers.values()]),
            'response_time_ms': min(layer.config.response_time_ms for layer in self.layers.values()),
            'round_trip_efficiency': self._calculate_system_efficiency()
        }
    
    def _check_soc_balance(self) -> Dict[str, float]:
        """SOC 밸런싱 체크"""
        soc_values = {name: layer.current_soc for name, layer in self.layers.items()}
        
        # 목표 SOC 범위 (레이어별 최적화)
        target_ranges = {
            'supercap': (0.4, 0.6),  # 즉시 응답용
            'bess': (0.2, 0.8),      # 일중 변동용  
            'rfb': (0.3, 0.7),       # 장주기용
            'caes': (0.4, 0.6),      # 주간 저장용
            'h2': (0.1, 0.9)         # 계절 저장용
        }
        
        balance_score = {}
        for layer_name, soc in soc_values.items():
            target_min, target_max = target_ranges[layer_name]
            if target_min <= soc <= target_max:
                balance_score[layer_name] = 1.0  # 최적
            elif soc < target_min:
                balance_score[layer_name] = soc / target_min
            else:  # soc > target_max
                balance_score[layer_name] = (1 - soc) / (1 - target_max)
        
        return {
            'soc_values': soc_values,
            'balance_scores': balance_score,
            'overall_balance': np.mean(list(balance_score.values())),
            'worst_layer': min(balance_score.keys(), key=lambda k: balance_score[k])
        }
    
    def _calculate_system_efficiency(self) -> float:
        """시스템 전체 효율 계산"""
        # 각 레이어별 가중평균 효율
        total_capacity = sum(layer.config.capacity_kwh for layer in self.layers.values())
        
        if total_capacity == 0:
            return 0.0
        
        weighted_eff = 0
        for layer in self.layers.values():
            weight = layer.config.capacity_kwh / total_capacity
            layer_eff = math.sqrt(layer.config.efficiency_charge * layer.config.efficiency_discharge)
            weighted_eff += weight * layer_eff
        
        # 현실적인 시스템 효율 (각 레이어 독립적이므로 가장 높은 효율층 기준)
        max_eff = max(math.sqrt(layer.config.efficiency_charge * layer.config.efficiency_discharge) 
                     for layer in self.layers.values())
        
        return max_eff  # 시스템은 가장 효율적인 레이어를 선택하여 운전
    
    def get_system_status(self) -> Dict[str, any]:
        """시스템 전체 상태 조회"""
        total_energy = sum(layer.current_soc * layer.config.capacity_kwh * layer.degradation_factor 
                          for layer in self.layers.values())
        total_capacity = sum(layer.config.capacity_kwh * layer.degradation_factor 
                           for layer in self.layers.values())
        
        return {
            'layers': {name: {
                'soc': layer.current_soc,
                'energy_kwh': layer.current_soc * layer.config.capacity_kwh * layer.degradation_factor,
                'power_kw': layer.current_power,
                'degradation': layer.degradation_factor,
                'cycle_count': layer.cycle_count,
                'temperature': layer.temperature
            } for name, layer in self.layers.items()},
            'system_total': {
                'energy_kwh': total_energy,
                'capacity_kwh': total_capacity,
                'average_soc': total_energy / total_capacity if total_capacity > 0 else 0,
                'total_power_kw': sum(layer.current_power for layer in self.layers.values()),
                'system_efficiency': self._calculate_system_efficiency()
            }
        }
    
    def estimate_lcoe(self, lifetime_years: int = 20, discount_rate: float = 0.05) -> Dict[str, float]:
        """레이어별 LCOE 추정"""
        lcoe_by_layer = {}
        
        for name, layer in self.layers.items():
            # CAPEX
            total_capex = layer.config.capex_per_kwh * layer.config.capacity_kwh
            
            # OPEX (NPV)
            annual_opex = layer.config.opex_per_kwh_year * layer.config.capacity_kwh
            pv_opex = sum(annual_opex / (1 + discount_rate) ** year 
                         for year in range(1, lifetime_years + 1))
            
            total_cost = total_capex + pv_opex
            
            # 연간 처리량 추정 (단순화: 용량의 100회 사이클/년)
            annual_throughput_kwh = layer.config.capacity_kwh * 100
            pv_throughput = sum(annual_throughput_kwh / (1 + discount_rate) ** year
                              for year in range(1, lifetime_years + 1))
            
            lcoe_by_layer[name] = total_cost / pv_throughput if pv_throughput > 0 else float('inf')
        
        return lcoe_by_layer


# 테스트 코드
if __name__ == "__main__":
    # HESS 시스템 생성
    hess = HESSModule()
    
    print("🔋 HESS System Initialized")
    print(f"Total Capacity: {hess.get_system_status()['system_total']['capacity_kwh']/1000:.0f} MWh")
    
    # 다양한 주파수의 전력 요청 테스트
    test_scenarios = [
        {"name": "Fast response (1 Hz)", "power": 5000, "freq": 1.0, "duration": 1},
        {"name": "Medium response (0.01 Hz)", "power": -10000, "freq": 0.01, "duration": 100},
        {"name": "Slow response (1e-5 Hz)", "power": 20000, "freq": 1e-5, "duration": 10000}
    ]
    
    for scenario in test_scenarios:
        print(f"\n📊 Testing {scenario['name']}")
        result = hess.operate_hess(
            power_request_kw=scenario['power'],
            duration_s=scenario['duration'],
            frequency_hz=scenario['freq']
        )
        
        print(f"  Requested: {result['power_request_kw']:,.0f} kW")
        print(f"  Delivered: {result['power_delivered_kw']:,.0f} kW")
        print(f"  Efficiency: {result['round_trip_efficiency']:.1%}")
        print(f"  Primary layers: {[k for k,v in result['power_allocation'].items() if abs(v) > 100]}")
    
    # LCOE 추정
    lcoe = hess.estimate_lcoe()
    print(f"\n💰 LCOE Estimates:")
    for layer, cost in lcoe.items():
        print(f"  {layer}: ${cost:.2f}/kWh")