"""
M1. PV 발전 모듈 (Photovoltaic Module)
4가지 PV 기술: c-Si, 탠덤, 3접합, 무한접합
능동 제어 옵션 포함
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import math

from config import PV_TYPES


class PVModule:
    """태양광 발전 모듈"""
    
    def __init__(self, 
                 pv_type: str = "c-Si",
                 capacity_mw: float = 100.0,
                 operating_years: float = 0.0,
                 active_control: bool = False):
        """
        PV 모듈 초기화
        
        Args:
            pv_type: PV 기술 타입 ('c-Si', 'tandem', 'triple', 'infinite')
            capacity_mw: 설치 용량 (MW)
            operating_years: 운영 연수 (열화 계산용)
            active_control: 능동 제어 활성화 여부
        """
        if pv_type not in PV_TYPES:
            raise ValueError(f"지원되지 않는 PV 타입: {pv_type}")
        
        self.pv_type = pv_type
        self.params = PV_TYPES[pv_type].copy()
        self.capacity_mw = capacity_mw
        self.operating_years = operating_years
        self.active_control = active_control
        
        # PV 면적 계산 (MW당)
        # area_per_100mw는 100MW당 ha → MW당 m² = (ha/100) * 10000
        self.area_per_mw = (self.params['area_per_100mw'] / 100) * 10000  # m²/MW
        self.total_area_m2 = self.area_per_mw * capacity_mw
        
        # 능동 제어 파라미터
        self.control_v_range = (0.7, 1.0)  # V_OC 대비 비율
        self.control_j_range = (0.5, 1.0)  # J_SC 대비 비율
        
    def calculate_power_output(self, 
                             ghi_w_per_m2: float,
                             temp_celsius: float,
                             wind_speed_ms: float = 2.0) -> Dict[str, float]:
        """
        PV 출력 계산
        
        Args:
            ghi_w_per_m2: 전천일사량 (W/m²)
            temp_celsius: 외기온도 (°C)
            wind_speed_ms: 풍속 (m/s, 패널 냉각 효과용)
            
        Returns:
            출력 전력 및 관련 정보 딕셔너리
        """
        # 1. 셀 온도 계산 (NOCT 모델)
        cell_temp = self._calculate_cell_temperature(
            temp_celsius, ghi_w_per_m2, wind_speed_ms
        )
        
        # 2. 온도 의존 효율 계산
        eta_temp = self._calculate_temperature_efficiency(cell_temp)
        
        # 3. 열화 효과 적용
        eta_degraded = self._apply_degradation(eta_temp)
        
        # 4. 출력 전력 계산
        if self.active_control:
            power_mw = self._calculate_controlled_power(
                eta_degraded, ghi_w_per_m2
            )
        else:
            power_mw = self._calculate_fixed_power(
                eta_degraded, ghi_w_per_m2
            )
        
        return {
            'power_mw': power_mw,
            'cell_temp_celsius': cell_temp,
            'efficiency_percent': eta_degraded,
            'capacity_factor': power_mw / self.capacity_mw if self.capacity_mw > 0 else 0,
            'ghi_w_per_m2': ghi_w_per_m2,
            'temp_celsius': temp_celsius
        }
    
    def _calculate_cell_temperature(self, 
                                  temp_ambient: float,
                                  ghi: float,
                                  wind_speed: float) -> float:
        """NOCT 모델을 사용한 셀 온도 계산"""
        noct = self.params['noct']
        
        # 풍속 효과 (풍속이 높을수록 냉각 효과)
        wind_factor = 1 - 0.04 * max(0, wind_speed - 2)  # 2m/s 이상에서 냉각 효과
        
        cell_temp = temp_ambient + (noct - 20) * (ghi / 800) * wind_factor
        
        return max(temp_ambient, cell_temp)  # 셀 온도는 외기온도보다 낮을 수 없음
    
    def _calculate_temperature_efficiency(self, cell_temp: float) -> float:
        """
        온도 의존 효율 계산 (IEC 61215 / IEC 61853 기반)
        
        공식: η(T) = η_STC × [1 + β_rel × (T_cell − 25°C)]
        
        여기서:
          - η_STC: STC(25°C, 1000 W/m²) 기준 효율 (% 단위, 예: 24.4%)
          - β_rel: **상대** 온도계수 (1/°C 단위)
                   config의 beta 값은 %/°C 단위이므로 /100 변환 필요
                   예) c-Si beta = −0.35 %/°C → β_rel = −0.0035 /°C
                   이는 셀 온도가 1°C 상승할 때 효율이 STC 대비 0.35% **상대적** 감소를 의미
          - T_cell: 셀 온도 (°C), _calculate_cell_temperature()에서 NOCT 모델로 산출
        
        Note: beta는 **상대(relative)** 온도계수임 (절대(absolute) 아님).
              절대 온도계수 β_abs = η_STC × β_rel 이므로,
              c-Si 예) β_abs = 24.4% × 0.0035 = 0.0854 %p/°C
        
        Ref: De Soto et al., Solar Energy 80(1), 2006
             IEC 61215:2021, IEC 61853-1:2011
        """
        eta_stc = self.params['eta_stc']       # STC 효율 (%, 예: 24.4)
        beta_pct_per_degC = self.params['beta'] # 상대 온도계수 (%/°C, 예: -0.35)
        
        # %/°C → 1/°C 변환: beta_rel = beta_pct_per_degC / 100
        beta_rel = beta_pct_per_degC / 100.0    # 예: -0.35 → -0.0035 /°C
        
        delta_T = cell_temp - 25.0              # STC 기준 온도차 (°C)
        
        eta_temp = eta_stc * (1.0 + beta_rel * delta_T)
        
        return max(0.0, eta_temp)  # 효율은 0% 이상
    
    def _apply_degradation(self, efficiency: float) -> float:
        """연간 열화율 적용"""
        delta = self.params['delta']  # %/yr
        
        # 지수 감소 모델: η(t) = η_0 × (1 - δ)^t
        degradation_factor = (1 - delta / 100) ** self.operating_years
        
        return efficiency * degradation_factor
    
    def _calculate_fixed_power(self, efficiency: float, ghi: float) -> float:
        """고정 출력 계산 (능동 제어 없음)"""
        if ghi <= 0 or efficiency <= 0:
            return 0.0
        # P_PV = η × A × G / 1e6 (MW 단위)
        power_mw = (efficiency / 100) * self.total_area_m2 * ghi / 1e6
        
        # NaN 방지
        if not np.isfinite(power_mw):
            return 0.0
        
        return min(power_mw, self.capacity_mw)  # 정격 용량 제한
    
    def _calculate_controlled_power(self, efficiency: float, ghi: float) -> float:
        """능동 제어 출력 계산"""
        base_power = self._calculate_fixed_power(efficiency, ghi)
        
        if base_power <= 0:
            return 0.0
        
        # 능동 제어 효과:
        # 1. MPPT 최적화: +3~7% 출력 향상
        # 2. 부분 음영 손실 저감: +2~5% 
        # 3. DC 버스 전압 안정화: +1~3%
        
        # 종합 성능 향상: 5~15% (일사량에 따라 가변)
        improvement_base = 0.05  # 최소 5%
        improvement_variable = 0.10 * (ghi / 1000)  # 일사량 비례 (최대 10%)
        improvement_random = np.random.uniform(0, 0.02)  # 제어 변동성 ±2%
        
        total_improvement = improvement_base + improvement_variable + improvement_random
        total_improvement = min(total_improvement, 0.15)  # 최대 15% 제한
        
        controlled_power = base_power * (1.0 + total_improvement)
        
        return min(controlled_power, self.capacity_mw)
    
    def simulate_time_series(self, 
                           weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        시계열 시뮬레이션 실행
        
        Args:
            weather_data: 기상 데이터 (ghi_w_per_m2, temp_celsius, wind_speed_ms 컬럼 필요)
            
        Returns:
            PV 출력 시계열 데이터
        """
        results = []
        
        for idx, row in weather_data.iterrows():
            output = self.calculate_power_output(
                ghi_w_per_m2=row['ghi_w_per_m2'],
                temp_celsius=row['temp_celsius'],
                wind_speed_ms=row.get('wind_speed_ms', 2.0)
            )
            
            output['timestamp'] = idx
            results.append(output)
        
        df = pd.DataFrame(results)
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_daily_statistics(self, pv_data: pd.DataFrame) -> Dict[str, float]:
        """일별 통계 계산"""
        if pv_data.empty:
            return {}
        
        # 일별 데이터 그룹화
        daily_data = pv_data.groupby(pv_data.index.date).agg({
            'power_mw': ['sum', 'max', 'mean'],
            'capacity_factor': 'mean',
            'cell_temp_celsius': ['max', 'mean']
        }).round(3)
        
        # 전체 기간 통계
        stats = {
            'total_generation_mwh': pv_data['power_mw'].sum(),
            'peak_power_mw': pv_data['power_mw'].max(),
            'average_power_mw': pv_data['power_mw'].mean(),
            'capacity_factor_avg': pv_data['capacity_factor'].mean(),
            'max_cell_temp_celsius': pv_data['cell_temp_celsius'].max(),
            'min_cell_temp_celsius': pv_data['cell_temp_celsius'].min(),
        }
        
        # 운전 시간 (출력 > 0)
        operating_hours = (pv_data['power_mw'] > 0).sum()
        stats['operating_hours'] = operating_hours
        stats['utilization_rate'] = operating_hours / len(pv_data) if len(pv_data) > 0 else 0
        
        return stats
    
    def get_monthly_generation(self, pv_data: pd.DataFrame) -> Dict[int, float]:
        """월별 발전량 계산"""
        if pv_data.empty:
            return {}
        
        monthly_gen = pv_data.groupby(pv_data.index.month)['power_mw'].sum()
        return monthly_gen.to_dict()
    
    def estimate_levelized_cost(self, 
                              capex_per_mw: float = 1000000,  # $/MW
                              opex_per_mw_year: float = 20000,  # $/MW/year  
                              discount_rate: float = 0.05,
                              lifetime_years: int = 25) -> float:
        """
        LCOE (Levelized Cost of Energy) 추정
        
        Args:
            capex_per_mw: MW당 초기 투자비 ($)
            opex_per_mw_year: MW당 연간 운영비 ($)
            discount_rate: 할인율
            lifetime_years: 수명 (년)
            
        Returns:
            LCOE ($/MWh)
        """
        # 총 투자비
        total_capex = capex_per_mw * self.capacity_mw
        
        # 연간 운영비 현재가치
        annual_opex = opex_per_mw_year * self.capacity_mw
        pv_opex = sum(annual_opex / (1 + discount_rate) ** year 
                     for year in range(1, lifetime_years + 1))
        
        # 총 비용
        total_cost = total_capex + pv_opex
        
        # 연간 발전량 추정 (한국 평균 이용률 15% 가정)
        annual_generation_mwh = self.capacity_mw * 8760 * 0.15
        
        # 총 발전량 현재가치
        pv_generation = sum(annual_generation_mwh / (1 + discount_rate) ** year
                          for year in range(1, lifetime_years + 1))
        
        # LCOE 계산
        lcoe = total_cost / pv_generation if pv_generation > 0 else float('inf')
        
        return lcoe
    
    def update_parameters(self, **kwargs) -> None:
        """파라미터 업데이트"""
        for key, value in kwargs.items():
            if key == 'pv_type' and value in PV_TYPES:
                self.pv_type = value
                self.params = PV_TYPES[value].copy()
                self.area_per_mw = (self.params['area_per_100mw'] / 100) * 10000
                self.total_area_m2 = self.area_per_mw * self.capacity_mw
            elif key == 'capacity_mw':
                self.capacity_mw = value
                self.total_area_m2 = self.area_per_mw * value
            elif key == 'operating_years':
                self.operating_years = value
            elif key == 'active_control':
                self.active_control = value


# 테스트 코드
if __name__ == "__main__":
    # 테스트 데이터
    from m10_weather import WeatherModule
    
    # 기상 데이터 생성
    weather = WeatherModule()
    weather_data = weather.generate_tmy_data()
    
    # PV 시스템 생성 및 시뮬레이션
    for pv_type in ['c-Si', 'tandem', 'triple', 'infinite']:
        pv = PVModule(pv_type=pv_type, capacity_mw=100)
        
        # 샘플 출력 계산
        sample_output = pv.calculate_power_output(
            ghi_w_per_m2=800, temp_celsius=25, wind_speed_ms=3
        )
        
        print(f"\n{pv.params['name']} (100MW):")
        print(f"  출력: {sample_output['power_mw']:.2f} MW")
        print(f"  효율: {sample_output['efficiency_percent']:.2f}%")
        print(f"  이용률: {sample_output['capacity_factor']:.2f}")
        print(f"  필요 면적: {pv.total_area_m2/10000:.1f} ha")