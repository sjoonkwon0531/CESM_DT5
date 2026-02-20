"""
M3. AIDC 부하 모듈 (AI Data Center Load Module)
GPU 기반 AI 데이터센터의 전력 수요 프로파일 생성
워크로드 믹스, PUE, 확률적 부하 패턴 포함
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import math

from config import GPU_TYPES, PUE_TIERS, WORKLOAD_TYPES


class AIDCModule:
    """AI 데이터센터 부하 모듈"""
    
    def __init__(self,
                 gpu_type: str = "H100",
                 gpu_count: int = 50000,
                 pue_tier: str = "tier2",
                 workload_mix: Dict[str, float] = None):
        """
        AIDC 모듈 초기화
        
        Args:
            gpu_type: GPU 종류 ('H100', 'B200', 'next_gen')
            gpu_count: GPU 수량
            pue_tier: PUE 티어 ('tier1', 'tier2', 'tier3', 'tier4')  
            workload_mix: 워크로드 믹스 비율 {'llm': 0.4, 'training': 0.4, 'moe': 0.2}
        """
        if gpu_type not in GPU_TYPES:
            raise ValueError(f"지원되지 않는 GPU 타입: {gpu_type}")
        
        if pue_tier not in PUE_TIERS:
            raise ValueError(f"지원되지 않는 PUE 티어: {pue_tier}")
        
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count
        self.pue_tier = pue_tier
        self.gpu_params = GPU_TYPES[gpu_type].copy()
        self.pue_params = PUE_TIERS[pue_tier].copy()
        
        # 워크로드 믹스 기본값
        if workload_mix is None:
            workload_mix = {'llm': 0.4, 'training': 0.4, 'moe': 0.2}
        
        # 믹스 비율 정규화
        total_mix = sum(workload_mix.values())
        self.workload_mix = {k: v/total_mix for k, v in workload_mix.items()}
        
        # IT 부하 계산 (GPU 전력만)
        self.max_it_power_mw = (self.gpu_count * 
                               self.gpu_params['power_w'] / 1e6)
        
        # 총 시설 부하 (PUE 적용)
        self.max_total_power_mw = self.max_it_power_mw * self.pue_params['pue']
    
    def calculate_load_at_time(self, 
                             hour_of_day: int,
                             day_of_week: int = 1,
                             random_seed: Optional[int] = None) -> Dict[str, float]:
        """
        특정 시간의 부하 계산
        
        Args:
            hour_of_day: 시간 (0-23)
            day_of_week: 요일 (0=월요일, 6=일요일)
            random_seed: 랜덤 시드 (재현성용)
            
        Returns:
            부하 정보 딕셔너리
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 워크로드별 활용률 계산
        workload_utils = {}
        for workload, ratio in self.workload_mix.items():
            util = self._calculate_workload_utilization(
                workload, hour_of_day, day_of_week
            )
            workload_utils[workload] = util
        
        # 전체 GPU 활용률 (가중평균)
        total_gpu_util = sum(
            ratio * workload_utils[workload] 
            for workload, ratio in self.workload_mix.items()
        )
        
        # IT 부하 계산
        it_power_mw = self.max_it_power_mw * total_gpu_util
        
        # 추가 IT 부하 (CPU, Memory, Network, Storage) - GPU 대비 10-15%
        additional_it_ratio = 0.12
        total_it_power_mw = it_power_mw * (1 + additional_it_ratio)
        
        # 총 시설 부하 (PUE 적용)  
        total_power_mw = total_it_power_mw * self.pue_params['pue']
        
        return {
            'total_power_mw': total_power_mw,
            'it_power_mw': total_it_power_mw,
            'gpu_power_mw': it_power_mw,
            'gpu_utilization': total_gpu_util,
            'pue': self.pue_params['pue'],
            'workload_utils': workload_utils.copy()
        }
    
    def _calculate_workload_utilization(self, 
                                      workload: str,
                                      hour: int,
                                      day_of_week: int) -> float:
        """워크로드별 GPU 활용률 계산"""
        if workload not in WORKLOAD_TYPES:
            return 0.5  # 기본값
        
        params = WORKLOAD_TYPES[workload]
        base_util = params['base_utilization']
        peak_util = params['peak_utilization'] 
        burst_freq = params['burst_frequency']  # per hour
        
        # 시간대별 패턴 (LLM은 주간 높음, Training은 야간 높음)
        if workload == 'llm':
            # LLM 추론: 주간 활성, 피크 시간 14-16시
            time_factor = 0.3 + 0.7 * max(0, math.cos(math.pi * (hour - 15) / 12))
        elif workload == 'training':  
            # AI 훈련: 야간 집중 (전력 요금 절약)
            time_factor = 0.8 + 0.2 * max(0, math.cos(math.pi * (hour - 3) / 12))
        else:  # MoE
            # MoE: 상대적으로 균등하지만 약간의 일간 변동
            time_factor = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(math.pi * hour / 12))
        
        # 요일별 패턴 (주말 약간 낮음)
        weekday_factor = 0.85 if day_of_week >= 5 else 1.0
        
        # 기본 활용률
        base_load = base_util * time_factor * weekday_factor
        
        # 버스트 패턴 (Poisson 프로세스 근사)
        burst_prob = burst_freq / 60  # 분당 확률
        if np.random.random() < burst_prob:
            # 버스트 발생 시 피크로 상승
            burst_intensity = np.random.beta(2, 3)  # 0-1 사이, 평균 0.4
            utilization = base_load + burst_intensity * (peak_util - base_load)
        else:
            # 정상 운영
            utilization = base_load
        
        # 노이즈 추가 (±5%)
        noise = 0.05 * (np.random.random() - 0.5)
        utilization += noise
        
        return np.clip(utilization, 0.05, 1.0)  # 5-100% 범위
    
    def simulate_time_series(self, 
                           hours: int = 8760,
                           start_hour: int = 0,
                           random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        시계열 부하 시뮬레이션
        
        Args:
            hours: 시뮬레이션 시간 (시간)
            start_hour: 시작 시간 (0-23)
            random_seed: 랜덤 시드
            
        Returns:
            부하 시계열 데이터
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        results = []
        
        for h in range(hours):
            current_hour = (start_hour + h) % 24
            current_day = ((start_hour + h) // 24) % 7
            
            load_data = self.calculate_load_at_time(
                hour_of_day=current_hour,
                day_of_week=current_day
            )
            
            load_data['hour'] = h
            load_data['hour_of_day'] = current_hour
            load_data['day_of_week'] = current_day
            results.append(load_data)
        
        df = pd.DataFrame(results)
        df.set_index('hour', inplace=True)
        
        return df
    
    def get_statistics(self, load_data: pd.DataFrame) -> Dict[str, float]:
        """부하 데이터 통계 계산"""
        if load_data.empty:
            return {}
        
        stats = {
            # 전력 통계
            'peak_power_mw': load_data['total_power_mw'].max(),
            'min_power_mw': load_data['total_power_mw'].min(),
            'avg_power_mw': load_data['total_power_mw'].mean(),
            'std_power_mw': load_data['total_power_mw'].std(),
            
            # 에너지 통계  
            'total_energy_mwh': load_data['total_power_mw'].sum(),
            'it_energy_mwh': load_data['it_power_mw'].sum(),
            'gpu_energy_mwh': load_data['gpu_power_mw'].sum(),
            
            # 활용률 통계
            'avg_gpu_utilization': load_data['gpu_utilization'].mean(),
            'peak_gpu_utilization': load_data['gpu_utilization'].max(),
            'min_gpu_utilization': load_data['gpu_utilization'].min(),
            
            # 부하율
            'load_factor': (load_data['total_power_mw'].mean() / 
                           load_data['total_power_mw'].max() 
                           if load_data['total_power_mw'].max() > 0 else 0),
            
            # PUE 
            'actual_pue': (load_data['total_power_mw'].sum() / 
                          load_data['it_power_mw'].sum() 
                          if load_data['it_power_mw'].sum() > 0 else 0)
        }
        
        return stats
    
    def get_workload_breakdown(self, load_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """워크로드별 분석"""
        if load_data.empty or 'workload_utils' not in load_data.columns:
            return {}
        
        breakdown = {}
        
        # 각 워크로드별 통계 (첫 번째 행에서 구조 확인)
        first_row_utils = load_data['workload_utils'].iloc[0]
        if isinstance(first_row_utils, dict):
            for workload in first_row_utils.keys():
                workload_utils = [
                    row_utils.get(workload, 0) 
                    for row_utils in load_data['workload_utils']
                    if isinstance(row_utils, dict)
                ]
                
                if workload_utils:
                    breakdown[workload] = {
                        'avg_utilization': np.mean(workload_utils),
                        'max_utilization': np.max(workload_utils),
                        'min_utilization': np.min(workload_utils),
                        'mix_ratio': self.workload_mix.get(workload, 0),
                        'avg_power_mw': (np.mean(workload_utils) * 
                                       self.workload_mix.get(workload, 0) * 
                                       self.max_it_power_mw)
                    }
        
        return breakdown
    
    def get_daily_profile(self, load_data: pd.DataFrame) -> Dict[int, float]:
        """시간대별 평균 부하 프로파일"""
        if 'hour_of_day' not in load_data.columns:
            return {}
        
        daily_profile = load_data.groupby('hour_of_day')['total_power_mw'].mean()
        return daily_profile.to_dict()
    
    def estimate_operating_cost(self,
                               annual_energy_mwh: float,
                               electricity_price_per_mwh: float = 80,  # $/MWh
                               demand_charge_per_mw_month: float = 15000,  # $/MW/month
                               facility_opex_ratio: float = 0.15) -> Dict[str, float]:
        """
        연간 운영비 추정
        
        Args:
            annual_energy_mwh: 연간 에너지 소비량 (MWh)
            electricity_price_per_mwh: 전력 단가 ($/MWh)
            demand_charge_per_mw_month: 수요 요금 ($/MW/month)
            facility_opex_ratio: 시설 운영비 비율 (전력비 대비)
            
        Returns:
            비용 분석 딕셔너리
        """
        # 전력 에너지 비용
        energy_cost = annual_energy_mwh * electricity_price_per_mwh
        
        # 수요 요금 (피크 기준)
        demand_cost = self.max_total_power_mw * demand_charge_per_mw_month * 12
        
        # 시설 운영비
        facility_cost = (energy_cost + demand_cost) * facility_opex_ratio
        
        # 총 운영비
        total_opex = energy_cost + demand_cost + facility_cost
        
        return {
            'energy_cost_usd': energy_cost,
            'demand_cost_usd': demand_cost,
            'facility_cost_usd': facility_cost,
            'total_opex_usd': total_opex,
            'cost_per_mwh': total_opex / annual_energy_mwh if annual_energy_mwh > 0 else 0,
            'cost_per_gpu_year': total_opex / self.gpu_count if self.gpu_count > 0 else 0
        }
    
    def update_parameters(self, **kwargs) -> None:
        """파라미터 업데이트"""
        if 'gpu_type' in kwargs and kwargs['gpu_type'] in GPU_TYPES:
            self.gpu_type = kwargs['gpu_type']
            self.gpu_params = GPU_TYPES[self.gpu_type].copy()
        
        if 'gpu_count' in kwargs:
            self.gpu_count = kwargs['gpu_count']
        
        if 'pue_tier' in kwargs and kwargs['pue_tier'] in PUE_TIERS:
            self.pue_tier = kwargs['pue_tier']
            self.pue_params = PUE_TIERS[self.pue_tier].copy()
        
        if 'workload_mix' in kwargs:
            workload_mix = kwargs['workload_mix']
            total_mix = sum(workload_mix.values())
            self.workload_mix = {k: v/total_mix for k, v in workload_mix.items()}
        
        # 전력 용량 재계산
        self.max_it_power_mw = (self.gpu_count * 
                               self.gpu_params['power_w'] / 1e6)
        self.max_total_power_mw = self.max_it_power_mw * self.pue_params['pue']


# 테스트 코드
if __name__ == "__main__":
    # AIDC 시스템 생성
    aidc = AIDCModule(
        gpu_type="H100",
        gpu_count=50000,
        pue_tier="tier2", 
        workload_mix={'llm': 0.5, 'training': 0.3, 'moe': 0.2}
    )
    
    print(f"AIDC 설정:")
    print(f"  GPU: {aidc.gpu_params['name']} × {aidc.gpu_count:,}")
    print(f"  최대 IT 부하: {aidc.max_it_power_mw:.1f} MW")
    print(f"  PUE: {aidc.pue_params['pue']} ({aidc.pue_params['name']})")
    print(f"  최대 총 부하: {aidc.max_total_power_mw:.1f} MW")
    
    # 샘플 부하 계산
    sample_load = aidc.calculate_load_at_time(hour_of_day=14, day_of_week=2)
    print(f"\n오후 2시 부하:")
    print(f"  총 부하: {sample_load['total_power_mw']:.1f} MW")
    print(f"  GPU 활용률: {sample_load['gpu_utilization']:.1%}")
    
    # 24시간 시뮬레이션
    daily_data = aidc.simulate_time_series(hours=24, random_seed=42)
    daily_profile = aidc.get_daily_profile(daily_data)
    
    print(f"\n일간 부하 프로파일 (MW):")
    for hour, power in daily_profile.items():
        print(f"  {hour:2d}시: {power:.1f}")