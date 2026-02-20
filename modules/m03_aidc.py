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
        """
        워크로드별 GPU 활용률 계산
        
        실제 AIDC 워크로드 특성 반영:
        - LLM 추론: Poisson arrival, 주간 피크, burst 패턴 (100ms~1s 단위 변동)
        - Training: Sustained high + 30분 주기 checkpoint spike
        - MoE: Expert 활성화 패턴 (20-30% 동시 활성), 불규칙 burst
        
        References:
        - Meta LLaMA 3 GPU failure analysis (GPU failure every 3h)
        - MLPerf benchmark power profiles
        - Google TPU workload characterization
        """
        if workload not in WORKLOAD_TYPES:
            return 0.5
        
        params = WORKLOAD_TYPES[workload]
        base_util = params['base_utilization']
        peak_util = params['peak_utilization'] 
        burst_freq = params['burst_frequency']
        
        if workload == 'llm':
            # LLM 추론: 사용자 트래픽 패턴 (강한 일간 변동)
            # 새벽 2-5시 최저(20%), 오전 9시 급상승, 14-16시 피크(95%), 자정까지 감소
            if 2 <= hour <= 5:
                time_factor = 0.15 + 0.10 * np.random.random()  # 15-25%
            elif 6 <= hour <= 8:
                time_factor = 0.25 + 0.25 * (hour - 6) / 2  # 급상승 25→50%
            elif 9 <= hour <= 11:
                time_factor = 0.55 + 0.25 * (hour - 9) / 2  # 55→80%
            elif 12 <= hour <= 13:
                time_factor = 0.70 + 0.10 * np.random.random()  # 점심 소폭 감소
            elif 14 <= hour <= 16:
                time_factor = 0.85 + 0.15 * np.random.random()  # 피크 85-100%
            elif 17 <= hour <= 19:
                time_factor = 0.65 + 0.15 * np.random.random()  # 퇴근 후 감소
            elif 20 <= hour <= 23:
                time_factor = 0.35 + 0.20 * np.random.random()  # 야간 35-55%
            else:  # 0-1시
                time_factor = 0.25 + 0.15 * np.random.random()  # 심야
            
            # LLM burst: Poisson arrival로 급격한 스파이크 (viral content, API surge)
            burst_prob = 0.15  # 시간당 15% 확률로 대형 burst
            if np.random.random() < burst_prob:
                burst_intensity = 0.6 + 0.4 * np.random.random()  # 강한 burst
                time_factor = min(1.0, time_factor + burst_intensity * (1.0 - time_factor))
            
        elif workload == 'training':
            # Training: 높은 base load + 30분 주기 checkpoint spike
            # 야간 배치 job 시작(22시), 새벽에 최대, 오전에 일부 종료
            if 22 <= hour or hour <= 6:
                time_factor = 0.85 + 0.10 * np.random.random()  # 야간 고부하 85-95%
            elif 7 <= hour <= 9:
                time_factor = 0.70 + 0.15 * np.random.random()  # 오전 전환기
            elif 10 <= hour <= 16:
                time_factor = 0.60 + 0.15 * np.random.random()  # 주간 중부하 60-75%
            else:  # 17-21시
                time_factor = 0.70 + 0.10 * np.random.random()  # 야간 배치 준비
            
            # Checkpoint spike: 매 시간 50% 확률 (30분 주기 checkpoint 중 하나)
            if np.random.random() < 0.50:
                # Checkpoint 시 GPU→CPU→Storage 대규모 I/O → 전력 spike 10-20%
                spike = 0.10 + 0.15 * np.random.random()
                time_factor = min(1.0, time_factor + spike)
            
            # GPU failure recovery: Meta 기준 3시간당 1회 → 시간당 33% 확률
            if np.random.random() < 0.05:  # 5% 확률로 부분 장애 → 전력 일시 감소
                time_factor *= (0.70 + 0.15 * np.random.random())
            
        else:  # MoE (Mixture of Experts)
            # MoE: Expert 활성화 패턴이 불규칙, 20-30%만 동시 활성
            # base가 낮지만 burst가 극심 (특정 expert 집중 활성화)
            base_moe = 0.30 + 0.15 * np.random.random()  # base 30-45%
            
            # 시간대별 약한 변동 (MoE는 추론+학습 혼합)
            diurnal = 0.10 * math.sin(math.pi * (hour - 6) / 12)
            time_factor = base_moe + diurnal
            
            # Expert activation burst: 높은 빈도, 큰 진폭
            if np.random.random() < 0.25:  # 25% 확률 대형 burst
                burst_size = 0.3 + 0.4 * np.random.random()  # 30-70% 증가
                time_factor = min(1.0, time_factor + burst_size)
            elif np.random.random() < 0.35:  # 추가 35% 확률 중형 burst
                burst_size = 0.10 + 0.15 * np.random.random()
                time_factor = min(1.0, time_factor + burst_size)
        
        # 요일별 패턴
        if day_of_week >= 5:  # 주말
            if workload == 'llm':
                weekday_factor = 0.70 + 0.10 * np.random.random()
            elif workload == 'training':
                weekday_factor = 1.05
            else:
                weekday_factor = 0.85
        else:
            weekday_factor = 1.0
        
        # time_factor가 이미 절대 활용률 → base_util은 스케일링용 (peak 기준)
        utilization = peak_util * time_factor * weekday_factor
        
        # 현실적 노이즈 (±8%, GPU 온도/throttling 등)
        noise = 0.08 * (2 * np.random.random() - 1)
        utilization += noise
        
        return float(np.clip(utilization, 0.05, 1.0))
    
    def simulate_minute_resolution(self,
                                  hour_of_day: int = 14,
                                  day_of_week: int = 2,
                                  minutes: int = 60,
                                  random_seed: Optional[int] = None) -> List[Dict]:
        """
        분단위 고해상도 시뮬레이션 (1시간 줌인용)
        
        실제 AIDC 전력 변동:
        - LLM 추론 burst: 100ms~1s 단위 (여기선 분 단위로 근사)
        - Training checkpoint: ~30초 spike
        - GPU throttling: 수초 단위 전력 감소
        - Cooling fan ramp: 10-30초 응답
        
        Returns:
            분단위 부하 리스트 [{minute, total_power_mw, gpu_utilization, event}, ...]
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 해당 시간의 base load 계산
        base_load = self.calculate_load_at_time(hour_of_day, day_of_week)
        base_power = base_load['total_power_mw']
        base_util = base_load['gpu_utilization']
        
        results = []
        current_power = base_power
        current_util = base_util
        
        # 이벤트 스케줄 생성
        # Checkpoint: 매 30분마다 (분 0, 30 근처)
        checkpoint_minutes = set()
        for cp in [0, 30]:
            for offset in range(-1, 3):
                checkpoint_minutes.add(cp + offset)
        
        for m in range(minutes):
            event = "normal"
            power_mod = 0.0
            util_mod = 0.0
            
            # 1) LLM burst (Poisson arrival, 분당 ~20% 확률)
            if np.random.random() < 0.20:
                burst = np.random.exponential(0.08) * base_power  # 평균 8% spike
                power_mod += burst
                util_mod += burst / base_power * base_util if base_power > 0 else 0
                event = "llm_burst"
            
            # 2) Training checkpoint spike (30분 주기)
            if m % 60 in checkpoint_minutes and self.workload_mix.get('training', 0) > 0:
                cp_spike = (0.10 + 0.10 * np.random.random()) * base_power * self.workload_mix['training']
                power_mod += cp_spike
                util_mod += 0.05 + 0.05 * np.random.random()
                event = "checkpoint"
            
            # 3) MoE expert activation (불규칙, 분당 15% 확률)
            if np.random.random() < 0.15 and self.workload_mix.get('moe', 0) > 0:
                expert_burst = (0.05 + 0.15 * np.random.random()) * base_power * self.workload_mix['moe']
                power_mod += expert_burst
                event = "expert_activation" if event == "normal" else event
            
            # 4) GPU thermal throttling (분당 5% 확률, 전력 감소)
            if np.random.random() < 0.05:
                throttle = -(0.03 + 0.05 * np.random.random()) * base_power
                power_mod += throttle
                event = "throttling"
            
            # 5) Micro-fluctuation (항상, ±3%)
            noise = base_power * 0.03 * (2 * np.random.random() - 1)
            power_mod += noise
            
            # 6) 가끔 큰 drop (GPU failure/restart, 분당 1%)
            if np.random.random() < 0.01:
                drop = -(0.10 + 0.10 * np.random.random()) * base_power
                power_mod += drop
                event = "gpu_failure"
            
            minute_power = max(0.05 * self.max_total_power_mw, current_power + power_mod)
            minute_util = np.clip(current_util + util_mod + 0.03 * (2 * np.random.random() - 1), 0.05, 1.0)
            
            # Smoothing: 현재값과 새값의 가중평균 (관성)
            alpha = 0.6  # 새 값 비중
            current_power = alpha * minute_power + (1 - alpha) * current_power
            current_util = alpha * minute_util + (1 - alpha) * current_util
            
            results.append({
                'minute': m,
                'total_power_mw': float(current_power),
                'gpu_utilization': float(current_util),
                'event': event
            })
        
        return results

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