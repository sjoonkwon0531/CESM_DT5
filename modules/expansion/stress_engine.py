"""
DT5 확장 - 스트레스 테스트 엔진
3-Way 시스템 비교: 기존그리드 vs 스마트그리드 vs CEMS 마이크로그리드
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """스트레스 시나리오 정의"""
    scenario_id: str
    name: str
    description: str
    intensity: float  # 0-1 scale
    duration_hours: int
    parameters: Dict[str, Any]


@dataclass 
class SystemKPI:
    """시스템 성능 지표"""
    robustness_score: float  # 0-100
    recovery_time_s: float
    max_power_deviation_pct: float
    outage_duration_s: float = 0
    disruption_cost_usd: float = 0


class PowerSystem(ABC):
    """전력 시스템 추상 기본 클래스"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.max_capacity_mw = config.get('max_capacity_mw', 100)
        
    @abstractmethod
    def calculate_supply(self, demand_mw: np.ndarray, stress_factors: Dict[str, Any]) -> np.ndarray:
        """스트레스 상황에서 전력 공급 계산"""
        pass
    
    @abstractmethod 
    def get_response_time_s(self) -> float:
        """시스템 응답시간"""
        pass
    
    @abstractmethod
    def calculate_backup_duration_s(self, load_mw: float) -> float:
        """백업 지속 시간 계산"""
        pass


class LegacyGrid(PowerSystem):
    """Ref1: 기존 그리드 — 한전 계통 할당, UPS만"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("기존 그리드", config)
        self.contract_capacity_mw = config.get('contract_capacity_mw', 80)  # 계약전력
        self.ups_capacity_kwh = config.get('ups_capacity_kwh', 1000)
        self.ups_efficiency = config.get('ups_efficiency', 0.9)
        
    def calculate_supply(self, demand_mw: np.ndarray, stress_factors: Dict[str, Any]) -> np.ndarray:
        """계약전력 제한, 그리드 차단 시 UPS만 활용"""
        grid_outage = stress_factors.get('grid_outage', 0.0)  # 0-1 정전 강도
        
        # 기본 공급: 계약전력 제한
        base_supply = np.minimum(demand_mw, self.contract_capacity_mw)
        
        # 그리드 차단 시 UPS 백업 (제한된 시간)
        if grid_outage > 0:
            ups_duration_h = self.calculate_backup_duration_s(demand_mw.mean()) / 3600
            supply = base_supply * (1 - grid_outage)  # 부분 정전
            # UPS 백업 처리는 별도 로직에서 (시간 제한)
        else:
            supply = base_supply
            
        return supply
    
    def get_response_time_s(self) -> float:
        """Legacy 시스템 응답시간 분해 모델"""
        # 구성요소별 지연시간
        detection_time = 30      # SCADA polling (30초)
        decision_time = 60       # 수동/반자동 판단 (60초)
        ups_switching = 10       # ATS 동작 (10초)
        grid_recovery = np.random.uniform(300, 600)  # 한전 긴급복구 (5-10분)
        
        total_response = detection_time + decision_time + ups_switching + grid_recovery
        return total_response  # 400-700초
    
    def calculate_backup_duration_s(self, load_mw: float) -> float:
        """UPS 백업 지속시간"""
        if load_mw <= 0:
            return float('inf')
        return (self.ups_capacity_kwh * self.ups_efficiency) / load_mw * 3600  # 초 변환
    
    def get_response_breakdown(self) -> Dict[str, float]:
        """응답시간 구성요소 상세 반환"""
        return {
            'detection_time': 30,
            'decision_time': 60,
            'ups_switching': 10,
            'grid_recovery_min': 300,
            'grid_recovery_max': 600,
            'total_min': 400,
            'total_max': 700
        }


class SmartGrid(PowerSystem):
    """Ref2: 스마트그리드 — AMI + DR (그리드위즈 수준)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("스마트그리드", config)
        self.dr_participation_rate = config.get('dr_participation', 0.7)  # 70% 참여
        self.dr_reduction_rate = config.get('dr_reduction', 0.2)  # 20% 평균 감축
        self.ups_capacity_kwh = config.get('ups_capacity_kwh', 2000)  # UPS 용량 증대
        
    def calculate_supply(self, demand_mw: np.ndarray, stress_factors: Dict[str, Any]) -> np.ndarray:
        """그리드 + DR 조합"""
        grid_outage = stress_factors.get('grid_outage', 0.0)
        
        # 기본 그리드 공급
        grid_supply = demand_mw * (1 - grid_outage)
        
        # DR 자원 활용 (부족분 보상)
        shortage = np.maximum(0, demand_mw - grid_supply)
        dr_available = demand_mw * self.dr_participation_rate * self.dr_reduction_rate
        dr_used = np.minimum(shortage, dr_available)
        
        total_supply = grid_supply + dr_used
        return total_supply
    
    def get_response_time_s(self) -> float:
        """Smart 시스템 응답시간 분해 모델"""
        # 구성요소별 지연시간
        detection_time = 5       # AMI 실시간 감지 (5초)
        dr_activation = 300      # DR 발동 요청 (5분 응답)
        load_reduction = 60      # 부하 감축 점진적 (60초)
        
        total_response = detection_time + dr_activation + load_reduction
        return total_response  # 약 365초
    
    def calculate_backup_duration_s(self, load_mw: float) -> float:
        """개선된 UPS + DR 조합"""
        if load_mw <= 0:
            return float('inf')
        ups_duration = (self.ups_capacity_kwh * 0.9) / load_mw * 3600
        dr_duration = 4 * 3600  # DR 2-4시간 지속 가능
        return ups_duration + dr_duration
    
    def get_response_breakdown(self) -> Dict[str, float]:
        """응답시간 구성요소 상세 반환"""
        return {
            'detection_time': 5,
            'dr_activation': 300,
            'load_reduction': 60,
            'total': 365
        }


class CEMSMicrogrid(PowerSystem):
    """Ours: 기존 DT5 코어 모듈 활용"""
    
    def __init__(self, config: Dict[str, Any], dt5_modules: Optional[Dict] = None):
        super().__init__("CEMS 마이크로그리드", config)
        self.dt5_modules = dt5_modules or {}
        
        # HESS 5계층 구성
        self.supercap_kwh = config.get('supercap_kwh', 50)
        self.bess_kwh = config.get('bess_kwh', 10000)
        self.ups_capacity_kwh = config.get('ups_capacity_kwh', 3000)
        
        # PV + BESS 결합
        self.pv_capacity_mw = config.get('pv_capacity_mw', 100)
        self.grid_backup_mw = config.get('grid_backup_mw', 20)
        
    def calculate_supply(self, demand_mw: np.ndarray, stress_factors: Dict[str, Any]) -> np.ndarray:
        """PV + 5계층 HESS + Grid 백업"""
        # PV 출력 (스트레스 적용)
        pv_reduction = stress_factors.get('pv_reduction', 0.0)
        pv_output = np.full_like(demand_mw, self.pv_capacity_mw * 0.25, dtype=float)  # 25% CF 가정
        pv_output *= (1 - pv_reduction)
        
        # BESS 방전 (부족분 보상)
        shortage = np.maximum(0, demand_mw - pv_output)
        bess_available_mw = min(50, self.bess_kwh / 4)  # 4시간 방전율
        bess_output = np.minimum(shortage, bess_available_mw)
        
        # Grid 백업 (최종 보상)
        grid_outage = stress_factors.get('grid_outage', 0.0)
        remaining_shortage = shortage - bess_output
        grid_backup = remaining_shortage * (1 - grid_outage)
        
        total_supply = pv_output + bess_output + grid_backup
        return total_supply
    
    def get_response_time_s(self) -> float:
        """CEMS 시스템 응답시간 분해 모델"""
        # 구성요소별 지연시간 (마이크로초/밀리초 단위)
        detection_time = 0.001   # μs 센싱 (1ms)
        supercap_response = 0.001  # Supercap ms급 응답 (1ms)
        bess_switching = np.random.uniform(1, 3)  # BESS 전환 (1-3초)
        ai_optimization = np.random.uniform(2, 5)  # AI-EMS 최적화 (2-5초)
        
        total_response = detection_time + supercap_response + bess_switching + ai_optimization
        return total_response  # 3-8초
    
    def calculate_backup_duration_s(self, load_mw: float) -> float:
        """5계층 HESS 통합 백업시간"""
        if load_mw <= 0:
            return float('inf')
            
        # 1단계: Supercap (초 단위)
        supercap_duration = self.supercap_kwh / load_mw * 3600
        
        # 2단계: BESS (시간 단위)  
        bess_duration = self.bess_kwh / load_mw * 3600
        
        # 3단계: UPS (추가 백업)
        ups_duration = self.ups_capacity_kwh / load_mw * 3600
        
        return supercap_duration + bess_duration + ups_duration
    
    def get_response_breakdown(self) -> Dict[str, float]:
        """응답시간 구성요소 상세 반환"""
        return {
            'detection_time': 0.001,
            'supercap_response': 0.001,
            'bess_switching_min': 1,
            'bess_switching_max': 3,
            'ai_optimization_min': 2,
            'ai_optimization_max': 5,
            'total_min': 3,
            'total_max': 8
        }


class StressTestEngine:
    """스트레스 테스트 엔진 메인 클래스"""
    
    def __init__(self, dt5_modules: Optional[Dict] = None):
        self.dt5_modules = dt5_modules or {}
        self.systems = {}
        self.results = {}
        
    def initialize_systems(self, config: Dict[str, Any]):
        """3개 시스템 초기화"""
        self.systems = {
            'legacy': LegacyGrid(config.get('legacy', {})),
            'smart': SmartGrid(config.get('smart', {})),
            'cems': CEMSMicrogrid(config.get('cems', {}), self.dt5_modules)
        }
        
    def create_scenario_library(self) -> Dict[str, StressScenario]:
        """4개 스트레스 시나리오 라이브러리"""
        scenarios = {
            'S1': StressScenario(
                scenario_id='S1',
                name='GPU 워크로드 급증',
                description='Poisson burst로 GPU 부하 30-80% 급증',
                intensity=0.5,
                duration_hours=2,
                parameters={
                    'gpu_burst_multiplier': 1.5,
                    'burst_duration_min': 30
                }
            ),
            'S2': StressScenario(
                scenario_id='S2', 
                name='PV 급감',
                description='구름/고장으로 PV 출력 50-80% 감소',
                intensity=0.6,
                duration_hours=4,
                parameters={
                    'pv_reduction_factor': 0.6,
                    'reduction_pattern': 'sudden'
                }
            ),
            'S3': StressScenario(
                scenario_id='S3',
                name='그리드 차단', 
                description='부분/완전 정전 상황',
                intensity=0.8,
                duration_hours=6,
                parameters={
                    'grid_outage_factor': 0.8,
                    'outage_pattern': 'complete'
                }
            ),
            'S4': StressScenario(
                scenario_id='S4',
                name='S1+S2 복합',
                description='GPU 급증 + PV 급감 동시 발생',
                intensity=0.9,
                duration_hours=3,
                parameters={
                    'gpu_burst_multiplier': 1.3,
                    'pv_reduction_factor': 0.5
                }
            )
        }
        return scenarios
    
    def generate_demand_profile(self, scenario: StressScenario, base_demand_mw: float = 80) -> np.ndarray:
        """시나리오별 수요 프로파일 생성"""
        time_points = scenario.duration_hours * 60  # 1분 해상도
        t = np.linspace(0, scenario.duration_hours, time_points)
        
        # 기본 수요 패턴
        base_profile = base_demand_mw * (1 + 0.1 * np.sin(2 * np.pi * t / 24))  # 일일 패턴
        
        # 시나리오별 스트레스 적용
        if 'gpu_burst' in scenario.parameters or 'gpu_burst_multiplier' in scenario.parameters:
            # Poisson burst 모델링
            burst_points = np.random.poisson(2, size=time_points)  # 평균 2회/시간
            burst_multiplier = scenario.parameters.get('gpu_burst_multiplier', 1.3)
            
            for i, burst in enumerate(burst_points):
                if burst > 0:
                    burst_duration = min(30, time_points - i)  # 최대 30분
                    end_idx = i + burst_duration
                    base_profile[i:end_idx] *= burst_multiplier
        
        return base_profile
    
    def run_stress_test(self, scenario: StressScenario, systems: List[str] = None) -> Dict[str, Any]:
        """스트레스 테스트 실행"""
        if systems is None:
            systems = list(self.systems.keys())
        
        # 수요 프로파일 생성
        demand_profile = self.generate_demand_profile(scenario)
        time_points = len(demand_profile)
        
        results = {}
        
        for system_name in systems:
            system = self.systems[system_name]
            logger.info(f"Running stress test for {system.name}")
            
            # 스트레스 팩터 준비
            stress_factors = {
                'pv_reduction': scenario.parameters.get('pv_reduction_factor', 0.0),
                'grid_outage': scenario.parameters.get('grid_outage_factor', 0.0)
            }
            
            # 전력 공급 계산
            supply_profile = system.calculate_supply(demand_profile, stress_factors)
            
            # KPI 계산
            kpi = self.calculate_system_kpi(demand_profile, supply_profile, system)
            
            results[system_name] = {
                'system': system,
                'demand_profile': demand_profile,
                'supply_profile': supply_profile,
                'kpi': kpi,
                'scenario': scenario
            }
        
        return results
    
    def calculate_system_kpi(self, demand: np.ndarray, supply: np.ndarray, system: PowerSystem) -> SystemKPI:
        """시스템 KPI 계산"""
        # Power deviation
        deviation = np.abs(supply - demand) / np.maximum(demand, 1e-6)
        max_deviation_pct = np.max(deviation) * 100
        
        # Outage detection
        outage_threshold = 0.95  # 95% 미만이면 정전으로 간주
        outage_mask = (supply / np.maximum(demand, 1e-6)) < outage_threshold
        outage_duration_s = np.sum(outage_mask) * 60  # 1분 해상도
        
        # Recovery time (정전 후 정상 복귀까지)
        recovery_time_s = self.calculate_recovery_time(supply, demand)
        
        # Robustness score
        survival_rate = 1 - (outage_duration_s / (len(demand) * 60))
        deviation_penalty = max_deviation_pct / 100 * 0.3
        robustness_score = max(0, (survival_rate - deviation_penalty) * 100)
        
        # Disruption cost
        disruption_cost_usd = outage_duration_s / 60 * 9000  # $9K/min (Tier IV DC 기준)
        
        return SystemKPI(
            robustness_score=robustness_score,
            recovery_time_s=recovery_time_s,
            max_power_deviation_pct=max_deviation_pct,
            outage_duration_s=outage_duration_s,
            disruption_cost_usd=disruption_cost_usd
        )
    
    def calculate_recovery_time(self, supply: np.ndarray, demand: np.ndarray) -> float:
        """복구 시간 계산"""
        normal_threshold = 0.95
        supply_ratio = supply / np.maximum(demand, 1e-6)
        
        # 첫 번째 정전 지점 찾기
        outage_points = np.where(supply_ratio < normal_threshold)[0]
        if len(outage_points) == 0:
            return 0.0  # 정전 없음
        
        first_outage = outage_points[0]
        
        # 복구 지점 찾기 (정전 이후 정상 복귀)
        post_outage = supply_ratio[first_outage:]
        recovery_points = np.where(post_outage >= normal_threshold)[0]
        
        if len(recovery_points) == 0:
            return float('inf')  # 복구 실패
        
        recovery_time_minutes = recovery_points[0]
        return recovery_time_minutes * 60  # 초 변환
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """3-Way 비교 리포트 생성"""
        systems = list(results.keys())
        comparison = {
            'scenario': results[systems[0]]['scenario'],
            'systems': systems,
            'kpi_comparison': {},
            'winner': {},
            'summary': {}
        }
        
        # KPI별 비교
        kpi_metrics = ['robustness_score', 'recovery_time_s', 'max_power_deviation_pct', 'disruption_cost_usd']
        
        for metric in kpi_metrics:
            comparison['kpi_comparison'][metric] = {}
            values = []
            
            for system in systems:
                value = getattr(results[system]['kpi'], metric)
                comparison['kpi_comparison'][metric][system] = value
                values.append((system, value))
            
            # Winner 결정 (낮을수록 좋은 지표들은 reverse)
            if metric in ['recovery_time_s', 'max_power_deviation_pct', 'disruption_cost_usd']:
                winner = min(values, key=lambda x: x[1])
            else:
                winner = max(values, key=lambda x: x[1])
            
            comparison['winner'][metric] = winner[0]
        
        # 종합 우승자
        cems_wins = sum(1 for winner in comparison['winner'].values() if winner == 'cems')
        total_metrics = len(kpi_metrics)
        
        comparison['summary'] = {
            'overall_winner': 'cems' if cems_wins >= total_metrics / 2 else 'contested',
            'cems_win_rate': cems_wins / total_metrics,
            'key_advantages': self._identify_key_advantages(comparison['kpi_comparison'])
        }
        
        return comparison
    
    def _identify_key_advantages(self, kpi_comparison: Dict) -> Dict[str, List[str]]:
        """시스템별 주요 장점 식별"""
        advantages = {system: [] for system in ['legacy', 'smart', 'cems']}
        
        for metric, values in kpi_comparison.items():
            best_value = min(values.values()) if metric in ['recovery_time_s', 'max_power_deviation_pct', 'disruption_cost_usd'] else max(values.values())
            
            for system, value in values.items():
                if value == best_value:
                    metric_name = {
                        'robustness_score': '강건성',
                        'recovery_time_s': '복구시간',
                        'max_power_deviation_pct': '전력안정성', 
                        'disruption_cost_usd': '비용효율성'
                    }.get(metric, metric)
                    advantages[system].append(metric_name)
        
        return advantages

# TODO: Phase 2 - GPU degradation integration
# TODO: Phase 2 - Cascading failure modeling
# TODO: Phase 2 - More sophisticated Poisson burst patterns
# TODO: Phase 2 - Weather correlation for PV scenarios