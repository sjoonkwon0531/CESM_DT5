"""
DT5 확장 - 통합 분석 엔진
스트레스 테스트 + 데이터 생존성 통합 분석
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from .stress_engine import StressTestEngine, StressScenario
from .data_survival import DataSurvivalAnalyzer, EnergySLACalculator, DEFAULT_SYSTEM_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class UnifiedKPI:
    """통합 성과 지표"""
    # 스트레스 테스트 KPI
    robustness_score: float
    recovery_time_s: float
    max_power_deviation_pct: float
    
    # 데이터 생존성 KPI
    data_survival_rate: float
    t2_total_min: float
    data_loss_cost_usd: float
    
    # 에너지 SLA KPI
    tier_4_compliant: bool
    availability_achieved: float
    
    # 통합 점수
    overall_score: float


class UnifiedExpansionAnalytics:
    """DT5 확장 통합 분석 엔진"""
    
    def __init__(self, dt5_modules: Optional[Dict] = None):
        self.dt5_modules = dt5_modules or {}
        
        # 서브 분석기들
        self.stress_engine = StressTestEngine(dt5_modules)
        self.data_analyzer = DataSurvivalAnalyzer({
            'gpu_count': 50000,
            'hbm_per_gpu_gb': 80,
            'hbm_utilization': 0.8,
            'ssd_count': 1000,
            'ssd_write_bw_gb_s': 5.5,
            'checkpoint_interval_min': 15
        })
        self.sla_calculator = EnergySLACalculator()
        
        # 결과 저장
        self.unified_results = {}
        
    def run_comprehensive_analysis(self, 
                                 scenario_configs: List[Dict[str, Any]], 
                                 system_configs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """종합 분석 실행"""
        if system_configs is None:
            system_configs = DEFAULT_SYSTEM_CONFIGS
        
        logger.info("Starting comprehensive DT5 expansion analysis")
        
        # 시스템 초기화
        self.stress_engine.initialize_systems(system_configs)
        
        comprehensive_results = {
            'stress_tests': {},
            'data_survival': {},
            'energy_sla': {},
            'unified_kpi': {},
            'executive_summary': {}
        }
        
        # 1. 스트레스 테스트 실행
        for scenario_config in scenario_configs:
            scenario = self._create_scenario_from_config(scenario_config)
            stress_results = self.stress_engine.run_stress_test(scenario)
            comparison_report = self.stress_engine.generate_comparison_report(stress_results)
            
            comprehensive_results['stress_tests'][scenario.scenario_id] = {
                'results': stress_results,
                'comparison': comparison_report
            }
        
        # 2. 데이터 생존성 분석
        data_survival_results = self.data_analyzer.compare_three_systems(system_configs)
        comprehensive_results['data_survival'] = data_survival_results
        
        # 3. 에너지 SLA 분석
        sla_results = self.sla_calculator.calculate_energy_sla(data_survival_results)
        comprehensive_results['energy_sla'] = sla_results
        
        # 4. 통합 KPI 계산
        unified_kpi = self._calculate_unified_kpi(comprehensive_results)
        comprehensive_results['unified_kpi'] = unified_kpi
        
        # 5. 경영진 요약
        executive_summary = self._generate_executive_summary(comprehensive_results)
        comprehensive_results['executive_summary'] = executive_summary
        
        self.unified_results = comprehensive_results
        
        logger.info("Comprehensive analysis completed")
        return comprehensive_results
    
    def _create_scenario_from_config(self, config: Dict[str, Any]) -> StressScenario:
        """설정에서 시나리오 생성"""
        return StressScenario(
            scenario_id=config.get('scenario_id', 'S1'),
            name=config.get('name', 'Custom Scenario'),
            description=config.get('description', ''),
            intensity=config.get('intensity', 0.5),
            duration_hours=config.get('duration_hours', 2),
            parameters=config.get('parameters', {})
        )
    
    def _calculate_unified_kpi(self, results: Dict[str, Any]) -> Dict[str, UnifiedKPI]:
        """통합 KPI 계산"""
        unified_kpi = {}
        systems = ['legacy', 'smart', 'cems']
        
        for system in systems:
            # 스트레스 테스트 결과 (대표 시나리오 S1 사용)
            if 'S1' in results['stress_tests']:
                stress_kpi = results['stress_tests']['S1']['results'][system]['kpi']
                robustness_score = stress_kpi.robustness_score
                recovery_time_s = stress_kpi.recovery_time_s
                max_deviation = stress_kpi.max_power_deviation_pct
            else:
                robustness_score = 0
                recovery_time_s = float('inf')
                max_deviation = 100
            
            # 데이터 생존성 결과
            if system in results['data_survival']:
                survival_result = results['data_survival'][system]['survival_result']
                t2_components = results['data_survival'][system]['t2_components']
                
                data_survival_rate = survival_result.data_survival_rate
                t2_total_min = t2_components.total_t2_s / 60
                data_loss_cost = survival_result.data_loss_cost_usd
            else:
                data_survival_rate = 0
                t2_total_min = 0
                data_loss_cost = float('inf')
            
            # 에너지 SLA 결과
            if system in results['energy_sla']:
                tier_4_sla = results['energy_sla'][system]['tier_4']
                tier_4_compliant = tier_4_sla.compliant
                availability_achieved = tier_4_sla.achieved_availability
            else:
                tier_4_compliant = False
                availability_achieved = 0
            
            # 통합 점수 계산 (0-100)
            overall_score = self._calculate_overall_score(
                robustness_score, recovery_time_s, max_deviation,
                data_survival_rate, tier_4_compliant, availability_achieved
            )
            
            unified_kpi[system] = UnifiedKPI(
                robustness_score=robustness_score,
                recovery_time_s=recovery_time_s,
                max_power_deviation_pct=max_deviation,
                data_survival_rate=data_survival_rate,
                t2_total_min=t2_total_min,
                data_loss_cost_usd=data_loss_cost,
                tier_4_compliant=tier_4_compliant,
                availability_achieved=availability_achieved,
                overall_score=overall_score
            )
        
        return unified_kpi
    
    def _calculate_overall_score(self, robustness: float, recovery_time: float, deviation: float,
                               survival_rate: float, tier_4_ok: bool, availability: float) -> float:
        """통합 점수 계산 (가중 평균)"""
        # 정규화
        robustness_norm = robustness / 100  # 0-1
        recovery_norm = max(0, 1 - recovery_time / 3600)  # 1시간 기준으로 정규화
        deviation_norm = max(0, 1 - deviation / 50)  # 50% 기준으로 정규화
        survival_norm = survival_rate  # 이미 0-1
        tier4_norm = 1.0 if tier_4_ok else 0.0
        availability_norm = availability  # 0-1
        
        # 가중치 적용
        weights = {
            'robustness': 0.2,
            'recovery': 0.15,
            'deviation': 0.15,
            'survival': 0.2,
            'tier4': 0.15,
            'availability': 0.15
        }
        
        overall_score = (
            weights['robustness'] * robustness_norm +
            weights['recovery'] * recovery_norm +
            weights['deviation'] * deviation_norm +
            weights['survival'] * survival_norm +
            weights['tier4'] * tier4_norm +
            weights['availability'] * availability_norm
        ) * 100
        
        return min(100, max(0, overall_score))
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """경영진 요약 보고서"""
        unified_kpi = results['unified_kpi']
        
        # 시스템별 점수 정렬
        system_scores = [(system, kpi.overall_score) for system, kpi in unified_kpi.items()]
        system_scores.sort(key=lambda x: x[1], reverse=True)
        
        # CEMS 우위 분석
        cems_score = unified_kpi.get('cems', UnifiedKPI(0,0,0,0,0,0,False,0,0)).overall_score
        legacy_score = unified_kpi.get('legacy', UnifiedKPI(0,0,0,0,0,0,False,0,0)).overall_score
        smart_score = unified_kpi.get('smart', UnifiedKPI(0,0,0,0,0,0,False,0,0)).overall_score
        
        cems_vs_legacy = cems_score - legacy_score
        cems_vs_smart = cems_score - smart_score
        
        # 주요 우위 영역 식별
        cems_kpi = unified_kpi.get('cems')
        advantages = []
        
        if cems_kpi:
            if cems_kpi.robustness_score > 90:
                advantages.append("시스템 강건성 (94점 이상)")
            if cems_kpi.t2_total_min > 60:
                advantages.append(f"데이터 백업 여유시간 ({cems_kpi.t2_total_min:.1f}분)")
            if cems_kpi.tier_4_compliant:
                advantages.append("Tier IV 에너지 SLA 달성")
            if cems_kpi.data_survival_rate > 0.99:
                advantages.append(f"데이터 생존율 ({cems_kpi.data_survival_rate:.1%})")
        
        # ROI 계산 (간단 모델)
        annual_savings = self._calculate_annual_savings(results)
        investment_cost = 15_000_000  # 15억원 (가정)
        roi_years = investment_cost / annual_savings if annual_savings > 0 else float('inf')
        
        executive_summary = {
            'overall_winner': system_scores[0][0] if system_scores else 'unknown',
            'winner_score': system_scores[0][1] if system_scores else 0,
            'cems_advantages': {
                'vs_legacy': cems_vs_legacy,
                'vs_smart': cems_vs_smart,
                'key_strengths': advantages
            },
            'business_impact': {
                'annual_savings_krw': annual_savings,
                'roi_years': roi_years,
                'risk_reduction': self._calculate_risk_reduction(results)
            },
            'recommendations': self._generate_recommendations(results)
        }
        
        return executive_summary
    
    def _calculate_annual_savings(self, results: Dict[str, Any]) -> float:
        """연간 비용 절감 계산"""
        # 데이터 손실 비용 절감
        cems_loss_cost = results['unified_kpi'].get('cems', UnifiedKPI(0,0,0,0,0,0,False,0,0)).data_loss_cost_usd
        legacy_loss_cost = results['unified_kpi'].get('legacy', UnifiedKPI(0,0,0,0,0,0,False,0,0)).data_loss_cost_usd
        
        # 연간 12회 이벤트 가정
        annual_events = 12
        data_loss_savings_usd = (legacy_loss_cost - cems_loss_cost) * annual_events
        
        # 정전 비용 절감 (스트레스 테스트 기반)
        if 'S3' in results['stress_tests']:  # 그리드 차단 시나리오
            cems_disruption = results['stress_tests']['S3']['results'].get('cems', {}).get('kpi', type('', (), {'disruption_cost_usd': 0})()).disruption_cost_usd
            legacy_disruption = results['stress_tests']['S3']['results'].get('legacy', {}).get('kpi', type('', (), {'disruption_cost_usd': 0})()).disruption_cost_usd
            
            disruption_savings_usd = (legacy_disruption - cems_disruption) * annual_events
        else:
            disruption_savings_usd = 0
        
        total_savings_usd = data_loss_savings_usd + disruption_savings_usd
        total_savings_krw = total_savings_usd * 1300  # USD → KRW 환율
        
        return total_savings_krw
    
    def _calculate_risk_reduction(self, results: Dict[str, Any]) -> Dict[str, float]:
        """위험 감소 계산"""
        cems_kpi = results['unified_kpi'].get('cems')
        legacy_kpi = results['unified_kpi'].get('legacy')
        
        if not cems_kpi or not legacy_kpi:
            return {'data_loss_risk': 0, 'outage_risk': 0}
        
        data_loss_risk_reduction = (1 - legacy_kpi.data_survival_rate) - (1 - cems_kpi.data_survival_rate)
        outage_risk_reduction = (1 - legacy_kpi.availability_achieved) - (1 - cems_kpi.availability_achieved)
        
        return {
            'data_loss_risk': data_loss_risk_reduction,
            'outage_risk': outage_risk_reduction
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """실행 권고사항"""
        recommendations = []
        
        # CEMS 우위가 명확한 경우
        cems_kpi = results['unified_kpi'].get('cems')
        if cems_kpi and cems_kpi.overall_score > 80:
            recommendations.append("CEMS 마이크로그리드 도입을 적극 권장")
            
            if cems_kpi.tier_4_compliant:
                recommendations.append("Tier IV 데이터센터 표준 달성으로 프리미엄 고객 유치 가능")
            
            if cems_kpi.data_survival_rate > 0.99:
                recommendations.append("99%+ 데이터 생존율로 AI 훈련 안정성 확보")
        
        # 개선 영역 식별
        if cems_kpi and cems_kpi.recovery_time_s > 300:
            recommendations.append("복구 시간 단축을 위한 BESS 용량 증설 검토")
        
        # Phase 2 확장 권고
        recommendations.append("GPU 열화 예측 모듈 추가 개발로 예방 정비 구현")
        recommendations.append("실시간 모니터링 시스템과 연계한 자동 대응 체계 구축")
        
        return recommendations
    
    def get_key_visualizations(self) -> Dict[str, Any]:
        """핵심 시각화 데이터"""
        if not self.unified_results:
            return {}
        
        return {
            'kpi_radar_chart': self._prepare_kpi_radar_data(),
            't2_breakdown_chart': self._prepare_t2_breakdown_data(),
            'scenario_comparison': self._prepare_scenario_comparison_data(),
            'sla_compliance_table': self._prepare_sla_table_data()
        }
    
    def _prepare_kpi_radar_data(self) -> Dict[str, Any]:
        """KPI 레이더 차트 데이터"""
        unified_kpi = self.unified_results.get('unified_kpi', {})
        
        metrics = ['robustness_score', 'data_survival_rate', 'availability_achieved']
        systems = ['legacy', 'smart', 'cems']
        
        radar_data = {
            'metrics': [m.replace('_', ' ').title() for m in metrics],
            'systems': {}
        }
        
        for system in systems:
            if system in unified_kpi:
                kpi = unified_kpi[system]
                values = [
                    kpi.robustness_score,
                    kpi.data_survival_rate * 100,
                    kpi.availability_achieved * 100
                ]
                radar_data['systems'][system] = values
        
        return radar_data
    
    def _prepare_t2_breakdown_data(self) -> pd.DataFrame:
        """t2 분해 차트 데이터"""
        data_survival = self.unified_results.get('data_survival', {})
        rows = []
        
        for system, results in data_survival.items():
            components = results.get('t2_components')
            if components:
                rows.append({
                    'system': system,
                    'PSU (ms)': components.psu_holdup_s * 1000,
                    'UPS (min)': components.ups_backup_s / 60,
                    'BESS (min)': components.bess_emergency_s / 60,
                    'Total (min)': components.total_t2_s / 60
                })
        
        return pd.DataFrame(rows)
    
    def _prepare_scenario_comparison_data(self) -> Dict[str, Any]:
        """시나리오 비교 데이터"""
        stress_tests = self.unified_results.get('stress_tests', {})
        
        comparison_data = {}
        for scenario_id, test_results in stress_tests.items():
            comparison = test_results.get('comparison', {})
            if 'summary' in comparison:
                comparison_data[scenario_id] = {
                    'winner': comparison['summary'].get('overall_winner', 'unknown'),
                    'cems_win_rate': comparison['summary'].get('cems_win_rate', 0)
                }
        
        return comparison_data
    
    def _prepare_sla_table_data(self) -> pd.DataFrame:
        """SLA 준수 테이블 데이터"""
        sla_results = self.unified_results.get('energy_sla', {})
        
        if not sla_results:
            return pd.DataFrame()
        
        return self.sla_calculator.generate_sla_comparison_table(sla_results)

# TODO: Phase 2 - Real-time risk monitoring integration
# TODO: Phase 2 - Advanced ROI modeling with uncertainty analysis
# TODO: Phase 2 - Machine learning for predictive analytics
# TODO: Phase 2 - Integration with external data sources