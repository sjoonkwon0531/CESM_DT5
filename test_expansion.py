"""
DT5 확장 모듈 테스트 스크립트
기본 기능 검증 및 데모 실행
"""

import sys
import traceback
import numpy as np
from modules.expansion import *
from modules.expansion.data_survival import DEFAULT_SYSTEM_CONFIGS, EnergySLACalculator

def test_stress_engine():
    """스트레스 엔진 기본 테스트"""
    print("🌊 Testing Stress Engine...")
    
    try:
        # 스트레스 엔진 초기화
        engine = StressTestEngine()
        
        # 시스템 설정
        system_config = {
            'legacy': {'contract_capacity_mw': 80, 'ups_capacity_kwh': 1000},
            'smart': {'dr_participation': 0.7, 'ups_capacity_kwh': 2000},
            'cems': {'pv_capacity_mw': 100, 'bess_kwh': 10000, 'ups_capacity_kwh': 3000}
        }
        
        engine.initialize_systems(system_config)
        print("  ✅ Systems initialized")
        
        # 시나리오 라이브러리 테스트
        scenarios = engine.create_scenario_library()
        print(f"  ✅ Scenario library created: {len(scenarios)} scenarios")
        
        # S1 시나리오 테스트
        s1_scenario = scenarios['S1']
        results = engine.run_stress_test(s1_scenario)
        print(f"  ✅ S1 stress test completed: {len(results)} systems")
        
        # 비교 리포트 생성
        comparison = engine.generate_comparison_report(results)
        print(f"  ✅ Comparison report generated")
        print(f"     Overall winner: {comparison['summary']['overall_winner']}")
        print(f"     CEMS win rate: {comparison['summary']['cems_win_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Stress engine test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_data_survival():
    """데이터 생존성 기본 테스트"""
    print("💾 Testing Data Survival Analyzer...")
    
    try:
        # 분석기 초기화
        config = {
            'gpu_count': 50000,
            'hbm_per_gpu_gb': 80,
            'hbm_utilization': 0.8,
            'ssd_count': 1000,
            'ssd_write_bw_gb_s': 5.5,
            'checkpoint_interval_min': 15
        }
        
        analyzer = DataSurvivalAnalyzer(config)
        print("  ✅ Analyzer initialized")
        
        # 3-Way 시스템 비교
        system_configs = DEFAULT_SYSTEM_CONFIGS
        results = analyzer.compare_three_systems(system_configs)
        print(f"  ✅ 3-way comparison completed: {len(results)} systems")
        
        # 결과 출력 (보정된 값)
        for system, result in results.items():
            survival = result['survival_result']
            t2 = result['t2_components']
            mc_result = result['mc_simulation']
            
            print(f"     {system}:")
            print(f"       t2 = {t2.total_t2_s/60:.1f}분 (PSU:{t2.psu_holdup_s:.3f}s + UPS:{t2.ups_backup_s/60:.1f}분 + BESS:{t2.bess_emergency_s/60:.1f}분)")
            print(f"       생존율 = {survival.data_survival_rate:.1%}")
            print(f"       MC 시뮬레이션 = {mc_result['mean_survival_rate']:.1%} (95% CI: {mc_result['percentile_5']:.1%}-{mc_result['percentile_95']:.1%})")
        
        # CEMS 우위 계산
        legacy_t2 = results['legacy']['t2_components'].total_t2_s / 60
        cems_t2 = results['cems']['t2_components'].total_t2_s / 60
        advantage_ratio = cems_t2 / legacy_t2
        print(f"     CEMS vs Legacy 우위: {advantage_ratio:.1f}배 (현실적 범위: 7-13배)")
        
        # SLA 계산기 테스트
        sla_calc = EnergySLACalculator()
        sla_results = sla_calc.calculate_energy_sla(results, t3_seconds=600)
        print("  ✅ Energy SLA calculated")
        
        # Tier IV 준수 여부
        for system, sla_data in sla_results.items():
            tier4_sla = sla_data['tier_4']
            status = "✅" if tier4_sla.compliant else "❌"
            print(f"     {system} Tier IV: {status} ({tier4_sla.achieved_availability:.4%})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data survival test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_unified_analytics():
    """통합 분석 테스트"""
    print("📊 Testing Unified Analytics...")
    
    try:
        # 통합 분석기 초기화
        analytics = UnifiedExpansionAnalytics()
        print("  ✅ Analytics initialized")
        
        # 테스트 시나리오 구성
        scenario_configs = [
            {
                'scenario_id': 'S1',
                'name': 'GPU 워크로드 급증',
                'description': 'Test scenario',
                'intensity': 0.6,
                'duration_hours': 2,
                'parameters': {'gpu_burst_multiplier': 1.5}
            }
        ]
        
        # 종합 분석 실행
        results = analytics.run_comprehensive_analysis(scenario_configs)
        print("  ✅ Comprehensive analysis completed")
        
        # 결과 검증
        assert 'stress_tests' in results
        assert 'data_survival' in results
        assert 'energy_sla' in results
        assert 'unified_kpi' in results
        assert 'executive_summary' in results
        
        print("  ✅ All result sections present")
        
        # 주요 결과 출력
        executive = results['executive_summary']
        print(f"     Overall winner: {executive['overall_winner']}")
        print(f"     Winner score: {executive['winner_score']:.1f}")
        
        # CEMS 우위 출력
        cems_adv = executive['cems_advantages']
        print(f"     CEMS vs Legacy: +{cems_adv['vs_legacy']:.1f}")
        print(f"     CEMS vs Smart: +{cems_adv['vs_smart']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Unified analytics test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_system_integration():
    """시스템 통합 테스트"""
    print("🔧 Testing System Integration...")
    
    try:
        # Legacy vs CEMS 직접 비교
        legacy = LegacyGrid({'contract_capacity_mw': 80, 'ups_capacity_kwh': 1000})
        cems = CEMSMicrogrid({'pv_capacity_mw': 100, 'bess_kwh': 10000})
        
        # 테스트 부하
        demand = np.full(120, 80.0, dtype=float)  # 80MW for 2 hours
        
        # 스트레스 조건
        stress_factors = {'pv_reduction': 0.3, 'grid_outage': 0.0}
        
        # 공급 계산
        legacy_supply = legacy.calculate_supply(demand, stress_factors)
        cems_supply = cems.calculate_supply(demand, stress_factors)
        
        # 응답 시간 비교 (분해 모델)
        legacy_response = legacy.get_response_time_s()
        cems_response = cems.get_response_time_s()
        legacy_breakdown = legacy.get_response_breakdown()
        cems_breakdown = cems.get_response_breakdown()
        
        print(f"  ✅ Supply calculation completed")
        print(f"     Legacy response: {legacy_response:.1f}s ({legacy_response/60:.1f}분)")
        print(f"       - 장애감지: {legacy_breakdown['detection_time']}s, 판단: {legacy_breakdown['decision_time']}s")
        print(f"       - UPS전환: {legacy_breakdown['ups_switching']}s, 그리드복구: {legacy_breakdown['grid_recovery_min']}-{legacy_breakdown['grid_recovery_max']}s")
        print(f"     CEMS response: {cems_response:.1f}s")
        print(f"       - 장애감지: {cems_breakdown['detection_time']}s, Supercap: {cems_breakdown['supercap_response']}s")
        print(f"       - BESS전환: {cems_breakdown['bess_switching_min']}-{cems_breakdown['bess_switching_max']}s, AI최적화: {cems_breakdown['ai_optimization_min']}-{cems_breakdown['ai_optimization_max']}s")
        
        advantage_ratio = legacy_response / cems_response
        print(f"     CEMS advantage: {advantage_ratio:.0f}x faster (현실적 범위: 50-200배)")
        
        # 백업 시간 비교
        legacy_backup = legacy.calculate_backup_duration_s(80)
        cems_backup = cems.calculate_backup_duration_s(80)
        
        print(f"     Legacy backup: {legacy_backup/60:.1f} min")
        print(f"     CEMS backup: {cems_backup/60:.1f} min")
        print(f"     CEMS advantage: {cems_backup/legacy_backup:.1f}x longer")
        
        return True
        
    except Exception as e:
        print(f"  ❌ System integration test failed: {str(e)}")
        traceback.print_exc()
        return False


def run_all_tests():
    """모든 테스트 실행"""
    print("🚀 DT5 Expansion Module Tests")
    print("=" * 50)
    
    tests = [
        ("Stress Engine", test_stress_engine),
        ("Data Survival", test_data_survival),
        ("Unified Analytics", test_unified_analytics),
        ("System Integration", test_system_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results.append((test_name, success))
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if success:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! MVP is ready for demo.")
        print("\n🎯 Next steps:")
        print("   1. Run: streamlit run app_expansion.py --server.port 8502")
        print("   2. Test all UI functionalities")
        print("   3. Prepare demo presentation")
    else:
        print("⚠️  Some tests failed. Please fix before demo.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)