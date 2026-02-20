"""
CEMS-DT 데모 스크립트
Streamlit 없이 콘솔에서 시뮬레이션 실행 및 결과 표시
"""
import pandas as pd
import numpy as np
import os

from modules import PVModule, AIDCModule, DCBusModule, WeatherModule
from config import PV_TYPES

def run_demo_simulation():
    """데모 시뮬레이션 실행"""
    print("=" * 60)
    print("CEMS Digital Twin - 데모 시뮬레이션")
    print("100MW급 AIDC 신재생 마이크로그리드")
    print("=" * 60)
    
    # 1. 기상 데이터 생성/로드
    print("\\n📊 1. 기상 데이터 준비...")
    weather = WeatherModule()
    weather_file = 'data/weather_sample.csv'
    
    if os.path.exists(weather_file):
        weather_data = weather.load_from_csv(weather_file)
        print(f"   ✓ 기존 기상 데이터 로드: {len(weather_data)} 시간")
    else:
        weather_data = weather.generate_tmy_data(year=2024, noise_level=0.1)
        weather_data.to_csv(weather_file)
        print(f"   ✓ 새 기상 데이터 생성: {len(weather_data)} 시간")
    
    # 기상 통계
    stats = weather.get_statistics()
    print(f"   - 연간 일사량: {stats['annual_ghi_kwh_per_m2']:.0f} kWh/m²")
    print(f"   - 온도 범위: {stats['temp_celsius_min']:.1f}~{stats['temp_celsius_max']:.1f}°C")
    
    # 시뮬레이션 기간 설정 (1주일)
    sim_hours = 168
    weather_subset = weather_data.head(sim_hours)
    
    # 2. PV 시스템 시뮬레이션
    print("\\n☀️ 2. PV 발전 시뮬레이션...")
    
    pv_scenarios = [
        ('c-Si', 'c-Si 단결정 실리콘 (24.4%)'),
        ('tandem', '탠덤 페로브스카이트 (34.85%)'),
        ('triple', '3접합 III-V (39.5%)'),
        ('infinite', '무한접합 이상적 (68.7%)')
    ]
    
    pv_results = {}
    
    for pv_type, pv_name in pv_scenarios:
        pv = PVModule(pv_type=pv_type, capacity_mw=100, active_control=False)
        pv_data = pv.simulate_time_series(weather_subset)
        pv_stats = pv.get_daily_statistics(pv_data)
        pv_results[pv_type] = {'data': pv_data, 'stats': pv_stats, 'module': pv}
        
        print(f"   {pv_name}:")
        print(f"     - 총 발전량: {pv_stats['total_generation_mwh']:.1f} MWh")
        print(f"     - 평균 이용률: {pv_stats['capacity_factor_avg']:.1%}")
        print(f"     - 필요 면적: {pv.total_area_m2/10000:.1f} ha")
    
    # 이후 시뮬레이션은 c-Si 기준으로 진행
    selected_pv = pv_results['c-Si']
    pv_data = selected_pv['data']
    
    # 3. AIDC 부하 시뮬레이션
    print("\\n🖥️ 3. AIDC 부하 시뮬레이션...")
    
    gpu_scenarios = [
        ('H100', 'NVIDIA H100 SXM', 50000),
        ('B200', 'NVIDIA B200 Blackwell', 40000), 
        ('next_gen', '차세대 GPU (2027+)', 35000)
    ]
    
    aidc_results = {}
    
    for gpu_type, gpu_name, gpu_count in gpu_scenarios:
        aidc = AIDCModule(
            gpu_type=gpu_type, 
            gpu_count=gpu_count,
            pue_tier='tier2',
            workload_mix={'llm': 0.4, 'training': 0.4, 'moe': 0.2}
        )
        aidc_data = aidc.simulate_time_series(hours=sim_hours, random_seed=42)
        aidc_stats = aidc.get_statistics(aidc_data)
        aidc_results[gpu_type] = {'data': aidc_data, 'stats': aidc_stats, 'module': aidc}
        
        print(f"   {gpu_name} × {gpu_count:,}:")
        print(f"     - 최대 IT 부하: {aidc.max_it_power_mw:.1f} MW")
        print(f"     - 최대 총 부하: {aidc.max_total_power_mw:.1f} MW")  
        print(f"     - 평균 부하율: {aidc_stats['load_factor']:.1%}")
        print(f"     - 실제 PUE: {aidc_stats['actual_pue']:.2f}")
    
    # H100 기준으로 이후 분석 진행
    selected_aidc = aidc_results['H100']
    aidc_data = selected_aidc['data']
    
    # 4. DC Bus 전력 균형 시뮬레이션
    print("\\n⚡ 4. DC Bus 전력 균형 시뮬레이션...")
    
    converter_scenarios = [
        ('default', 'SiC 기본 변환기'),
        ('advanced', 'GaN+ 고효율 변환기')
    ]
    
    dcbus_results = {}
    
    for conv_tech, conv_name in converter_scenarios:
        dcbus = DCBusModule(
            converter_tech=conv_tech,
            grid_capacity_mw=20
        )
        
        dcbus_data = dcbus.simulate_time_series(
            pv_data=pv_data,
            aidc_data=aidc_data,
            bess_capacity_mw=200,
            h2_electrolyzer_mw=50,
            h2_fuelcell_mw=30
        )
        
        dcbus_summary = dcbus.get_energy_flows_summary(dcbus_data)
        dcbus_results[conv_tech] = {'data': dcbus_data, 'summary': dcbus_summary, 'module': dcbus}
        
        print(f"   {conv_name}:")
        print(f"     - 시스템 효율: {dcbus_summary['system_efficiency']:.1%}")
        print(f"     - 그리드 독립도: {dcbus_summary['grid_independence_ratio']:.1%}")
        print(f"     - PV 출력제한: {dcbus_summary['curtailment_ratio']:.1%}")
        print(f"     - 총 변환손실: {dcbus_summary['total_losses_mwh']:.1f} MWh")
    
    # 기본 변환기로 상세 분석
    selected_dcbus = dcbus_results['default']
    dcbus_data = selected_dcbus['data']
    
    # 5. 종합 분석
    print("\\n📈 5. 종합 성능 분석...")
    
    # 전력 균형 분석
    total_pv_gen = pv_data['power_mw'].sum()
    total_aidc_load = aidc_data['total_power_mw'].sum()
    total_grid_import = dcbus_data['grid_import_mw'].sum()
    total_grid_export = dcbus_data['grid_export_mw'].sum()
    
    print(f"\\n   ** 에너지 수지 (1주간) **")
    print(f"   - PV 총 발전량: {total_pv_gen:.1f} MWh")
    print(f"   - AIDC 총 소비량: {total_aidc_load:.1f} MWh") 
    print(f"   - 에너지 자립률: {min(total_pv_gen/total_aidc_load*100, 100):.1f}%")
    print(f"   - 그리드 구매: {total_grid_import:.1f} MWh")
    print(f"   - 그리드 판매: {total_grid_export:.1f} MWh")
    
    # 피크 분석
    pv_peak = pv_data['power_mw'].max()
    aidc_peak = aidc_data['total_power_mw'].max()
    max_mismatch = (pv_data['power_mw'] - aidc_data['total_power_mw']).abs().max()
    
    print(f"\\n   ** 피크 성능 **")
    print(f"   - PV 피크 출력: {pv_peak:.1f} MW")
    print(f"   - AIDC 피크 부하: {aidc_peak:.1f} MW")
    print(f"   - 최대 전력 미스매치: {max_mismatch:.1f} MW")
    
    # 시간별 패턴 분석
    hourly_mismatch = pv_data['power_mw'].values - aidc_data['total_power_mw'].values
    surplus_hours = (hourly_mismatch > 0).sum()
    deficit_hours = (hourly_mismatch < 0).sum()
    
    print(f"\\n   ** 시간대별 패턴 **")
    print(f"   - 잉여 전력 시간: {surplus_hours}h ({surplus_hours/sim_hours:.1%})")
    print(f"   - 부족 전력 시간: {deficit_hours}h ({deficit_hours/sim_hours:.1%})")
    
    # 6. PV 기술별 비교 요약
    print("\\n🔬 6. PV 기술별 성능 비교...")
    print(f"{'기술':<15} {'발전량(MWh)':<12} {'이용률':<8} {'면적(ha)':<10}")
    print("-" * 50)
    
    for pv_type, pv_name in pv_scenarios:
        result = pv_results[pv_type]
        gen = result['stats']['total_generation_mwh']
        cf = result['stats']['capacity_factor_avg']
        area = result['module'].total_area_m2 / 10000
        print(f"{pv_type:<15} {gen:<12.1f} {cf:<8.1%} {area:<10.1f}")
    
    # 7. 결론 및 권장사항
    print("\\n💡 7. 결론 및 권장사항...")
    
    # 최적 PV 기술 선정 (발전량 대비 면적 효율)
    best_pv = max(pv_results.keys(), 
                  key=lambda x: pv_results[x]['stats']['total_generation_mwh'] / 
                              (pv_results[x]['module'].total_area_m2 / 10000))
    
    print(f"   ✅ 권장 PV 기술: {PV_TYPES[best_pv]['name']}")
    print(f"      (면적 대비 발전량 최적화)")
    
    # 시스템 자립도 평가
    self_sufficiency = min(total_pv_gen / total_aidc_load, 1.0)
    if self_sufficiency > 0.8:
        print("   ✅ 높은 에너지 자립도 달성 가능")
    elif self_sufficiency > 0.6:
        print("   ⚠️ 보통 수준의 에너지 자립도")
    else:
        print("   ❌ 추가 신재생 설비 확충 필요")
    
    # BESS 필요성 평가
    if max_mismatch > 50:  # 50MW 이상 미스매치
        print("   ✅ BESS 필수: 전력 평준화를 위한 대용량 저장 시스템 필요")
    else:
        print("   📝 BESS 검토: 소규모 완충 저장 시스템으로 충분")
    
    print("\\n" + "=" * 60)
    print("시뮬레이션 완료! 상세 데이터는 CSV 파일로 저장됨.")
    print("=" * 60)
    
    return {
        'weather': weather_subset,
        'pv': pv_results,
        'aidc': aidc_results, 
        'dcbus': dcbus_results
    }


def save_demo_results(results):
    """데모 결과를 CSV로 저장"""
    print("\\n💾 결과 데이터 저장 중...")
    
    # 기상 데이터
    results['weather'].to_csv('data/demo_weather.csv')
    print("   ✓ 기상 데이터: data/demo_weather.csv")
    
    # PV 데이터 (c-Si 기준)
    results['pv']['c-Si']['data'].to_csv('data/demo_pv.csv')
    print("   ✓ PV 데이터: data/demo_pv.csv")
    
    # AIDC 데이터 (H100 기준)
    results['aidc']['H100']['data'].to_csv('data/demo_aidc.csv') 
    print("   ✓ AIDC 데이터: data/demo_aidc.csv")
    
    # DC Bus 데이터
    results['dcbus']['default']['data'].to_csv('data/demo_dcbus.csv')
    print("   ✓ DC Bus 데이터: data/demo_dcbus.csv")


if __name__ == "__main__":
    # 데모 실행
    results = run_demo_simulation()
    
    # 결과 저장
    save_demo_results(results)
    
    print("\\n🎯 다음 단계:")
    print("   1. Streamlit 설치 후 'streamlit run app.py' 실행")
    print("   2. 웹 브라우저에서 http://localhost:8501 접속")
    print("   3. 대화형 시뮬레이션 및 시각화 체험")