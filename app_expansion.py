"""
DT5 확장 기능 Streamlit 앱
3-Way 스트레스 테스트 + 데이터 생존 분석 MVP 데모
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import logging

# 기존 DT5 모듈
from modules import PVModule, AIDCModule, DCBusModule, WeatherModule
from config import GPU_TYPES, PUE_TIERS, COLOR_PALETTE

# 확장 모듈
from modules.expansion import (
    StressTestEngine, UnifiedExpansionAnalytics,
    DataSurvivalAnalyzer, DEFAULT_SYSTEM_CONFIGS
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit 페이지 설정
st.set_page_config(
    page_title="DT5 Expansion Analytics", 
    page_icon="🚀",
    layout="wide"
)

# 세션 상태 초기화
if 'expansion_results' not in st.session_state:
    st.session_state.expansion_results = None
if 'dt5_modules' not in st.session_state:
    st.session_state.dt5_modules = {}


def initialize_dt5_modules():
    """기존 DT5 모듈 초기화"""
    try:
        # 기본 설정으로 모듈 초기화
        modules = {
            'pv': PVModule(pv_type='c-Si', capacity_mw=100),
            'aidc': AIDCModule(gpu_type='H100', gpu_count=50000, pue_tier='tier3'),
            'dcbus': DCBusModule(converter_tech='default', grid_capacity_mw=20),
            'weather': WeatherModule()
        }
        st.session_state.dt5_modules = modules
        return True
    except Exception as e:
        st.error(f"DT5 모듈 초기화 실패: {str(e)}")
        return False


def create_main_dashboard():
    """메인 대시보드 구성"""
    st.title("🚀 DT5 확장 분석 - MVP 데모")
    st.markdown("### 3-Way 스트레스 테스트 + 데이터 생존 분석")
    
    # 사이드바 - 기능 선택
    with st.sidebar:
        st.header("📋 분석 기능")
        analysis_type = st.radio(
            "분석 유형 선택",
            ["🌊 스트레스 테스트", "💾 데이터 생존성", "📊 통합 대시보드"],
            key="analysis_type"
        )
        
        st.markdown("---")
        st.subheader("⚙️ 시스템 설정")
        
        # GPU 설정
        gpu_count = st.number_input(
            "GPU 수량", 
            min_value=1000, max_value=100000, value=50000, step=5000,
            key="gpu_count"
        )
        
        # 체크포인트 설정
        checkpoint_interval = st.slider(
            "체크포인트 간격 (분)", 
            min_value=5, max_value=30, value=15,
            key="checkpoint_interval"
        )
        
        # SSD 설정
        ssd_count = st.number_input(
            "SSD 수량",
            min_value=100, max_value=5000, value=1000, step=100,
            key="ssd_count"
        )
    
    # 메인 영역 - 분석 타입별 UI
    if analysis_type == "🌊 스트레스 테스트":
        render_stress_test_page()
    elif analysis_type == "💾 데이터 생존성":
        render_data_survival_page()
    else:
        render_unified_dashboard()


def render_stress_test_page():
    """스트레스 테스트 UI"""
    st.subheader("🌊 3-Way 스트레스 테스트 비교")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("#### 시나리오 설정")
        
        # 시나리오 선택
        scenarios = {
            'S1': 'GPU 워크로드 급증 (+30~80%)',
            'S2': 'PV 급감 (구름 -50~80%)',
            'S3': '그리드 차단 (부분/완전 정전)',
            'S4': 'S1+S2 복합 시나리오'
        }
        
        selected_scenarios = st.multiselect(
            "적용할 시나리오",
            options=list(scenarios.keys()),
            default=['S1'],
            format_func=lambda x: scenarios[x]
        )
        
        # 강도 설정
        col_int, col_dur = st.columns(2)
        with col_int:
            intensity = st.select_slider(
                "스트레스 강도",
                options=[0, 25, 50, 75, 100],
                value=50,
                format_func=lambda x: {0: '약함', 25: '중간', 50: '강함', 75: '극한', 100: '최대'}[x]
            )
        
        with col_dur:
            duration_hours = st.selectbox(
                "지속 시간",
                options=[1, 2, 4, 6, 12, 24],
                index=1,
                format_func=lambda x: f"{x}시간"
            )
        
        # 시뮬레이션 실행 버튼
        if st.button("🎯 스트레스 테스트 실행", type="primary", key="run_stress"):
            run_stress_simulation(selected_scenarios, intensity/100, duration_hours)
    
    with col2:
        st.write("#### 3-Way 시스템 비교 결과")
        
        if st.session_state.expansion_results and 'stress_tests' in st.session_state.expansion_results:
            display_stress_test_results()
        else:
            st.info("좌측에서 시나리오를 선택하고 '스트레스 테스트 실행' 버튼을 눌러주세요.")
            
            # 예시 차트 표시
            display_example_stress_charts()
    
    # 하단: KPI 메트릭 카드
    st.write("#### 📈 시스템 성능 지표 (KPI)")
    display_kpi_metrics()


def render_data_survival_page():
    """데이터 생존성 분석 UI"""
    st.subheader("💾 데이터 생존성 분석")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("#### 시스템 구성")
        
        # 하드웨어 설정
        gpu_count = st.session_state.gpu_count
        st.metric("GPU 수량", f"{gpu_count:,} 개")
        
        hbm_per_gpu = st.selectbox(
            "GPU당 HBM 용량",
            options=[80, 192, 256],
            index=0,
            format_func=lambda x: f"{x} GB"
        )
        
        ssd_count = st.session_state.ssd_count
        st.metric("SSD 수량", f"{ssd_count:,} 개")
        
        ssd_bandwidth = st.selectbox(
            "SSD 쓰기 대역폭",
            options=[3.5, 5.5, 7.4],
            index=1,
            format_func=lambda x: f"{x} GB/s"
        )
        
        checkpoint_interval = st.session_state.checkpoint_interval
        st.metric("체크포인트 간격", f"{checkpoint_interval} 분")
        
        # 분석 실행 버튼
        if st.button("📊 데이터 생존성 분석 실행", type="primary", key="run_survival"):
            run_survival_analysis(hbm_per_gpu, ssd_bandwidth)
    
    with col2:
        st.write("#### 생존성 분석 결과")
        
        if st.session_state.expansion_results and 'data_survival' in st.session_state.expansion_results:
            display_survival_results()
        else:
            st.info("좌측에서 '데이터 생존성 분석 실행' 버튼을 눌러주세요.")
            display_example_survival_charts()
    
    # 하단: 3-Way t2 비교
    st.write("#### ⏱️ 3-Way t2 버팀시간 비교")
    display_t2_breakdown()


def render_unified_dashboard():
    """통합 대시보드"""
    st.subheader("📊 DT5 확장 통합 대시보드")
    
    # 상단: 핵심 KPI
    st.write("#### 🎯 핵심 성과 지표")
    display_unified_kpi()
    
    # 중간: 비교 분석
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 📈 종합 점수 비교")
        display_overall_score_chart()
    
    with col2:
        st.write("#### 🏆 우위 분석")
        display_advantage_analysis()
    
    # 하단: 투자 타당성
    st.write("#### 💰 투자 타당성 분석")
    display_roi_analysis()


def run_stress_simulation(scenarios, intensity, duration_hours):
    """스트레스 시뮬레이션 실행"""
    with st.spinner("스트레스 테스트 실행 중..."):
        try:
            # DT5 모듈 초기화
            if not st.session_state.dt5_modules:
                if not initialize_dt5_modules():
                    return
            
            # 확장 분석 엔진 초기화
            analytics = UnifiedExpansionAnalytics(st.session_state.dt5_modules)
            
            # 시나리오 구성
            scenario_configs = []
            for scenario_id in scenarios:
                config = {
                    'scenario_id': scenario_id,
                    'name': get_scenario_name(scenario_id),
                    'description': get_scenario_description(scenario_id),
                    'intensity': intensity,
                    'duration_hours': duration_hours,
                    'parameters': get_scenario_parameters(scenario_id, intensity)
                }
                scenario_configs.append(config)
            
            # 종합 분석 실행
            results = analytics.run_comprehensive_analysis(scenario_configs)
            st.session_state.expansion_results = results
            
            st.success("스트레스 테스트 완료!")
            
        except Exception as e:
            st.error(f"시뮬레이션 오류: {str(e)}")
            logger.error(f"Stress simulation error: {e}")


def run_survival_analysis(hbm_per_gpu, ssd_bandwidth):
    """데이터 생존성 분석 실행"""
    with st.spinner("데이터 생존성 분석 중..."):
        try:
            # 분석 설정 업데이트
            aidc_config = {
                'gpu_count': st.session_state.gpu_count,
                'hbm_per_gpu_gb': hbm_per_gpu,
                'hbm_utilization': 0.8,
                'ssd_count': st.session_state.ssd_count,
                'ssd_write_bw_gb_s': ssd_bandwidth,
                'checkpoint_interval_min': st.session_state.checkpoint_interval
            }
            
            analyzer = DataSurvivalAnalyzer(aidc_config)
            survival_results = analyzer.compare_three_systems(DEFAULT_SYSTEM_CONFIGS)
            
            # 결과 저장
            if 'expansion_results' not in st.session_state or not st.session_state.expansion_results:
                st.session_state.expansion_results = {}
            
            st.session_state.expansion_results['data_survival'] = survival_results
            
            st.success("데이터 생존성 분석 완료!")
            
        except Exception as e:
            st.error(f"분석 오류: {str(e)}")
            logger.error(f"Survival analysis error: {e}")


def display_stress_test_results():
    """스트레스 테스트 결과 표시"""
    results = st.session_state.expansion_results['stress_tests']
    
    if not results:
        st.warning("스트레스 테스트 결과가 없습니다.")
        return
    
    # 첫 번째 시나리오 결과 사용
    scenario_id = list(results.keys())[0]
    scenario_results = results[scenario_id]['results']
    
    # 3-Way 시계열 차트
    create_three_way_timeseries(scenario_results)


def display_survival_results():
    """데이터 생존성 결과 표시"""
    survival_data = st.session_state.expansion_results['data_survival']
    
    # 생존율 비교 차트
    fig = create_survival_comparison_chart(survival_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # 주요 메트릭
    col1, col2, col3 = st.columns(3)
    
    for i, (system, data) in enumerate(survival_data.items()):
        survival_result = data['survival_result']
        t2_components = data['t2_components']
        
        with [col1, col2, col3][i]:
            system_name = {'legacy': '기존그리드', 'smart': '스마트그리드', 'cems': 'CEMS'}[system]
            
            st.metric(
                f"{system_name} 생존율",
                f"{survival_result.data_survival_rate:.1%}",
                delta=f"t2: {t2_components.total_t2_s/60:.1f}분"
            )


def display_kpi_metrics():
    """KPI 메트릭 카드 표시"""
    if not st.session_state.expansion_results or 'unified_kpi' not in st.session_state.expansion_results:
        # 예시 메트릭
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Robustness Score", "92.4", "↑5.2")
        with col2:
            st.metric("Recovery Time", "12.3분", "↓2.1분")
        with col3:
            st.metric("Max Deviation", "8.7%", "↓1.2%")
        with col4:
            st.metric("Data Survival", "99.8%", "↑0.3%")
        with col5:
            st.metric("Disruption Cost", "$2.1K", "↓$0.8K")
        return
    
    unified_kpi = st.session_state.expansion_results['unified_kpi']
    cems_kpi = unified_kpi.get('cems')
    legacy_kpi = unified_kpi.get('legacy')
    
    if not cems_kpi or not legacy_kpi:
        st.warning("KPI 데이터가 불완전합니다.")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_robustness = cems_kpi.robustness_score - legacy_kpi.robustness_score
        st.metric(
            "Robustness Score",
            f"{cems_kpi.robustness_score:.1f}",
            delta=f"{delta_robustness:+.1f}"
        )
    
    with col2:
        delta_recovery = legacy_kpi.recovery_time_s - cems_kpi.recovery_time_s
        st.metric(
            "Recovery Time",
            f"{cems_kpi.recovery_time_s/60:.1f}분",
            delta=f"{delta_recovery/60:+.1f}분"
        )
    
    with col3:
        delta_deviation = legacy_kpi.max_power_deviation_pct - cems_kpi.max_power_deviation_pct
        st.metric(
            "Max Deviation", 
            f"{cems_kpi.max_power_deviation_pct:.1f}%",
            delta=f"{delta_deviation:+.1f}%"
        )
    
    with col4:
        delta_survival = cems_kpi.data_survival_rate - legacy_kpi.data_survival_rate
        st.metric(
            "Data Survival",
            f"{cems_kpi.data_survival_rate:.1%}",
            delta=f"{delta_survival:+.1%}"
        )
    
    with col5:
        delta_cost = legacy_kpi.data_loss_cost_usd - cems_kpi.data_loss_cost_usd
        st.metric(
            "Data Loss Cost",
            f"${cems_kpi.data_loss_cost_usd/1000:.1f}K",
            delta=f"${delta_cost/1000:+.1f}K"
        )


def display_unified_kpi():
    """통합 KPI 표시"""
    if not st.session_state.expansion_results or 'unified_kpi' not in st.session_state.expansion_results:
        # 예시 데이터
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("시스템 강건성", "94.2%", "↑7.1%")
        with col2:
            st.metric("데이터 안전성", "99.8%", "↑12.1%")
        with col3:
            st.metric("에너지 SLA", "Tier IV", "달성")
        with col4:
            st.metric("예상 손실", "$45K", "↓$23K")
        with col5:
            st.metric("ROI", "2.3년", "투자 회수")
        with col6:
            st.metric("종합 점수", "88.5", "A 등급")
        return
    
    unified_kpi = st.session_state.expansion_results['unified_kpi']
    cems_kpi = unified_kpi.get('cems')
    
    if not cems_kpi:
        st.warning("CEMS KPI 데이터가 없습니다.")
        return
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("시스템 강건성", f"{cems_kpi.robustness_score:.1f}%")
    with col2:
        st.metric("데이터 안전성", f"{cems_kpi.data_survival_rate:.1%}")
    with col3:
        st.metric("에너지 SLA", "Tier IV" if cems_kpi.tier_4_compliant else "미달성")
    with col4:
        st.metric("데이터 손실 비용", f"${cems_kpi.data_loss_cost_usd/1000:.0f}K")
    with col5:
        executive = st.session_state.expansion_results.get('executive_summary', {})
        roi_years = executive.get('business_impact', {}).get('roi_years', 0)
        st.metric("ROI", f"{roi_years:.1f}년" if roi_years < 10 else "10년+")
    with col6:
        grade = "A" if cems_kpi.overall_score >= 85 else "B" if cems_kpi.overall_score >= 70 else "C"
        st.metric("종합 점수", f"{cems_kpi.overall_score:.1f}", f"{grade} 등급")


def display_t2_breakdown():
    """t2 분해 차트"""
    if not st.session_state.expansion_results or 'data_survival' not in st.session_state.expansion_results:
        # 예시 차트
        systems = ['기존그리드', '스마트그리드', 'CEMS']
        psu = [18/1000/60, 18/1000/60, 18/1000/60]  # ms를 분으로
        ups = [10, 20, 30]
        bess = [0, 0, 60]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='PSU Holdup', x=systems, y=psu, marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(name='UPS Backup', x=systems, y=ups, marker_color='#4ECDC4'))
        fig.add_trace(go.Bar(name='BESS Emergency', x=systems, y=bess, marker_color='#45B7D1'))
        
        fig.update_layout(
            barmode='stack',
            title='t2 버팀시간 분해 (예시)',
            xaxis_title='시스템',
            yaxis_title='시간 (분)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return
    
    survival_data = st.session_state.expansion_results['data_survival']
    fig = create_t2_breakdown_chart(survival_data)
    st.plotly_chart(fig, use_container_width=True)


# 헬퍼 함수들
def get_scenario_name(scenario_id):
    names = {
        'S1': 'GPU 워크로드 급증',
        'S2': 'PV 급감',
        'S3': '그리드 차단',
        'S4': 'S1+S2 복합'
    }
    return names.get(scenario_id, 'Unknown Scenario')


def get_scenario_description(scenario_id):
    descriptions = {
        'S1': 'Poisson burst로 GPU 부하 30-80% 급증',
        'S2': '구름/고장으로 PV 출력 50-80% 감소',
        'S3': '부분/완전 정전 상황',
        'S4': 'GPU 급증 + PV 급감 동시 발생'
    }
    return descriptions.get(scenario_id, '')


def get_scenario_parameters(scenario_id, intensity):
    base_params = {
        'S1': {'gpu_burst_multiplier': 1.3 + intensity * 0.5},
        'S2': {'pv_reduction_factor': 0.3 + intensity * 0.5},
        'S3': {'grid_outage_factor': 0.5 + intensity * 0.5},
        'S4': {'gpu_burst_multiplier': 1.2 + intensity * 0.3, 'pv_reduction_factor': 0.2 + intensity * 0.3}
    }
    return base_params.get(scenario_id, {})


def create_three_way_timeseries(scenario_results):
    """3-Way 시계열 차트 생성"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=['기존 그리드', '스마트그리드', 'CEMS 마이크로그리드'],
        vertical_spacing=0.05
    )
    
    systems = ['legacy', 'smart', 'cems']
    colors = ['red', 'orange', 'green']
    
    for i, (system, color) in enumerate(zip(systems, colors)):
        if system in scenario_results:
            result = scenario_results[system]
            demand = result['demand_profile']
            supply = result['supply_profile']
            
            time_points = np.arange(len(demand))
            
            # 수요 라인
            fig.add_trace(
                go.Scatter(
                    x=time_points, y=demand,
                    name=f'{system} 수요',
                    line=dict(color='black', dash='dash'),
                    showlegend=(i == 0)
                ), row=i+1, col=1
            )
            
            # 공급 라인
            fig.add_trace(
                go.Scatter(
                    x=time_points, y=supply,
                    name=f'{system} 공급',
                    line=dict(color=color),
                    fill='tonexty',
                    showlegend=(i == 0)
                ), row=i+1, col=1
            )
    
    fig.update_layout(height=600, title="3-Way 스트레스 테스트 비교")
    fig.update_xaxes(title_text="시간 (분)", row=3, col=1)
    fig.update_yaxes(title_text="전력 (MW)")
    
    st.plotly_chart(fig, use_container_width=True)


def create_survival_comparison_chart(survival_data):
    """생존율 비교 차트"""
    systems = []
    survival_rates = []
    t2_times = []
    
    system_names = {'legacy': '기존그리드', 'smart': '스마트그리드', 'cems': 'CEMS'}
    
    for system, data in survival_data.items():
        systems.append(system_names[system])
        survival_rates.append(data['survival_result'].data_survival_rate * 100)
        t2_times.append(data['t2_components'].total_t2_s / 60)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            name='데이터 생존율',
            x=systems,
            y=survival_rates,
            marker_color=['#FF6B6B', '#FFD93D', '#6BCF7F'],
            text=[f"{rate:.1f}%" for rate in survival_rates],
            textposition='outside'
        )
    )
    
    fig.update_layout(
        title='3-Way 데이터 생존율 비교',
        xaxis_title='시스템',
        yaxis_title='생존율 (%)',
        height=400,
        yaxis=dict(range=[0, 105])
    )
    
    return fig


def create_t2_breakdown_chart(survival_data):
    """t2 분해 차트"""
    system_names = {'legacy': '기존그리드', 'smart': '스마트그리드', 'cems': 'CEMS'}
    
    systems = []
    psu_times = []
    ups_times = []
    bess_times = []
    
    for system, data in survival_data.items():
        components = data['t2_components']
        systems.append(system_names[system])
        psu_times.append(components.psu_holdup_s / 60)  # 분 변환
        ups_times.append(components.ups_backup_s / 60)
        bess_times.append(components.bess_emergency_s / 60)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='PSU Holdup', x=systems, y=psu_times, marker_color='#FF6B6B'))
    fig.add_trace(go.Bar(name='UPS Backup', x=systems, y=ups_times, marker_color='#4ECDC4'))
    fig.add_trace(go.Bar(name='BESS Emergency', x=systems, y=bess_times, marker_color='#45B7D1'))
    
    fig.update_layout(
        barmode='stack',
        title='t2 버팀시간 분해 비교',
        xaxis_title='시스템',
        yaxis_title='시간 (분)',
        height=400
    )
    
    return fig


def display_example_stress_charts():
    """예시 스트레스 차트"""
    # 예시 데이터 생성
    time_points = np.arange(0, 120)  # 2시간
    base_demand = 80  # 80MW
    
    # S1 시나리오 시뮬레이션
    demand = base_demand * (1 + 0.1 * np.sin(2 * np.pi * time_points / 60))
    
    # GPU burst at 30분
    burst_start, burst_end = 30, 60
    demand[burst_start:burst_end] *= 1.5
    
    # 시스템별 공급
    legacy_supply = np.minimum(demand, 80)  # 계약전력 제한
    smart_supply = demand * 0.95  # DR로 95% 공급
    cems_supply = demand.copy()  # 완전 공급
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=['기존 그리드 (예시)', '스마트그리드 (예시)', 'CEMS 마이크로그리드 (예시)'],
        vertical_spacing=0.05
    )
    
    supplies = [legacy_supply, smart_supply, cems_supply]
    colors = ['red', 'orange', 'green']
    
    for i, (supply, color) in enumerate(zip(supplies, colors)):
        # 수요 라인
        fig.add_trace(
            go.Scatter(
                x=time_points, y=demand,
                name='수요' if i == 0 else None,
                line=dict(color='black', dash='dash'),
                showlegend=(i == 0)
            ), row=i+1, col=1
        )
        
        # 공급 라인
        fig.add_trace(
            go.Scatter(
                x=time_points, y=supply,
                name='공급' if i == 0 else None,
                line=dict(color=color),
                fill='tonexty',
                showlegend=(i == 0)
            ), row=i+1, col=1
        )
    
    fig.update_layout(height=600, title="3-Way 스트레스 테스트 비교 (예시)")
    fig.update_xaxes(title_text="시간 (분)", row=3, col=1)
    fig.update_yaxes(title_text="전력 (MW)")
    
    st.plotly_chart(fig, use_container_width=True)


def display_example_survival_charts():
    """예시 생존성 차트"""
    systems = ['기존그리드', '스마트그리드', 'CEMS']
    survival_rates = [85.2, 92.7, 99.8]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            name='데이터 생존율',
            x=systems,
            y=survival_rates,
            marker_color=['#FF6B6B', '#FFD93D', '#6BCF7F'],
            text=[f"{rate:.1f}%" for rate in survival_rates],
            textposition='outside'
        )
    )
    
    fig.update_layout(
        title='3-Way 데이터 생존율 비교 (예시)',
        xaxis_title='시스템',
        yaxis_title='생존율 (%)',
        height=400,
        yaxis=dict(range=[0, 105])
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_overall_score_chart():
    """종합 점수 차트"""
    if not st.session_state.expansion_results or 'unified_kpi' not in st.session_state.expansion_results:
        # 예시 데이터
        systems = ['기존그리드', '스마트그리드', 'CEMS']
        scores = [52.3, 71.8, 88.5]
    else:
        unified_kpi = st.session_state.expansion_results['unified_kpi']
        systems = []
        scores = []
        system_names = {'legacy': '기존그리드', 'smart': '스마트그리드', 'cems': 'CEMS'}
        
        for system, kpi in unified_kpi.items():
            systems.append(system_names[system])
            scores.append(kpi.overall_score)
    
    fig = px.bar(
        x=systems, y=scores,
        title='종합 성능 점수',
        color=scores,
        color_continuous_scale=['red', 'yellow', 'green'],
        text=[f"{score:.1f}" for score in scores]
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_advantage_analysis():
    """우위 분석 표시"""
    if not st.session_state.expansion_results or 'executive_summary' not in st.session_state.expansion_results:
        # 예시 데이터
        st.markdown("#### CEMS의 주요 우위")
        st.markdown("- ✅ 시스템 강건성 (94점 이상)")
        st.markdown("- ✅ 데이터 백업 여유시간 (78.3분)")
        st.markdown("- ✅ Tier IV 에너지 SLA 달성")
        st.markdown("- ✅ 데이터 생존율 (99.8%)")
        
        st.markdown("#### 경쟁 시스템 대비 우위")
        st.markdown("- 📈 기존그리드 대비: **+36.2점**")
        st.markdown("- 📈 스마트그리드 대비: **+16.7점**")
        return
    
    executive = st.session_state.expansion_results['executive_summary']
    cems_advantages = executive.get('cems_advantages', {})
    
    st.markdown("#### CEMS의 주요 우위")
    for strength in cems_advantages.get('key_strengths', []):
        st.markdown(f"- ✅ {strength}")
    
    st.markdown("#### 경쟁 시스템 대비 우위")
    vs_legacy = cems_advantages.get('vs_legacy', 0)
    vs_smart = cems_advantages.get('vs_smart', 0)
    
    st.markdown(f"- 📈 기존그리드 대비: **{vs_legacy:+.1f}점**")
    st.markdown(f"- 📈 스마트그리드 대비: **{vs_smart:+.1f}점**")


def display_roi_analysis():
    """ROI 분석 표시"""
    if not st.session_state.expansion_results or 'executive_summary' not in st.session_state.expansion_results:
        # 예시 데이터
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("연간 절감액", "45억원")
        with col2:
            st.metric("초기 투자비", "15억원")
        with col3:
            st.metric("투자 회수", "2.3년")
        with col4:
            st.metric("ROI", "300%")
        
        st.markdown("#### 💡 주요 절감 요소")
        st.markdown("- 데이터 손실 비용 절감: 연간 28억원")
        st.markdown("- 정전 비용 절감: 연간 12억원")
        st.markdown("- 유지보수 비용 절감: 연간 5억원")
        return
    
    executive = st.session_state.expansion_results['executive_summary']
    business_impact = executive.get('business_impact', {})
    
    annual_savings = business_impact.get('annual_savings_krw', 0)
    roi_years = business_impact.get('roi_years', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("연간 절감액", f"{annual_savings/100000000:.1f}억원")
    with col2:
        st.metric("초기 투자비", "15억원")
    with col3:
        st.metric("투자 회수", f"{roi_years:.1f}년")
    with col4:
        roi_pct = (annual_savings * roi_years - 1500000000) / 1500000000 * 100 if roi_years > 0 else 0
        st.metric("ROI", f"{roi_pct:.0f}%")


# 시스템 상태 다이어그램 (ASCII art)
def display_system_diagram():
    """시스템 다이어그램"""
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                    CEMS 마이크로그리드                        │
    │                                                             │
    │  ☀️ PV ──────► ⚡ DC BUS ◄──── 🔋 BESS                      │
    │  🟢 (100MW)     🟡 (운영중)    🟢 (SoC 85%)                │
    │                     │                                       │
    │                     ▼                                       │
    │                 🖥️ AIDC      🏭 Grid                       │
    │                 🟢 (80MW)     🟢 (20MW)                     │
    │                                                             │
    │  범례: 🟢 정상  🟡 주의  🔴 위험  ⚫ 다운                    │
    └─────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    create_main_dashboard()