"""
CEMS Digital Twin - Streamlit 메인 앱
100MW급 AIDC 신재생 마이크로그리드 시뮬레이션
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os


def _to_list(v):
    """Convert numpy arrays to Python lists for Plotly compatibility."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _safe_dict(d):
    """Convert all numpy arrays in a dict to Python lists."""
    if isinstance(d, dict):
        return {k: _to_list(v) for k, v in d.items()}
    return d

# 모듈 임포트
from modules import (
    PVModule, AIDCModule, DCBusModule, WeatherModule,
    HESSModule, H2SystemModule, GridInterfaceModule
)
from config import (
    PV_TYPES, GPU_TYPES, PUE_TIERS, WORKLOAD_TYPES, 
    CONVERTER_EFFICIENCY, UI_CONFIG, COLOR_PALETTE,
    HESS_LAYER_CONFIGS, H2_SYSTEM_CONFIG, GRID_TARIFF_CONFIG
)

# Streamlit 페이지 설정
st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout']
)

# 세션 상태 초기화
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None


@st.cache_data
def load_weather_data():
    """기상 데이터 로드 (캐시)"""
    weather_file = 'data/weather_sample.csv'
    if os.path.exists(weather_file):
        weather = WeatherModule()
        return weather.load_from_csv(weather_file)
    else:
        weather = WeatherModule()
        data = weather.generate_tmy_data(year=2024, noise_level=0.1)
        data.to_csv(weather_file)
        return data


def create_main_dashboard():
    """메인 대시보드 구성"""
    st.title("⚡ CEMS Digital Twin")
    st.markdown("### 100MW급 AIDC 신재생 마이크로그리드 시뮬레이션")
    
    # 사이드바 - 시스템 파라미터 설정
    with st.sidebar:
        st.header("🔧 시스템 설정")
        
        # M1. PV 모듈 설정
        st.subheader("🌞 M1. PV 발전")
        pv_type = st.selectbox(
            "PV 기술", 
            options=list(PV_TYPES.keys()),
            format_func=lambda x: PV_TYPES[x]['name'],
            key="pv_type"
        )
        
        pv_capacity = st.slider(
            "PV 용량 (MW)", 
            min_value=50, max_value=200, value=100, step=10,
            key="pv_capacity"
        )
        
        pv_active_control = st.checkbox(
            "능동 제어 (V,J 1ms 제어)", 
            value=False,
            key="pv_active"
        )
        
        # M3. AIDC 부하 설정
        st.subheader("🖥️ M3. AIDC 부하")
        gpu_type = st.selectbox(
            "GPU 종류",
            options=list(GPU_TYPES.keys()),
            format_func=lambda x: GPU_TYPES[x]['name'],
            key="gpu_type"
        )
        
        gpu_count = st.slider(
            "GPU 수량",
            min_value=10000, max_value=100000, value=50000, step=5000,
            format="%d",
            key="gpu_count"
        )
        
        pue_tier = st.selectbox(
            "PUE Tier",
            options=list(PUE_TIERS.keys()),
            format_func=lambda x: PUE_TIERS[x]['name'],
            key="pue_tier"
        )
        
        st.write("**워크로드 믹스**")
        llm_ratio = st.slider("LLM 추론 비율", 0.0, 1.0, 0.4, 0.1, key="llm_ratio")
        training_ratio = st.slider("AI 훈련 비율", 0.0, 1.0, 0.4, 0.1, key="training_ratio")
        moe_ratio = st.slider("MoE 비율", 0.0, 1.0, 0.2, 0.1, key="moe_ratio")
        
        # 정규화
        total_ratio = llm_ratio + training_ratio + moe_ratio
        if total_ratio > 0:
            workload_mix = {
                'llm': llm_ratio / total_ratio,
                'training': training_ratio / total_ratio,
                'moe': moe_ratio / total_ratio
            }
        else:
            workload_mix = {'llm': 1.0, 'training': 0.0, 'moe': 0.0}
        
        # M4. DC Bus 설정
        st.subheader("⚡ M4. DC Bus")
        converter_tech = st.selectbox(
            "변환기 기술",
            options=['default', 'advanced'],
            format_func=lambda x: 'SiC (기본)' if x == 'default' else 'GaN+ (고효율)',
            key="converter_tech"
        )
        
        grid_capacity = st.slider(
            "그리드 연계 용량 (MW)",
            min_value=0, max_value=30, value=20, step=5,
            key="grid_capacity"
        )
        
        # 시뮬레이션 설정
        st.subheader("⚙️ 시뮬레이션")
        sim_hours = st.selectbox(
            "시뮬레이션 기간",
            options=[24, 168, 720, 8760],
            format_func=lambda x: {24: '1일', 168: '1주', 720: '1개월', 8760: '1년'}[x],
            index=1,  # 기본값: 1주
            key="sim_hours"
        )
        
        # 시뮬레이션 실행 버튼
        if st.button("🚀 시뮬레이션 실행", type="primary"):
            run_simulation()
    
    # 메인 영역 - 결과 표시
    if st.session_state.simulation_data is not None:
        display_results()
    else:
        st.info("좌측 사이드바에서 파라미터를 설정하고 '시뮬레이션 실행' 버튼을 눌러주세요.")


def run_simulation():
    """시뮬레이션 실행"""
    with st.spinner("시뮬레이션 실행 중..."):
        try:
            # 기상 데이터 로드
            weather_data = load_weather_data()
            st.session_state.weather_data = weather_data
            
            # 시뮬레이션 기간 설정
            sim_hours = st.session_state.sim_hours
            weather_subset = weather_data.head(sim_hours)
            
            # 모듈 초기화
            pv = PVModule(
                pv_type=st.session_state.pv_type,
                capacity_mw=st.session_state.pv_capacity,
                active_control=st.session_state.pv_active
            )
            
            aidc = AIDCModule(
                gpu_type=st.session_state.gpu_type,
                gpu_count=st.session_state.gpu_count,
                pue_tier=st.session_state.pue_tier,
                workload_mix=st.session_state.get('workload_mix', {'llm': 0.4, 'training': 0.4, 'moe': 0.2})
            )
            
            dcbus = DCBusModule(
                converter_tech=st.session_state.converter_tech,
                grid_capacity_mw=st.session_state.grid_capacity
            )
            
            # Week 2 모듈 초기화
            hess = HESSModule()
            h2_system = H2SystemModule()
            grid = GridInterfaceModule(connection_capacity_mw=st.session_state.grid_capacity)
            
            # PV 시뮬레이션
            pv_data = pv.simulate_time_series(weather_subset)
            
            # AIDC 시뮬레이션
            aidc_data = aidc.simulate_time_series(hours=sim_hours, random_seed=42)
            
            # HESS 시뮬레이션 (기본 운전)
            hess_data = []
            for i in range(min(sim_hours, len(pv_data))):
                # 간단한 HESS 운전: PV 변동에 따른 응답
                pv_power = pv_data.iloc[i]['power_mw'] * 1000  # kW
                operation_result = hess.operate_hess(
                    power_request_kw=pv_power * 0.1,  # PV의 10%를 HESS로
                    duration_s=3600,
                    frequency_hz=0.01
                )
                hess_data.append({
                    'timestamp': pv_data.index[i],
                    'power_delivered_kw': operation_result['power_delivered_kw'],
                    'average_soc': operation_result['average_soc'],
                    'system_efficiency': operation_result['round_trip_efficiency']
                })
            hess_df = pd.DataFrame(hess_data).set_index('timestamp')
            
            # H₂ 시스템 시뮬레이션 (일부 잉여 전력으로 P2G 운전)  
            h2_data = []
            for i in range(0, min(sim_hours, len(pv_data)), 4):  # 4시간마다 운전
                if i + 4 <= len(pv_data):
                    avg_pv = pv_data.iloc[i:i+4]['power_mw'].mean()
                    if avg_pv > 40:  # 40MW 이상일 때 P2G
                        p2g_result = h2_system.power_to_gas((avg_pv - 40) * 1000, 2)  # 여분을 P2G
                        h2_data.append({
                            'timestamp': pv_data.index[i],
                            'operation': 'P2G',
                            'power_kw': p2g_result['electrical_input_kw'],
                            'h2_kg': p2g_result['h2_produced_kg'],
                            'efficiency': p2g_result['electrical_efficiency']
                        })
                    elif avg_pv < 20:  # 20MW 미만일 때 G2P
                        try:
                            g2p_result = h2_system.gas_to_power(10000, 2)  # 10MW G2P
                            h2_data.append({
                                'timestamp': pv_data.index[i],
                                'operation': 'G2P', 
                                'power_kw': g2p_result['electrical_output_kw'],
                                'h2_kg': -g2p_result['h2_consumed_kg'],
                                'efficiency': g2p_result['electrical_efficiency']
                            })
                        except:
                            pass  # H2 부족시 건너뛰기
            h2_df = pd.DataFrame(h2_data).set_index('timestamp') if h2_data else pd.DataFrame()
            
            # 그리드 시뮬레이션 (매 시간 잉여/부족 전력 거래)
            grid_data = []
            for i in range(min(sim_hours, len(pv_data))):
                pv_power = pv_data.iloc[i]['power_mw']
                aidc_power = aidc_data.iloc[i]['total_power_mw']
                surplus = pv_power - aidc_power
                
                if abs(surplus) > 1:  # 1MW 이상 차이날 때 거래
                    try:
                        transaction = grid.execute_grid_transaction(
                            requested_power_mw=-surplus,  # 잉여면 판매(음수), 부족이면 구매(양수)
                            hour=i % 24,
                            season="summer"
                        )
                        if transaction['success']:
                            grid_data.append({
                                'timestamp': pv_data.index[i],
                                'power_mw': transaction['power_delivered_mw'],
                                'revenue_krw': transaction['revenue']['total_revenue_krw'],
                                'smp_price': transaction['revenue']['smp_price_krw_per_mwh']
                            })
                    except:
                        pass  # 거래 실패시 건너뛰기
            grid_df = pd.DataFrame(grid_data).set_index('timestamp') if grid_data else pd.DataFrame()
            
            # DC Bus 시뮬레이션 (전력 균형)
            dcbus_data = dcbus.simulate_time_series(
                pv_data=pv_data,
                aidc_data=aidc_data,
                bess_capacity_mw=200,  # 기본값
                h2_electrolyzer_mw=50,
                h2_fuelcell_mw=30
            )
            
            # 결과 통합
            simulation_result = {
                'weather': weather_subset,
                'pv': pv_data,
                'aidc': aidc_data,
                'dcbus': dcbus_data,
                'hess': hess_df,
                'h2': h2_df,
                'grid': grid_df,
                'modules': {
                    'pv': pv, 'aidc': aidc, 'dcbus': dcbus,
                    'hess': hess, 'h2': h2_system, 'grid': grid
                }
            }
            
            st.session_state.simulation_data = simulation_result
            st.success("시뮬레이션 완료!")
            
        except Exception as e:
            st.error(f"시뮬레이션 오류: {str(e)}")


def display_results():
    """시뮬레이션 결과 표시"""
    data = st.session_state.simulation_data
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 전력 균형", "☀️ PV 발전", "🖥️ AIDC 부하", 
        "🔄 DC Bus", "🔋 HESS", "⚡ H₂ 시스템", "🔌 그리드", "📈 통계 분석"
    ])
    
    with tab1:
        display_power_balance(data)
    
    with tab2:
        display_pv_results(data)
    
    with tab3:
        display_aidc_results(data)
    
    with tab4:
        display_dcbus_results(data)
    
    with tab5:
        display_hess_results(data)
    
    with tab6:
        display_h2_results(data)
    
    with tab7:
        display_grid_results(data)
    
    with tab8:
        display_statistics(data)


def display_power_balance(data):
    """전력 균형 결과 표시"""
    st.subheader("⚖️ 전력 공급 vs 수요")
    
    pv_data = _safe_dict(data['pv'])
    aidc_data = _safe_dict(data['aidc'])
    dcbus_data = _safe_dict(data['dcbus'])
    
    # 시간축 생성
    hours = list(range(len(pv_data['power_mw'])))
    
    # 메인 전력 균형 차트
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['전력 공급 vs 수요 (MW)', '전력 미스매치 (MW)'],
        vertical_spacing=0.1
    )
    
    # 상단: 공급 vs 수요
    fig.add_trace(
        go.Scatter(
            x=hours, y=pv_data['power_mw'],
            name='PV 발전', fill='tonexty',
            line=dict(color=COLOR_PALETTE['pv'])
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hours, y=aidc_data['total_power_mw'],
            name='AIDC 부하',
            line=dict(color=COLOR_PALETTE['aidc'])
        ), row=1, col=1
    )
    
    # 하단: 미스매치
    mismatch = [p - a for p, a in zip(pv_data['power_mw'], aidc_data['total_power_mw'])]
    colors = [COLOR_PALETTE['surplus'] if x >= 0 else COLOR_PALETTE['deficit'] for x in mismatch]
    
    fig.add_trace(
        go.Scatter(
            x=hours, y=mismatch,
            name='잉여/부족',
            fill='tozeroy',
            line=dict(color='gray'),
            fillcolor='rgba(144, 238, 144, 0.3)'  # 연한 녹색
        ), row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="전력 균형 분석"
    )
    
    fig.update_xaxes(title_text="시간 (hour)", row=2, col=1)
    fig.update_yaxes(title_text="전력 (MW)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 요약 통계
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "총 PV 발전량", 
            f"{sum(pv_data['power_mw']):.0f} MWh",
            delta=f"CF: {sum(pv_data['capacity_factor'])/len(pv_data['capacity_factor']):.1%}"
        )
    
    with col2:
        st.metric(
            "총 AIDC 소비량",
            f"{sum(aidc_data['total_power_mw']):.0f} MWh",
            delta=f"평균: {sum(aidc_data['total_power_mw'])/len(aidc_data['total_power_mw']):.1f} MW"
        )
    
    with col3:
        surplus_hours = sum(1 for x in mismatch if x > 0)
        st.metric(
            "잉여 전력 시간",
            f"{surplus_hours}h",
            delta=f"{surplus_hours/len(mismatch):.1%} of time"
        )
    
    with col4:
        deficit_hours = sum(1 for x in mismatch if x < 0)
        st.metric(
            "부족 전력 시간",
            f"{deficit_hours}h", 
            delta=f"{deficit_hours/len(mismatch):.1%} of time"
        )


def display_pv_results(data):
    """PV 발전 결과 표시"""
    st.subheader("☀️ PV 발전 분석")
    
    pv_data = _safe_dict(data['pv'])
    weather_data = _safe_dict(data['weather'])
    pv_module = data['modules']['pv']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # PV 출력 및 일사량
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['PV 발전량 (MW)', '일사량 (W/m²)'],
            specs=[[{"secondary_y": True}], [{}]]
        )
        
        hours = list(range(len(pv_data['power_mw'])))
        
        # PV 출력
        fig.add_trace(
            go.Scatter(
                x=hours, y=pv_data['power_mw'],
                name='PV 출력', 
                line=dict(color=COLOR_PALETTE['pv'])
            ), row=1, col=1
        )
        
        # 셀 온도 (보조 축)
        fig.add_trace(
            go.Scatter(
                x=hours, y=pv_data['cell_temp_celsius'],
                name='셀 온도', yaxis='y2',
                line=dict(color='red', dash='dot')
            ), row=1, col=1
        )
        
        # 일사량
        fig.add_trace(
            go.Scatter(
                x=hours, y=weather_data['ghi_w_per_m2'],
                name='일사량',
                fill='tonexty',
                line=dict(color='orange')
            ), row=2, col=1
        )
        
        fig.update_layout(height=500, title="PV 성능 분석")
        fig.update_xaxes(title_text="시간 (hour)", row=2, col=1)
        fig.update_yaxes(title_text="전력 (MW)", row=1, col=1)
        fig.update_yaxes(title_text="온도 (°C)", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="일사량 (W/m²)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # PV 시스템 정보
        st.write("**시스템 정보**")
        st.write(f"- 기술: {pv_module.params['name']}")
        st.write(f"- 용량: {pv_module.capacity_mw} MW") 
        st.write(f"- 효율: {pv_module.params['eta_stc']}%")
        st.write(f"- 면적: {pv_module.total_area_m2/10000:.1f} ha")
        st.write(f"- 능동제어: {'ON' if pv_module.active_control else 'OFF'}")
        
        # 성능 지표
        stats = pv_module.get_daily_statistics(pv_data)
        
        st.write("**성능 지표**")
        st.metric("총 발전량", f"{stats.get('total_generation_mwh', 0):.1f} MWh")
        st.metric("평균 이용률", f"{stats.get('capacity_factor_avg', 0):.1%}")
        st.metric("최대 셀온도", f"{stats.get('max_cell_temp_celsius', 0):.1f} °C")
        st.metric("운전 시간", f"{stats.get('operating_hours', 0)} h")


def display_aidc_results(data):
    """AIDC 부하 결과 표시"""
    st.subheader("🖥️ AIDC 부하 분석")
    
    aidc_data = _safe_dict(data['aidc'])
    aidc_module = data['modules']['aidc']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 부하 프로파일
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['전력 소비 프로파일 (MW)', 'GPU 활용률 (%)']
        )
        
        hours = list(range(len(aidc_data['total_power_mw'])))
        
        # 전력 소비
        fig.add_trace(
            go.Scatter(
                x=hours, y=aidc_data['total_power_mw'],
                name='총 소비전력',
                line=dict(color=COLOR_PALETTE['aidc'])
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=hours, y=aidc_data['it_power_mw'],
                name='IT 전력',
                line=dict(color='blue', dash='dash')
            ), row=1, col=1
        )
        
        # GPU 활용률
        fig.add_trace(
            go.Scatter(
                x=hours, y=aidc_data['gpu_utilization'] * 100,
                name='GPU 활용률',
                fill='tonexty',
                line=dict(color='green')
            ), row=2, col=1
        )
        
        fig.update_layout(height=500, title="AIDC 부하 분석")
        fig.update_xaxes(title_text="시간 (hour)", row=2, col=1)
        fig.update_yaxes(title_text="전력 (MW)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # AIDC 시스템 정보
        st.write("**시스템 정보**")
        st.write(f"- GPU: {aidc_module.gpu_params['name']}")
        st.write(f"- 수량: {aidc_module.gpu_count:,} 개")
        st.write(f"- GPU 전력: {aidc_module.gpu_params['power_w']} W")
        st.write(f"- PUE: {aidc_module.pue_params['pue']} ({aidc_module.pue_params['name']})")
        st.write(f"- 최대 IT 부하: {aidc_module.max_it_power_mw:.1f} MW")
        st.write(f"- 최대 총 부하: {aidc_module.max_total_power_mw:.1f} MW")
        
        # 워크로드 믹스
        st.write("**워크로드 믹스**")
        for workload, ratio in aidc_module.workload_mix.items():
            name = WORKLOAD_TYPES[workload]['name']
            st.write(f"- {name}: {ratio:.1%}")
        
        # 부하 통계
        stats = aidc_module.get_statistics(aidc_data)
        
        st.write("**부하 통계**")
        st.metric("평균 전력", f"{stats.get('avg_power_mw', 0):.1f} MW")
        st.metric("피크 전력", f"{stats.get('peak_power_mw', 0):.1f} MW")
        st.metric("부하율", f"{stats.get('load_factor', 0):.1%}")
        st.metric("실제 PUE", f"{stats.get('actual_pue', 0):.2f}")


def display_dcbus_results(data):
    """DC Bus 결과 표시"""
    st.subheader("🔄 DC Bus 전력 분배")
    
    dcbus_data = _safe_dict(data['dcbus'])
    dcbus_module = data['modules']['dcbus']
    
    # 전력 흐름 Sankey 다이어그램 (단순화)
    hours = list(range(len(dcbus_data['bess_charge_mw'])))
    
    # 전력 흐름 분석
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            'BESS 충방전 (MW)',
            '그리드 거래 (MW)', 
            'BESS SoC (%)'
        ]
    )
    
    # BESS 충방전
    fig.add_trace(
        go.Scatter(
            x=hours, y=dcbus_data['bess_charge_mw'],
            name='BESS 충전', fill='tonexty',
            line=dict(color='green')
        ), row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hours, y=-dcbus_data['bess_discharge_mw'],
            name='BESS 방전', fill='tonexty',
            line=dict(color='red')
        ), row=1, col=1
    )
    
    # 그리드 거래  
    fig.add_trace(
        go.Scatter(
            x=hours, y=dcbus_data['grid_export_mw'],
            name='그리드 판매', fill='tonexty',
            line=dict(color='blue')
        ), row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=hours, y=-dcbus_data['grid_import_mw'],
            name='그리드 구매', fill='tonexty',
            line=dict(color='orange')  
        ), row=2, col=1
    )
    
    # BESS SoC
    if 'bess_soc' in dcbus_data.columns:
        fig.add_trace(
            go.Scatter(
                x=hours, y=dcbus_data['bess_soc'] * 100,
                name='BESS SoC',
                line=dict(color='purple')
            ), row=3, col=1
        )
    
    fig.update_layout(height=700, title="DC Bus 전력 흐름")
    fig.update_xaxes(title_text="시간 (hour)", row=3, col=1)
    fig.update_yaxes(title_text="전력 (MW)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # DC Bus 통계
    summary = dcbus_module.get_energy_flows_summary(dcbus_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "시스템 효율",
            f"{summary.get('system_efficiency', 0):.1%}"
        )
    
    with col2:
        st.metric(
            "그리드 독립도",
            f"{summary.get('grid_independence_ratio', 0):.1%}"
        )
    
    with col3:
        st.metric(
            "PV 출력제한",
            f"{summary.get('curtailment_ratio', 0):.1%}"
        )
    
    with col4:
        st.metric(
            "총 변환손실",
            f"{summary.get('total_losses_mwh', 0):.1f} MWh"
        )


def display_statistics(data):
    """통계 분석 표시"""
    st.subheader("📈 종합 통계 분석")
    
    # 데이터 준비 (numpy 유지 for 계산, plotly에 넘길때만 변환)
    pv_data_raw = data['pv']
    aidc_data_raw = data['aidc']
    pv_data = _safe_dict(pv_data_raw)
    aidc_data = _safe_dict(data['aidc'])
    dcbus_data = _safe_dict(data['dcbus'])
    
    # 시간별 히트맵 (잉여/부족 전력)
    st.subheader("⏰ 시간대별 전력 미스매치 패턴")
    
    if len(pv_data['power_mw']) >= 168:  # 1주 이상 데이터
        # 주간 패턴 분석
        pv_hourly = np.array(pv_data['power_mw']).reshape(-1, 24)[:7]  # 1주일
        aidc_hourly = np.array(aidc_data['total_power_mw']).reshape(-1, 24)[:7]
        mismatch_hourly = pv_hourly - aidc_hourly
        
        fig = px.imshow(
            mismatch_hourly,
            x=[f"{h:02d}:00" for h in range(24)],
            y=['월', '화', '수', '목', '금', '토', '일'],
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title="주간 전력 미스매치 히트맵 (MW)",
            labels=dict(x="시간", y="요일", color="미스매치 (MW)")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 월별/계절별 통계 (연간 시뮬레이션인 경우)
    if len(pv_data) >= 8760:
        st.subheader("📅 월별 에너지 수지")
        
        # 월별 집계 로직 구현
        # (간단히 하기 위해 생략, 실제로는 날짜 인덱스 기반 그룹화 필요)
        pass
    
    # 핵심 KPI 요약
    st.subheader("🎯 핵심 성능 지표")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**에너지 지표**")
        total_pv = sum(pv_data['power_mw'])
        total_aidc = sum(aidc_data['total_power_mw'])
        
        st.metric("PV 발전량", f"{total_pv:.0f} MWh")
        st.metric("AIDC 소비량", f"{total_aidc:.0f} MWh") 
        st.metric("에너지 자립률", f"{min(total_pv/total_aidc*100, 100):.1f}%" if total_aidc > 0 else "N/A")
        
        # 그리드 의존도
        grid_import = sum(dcbus_data['grid_import_mw'])
        grid_dependence = grid_import / total_aidc * 100 if total_aidc > 0 else 0
        st.metric("그리드 의존도", f"{grid_dependence:.1f}%")
    
    with col2:
        st.write("**효율성 지표**")
        
        # 시스템 전체 효율
        dcbus_module = data['modules']['dcbus']
        summary = dcbus_module.get_energy_flows_summary(dcbus_data)
        
        st.metric("시스템 효율", f"{summary.get('system_efficiency', 0)*100:.1f}%")
        st.metric("변환 손실", f"{summary.get('total_losses_mwh', 0):.1f} MWh")
        
        # 평균 용량 이용률
        avg_pv_cf = sum(pv_data['capacity_factor']) / len(pv_data['capacity_factor']) if pv_data['capacity_factor'] else 0
        aidc_mean = sum(aidc_data['total_power_mw']) / len(aidc_data['total_power_mw']) if aidc_data['total_power_mw'] else 0
        aidc_max = max(aidc_data['total_power_mw']) if aidc_data['total_power_mw'] else 0
        avg_aidc_cf = aidc_mean / aidc_max if aidc_max > 0 else 0
        
        st.metric("PV 이용률", f"{avg_pv_cf:.1%}")
        st.metric("AIDC 부하율", f"{avg_aidc_cf:.1%}")


def display_hess_results(data):
    """HESS 결과 표시"""
    st.subheader("🔋 HESS (Hybrid Energy Storage System)")
    
    if 'hess' not in data or data['hess'].empty:
        st.warning("HESS 데이터가 없습니다.")
        return
    
    hess_data = data['hess']
    hess_module = data['modules']['hess']
    
    # 시스템 상태
    system_status = hess_module.get_system_status()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "총 저장 용량", 
            f"{system_status['system_total']['capacity_kwh']/1000:.0f} MWh",
            f"평균 SOC: {system_status['system_total']['average_soc']:.1%}"
        )
    with col2:
        st.metric(
            "시스템 효율",
            f"{system_status['system_total']['system_efficiency']:.1%}"
        )
    with col3:
        avg_power = hess_data['power_delivered_kw'].mean()
        st.metric(
            "평균 운전 전력",
            f"{avg_power/1000:.1f} MW"
        )
    
    # 레이어별 상태 차트
    st.subheader("레이어별 SOC 상태")
    layer_soc_data = pd.DataFrame({
        layer: [info['soc']] 
        for layer, info in system_status['layers'].items()
    })
    
    fig = px.bar(
        x=layer_soc_data.columns,
        y=layer_soc_data.iloc[0],
        title="HESS 레이어별 SOC",
        labels={'x': '레이어', 'y': 'SOC (%)'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 시간별 운전 차트
    st.subheader("HESS 운전 이력")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hess_data.index,
        y=hess_data['power_delivered_kw'] / 1000,
        mode='lines',
        name='운전 전력 (MW)',
        line=dict(color=COLOR_PALETTE['bess'])
    ))
    
    fig.add_trace(go.Scatter(
        x=hess_data.index,
        y=hess_data['average_soc'] * 100,
        mode='lines',
        name='평균 SOC (%)',
        yaxis='y2',
        line=dict(color=COLOR_PALETTE['pv'])
    ))
    
    fig.update_layout(
        title="HESS 전력 및 SOC",
        xaxis_title="시간",
        yaxis_title="전력 (MW)",
        yaxis2=dict(
            title="SOC (%)",
            overlaying='y',
            side='right'
        ),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def display_h2_results(data):
    """H₂ 시스템 결과 표시"""
    st.subheader("⚡ H₂ System (Power-to-Gas-to-Power)")
    
    h2_module = data['modules']['h2']
    system_status = h2_module.get_system_status()
    
    # 시스템 상태 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "H₂ 저장량",
            f"{system_status['storage']['inventory_kg']:,.0f} kg",
            f"충전율: {system_status['storage']['fill_level']:.1%}"
        )
    with col2:
        st.metric(
            "저장 용량",
            f"{system_status['storage']['capacity_kg']:,.0f} kg",
            f"유형: {system_status['storage']['storage_type']}"
        )
    with col3:
        st.metric(
            "SOEC 상태",
            "온라인" if system_status['soec']['online'] else "오프라인",
            f"열화율: {system_status['soec']['degradation']:.1%}"
        )
    with col4:
        st.metric(
            "SOFC 상태",
            "온라인" if system_status['sofc']['online'] else "오프라인",
            f"열화율: {system_status['sofc']['degradation']:.1%}"
        )
    
    # Round-trip 효율
    try:
        rt_eff = h2_module.calculate_round_trip_efficiency()
        if 'error' not in rt_eff:
            st.subheader("Round-Trip 효율")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "전기 효율",
                    f"{rt_eff['electrical_round_trip_efficiency']:.1%}",
                    "전기 → H₂ → 전기"
                )
            with col2:
                st.metric(
                    "CHP 효율",
                    f"{rt_eff['chp_round_trip_efficiency']:.1%}",
                    "열 회수 포함"
                )
    except:
        st.info("Round-trip 효율 계산을 위한 데이터가 부족합니다.")
    
    # H₂ 운전 이력
    if 'h2' in data and not data['h2'].empty:
        h2_data = data['h2']
        
        st.subheader("H₂ 운전 이력")
        
        # P2G vs G2P 운전량
        p2g_data = h2_data[h2_data['operation'] == 'P2G']
        g2p_data = h2_data[h2_data['operation'] == 'G2P']
        
        col1, col2 = st.columns(2)
        with col1:
            if not p2g_data.empty:
                st.metric(
                    "P2G 운전",
                    f"{len(p2g_data)} 회",
                    f"총 {p2g_data['h2_kg'].sum():.1f} kg H₂ 생산"
                )
        with col2:
            if not g2p_data.empty:
                st.metric(
                    "G2P 운전",
                    f"{len(g2p_data)} 회",
                    f"총 {abs(g2p_data['h2_kg'].sum()):.1f} kg H₂ 소비"
                )
        
        # 운전 차트
        fig = go.Figure()
        
        if not p2g_data.empty:
            fig.add_trace(go.Scatter(
                x=p2g_data.index,
                y=p2g_data['power_kw'] / 1000,
                mode='markers',
                marker=dict(size=10, color=COLOR_PALETTE['pv']),
                name='P2G (MW)'
            ))
        
        if not g2p_data.empty:
            fig.add_trace(go.Scatter(
                x=g2p_data.index,
                y=g2p_data['power_kw'] / 1000,
                mode='markers',
                marker=dict(size=10, color=COLOR_PALETTE['h2']),
                name='G2P (MW)'
            ))
        
        fig.update_layout(
            title="H₂ 시스템 운전 이력",
            xaxis_title="시간",
            yaxis_title="전력 (MW)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("H₂ 운전 데이터가 없습니다.")


def display_grid_results(data):
    """그리드 결과 표시"""
    st.subheader("🔌 Grid Interface")
    
    grid_module = data['modules']['grid']
    
    # 거래 통계
    try:
        stats = grid_module.get_trading_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "총 거래 횟수",
                f"{stats['total_transactions']} 회"
            )
        with col2:
            st.metric(
                "구매 전력량",
                f"{stats['total_energy_imported_mwh']:.1f} MWh"
            )
        with col3:
            st.metric(
                "판매 전력량", 
                f"{stats['total_energy_exported_mwh']:.1f} MWh"
            )
        with col4:
            st.metric(
                "총 수익",
                f"{stats['total_revenue_krw']:,.0f} ₩"
            )
        
        # 에너지 균형
        net_balance = stats['net_energy_balance_mwh']
        balance_type = "순 구매" if net_balance > 0 else "순 판매"
        st.metric(
            "에너지 균형",
            f"{abs(net_balance):.1f} MWh ({balance_type})",
            f"평균 SMP: {stats['average_smp_price_krw_per_mwh']:,.0f} ₩/MWh"
        )
        
    except:
        st.info("거래 통계를 계산할 수 없습니다.")
    
    # 그리드 거래 이력
    if 'grid' in data and not data['grid'].empty:
        grid_data = data['grid']
        
        st.subheader("그리드 거래 이력")
        
        # 매매 구분
        buy_data = grid_data[grid_data['power_mw'] > 0]  # 구매 (양수)
        sell_data = grid_data[grid_data['power_mw'] < 0]  # 판매 (음수)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "구매 거래",
                f"{len(buy_data)} 회",
                f"총 비용: {abs(buy_data['revenue_krw'].sum()):,.0f} ₩" if not buy_data.empty else ""
            )
        with col2:
            st.metric(
                "판매 거래", 
                f"{len(sell_data)} 회",
                f"총 수익: {sell_data['revenue_krw'].sum():,.0f} ₩" if not sell_data.empty else ""
            )
        
        # 거래 차트
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=grid_data.index,
            y=grid_data['power_mw'],
            mode='markers+lines',
            marker=dict(
                size=8,
                color=grid_data['power_mw'],
                colorscale='RdYlBu',
                colorbar=dict(title="전력 (MW)")
            ),
            name='거래 전력',
            line=dict(color=COLOR_PALETTE['grid'])
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            title="그리드 거래 전력 (양수: 구매, 음수: 판매)",
            xaxis_title="시간",
            yaxis_title="전력 (MW)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # SMP 가격 차트
        st.subheader("SMP 가격 추이")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=grid_data.index,
            y=grid_data['smp_price'],
            mode='lines',
            name='SMP 가격',
            line=dict(color=COLOR_PALETTE['surplus'])
        ))
        
        fig2.update_layout(
            title="시간대별 SMP 가격",
            xaxis_title="시간", 
            yaxis_title="SMP 가격 (₩/MWh)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.info("그리드 거래 데이터가 없습니다.")


if __name__ == "__main__":
    create_main_dashboard()