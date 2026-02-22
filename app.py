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
    HESSModule, H2SystemModule, GridInterfaceModule,
    AIEMSModule, CarbonAccountingModule, EconomicsModule,
    PolicySimulator, IndustryModel, InvestmentDashboard
)
from config import (
    PV_TYPES, GPU_TYPES, PUE_TIERS, WORKLOAD_TYPES, 
    CONVERTER_EFFICIENCY, UI_CONFIG, COLOR_PALETTE,
    HESS_LAYER_CONFIGS, H2_SYSTEM_CONFIG, GRID_TARIFF_CONFIG,
    AI_EMS_CONFIG, CARBON_CONFIG, ECONOMICS_CONFIG
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

# 클린 화이트 테마 (Streamlit 기본)


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
        
        # 언어 선택
        language = st.selectbox(
            "🌐 Language", ["KO", "EN", "CN"],
            key="language", index=0)
        
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
        
        # 정규화 및 세션 상태에 저장
        total_ratio = llm_ratio + training_ratio + moe_ratio
        if total_ratio > 0:
            workload_mix = {
                'llm': llm_ratio / total_ratio,
                'training': training_ratio / total_ratio,
                'moe': moe_ratio / total_ratio
            }
        else:
            workload_mix = {'llm': 1.0, 'training': 0.0, 'moe': 0.0}
        
        st.session_state.workload_mix = workload_mix
        
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
        
        # Week 3: 경제/탄소 파라미터
        st.subheader("💰 M9. 경제/탄소")
        carbon_price = st.slider(
            "탄소가격 (₩/tCO₂)", 10000, 100000, 25000, 5000, key="carbon_price"
        )
        discount_rate = st.slider(
            "할인율 (%)", 1.0, 15.0, 5.0, 0.5, key="discount_rate"
        )
        electricity_price = st.slider(
            "전력단가 (₩/MWh)", 50000, 150000, 80000, 5000, key="elec_price"
        )
        learning_curve_on = st.checkbox(
            "학습곡선 적용", value=False, key="learning_curve"
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
                workload_mix=st.session_state.workload_mix
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
            
            # Week 3: AI-EMS 디스패치
            ems = AIEMSModule()
            ems_dispatches = []
            ems_soc = 0.5
            ems_h2 = 0.5
            for i in range(min(sim_hours, len(pv_data))):
                pv_mw = pv_data.iloc[i]['power_mw']
                aidc_mw = aidc_data.iloc[i]['total_power_mw']
                hour = i % 24
                cmd = ems.execute_dispatch(
                    pv_power_mw=pv_mw, aidc_load_mw=aidc_mw,
                    hess_soc=ems_soc, h2_storage_level=ems_h2,
                    grid_price_krw=st.session_state.get('elec_price', 80000),
                    hour_of_day=hour,
                )
                ems_dispatches.append(cmd.to_dict())
                ems_soc = float(np.clip(ems_soc + (cmd.pv_to_hess_mw - cmd.hess_to_aidc_mw) / 2000, 0, 1))
                ems_h2 = float(np.clip(ems_h2 + (cmd.h2_electrolyzer_mw - cmd.h2_fuelcell_mw) / 5000, 0, 1))
            ems_df = pd.DataFrame(ems_dispatches)
            ems_kpi = ems.calculate_kpi()

            # Week 3: 탄소 회계
            carbon = CarbonAccountingModule(
                k_ets_price=st.session_state.get('carbon_price', 25000)
            )
            carbon_records = []
            for i in range(min(sim_hours, len(pv_data))):
                grid_mwh = ems_dispatches[i]['grid_to_aidc_mw']
                pv_self_mwh = ems_dispatches[i]['pv_to_aidc_mw']
                rec = carbon.calculate_hourly_emissions(grid_mwh, pv_self_mwh, hour=i)
                carbon_records.append(rec.to_dict())
            carbon_df = pd.DataFrame(carbon_records)

            # Week 3: 경제성
            economics = EconomicsModule()

            # 결과 통합
            simulation_result = {
                'weather': weather_subset,
                'pv': pv_data,
                'aidc': aidc_data,
                'dcbus': dcbus_data,
                'hess': hess_df,
                'h2': h2_df,
                'grid': grid_df,
                'ems_df': ems_df,
                'ems_kpi': ems_kpi,
                'carbon_df': carbon_df,
                'modules': {
                    'pv': pv, 'aidc': aidc, 'dcbus': dcbus,
                    'hess': hess, 'h2': h2_system, 'grid': grid,
                    'ems': ems, 'carbon': carbon, 'economics': economics
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, \
        tab12, tab13, tab14, tab15 = st.tabs([
        "📊 전력 균형", "☀️ PV 발전", "🖥️ AIDC 부하", 
        "🔄 DC Bus", "🔋 HESS", "⚡ H₂ 시스템", "🔌 그리드",
        "🤖 AI-EMS", "🌍 탄소 회계", "💰 경제성", "📈 통계 분석",
        "🏛️ 정책 시뮬레이터", "🏭 산업 상용화", "📋 투자 대시보드", "📚 References"
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
        display_ems_results(data)
    
    with tab9:
        display_carbon_results(data)
    
    with tab10:
        display_economics_results(data)
    
    with tab11:
        display_statistics(data)
    
    with tab12:
        display_policy_simulator()
    
    with tab13:
        display_industry_model()
    
    with tab14:
        display_investment_dashboard()
    
    with tab15:
        display_references()


def display_static_energy_flow_sankey(data):
    """정적 에너지 흐름 요약 Sankey 다이어그램 표시 (전체 시뮬레이션 기간 누적)"""
    
    # 데이터 추출
    pv_data = _safe_dict(data['pv'])
    aidc_data = _safe_dict(data['aidc'])
    dcbus_data = _safe_dict(data['dcbus'])
    
    # 전체 시뮬레이션 기간의 누적 에너지 계산 (MWh)
    import pandas as pd
    
    def _safe_sum(d, key):
        """Safely sum a list, Series, or array from a dict."""
        val = d.get(key, [])
        if val is None:
            return 0
        if isinstance(val, pd.Series):
            return float(val.sum())
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return float(sum(val))
        try:
            return float(sum(val))
        except (TypeError, ValueError):
            return 0
    
    pv_total = _safe_sum(pv_data, 'power_mw')
    aidc_total = _safe_sum(aidc_data, 'total_power_mw') or _safe_sum(aidc_data, 'power_mw')
    
    # HESS 데이터
    hess_charge_total = _safe_sum(dcbus_data, 'bess_charge_mw')
    hess_discharge_total = _safe_sum(dcbus_data, 'bess_discharge_mw')
    
    # Grid 데이터  
    grid_import_total = _safe_sum(dcbus_data, 'grid_import_mw')
    grid_export_total = _safe_sum(dcbus_data, 'grid_export_mw')
    
    # H2 시스템 데이터
    # H2 데이터 — DC Bus가 이미 정확한 값을 가지고 있음
    h2_electrolyzer_total = _safe_sum(dcbus_data, 'h2_electrolyzer_mw')
    h2_fuelcell_total = _safe_sum(dcbus_data, 'h2_fuelcell_mw')
    
    # Curtailment (출력제한) — DC Bus 실제 데이터 사용
    curtailment_total = _safe_sum(dcbus_data, 'curtailment_mw')
    
    # === Sankey 다이어그램 (GDI 스타일: 깔끔한 좌→우, 세련된 색상) ===
    
    # 노드: 0-3 좌측(소스), 4 중앙(DC Bus), 5-9 우측(싱크)
    node_labels = [
        "Solar PV",        # 0
        "HESS 방전",       # 1
        "H₂ Fuel Cell",   # 2
        "Grid Import",     # 3
        "DC Bus",          # 4
        "AIDC",            # 5
        "HESS 충전",       # 6
        "H₂ 전해조",       # 7
        "Grid Export",     # 8
        "Curtailment",     # 9
    ]
    
    # 세련된 GDI 톤 (파스텔 + 다크 배경 조화)
    node_colors = [
        "#e6a817",  # PV — 머스타드 골드
        "#2dd4bf",  # HESS 방전 — 틸
        "#4ade80",  # H2 FC — 소프트 그린
        "#818cf8",  # Grid Import — 인디고
        "#475569",  # DC Bus — 슬레이트 그레이
        "#f87171",  # AIDC — 소프트 레드
        "#2dd4bf",  # HESS 충전 — 틸
        "#4ade80",  # H2 전해조 — 소프트 그린
        "#818cf8",  # Grid Export — 인디고
        "#64748b",  # Curtailment — 슬레이트
    ]
    
    # 링크 구성 (값 > 0.1 인 것만)
    links = [
        (0, 4, pv_total,              "rgba(230,168,23,0.35)"),
        (1, 4, hess_discharge_total,  "rgba(45,212,191,0.35)"),
        (2, 4, h2_fuelcell_total,     "rgba(74,222,128,0.35)"),
        (3, 4, grid_import_total,     "rgba(129,140,248,0.35)"),
        (4, 5, aidc_total,            "rgba(248,113,113,0.35)"),
        (4, 6, hess_charge_total,     "rgba(45,212,191,0.35)"),
        (4, 7, h2_electrolyzer_total, "rgba(74,222,128,0.35)"),
        (4, 8, grid_export_total,     "rgba(129,140,248,0.35)"),
        (4, 9, curtailment_total,     "rgba(100,116,139,0.35)"),
    ]
    
    source_nodes = [s for s, t, v, c in links if v > 0.1]
    target_nodes = [t for s, t, v, c in links if v > 0.1]
    values =       [v for s, t, v, c in links if v > 0.1]
    link_colors =  [c for s, t, v, c in links if v > 0.1]
    
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=30,
            thickness=25,
            line=dict(color="rgba(255,255,255,0.08)", width=0.5),
            label=node_labels,
            color=node_colors,
            x=[0.01, 0.01, 0.01, 0.01,  0.45,  0.99, 0.99, 0.99, 0.99, 0.99],
            y=[0.2,  0.4,  0.6,  0.8,   0.5,   0.1,  0.35, 0.55, 0.75, 0.95],
        ),
        link=dict(
            source=source_nodes,
            target=target_nodes,
            value=values,
            color=link_colors,
        ),
    )])
    
    fig.update_layout(
        title=dict(text="에너지 흐름 요약 (전체 시뮬레이션 기간)", font=dict(size=14)),
        font=dict(size=11),
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    # 요약 메트릭 표시
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pv_len = len(pv_data.get('power_mw', [])) or 1
        st.metric("☀️ PV 발전", f"{pv_total:.0f} MWh", 
                 delta=f"평균: {pv_total/pv_len:.1f} MW")
    
    with col2:
        aidc_len = len(aidc_data.get('total_power_mw', aidc_data.get('power_mw', []))) or 1
        st.metric("🖥️ AIDC 소비", f"{aidc_total:.0f} MWh",
                 delta=f"평균: {aidc_total/aidc_len:.1f} MW")
    
    with col3:
        hess_net = hess_discharge_total - hess_charge_total
        st.metric("🔋 HESS 순", f"{hess_net:+.0f} MWh", 
                 delta=f"{'방전' if hess_net > 0 else '충전'} 우세")
    
    with col4:
        h2_net = h2_fuelcell_total - h2_electrolyzer_total
        st.metric("💧 H₂ 순", f"{h2_net:+.0f} MWh",
                 delta=f"{'발전' if h2_net > 0 else '전해'} 우세")
    
    with col5:
        grid_net = grid_export_total - grid_import_total
        st.metric("🔌 Grid 순", f"{grid_net:+.0f} MWh",
                 delta=f"{'수출' if grid_net > 0 else '수입'} 우세")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")  # 구분선


def display_power_balance(data):
    """전력 균형 결과 표시"""
    
    # ⚡ 정적 에너지 흐름 요약 Sankey 다이어그램
    st.subheader("⚡ 에너지 흐름 요약")
    
    # 정적 Sankey 표시
    try:
        display_static_energy_flow_sankey(data)
    except Exception as e:
        st.warning(f"에너지 흐름 다이어그램 로딩 중 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
    
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
        template='plotly_white',
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
        
        fig.update_layout(height=500, title="PV 성능 분석", template='plotly_white')
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
        
        fig.update_layout(height=500, title="AIDC 부하 분석", template='plotly_white')
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
        stats = aidc_module.get_statistics(data['aidc'])
        
        st.write("**부하 통계**")
        st.metric("평균 전력", f"{stats.get('avg_power_mw', 0):.1f} MW")
        st.metric("피크 전력", f"{stats.get('peak_power_mw', 0):.1f} MW")
        st.metric("부하율", f"{stats.get('load_factor', 0):.1%}")
        st.metric("실제 PUE", f"{stats.get('actual_pue', 0):.2f}")
    
    # 분단위 줌인 차트
    st.subheader("🔬 AIDC 부하 줌인 (분단위 해상도)")
    st.caption("특정 시간대의 분단위 전력 변동을 시뮬레이션합니다. LLM burst, checkpoint spike, GPU throttling 등 실제 AIDC 이벤트를 반영합니다.")
    
    zoom_col1, zoom_col2 = st.columns([1, 3])
    with zoom_col1:
        zoom_hour = st.selectbox("줌인 시간대", list(range(24)), index=14, format_func=lambda h: f"{h:02d}:00")
    
    minute_data = aidc_module.simulate_minute_resolution(
        hour_of_day=zoom_hour, day_of_week=2, minutes=60, random_seed=zoom_hour * 7
    )
    
    minutes = [d['minute'] for d in minute_data]
    powers = [d['total_power_mw'] for d in minute_data]
    events = [d['event'] for d in minute_data]
    
    # 이벤트별 색상
    event_colors = {
        'normal': 'rgba(100,100,100,0.3)',
        'llm_burst': 'rgba(255,100,100,0.8)',
        'checkpoint': 'rgba(255,200,0,0.8)',
        'expert_activation': 'rgba(100,200,255,0.8)',
        'throttling': 'rgba(150,150,255,0.8)',
        'gpu_failure': 'rgba(255,0,0,0.9)'
    }
    marker_colors = [event_colors.get(e, 'gray') for e in events]
    
    fig_zoom = go.Figure()
    fig_zoom.add_trace(go.Scatter(
        x=minutes, y=powers,
        mode='lines+markers',
        line=dict(color=COLOR_PALETTE['aidc'], width=1.5),
        marker=dict(size=5, color=marker_colors),
        name='전력 (MW)',
        hovertemplate='%{x}분: %{y:.2f} MW<br>이벤트: %{text}',
        text=events
    ))
    
    fig_zoom.update_layout(
        height=350,
        title=f"AIDC 부하 분단위 프로파일 ({zoom_hour:02d}:00-{zoom_hour:02d}:59)",
        template='plotly_white',
        xaxis_title="분 (minute)",
        yaxis_title="전력 (MW)",
        showlegend=False
    )
    
    st.plotly_chart(fig_zoom, use_container_width=True)
    
    # 이벤트 범례
    event_counts = {}
    for e in events:
        event_counts[e] = event_counts.get(e, 0) + 1
    
    event_labels = {
        'normal': '정상 운영', 'llm_burst': '🔴 LLM Burst',
        'checkpoint': '🟡 Checkpoint Spike', 'expert_activation': '🔵 Expert Activation',
        'throttling': '🟣 GPU Throttling', 'gpu_failure': '⛔ GPU Failure'
    }
    
    legend_parts = [f"{event_labels.get(k,k)}: {v}회" for k, v in event_counts.items() if k != 'normal']
    if legend_parts:
        st.caption("이벤트: " + " | ".join(legend_parts))


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
    
    fig.update_layout(height=700, title="DC Bus 전력 흐름", template='plotly_white')
    fig.update_xaxes(title_text="시간 (hour)", row=3, col=1)
    fig.update_yaxes(title_text="전력 (MW)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # DC Bus 통계
    summary = dcbus_module.get_energy_flows_summary(data['dcbus'])
    
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


def display_ems_results(data):
    """AI-EMS 결과 표시"""
    st.subheader("🤖 AI-EMS 디스패치")
    
    if 'ems_df' not in data or data['ems_df'].empty:
        st.warning("AI-EMS 데이터가 없습니다.")
        return
    
    ems_df = data['ems_df']
    kpi = data.get('ems_kpi', {})
    
    # KPI 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("자급률", f"{kpi.get('self_sufficiency_ratio', 0):.1%}")
    with col2:
        st.metric("피크 감축률", f"{kpi.get('peak_reduction_ratio', 0):.1%}")
    with col3:
        st.metric("재생에너지 비율", f"{kpi.get('renewable_fraction', 0):.1%}")
    with col4:
        st.metric("평균 응답시간", f"{kpi.get('avg_response_time_ms', 0):.2f} ms")
    
    # 디스패치 Stacked Bar
    hours = list(range(len(ems_df)))
    fig = go.Figure()
    
    for col_name, label, color in [
        ('pv_to_aidc_mw', 'PV→AIDC', COLOR_PALETTE['pv']),
        ('hess_to_aidc_mw', 'HESS→AIDC', COLOR_PALETTE['bess']),
        ('grid_to_aidc_mw', 'Grid→AIDC', COLOR_PALETTE['grid']),
        ('h2_fuelcell_mw', 'H₂→AIDC', COLOR_PALETTE['h2']),
    ]:
        if col_name in ems_df.columns:
            fig.add_trace(go.Bar(
                x=hours, y=ems_df[col_name].tolist(),
                name=label, marker_color=color,
            ))
    
    fig.update_layout(
        barmode='stack', height=450,
        title="AIDC 공급원 구성 (Stacked)",
        template='plotly_white',
        xaxis_title="시간", yaxis_title="전력 (MW)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 잉여 전력 배분
    fig2 = go.Figure()
    for col_name, label, color in [
        ('pv_to_hess_mw', 'PV→HESS', COLOR_PALETTE['bess']),
        ('pv_to_grid_mw', 'PV→Grid', COLOR_PALETTE['grid']),
        ('h2_electrolyzer_mw', 'PV→H₂', COLOR_PALETTE['h2']),
        ('curtailment_mw', 'Curtailment', '#999999'),
    ]:
        if col_name in ems_df.columns:
            fig2.add_trace(go.Bar(
                x=hours, y=ems_df[col_name].tolist(),
                name=label, marker_color=color,
            ))
    fig2.update_layout(
        barmode='stack', height=350,
        title="잉여 전력 배분",
        template='plotly_white',
        xaxis_title="시간", yaxis_title="전력 (MW)"
    )
    st.plotly_chart(fig2, use_container_width=True)


def display_carbon_results(data):
    """탄소 회계 결과 표시"""
    st.subheader("🌍 탄소 배출 대시보드")
    
    if 'carbon_df' not in data or data['carbon_df'].empty:
        st.warning("탄소 데이터가 없습니다.")
        return
    
    carbon_df = data['carbon_df']
    carbon_module = data['modules'].get('carbon')
    
    # 총 배출 요약
    total_s1 = carbon_df['scope1_tco2'].sum()
    total_s2 = carbon_df['scope2_tco2'].sum()
    total_s3 = carbon_df['scope3_tco2'].sum()
    total_avoided = carbon_df['avoided_tco2'].sum()
    total_net = carbon_df['net_tco2'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Scope 2 배출", f"{total_s2:.1f} tCO₂")
    with col2:
        st.metric("Scope 3 배출", f"{total_s3:.1f} tCO₂")
    with col3:
        st.metric("회피 배출", f"{total_avoided:.1f} tCO₂", delta=f"-{total_avoided:.0f}")
    with col4:
        st.metric("순 배출", f"{total_net:.1f} tCO₂")
    
    # Scope 파이차트
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(
            values=[total_s1, total_s2, total_s3],
            names=['Scope 1 (직접)', 'Scope 2 (전력)', 'Scope 3 (공급망)'],
            title="배출 구성 (Scope 1/2/3)",
            color_discrete_sequence=[COLOR_PALETTE['scope1'], COLOR_PALETTE['scope2'], COLOR_PALETTE['scope3']]
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 시계열
        fig_ts = go.Figure()
        hours = list(range(len(carbon_df)))
        fig_ts.add_trace(go.Scatter(
            x=hours, y=carbon_df['scope2_tco2'].cumsum().tolist(),
            name='누적 Scope 2', fill='tozeroy',
            line=dict(color=COLOR_PALETTE['scope2'])
        ))
        fig_ts.add_trace(go.Scatter(
            x=hours, y=carbon_df['avoided_tco2'].cumsum().tolist(),
            name='누적 회피', fill='tozeroy',
            line=dict(color=COLOR_PALETTE['carbon'])
        ))
        fig_ts.update_layout(title="누적 탄소 배출/회피", height=400,
                             xaxis_title="시간", yaxis_title="tCO₂", template='plotly_white')
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # K-ETS / CBAM 분석
    if carbon_module:
        st.subheader("K-ETS & CBAM 시나리오")
        col1, col2 = st.columns(2)
        with col1:
            kets = carbon_module.calculate_k_ets_cost_or_revenue(total_net, baseline_tco2=total_s2 * 0.9)
            if kets["status"] == "credit_available":
                st.success(f"탄소크레딧 판매 가능: {kets['surplus_tco2']:.0f} tCO₂ → {kets['revenue_krw']:,.0f}₩")
            else:
                st.warning(f"배출권 구매 필요: {kets['excess_tco2']:.0f} tCO₂ → {kets['cost_krw']:,.0f}₩")
        with col2:
            cbam = carbon_module.calculate_cbam_cost(100)
            st.info(f"CBAM 예시 (100 tCO₂ 수출): {cbam['cbam_cost_krw']:,.0f}₩ ({cbam['cbam_cost_eur']:,.0f}€)")


def display_economics_results(data):
    """경제성 대시보드"""
    st.subheader("💰 경제성 분석")
    
    econ = data['modules'].get('economics')
    if not econ:
        st.warning("경제성 모듈이 없습니다.")
        return
    
    # Base case
    with st.spinner("경제성 분석 중..."):
        base = econ.run_base_case()
    
    # 헤드라인 메트릭
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CAPEX", f"{base['capex_billion_krw']:,.0f}억원")
    with col2:
        st.metric("IRR", f"{base['irr_pct']:.1f}%")
    with col3:
        st.metric("NPV", f"{base['npv_billion_krw']:,.0f}억원")
    with col4:
        st.metric("회수기간", f"{base['payback_years']:.1f}년")
    
    # CAPEX 구성
    col1, col2 = st.columns(2)
    with col1:
        items = base['capex_breakdown']
        fig_capex = px.pie(
            values=list(items.values()),
            names=list(items.keys()),
            title="CAPEX 구성",
        )
        st.plotly_chart(fig_capex, use_container_width=True)
    
    with col2:
        # 연간 현금흐름
        cfs = base['annual_cashflows']
        cumulative = np.cumsum(cfs).tolist()
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=list(range(1, len(cfs)+1)), y=cfs,
            name='연간 순현금흐름', marker_color=COLOR_PALETTE['economics']
        ))
        fig_cf.add_trace(go.Scatter(
            x=list(range(1, len(cumulative)+1)), y=cumulative,
            name='누적', line=dict(color='red')
        ))
        fig_cf.add_hline(y=base['capex_billion_krw'], line_dash="dash", line_color="gray",
                         annotation_text="CAPEX")
        fig_cf.update_layout(title="연간 현금흐름 (억원)", height=400,
                             xaxis_title="연차", yaxis_title="억원", template='plotly_white')
        st.plotly_chart(fig_cf, use_container_width=True)
    
    # Monte Carlo
    st.subheader("📊 Monte Carlo 민감도 분석")
    mc_iterations = st.selectbox("MC 반복 횟수", [100, 1000, 5000, 10000], index=1)
    
    if st.button("Monte Carlo 실행"):
        with st.spinner(f"Monte Carlo {mc_iterations}회 실행 중..."):
            mc = econ.run_monte_carlo(n_iterations=mc_iterations)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("IRR 평균", f"{mc['irr_mean']*100:.1f}%")
            st.metric("IRR 범위 (5-95%)", f"{mc['irr_p5']*100:.1f}% ~ {mc['irr_p95']*100:.1f}%")
            st.metric("NPV>0 확률", f"{mc['prob_positive_npv']*100:.1f}%")
        
        with col2:
            # IRR 히스토그램
            fig_hist = px.histogram(
                x=[x*100 for x in mc['irr_distribution']],
                nbins=50, title="IRR 분포",
                labels={'x': 'IRR (%)', 'y': '빈도'}
            )
            fig_hist.add_vline(x=mc['irr_mean']*100, line_dash="dash", line_color="red",
                              annotation_text=f"평균 {mc['irr_mean']*100:.1f}%")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # 토네이도 차트
    st.subheader("🌪️ 민감도 토네이도")
    tornado = econ.sensitivity_tornado(base['irr'])
    
    fig_tornado = go.Figure()
    for item in reversed(tornado):
        fig_tornado.add_trace(go.Bar(
            y=[item['variable']], x=[item['irr_high']*100 - base['irr_pct']],
            orientation='h', name=f"{item['variable']} (상)", marker_color='green',
            showlegend=False
        ))
        fig_tornado.add_trace(go.Bar(
            y=[item['variable']], x=[item['irr_low']*100 - base['irr_pct']],
            orientation='h', name=f"{item['variable']} (하)", marker_color='red',
            showlegend=False
        ))
    fig_tornado.update_layout(
        title=f"IRR 민감도 (Base: {base['irr_pct']:.1f}%)",
        xaxis_title="IRR 변동 (%p)", barmode='overlay', height=400, template='plotly_white'
    )
    st.plotly_chart(fig_tornado, use_container_width=True)
    
    # 과장 금지 경고
    report = econ.get_summary_report(base)
    st.info(report["confidence_note"])



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
            mismatch_hourly.tolist(),  # Convert numpy array to list
            x=[f"{h:02d}:00" for h in range(24)],
            y=['월', '화', '수', '목', '금', '토', '일'],
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title="주간 전력 미스매치 히트맵 (MW)",
            labels=dict(x="시간", y="요일", color="미스매치 (MW)")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 월별/계절별 통계 (연간 시뮬레이션인 경우)
    if len(pv_data['power_mw']) >= 8760:
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
        summary = dcbus_module.get_energy_flows_summary(data['dcbus'])
        
        st.metric("시스템 효율", f"{summary.get('system_efficiency', 0)*100:.1f}%")
        st.metric("변환 손실", f"{summary.get('total_losses_mwh', 0):.1f} MWh")
        
        # 평균 용량 이용률
        avg_pv_cf = sum(pv_data['capacity_factor']) / len(pv_data['capacity_factor']) if len(pv_data['capacity_factor']) > 0 else 0
        aidc_mean = sum(aidc_data['total_power_mw']) / len(aidc_data['total_power_mw']) if len(aidc_data['total_power_mw']) > 0 else 0
        aidc_max = max(aidc_data['total_power_mw']) if len(aidc_data['total_power_mw']) > 0 else 0
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
        x=list(layer_soc_data.columns),
        y=layer_soc_data.iloc[0].tolist(),
        title="HESS 레이어별 SOC",
        labels={'x': '레이어', 'y': 'SOC (%)'}
    )
    fig.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # 시간별 운전 차트
    st.subheader("HESS 운전 이력")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(hess_data.index),
        y=(hess_data['power_delivered_kw'] / 1000).tolist(),
        mode='lines',
        name='운전 전력 (MW)',
        line=dict(color=COLOR_PALETTE['bess'])
    ))
    
    fig.add_trace(go.Scatter(
        x=list(hess_data.index),
        y=(hess_data['average_soc'] * 100).tolist(),
        mode='lines',
        name='평균 SOC (%)',
        yaxis='y2',
        line=dict(color=COLOR_PALETTE['pv'])
    ))
    
    fig.update_layout(
        title="HESS 전력 및 SOC",
        xaxis_title="시간",
        template='plotly_white',
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
                x=list(p2g_data.index),
                y=(p2g_data['power_kw'] / 1000).tolist(),
                mode='markers',
                marker=dict(size=10, color=COLOR_PALETTE['pv']),
                name='P2G (MW)'
            ))
        
        if not g2p_data.empty:
            fig.add_trace(go.Scatter(
                x=list(g2p_data.index),
                y=(g2p_data['power_kw'] / 1000).tolist(),
                mode='markers',
                marker=dict(size=10, color=COLOR_PALETTE['h2']),
                name='G2P (MW)'
            ))
        
        fig.update_layout(
            title="H₂ 시스템 운전 이력",
            xaxis_title="시간",
            template='plotly_white',
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
            x=list(grid_data.index),
            y=grid_data['power_mw'].tolist(),
            mode='markers+lines',
            marker=dict(
                size=8,
                color=grid_data['power_mw'].tolist(),
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
            template='plotly_white',
            yaxis_title="전력 (MW)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # SMP 가격 차트
        st.subheader("SMP 가격 추이")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(grid_data.index),
            y=grid_data['smp_price'].tolist(),
            mode='lines',
            name='SMP 가격',
            line=dict(color=COLOR_PALETTE['surplus'])
        ))
        
        fig2.update_layout(
            title="시간대별 SMP 가격",
            xaxis_title="시간",
            template='plotly_white', 
            yaxis_title="SMP 가격 (₩/MWh)",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.info("그리드 거래 데이터가 없습니다.")


# ═══════════════════════════════════════════════════════════════
# 다국어 지원 (i18n) — KO 완성, EN/CN 키만 준비
# ═══════════════════════════════════════════════════════════════
I18N = {
    "KO": {
        "policy_tab": "🏛️ 정책 시뮬레이터",
        "industry_tab": "🏭 산업 상용화",
        "investment_tab": "📋 투자 대시보드",
        "references_tab": "📚 References",
        "carbon_price": "탄소가격 (₩/tCO₂)",
        "rec_price": "REC 가격 (₩/MWh)",
        "subsidy_rate": "보조금 비율 (%)",
        "csp_select": "CSP 선택",
        "go_decision": "투자 판정",
        "base_scenario": "Base (현행)",
        "combined_scenario": "복합 (정책 강화)",
        "optimal_scenario": "최적 (보조금+정책)",
        "irr": "IRR (%)",
        "npv": "NPV (억원)",
        "payback": "회수 기간 (년)",
        "capex": "CAPEX (억원)",
        "annual_revenue": "연간 수익 (억원)",
        "co2_reduction": "CO₂ 감축 (tCO₂/년)",
    },
    "EN": {
        "policy_tab": "🏛️ Policy Simulator",
        "industry_tab": "🏭 Industry Model",
        "investment_tab": "📋 Investment Dashboard",
        "references_tab": "📚 References",
        "carbon_price": "Carbon Price (₩/tCO₂)",
        "rec_price": "REC Price (₩/MWh)",
        "subsidy_rate": "Subsidy Rate (%)",
        "csp_select": "Select CSP",
        "go_decision": "Investment Decision",
    },
    "CN": {
        "policy_tab": "🏛️ 政策模拟器",
        "industry_tab": "🏭 产业商用化",
        "investment_tab": "📋 投资决策面板",
        "references_tab": "📚 参考资料",
        "carbon_price": "碳价格 (₩/tCO₂)",
    },
}


def _t(key: str) -> str:
    """다국어 텍스트 반환"""
    lang = st.session_state.get("language", "KO")
    return I18N.get(lang, I18N["KO"]).get(key, I18N["KO"].get(key, key))


# ═══════════════════════════════════════════════════════════════
# Week 4 탭: 정책 시뮬레이터
# ═══════════════════════════════════════════════════════════════

def display_statistics(data):
    """📈 통계 분석 탭"""
    st.subheader("📈 통합 통계 분석")
    
    try:
        import pandas as pd
        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np
        
        pv_data = _safe_dict(data.get('pv', {}))
        aidc_data = _safe_dict(data.get('aidc', {}))
        grid_df = data.get('grid', pd.DataFrame())
        ems_kpi = data.get('ems_kpi', {})
        carbon_df = data.get('carbon_df', pd.DataFrame())
        pv_module = data['modules']['pv']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        pv_power = pv_data.get('power_mw', [])
        aidc_power = aidc_data.get('power_mw', [])
        
        with col1:
            if len(pv_power) > 0:
                pv_cap = pv_module.capacity_mw if hasattr(pv_module, 'capacity_mw') else 100
                cf = np.mean(pv_power) / pv_cap if pv_cap > 0 else 0
                st.metric("PV Capacity Factor", f"{cf*100:.1f}%")
            else:
                st.metric("PV Capacity Factor", "N/A")
        with col2:
            if len(aidc_power) > 0:
                st.metric("평균 AIDC 부하", f"{np.mean(aidc_power):.1f} MW")
            else:
                st.metric("평균 AIDC 부하", "N/A")
        with col3:
            if ems_kpi:
                ss = ems_kpi.get('self_sufficiency_pct', 0)
                st.metric("자급률", f"{ss:.1f}%")
            else:
                st.metric("자급률", "N/A")
        with col4:
            if ems_kpi:
                curt = ems_kpi.get('curtailment_pct', 0)
                st.metric("Curtailment", f"{curt:.1f}%")
            else:
                st.metric("Curtailment", "N/A")
        
        st.divider()
        
        # Time series: combined power flow
        st.subheader("⏱️ 시간별 에너지 흐름 요약")
        
        if len(pv_power) > 0 and len(aidc_power) > 0:
            hours = list(range(len(pv_power)))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hours, y=pv_power, name='☀️ PV 발전', line=dict(color='#f59e0b')))
            fig.add_trace(go.Scatter(x=hours, y=aidc_power, name='🖥️ AIDC 부하', line=dict(color='#ef4444')))
            
            if isinstance(grid_df, pd.DataFrame) and 'import_mw' in grid_df.columns:
                fig.add_trace(go.Scatter(x=hours[:len(grid_df)], y=grid_df['import_mw'].tolist(), 
                                         name='📥 그리드 수입', line=dict(color='#3b82f6', dash='dash')))
            if isinstance(grid_df, pd.DataFrame) and 'export_mw' in grid_df.columns:
                fig.add_trace(go.Scatter(x=hours[:len(grid_df)], y=grid_df['export_mw'].tolist(), 
                                         name='📤 그리드 수출', line=dict(color='#22c55e', dash='dash')))
            
            fig.update_layout(title="시간별 전력 흐름", xaxis_title="시간 (h)", 
                            yaxis_title="전력 (MW)", height=450, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("시뮬레이션을 먼저 실행해주세요.")
        
        # Distribution analysis
        st.subheader("📊 분포 분석")
        col1, col2 = st.columns(2)
        
        with col1:
            if len(pv_power) > 0:
                fig = px.histogram(x=pv_power, nbins=30, title="PV 발전량 분포 (MW)",
                                   labels={'x': 'MW', 'y': 'Count'}, color_discrete_sequence=['#f59e0b'],
                                   template='plotly_white')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(aidc_power) > 0:
                fig = px.histogram(x=aidc_power, nbins=30, title="AIDC 부하 분포 (MW)",
                                   labels={'x': 'MW', 'y': 'Count'}, color_discrete_sequence=['#ef4444'],
                                   template='plotly_white')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Key statistics table
        st.subheader("📋 주요 통계")
        if len(pv_power) > 0 and len(aidc_power) > 0:
            stats_data = {
                '항목': ['PV 발전', 'AIDC 부하'],
                '평균 (MW)': [f"{np.mean(pv_power):.2f}", f"{np.mean(aidc_power):.2f}"],
                '최대 (MW)': [f"{np.max(pv_power):.2f}", f"{np.max(aidc_power):.2f}"],
                '최소 (MW)': [f"{np.min(pv_power):.2f}", f"{np.min(aidc_power):.2f}"],
                '표준편차': [f"{np.std(pv_power):.2f}", f"{np.std(aidc_power):.2f}"],
            }
            
            if isinstance(carbon_df, pd.DataFrame) and 'total_tCO2' in carbon_df.columns:
                total_co2 = carbon_df['total_tCO2'].sum()
                stats_data['항목'].append('탄소 배출')
                stats_data['평균 (MW)'].append(f"{carbon_df['total_tCO2'].mean():.3f} tCO₂/h")
                stats_data['최대 (MW)'].append(f"{carbon_df['total_tCO2'].max():.3f} tCO₂/h")
                stats_data['최소 (MW)'].append(f"{carbon_df['total_tCO2'].min():.3f} tCO₂/h")
                stats_data['표준편차'].append(f"{carbon_df['total_tCO2'].std():.3f}")
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"통계 분석 오류: {e}")
        import traceback
        st.code(traceback.format_exc())

def display_policy_simulator():
    """정책 시뮬레이터 탭"""
    st.subheader("🏛️ 정책 시뮬레이터")
    st.markdown("K-ETS, REC, CBAM, RE100, 전력수급계획 시나리오 분석")

    sim = PolicySimulator()

    col1, col2, col3 = st.columns(3)
    with col1:
        carbon_price = st.slider(
            "K-ETS 탄소가격 (₩/tCO₂)", 10_000, 150_000, 25_000, 5_000,
            key="policy_carbon")
    with col2:
        rec_price = st.slider(
            "REC 가격 (₩/MWh)", 10_000, 80_000, 25_000, 5_000,
            key="policy_rec")
    with col3:
        subsidy_pct = st.slider(
            "보조금 비율 (%)", 0, 30, 0, 5, key="policy_subsidy") / 100

    # K-ETS 시나리오
    st.markdown("### K-ETS 탄소가격 시나리오")
    k_ets_results = sim.k_ets_scenarios_compare()
    cols = st.columns(3)
    for i, (label, result) in enumerate(zip(
            ["현행 25,000", "중간 50,000", "강화 100,000"], k_ets_results)):
        with cols[i]:
            st.metric(label=f"{label} ₩/tCO₂",
                      value=f"{result['annual_revenue_billion_krw']:.0f}억/년",
                      delta=f"NPV {result['npv_billion_krw']:.0f}억")

    # CBAM
    st.markdown("### CBAM 영향")
    cbam = sim.cbam_impact(eu_carbon_price_eur=80)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("CBAM 비용 (CEMS 없이)", f"{cbam['cbam_cost_without_cems_billion_krw']:.1f}억/년")
    with col_b:
        st.metric("CBAM 절감 (CEMS 적용)", f"{cbam['cbam_savings_billion_krw']:.1f}억/년")

    # RE100
    st.markdown("### RE100 달성률")
    re100 = sim.re100_achievement()
    st.progress(min(re100["achievement_pct"] / 100, 1.0))
    st.write(f"달성률: **{re100['achievement_pct']}%** | 부족: {re100['gap_mwh']:,.0f} MWh")

    # 정책 조합 히트맵
    st.markdown("### 정책 조합 IRR 히트맵")
    hm = sim.policy_heatmap_data()
    fig = go.Figure(data=go.Heatmap(
        z=hm["irr_matrix"],
        x=[f"{p/1000:.0f}k" for p in hm["rec_prices"]],
        y=[f"{p/1000:.0f}k" for p in hm["carbon_prices"]],
        colorscale="RdYlGn",
        text=[[f"{v:.1f}%" for v in row] for row in hm["irr_matrix"]],
        texttemplate="%{text}",
        colorbar=dict(title="IRR (%)"),
    ))
    fig.update_layout(
        title="탄소가격 × REC 가격 → IRR (%)",
        xaxis_title="REC 가격 (₩/MWh)",
        template='plotly_white',
        yaxis_title="K-ETS 탄소가격 (₩/tCO₂)",
        height=400)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Week 4 탭: 산업 상용화
# ═══════════════════════════════════════════════════════════════
def display_industry_model():
    """산업 상용화 탭"""
    st.subheader("🏭 산업 상용화 모델")
    st.markdown("CSP별 맞춤 분석 + BYOG + 스케일링")

    model = IndustryModel()

    from modules.m12_industry import CSP_PROFILES
    csp_keys = list(CSP_PROFILES.keys())
    csp_names = [CSP_PROFILES[k]["name"] for k in csp_keys]

    csp_selected = st.selectbox(
        "CSP 선택", csp_keys,
        format_func=lambda x: f"{CSP_PROFILES[x]['name']} ({CSP_PROFILES[x]['description']})",
        key="csp_select")

    col1, col2 = st.columns(2)
    with col1:
        ind_subsidy = st.slider("보조금 (%)", 0, 30, 0, 5, key="ind_subsidy") / 100
    with col2:
        ind_carbon = st.slider("탄소가격 (₩/tCO₂)", 10_000, 150_000, 25_000, 5_000,
                                key="ind_carbon")

    # 선택된 CSP 분석
    result = model.csp_analysis(csp_selected, subsidy_pct=ind_subsidy,
                                 carbon_price_krw=ind_carbon)

    st.markdown(f"### {result['csp_name']} 분석 결과")
    cols = st.columns(4)
    cols[0].metric("에너지 CAPEX", f"{result['energy_capex_billion_krw']:,.0f}억")
    cols[1].metric("연간 수익", f"{result['annual_revenue_billion_krw']:,.0f}억")
    cols[2].metric("IRR", f"{result['irr_pct']:.1f}%" if result['irr_pct'] else "N/A")
    cols[3].metric("Payback", f"{result['payback_years']:.1f}년")

    col_a, col_b = st.columns(2)
    col_a.metric("연간 CO₂ 감축", f"{result['annual_co2_reduction_ton']:,.0f} tCO₂")
    col_b.metric("20년 CO₂ 감축", f"{result['lifetime_co2_reduction_kton']:,.0f} 천tCO₂")

    # 전체 CSP 비교
    st.markdown("### 전체 CSP 비교")
    all_csp = model.all_csp_comparison(subsidy_pct=ind_subsidy, carbon_price_krw=ind_carbon)

    fig = go.Figure()
    names = [c["csp_name"] for c in all_csp]
    fig.add_trace(go.Bar(name="에너지 CAPEX (억)", x=names,
                         y=[c["energy_capex_billion_krw"] for c in all_csp]))
    fig.add_trace(go.Bar(name="연간 수익 (억)", x=names,
                         y=[c["annual_revenue_billion_krw"] for c in all_csp]))
    fig.update_layout(barmode="group", height=400,
                      title="CSP별 CAPEX vs 연간 수익", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # 스케일링 분석
    st.markdown("### 스케일링 분석 (규모의 경제)")
    scaling = model.scaling_analysis()
    fig2 = go.Figure()
    caps = [s["capacity_mw"] for s in scaling]
    fig2.add_trace(go.Scatter(x=caps, y=[s["irr_pct"] or 0 for s in scaling],
                              mode="lines+markers", name="IRR (%)"))
    fig2.update_layout(title="용량별 IRR", xaxis_title="용량 (MW)",
                       yaxis_title="IRR (%)", height=350, template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Week 4 탭: 투자 대시보드
# ═══════════════════════════════════════════════════════════════
def display_investment_dashboard():
    """투자 의사결정 대시보드"""
    st.subheader("📋 투자 의사결정 대시보드")

    dash = InvestmentDashboard()

    # What-if 슬라이더
    st.markdown("### NPV/IRR What-if 분석")
    col1, col2, col3 = st.columns(3)
    with col1:
        capex_var = st.slider("CAPEX 변동 (%)", -30, 30, 0, 5, key="inv_capex") / 100
    with col2:
        rev_var = st.slider("수익 변동 (%)", -30, 30, 0, 5, key="inv_rev") / 100
    with col3:
        inv_dr = st.slider("할인율 (%)", 3, 10, 5, 1, key="inv_dr") / 100

    whatif = dash.whatif_analysis(capex_variation=capex_var,
                                  revenue_variation=rev_var,
                                  discount_rate=inv_dr)

    cols = st.columns(3)
    cols[0].metric("NPV", f"{whatif['npv_billion_krw']:,.0f}억",
                   delta=f"{'양' if whatif['npv_billion_krw'] > 0 else '음'}수")
    cols[1].metric("IRR", f"{whatif['irr_pct']:.1f}%" if whatif["irr_pct"] else "N/A")
    cols[2].metric("Payback", f"{whatif['payback_years']:.1f}년")

    # MC 히스토그램
    st.markdown("### Monte Carlo 시뮬레이션 (10,000회)")
    mc = dash.monte_carlo(n_iterations=10_000)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_irr = go.Figure()
        fig_irr.add_trace(go.Histogram(
            x=mc["irr_distribution"], nbinsx=50,
            marker_color="#2E8B57", name="IRR"))
        fig_irr.add_vline(x=mc["irr_mean_pct"], line_dash="dash",
                          annotation_text=f"Mean: {mc['irr_mean_pct']:.1f}%")
        fig_irr.update_layout(title="IRR 분포", xaxis_title="IRR (%)",
                              yaxis_title="빈도", height=350, template='plotly_white')
        st.plotly_chart(fig_irr, use_container_width=True)

    with col_b:
        fig_npv = go.Figure()
        fig_npv.add_trace(go.Histogram(
            x=mc["npv_distribution"], nbinsx=50,
            marker_color="#DAA520", name="NPV"))
        fig_npv.add_vline(x=0, line_dash="solid", line_color="red",
                          annotation_text="BEP")
        fig_npv.update_layout(title="NPV 분포", xaxis_title="NPV (억원)",
                              yaxis_title="빈도", height=350, template='plotly_white')
        st.plotly_chart(fig_npv, use_container_width=True)

    st.info(f"P(NPV>0) = **{mc['prob_positive_npv_pct']:.1f}%** | "
            f"IRR p5-p95 = [{mc['irr_p5_pct']:.1f}%, {mc['irr_p95_pct']:.1f}%]")

    # 시나리오 비교
    st.markdown("### 시나리오 비교")
    scenarios = dash.scenario_comparison()
    import pandas as pd
    df = pd.DataFrame(scenarios)
    st.dataframe(df[["scenario", "capex_billion_krw", "annual_revenue_billion_krw",
                      "irr_pct", "npv_billion_krw", "payback_years"]],
                 hide_index=True)

    # Go/No-Go 신호등
    st.markdown("### 투자 의사결정 (Go/No-Go)")
    decision = dash.go_nogo_decision(
        irr_pct=whatif["irr_pct"] or 0,
        npv_billion=whatif["npv_billion_krw"],
        payback_years=whatif["payback_years"],
        prob_positive_npv_pct=mc["prob_positive_npv_pct"])

    color_map = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
    st.markdown(f"## {color_map.get(decision['color'], '⚪')} {decision['decision']}")
    st.write(decision["recommendation"])

    for name, crit in decision["criteria"].items():
        icon = "✅" if crit["pass"] else "❌"
        st.write(f"{icon} {crit['label']}")

    # 보조금 민감도
    st.markdown("### 보조금 민감도")
    sub_results = dash.subsidy_sensitivity()
    fig_sub = go.Figure()
    fig_sub.add_trace(go.Bar(
        x=[f"{r['subsidy_pct']:.0f}%" for r in sub_results],
        y=[r["irr_pct"] or 0 for r in sub_results],
        marker_color=["#DC143C" if (r["irr_pct"] or 0) < 5 else "#2E8B57"
                      for r in sub_results],
        text=[f"{r['irr_pct']:.1f}%" if r["irr_pct"] else "N/A" for r in sub_results],
        textposition="auto"))
    fig_sub.update_layout(title="보조금 비율별 IRR",
                          xaxis_title="보조금", yaxis_title="IRR (%)",
                          height=350, template='plotly_white')
    st.plotly_chart(fig_sub, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# Week 4 탭: References
# ═══════════════════════════════════════════════════════════════
def display_references():
    """참고 자료 탭"""
    st.subheader("📚 References")

    st.markdown("""
### 데이터셋 출처
- **한국 기상청 (KMA)**: 일사량, 기온 데이터 — [data.kma.go.kr](https://data.kma.go.kr)
- **한국전력거래소 (KPX)**: SMP 가격, 전력수급 — [epsis.kpx.or.kr](https://epsis.kpx.or.kr)
- **에너지경제연구원 (KEEI)**: 에너지 통계 — [keei.re.kr](https://www.keei.re.kr)

### 참고 논문/보고서
1. NREL (2024), *Utility-Scale Solar PV LCOE*, Annual Technology Baseline
2. BloombergNEF (2024), *Lithium-Ion Battery Pack Prices*
3. IEA (2024), *Global Hydrogen Review*
4. McKinsey (2024), *The Green Data Center Revolution*
5. 한국에너지공단 (2024), *신재생에너지 백서*

### 정책 자료
- **K-ETS**: [환경부 온실가스종합정보센터](https://ngms.gir.go.kr) — 배출권거래제 운영
- **전력수급기본계획**: [산업통상자원부](https://motie.go.kr) — 제11차 전력수급기본계획
- **CBAM**: [EU CBAM Regulation (2023/956)](https://eur-lex.europa.eu) — Carbon Border Adjustment Mechanism
- **RE100**: [The Climate Group RE100](https://www.there100.org) — 글로벌 RE100 이니셔티브
- **REC 시장**: [한국에너지공단 신재생에너지센터](https://www.knrec.or.kr)

### 기술 참고
- NVIDIA H100/B200 Datasheet
- Samsung SDI ESS Battery Specifications
- Bloom Energy SOFC Technical Data
- Nel Hydrogen Electrolyzer Specifications

### 경제성 모델 가정
| 항목 | 값 | 출처 |
|------|------|------|
| 할인율 | 5% | 한국개발연구원 (KDI) |
| PV CAPEX | 1,500억/100MW | IRENA 2024 |
| BESS CAPEX | 4,000억/2GWh | BloombergNEF 2024 |
| 그리드 배출계수 | 0.4594 tCO₂/MWh | 환경부 2024 |
| K-ETS 탄소가격 | 25,000 ₩/tCO₂ | KRX 2024 |
| SMP 기준가 | 80,000 ₩/MWh | KPX 2024 평균 |
    """)


if __name__ == "__main__":
    create_main_dashboard()