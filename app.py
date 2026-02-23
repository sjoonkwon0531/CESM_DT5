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
    AI_EMS_CONFIG, CARBON_CONFIG, ECONOMICS_CONFIG,
    INTERNATIONAL_BENCHMARKS, BENCHMARK_API_SOURCES, BENCHMARK_LAST_UPDATED
)
import copy
import json

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

# 탭 바 가로 스크롤 + 클린 화이트 테마
st.markdown("""
<style>
/* 탭 바: 가로 스크롤 강제 */
div[data-testid="stTabs"] > div[role="tablist"] {
    overflow-x: scroll !important;
    overflow-y: hidden !important;
    flex-wrap: nowrap !important;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: auto;
    scrollbar-color: #64748b #e2e8f0;
    padding-bottom: 8px;
    gap: 2px !important;
    display: flex !important;
}
div[data-testid="stTabs"] > div[role="tablist"]::-webkit-scrollbar {
    height: 10px !important;
    display: block !important;
}
div[data-testid="stTabs"] > div[role="tablist"]::-webkit-scrollbar-track {
    background: #e2e8f0;
    border-radius: 5px;
}
div[data-testid="stTabs"] > div[role="tablist"]::-webkit-scrollbar-thumb {
    background: #64748b;
    border-radius: 5px;
    border: 2px solid #e2e8f0;
}
div[data-testid="stTabs"] > div[role="tablist"]::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
div[data-testid="stTabs"] > div[role="tablist"] button {
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    min-width: fit-content !important;
    font-size: 0.82rem;
    padding: 0.35rem 0.7rem !important;
}
div[data-testid="stTabs"] {
    max-width: 100% !important;
    overflow: visible !important;
}
/* 탭 내부 gap 제거 */
div[data-testid="stTabs"] > div[role="tablist"] > div {
    flex-shrink: 0 !important;
}
</style>
""", unsafe_allow_html=True)


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
            min_value=500, max_value=100000, value=50000, step=500,
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
        st.caption("세 비율의 합 = 1.0 (자동 정규화)")
        llm_ratio = st.slider("LLM 추론 비율", 0.0, 1.0, 0.4, 0.05, key="llm_ratio")
        # 남은 비율 계산하여 training+moe 상한 제한
        remaining_after_llm = 1.0 - llm_ratio
        training_default = min(0.4, remaining_after_llm)
        training_ratio = st.slider("AI 훈련 비율", 0.0, remaining_after_llm, 
                                    training_default, 0.05, key="training_ratio")
        remaining_after_train = max(0.0, remaining_after_llm - training_ratio)
        moe_ratio = remaining_after_train  # 자동 결정
        
        # 합계 표시
        total_ratio = llm_ratio + training_ratio + moe_ratio
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("추론", f"{llm_ratio:.0%}")
        with col_r2:
            st.metric("훈련", f"{training_ratio:.0%}")
        with col_r3:
            st.metric("MoE", f"{moe_ratio:.0%}")
        
        if abs(total_ratio - 1.0) > 0.01:
            st.warning(f"⚠️ 합계 {total_ratio:.2f} ≠ 1.0")
        
        workload_mix = {
            'llm': llm_ratio,
            'training': training_ratio,
            'moe': moe_ratio
        }
        
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
                # SOC 업데이트: HESS 용량 ~200 MWh, H₂ 저장 ~5000 kg (~167 MWh)
                # 1시간 운전이므로 MWh = MW × 1h
                hess_capacity_mwh = 200.0  # Supercap + BESS 합산
                h2_capacity_mwh = 167.0    # 5000 kg × 33.3 kWh/kg
                ems_soc = float(np.clip(ems_soc + (cmd.pv_to_hess_mw - cmd.hess_to_aidc_mw) / hess_capacity_mwh, 0, 1))
                ems_h2 = float(np.clip(ems_h2 + (cmd.h2_electrolyzer_mw - cmd.h2_fuelcell_mw) / h2_capacity_mwh, 0, 1))
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
        tab12, tab13, tab14, tab15, tab16, tab17, tab18 = st.tabs([
        "📊 전력", "☀️ PV", "🖥️ AIDC", 
        "🔄 Bus", "🔋 HESS", "⚡ H₂", "🔌 Grid",
        "🤖 EMS", "🌍 탄소", "💰 경제", "📈 통계",
        "🏛️ 정책", "🏭 산업", "📋 투자",
        "🌏 국제", "🦆 Duck", "📥 다운", "📚 Ref"
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
        display_international_comparison(data)
    
    with tab16:
        display_duck_curve(data)
    
    with tab17:
        display_data_download(data)
    
    with tab18:
        display_references()


def display_static_energy_flow_sankey(data):
    """정적 에너지 흐름 Sankey — 장치 raw 값 + 변환손실로 완벽 밸런스 보장.
    
    원리: 좌측(소스) raw 합 = 우측(싱크) raw 합 + 변환 손실
    변환 손실 = Σ입력 - Σ출력 (항상 ≥ 0, 자동 계산)
    → Sankey 좌우가 수학적으로 정확히 일치.
    """
    import pandas as pd
    
    dcbus_data = _safe_dict(data['dcbus'])
    
    def _ssum(d, key):
        val = d.get(key, [])
        if val is None:
            return 0.0
        if isinstance(val, pd.Series):
            return float(val.sum())
        try:
            return float(sum(val))
        except (TypeError, ValueError):
            return 0.0
    
    # --- 모든 값은 DC Bus가 기록한 장치 측 raw 값 (MWh) ---
    pv = _ssum(dcbus_data, 'pv_power_mw')
    if pv < 0.1:
        pv = _ssum(_safe_dict(data['pv']), 'power_mw')
    
    bess_disch = _ssum(dcbus_data, 'bess_discharge_mw')
    bess_chg   = _ssum(dcbus_data, 'bess_charge_mw')
    h2_fc      = _ssum(dcbus_data, 'h2_fuelcell_mw')
    h2_elec    = _ssum(dcbus_data, 'h2_electrolyzer_mw')
    grid_imp   = _ssum(dcbus_data, 'grid_import_mw')
    grid_exp   = _ssum(dcbus_data, 'grid_export_mw')
    curtail    = _ssum(dcbus_data, 'curtailment_mw')
    conv_loss  = _ssum(dcbus_data, 'conversion_loss_mw')
    
    aidc_dict  = _safe_dict(data['aidc'])
    aidc = _ssum(aidc_dict, 'total_power_mw') or _ssum(aidc_dict, 'power_mw')
    
    # 변환 손실: 입력 합 - 출력 합 (DC Bus가 기록한 값 사용, 없으면 잔차로 계산)
    total_in  = pv + bess_disch + h2_fc + grid_imp
    total_out = aidc + bess_chg + h2_elec + grid_exp + curtail
    if conv_loss < 0.1:
        conv_loss = max(0, total_in - total_out)
    
    # 최종 밸런스 보정: 출력 + 손실 = 입력 (수학적 보장)
    total_out_with_loss = total_out + conv_loss
    if total_in > 0 and abs(total_in - total_out_with_loss) > 0.01:
        conv_loss += (total_in - total_out_with_loss)
        conv_loss = max(0, conv_loss)
    
    # --- Sankey 구성 ---
    # 라벨에 절대치 + 비중 표시
    def _lbl(name, val, ref):
        pct = (val / ref * 100) if ref > 0 else 0
        return f"{name}<br>{val:,.0f} MWh ({pct:.1f}%)"
    
    # 0-3: 소스(좌), 4: DC Bus(중), 5-10: 싱크(우)
    labels = [
        _lbl("☀️ Solar PV", pv, total_in),
        _lbl("🔋 HESS 방전", bess_disch, total_in),
        _lbl("💧 H₂ FC", h2_fc, total_in),
        _lbl("🔌 Grid 수입", grid_imp, total_in),
        f"⚡ DC Bus<br>{total_in:,.0f} MWh",
        _lbl("🖥️ AIDC", aidc, total_in),
        _lbl("🔋 HESS 충전", bess_chg, total_in),
        _lbl("💧 H₂ 전해조", h2_elec, total_in),
        _lbl("🔌 Grid 수출", grid_exp, total_in),
        _lbl("⛔ Curtailment", curtail, total_in),
        _lbl("🔥 변환 손실", conv_loss, total_in),
    ]
    
    colors = [
        "#d97706", "#0d9488", "#059669", "#4f46e5",  # 소스
        "#1e40af",                                     # DC Bus — 진한 블루
        "#dc2626", "#0d9488", "#059669", "#4f46e5",  # 싱크
        "#94a3b8", "#78716c",                          # Curtail, 손실
    ]
    
    # (source, target, value, link_color) — 0 이하인 건 자동 제외
    raw_links = [
        (0, 4, pv,         "rgba(217,119,6,0.45)"),
        (1, 4, bess_disch, "rgba(13,148,136,0.45)"),
        (2, 4, h2_fc,      "rgba(5,150,105,0.45)"),
        (3, 4, grid_imp,   "rgba(79,70,229,0.45)"),
        (4, 5, aidc,       "rgba(220,38,38,0.45)"),
        (4, 6, bess_chg,   "rgba(13,148,136,0.45)"),
        (4, 7, h2_elec,    "rgba(5,150,105,0.45)"),
        (4, 8, grid_exp,   "rgba(79,70,229,0.45)"),
        (4, 9, curtail,    "rgba(148,163,184,0.35)"),
        (4, 10, conv_loss, "rgba(120,113,108,0.3)"),
    ]
    
    flt = [(s, t, v, c) for s, t, v, c in raw_links if v > 0.1]
    
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        textfont=dict(size=13, color="#1e293b", family="Arial, sans-serif"),
        node=dict(
            pad=30,
            thickness=28,
            line=dict(color="#cbd5e1", width=1),
            label=labels,
            color=colors,
        ),
        link=dict(
            source=[s for s, t, v, c in flt],
            target=[t for s, t, v, c in flt],
            value=[v for s, t, v, c in flt],
            color=[c for s, t, v, c in flt],
        ),
    )])
    
    fig.update_layout(
        title=dict(
            text=f"에너지 흐름 요약 · 입력 {total_in:,.0f} MWh → 출력 {total_out:,.0f} MWh + 손실 {conv_loss:,.0f} MWh",
            font=dict(size=13, color="#334155"),
        ),
        font=dict(size=12, family="Arial, sans-serif", color="#1e293b"),
        height=500,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    
    # 요약 메트릭
    sim_hours = max(len(_safe_dict(data['pv']).get('power_mw', [1])), 1)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("☀️ PV 발전", f"{pv:.0f} MWh", 
                 delta=f"평균: {pv/sim_hours:.1f} MW")
    with col2:
        st.metric("🖥️ AIDC 소비", f"{aidc:.0f} MWh",
                 delta=f"평균: {aidc/sim_hours:.1f} MW")
    with col3:
        hess_net = bess_disch - bess_chg
        st.metric("🔋 HESS 순", f"{hess_net:+.0f} MWh", 
                 delta=f"{'방전' if hess_net > 0 else '충전'} 우세")
    with col4:
        h2_net = h2_fc - h2_elec
        st.metric("💧 H₂ 순", f"{h2_net:+.0f} MWh",
                 delta=f"{'발전' if h2_net > 0 else '전해'} 우세")
    with col5:
        grid_net = grid_exp - grid_imp
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
        aidc_power = aidc_data.get('total_power_mw', aidc_data.get('power_mw', []))
        
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
            
            # DC Bus 데이터에서 추가 시계열
            dcbus_data_stat = _safe_dict(data.get('dcbus', {}))
            bess_disch_ts = dcbus_data_stat.get('bess_discharge_mw', [])
            bess_chg_ts = dcbus_data_stat.get('bess_charge_mw', [])
            grid_imp_ts = dcbus_data_stat.get('grid_import_mw', [])
            grid_exp_ts = dcbus_data_stat.get('grid_export_mw', [])
            curtail_ts = dcbus_data_stat.get('curtailment_mw', [])
            
            if len(bess_disch_ts) > 0:
                fig.add_trace(go.Scatter(x=hours[:len(bess_disch_ts)], y=bess_disch_ts,
                                         name='🔋 HESS 방전', line=dict(color='#0d9488', dash='dot')))
            if len(bess_chg_ts) > 0:
                bess_chg_neg = [-v for v in bess_chg_ts]
                fig.add_trace(go.Scatter(x=hours[:len(bess_chg_neg)], y=bess_chg_neg,
                                         name='🔋 HESS 충전', line=dict(color='#0d9488', dash='dash')))
            if len(grid_imp_ts) > 0:
                fig.add_trace(go.Scatter(x=hours[:len(grid_imp_ts)], y=grid_imp_ts,
                                         name='📥 Grid 수입', line=dict(color='#3b82f6', dash='dash')))
            if len(grid_exp_ts) > 0:
                grid_exp_neg = [-v for v in grid_exp_ts]
                fig.add_trace(go.Scatter(x=hours[:len(grid_exp_neg)], y=grid_exp_neg,
                                         name='📤 Grid 수출', line=dict(color='#22c55e', dash='dash')))
            if len(curtail_ts) > 0:
                fig.add_trace(go.Scatter(x=hours[:len(curtail_ts)], y=curtail_ts,
                                         name='⛔ Curtailment', line=dict(color='#94a3b8', dash='dot'),
                                         fill='tozeroy', fillcolor='rgba(148,163,184,0.1)'))
            
            fig.update_layout(title="시간별 전력 흐름 (전체 시스템)", xaxis_title="시간 (h)", 
                            yaxis_title="전력 (MW)", height=500, template='plotly_white',
                            legend=dict(orientation="h", yanchor="bottom", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
            
            # === 24h × Day 히트맵 (컬러맵) ===
            n_hours = len(pv_power)
            n_days = max(1, n_hours // 24)
            if n_days >= 2:
                st.subheader("🗓️ 일별 × 시간별 에너지 패턴 (히트맵)")
                
                def _build_heatmap_matrix(series, n_days):
                    """시계열을 (days × 24h) 행렬로 변환"""
                    arr = np.array(series[:n_days * 24])
                    return arr.reshape(n_days, 24)
                
                heatmap_vars = {
                    'PV 발전 (MW)': (pv_power, 'YlOrRd'),
                    'AIDC 부하 (MW)': (aidc_power, 'Blues'),
                    '잉여/부족 (MW)': ([p - a for p, a in zip(pv_power[:n_days*24], aidc_power[:n_days*24])], 'RdBu'),
                }
                
                # HESS SOC도 있으면 추가
                hess_soc_ts = dcbus_data_stat.get('bess_soc', [])
                if len(hess_soc_ts) >= n_days * 24:
                    heatmap_vars['HESS SoC (%)'] = ([v * 100 for v in hess_soc_ts[:n_days*24]], 'Greens')
                
                hm_select = st.selectbox(
                    "히트맵 변수 선택", list(heatmap_vars.keys()),
                    key="heatmap_var_select"
                )
                
                hm_series, hm_cmap = heatmap_vars[hm_select]
                mat = _build_heatmap_matrix(hm_series, n_days)
                
                # 잉여/부족은 0 기준 대칭 컬러스케일
                zmid = 0 if '잉여' in hm_select else None
                
                day_labels = [f"Day {d+1}" for d in range(n_days)]
                hour_labels = [f"{h:02d}:00" for h in range(24)]
                
                fig_hm = go.Figure(data=go.Heatmap(
                    z=mat, x=hour_labels, y=day_labels,
                    colorscale=hm_cmap, zmid=zmid,
                    colorbar=dict(title=hm_select),
                    hovertemplate='%{y}, %{x}<br>%{z:.1f}<extra></extra>'
                ))
                fig_hm.update_layout(
                    title=f"{hm_select} — 24시간 × {n_days}일 패턴",
                    xaxis_title="시간 (Hour of Day)",
                    yaxis_title="일차",
                    height=max(300, n_days * 28 + 150),
                    template='plotly_white',
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig_hm, use_container_width=True)
                
                # 시간대별 평균 프로파일 (일 평균)
                col_hm1, col_hm2 = st.columns(2)
                with col_hm1:
                    hourly_avg = mat.mean(axis=0)
                    fig_avg = go.Figure()
                    fig_avg.add_trace(go.Bar(
                        x=hour_labels, y=hourly_avg,
                        marker_color='#f59e0b' if 'PV' in hm_select else '#3b82f6',
                        name='시간대별 평균'
                    ))
                    fig_avg.update_layout(
                        title=f"시간대별 평균 {hm_select}",
                        height=300, template='plotly_white',
                        xaxis_title="시간", yaxis_title=hm_select
                    )
                    st.plotly_chart(fig_avg, use_container_width=True)
                
                with col_hm2:
                    daily_avg = mat.mean(axis=1)
                    fig_daily = go.Figure()
                    fig_daily.add_trace(go.Bar(
                        x=day_labels, y=daily_avg,
                        marker_color='#10b981',
                        name='일별 평균'
                    ))
                    fig_daily.update_layout(
                        title=f"일별 평균 {hm_select}",
                        height=300, template='plotly_white',
                        xaxis_title="일차", yaxis_title=hm_select
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
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
# 데이터 다운로드 탭
# ═══════════════════════════════════════════════════════════════
def display_data_download(data):
    """시뮬레이션 데이터 다운로드"""
    st.subheader("📥 시뮬레이션 데이터 다운로드")

    if data is None:
        st.info("시뮬레이션을 먼저 실행해주세요.")
        return

    import io

    # --- 1. 개별 모듈 CSV 다운로드 ---
    st.markdown("### 📄 개별 모듈 데이터 (CSV)")

    download_items = {}

    # PV
    pv_raw = data.get('pv')
    if pv_raw is not None:
        if isinstance(pv_raw, pd.DataFrame):
            download_items['PV_발전'] = pv_raw
        else:
            download_items['PV_발전'] = pd.DataFrame(_safe_dict(pv_raw))

    # AIDC
    aidc_raw = data.get('aidc')
    if aidc_raw is not None:
        if isinstance(aidc_raw, pd.DataFrame):
            download_items['AIDC_부하'] = aidc_raw
        else:
            download_items['AIDC_부하'] = pd.DataFrame(_safe_dict(aidc_raw))

    # DC Bus
    dcbus_raw = data.get('dcbus')
    if dcbus_raw is not None:
        if isinstance(dcbus_raw, dict):
            download_items['DC_Bus'] = pd.DataFrame(_safe_dict(dcbus_raw))
        elif isinstance(dcbus_raw, pd.DataFrame):
            download_items['DC_Bus'] = dcbus_raw

    # HESS
    hess_raw = data.get('hess')
    if hess_raw is not None and isinstance(hess_raw, pd.DataFrame) and len(hess_raw) > 0:
        download_items['HESS'] = hess_raw

    # H₂
    h2_raw = data.get('h2')
    if h2_raw is not None and isinstance(h2_raw, pd.DataFrame) and len(h2_raw) > 0:
        download_items['H2_시스템'] = h2_raw

    # Grid
    grid_raw = data.get('grid')
    if grid_raw is not None and isinstance(grid_raw, pd.DataFrame) and len(grid_raw) > 0:
        download_items['Grid'] = grid_raw

    # AI-EMS
    ems_raw = data.get('ems_df')
    if ems_raw is not None and isinstance(ems_raw, pd.DataFrame) and len(ems_raw) > 0:
        download_items['AI_EMS_Dispatch'] = ems_raw

    # Carbon
    carbon_raw = data.get('carbon_df')
    if carbon_raw is not None and isinstance(carbon_raw, pd.DataFrame) and len(carbon_raw) > 0:
        download_items['탄소_회계'] = carbon_raw

    # Weather
    weather_raw = data.get('weather')
    if weather_raw is not None and isinstance(weather_raw, pd.DataFrame):
        download_items['기상_데이터'] = weather_raw

    cols = st.columns(3)
    for idx, (name, df) in enumerate(download_items.items()):
        with cols[idx % 3]:
            csv_buf = df.to_csv(index=True).encode('utf-8-sig')
            st.download_button(
                label=f"⬇️ {name}.csv",
                data=csv_buf,
                file_name=f"CEMS_DT5_{name}.csv",
                mime="text/csv",
                key=f"dl_csv_{name}"
            )
            st.caption(f"{len(df)} rows × {len(df.columns)} cols")

    # --- 2. 전체 통합 Excel ---
    st.markdown("### 📊 전체 통합 Excel (다중 시트)")

    try:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            for name, df in download_items.items():
                sheet_name = name[:31]  # Excel 시트명 31자 제한
                df.to_excel(writer, sheet_name=sheet_name, index=True)

            # 요약 시트 추가
            summary_data = {}
            pv_dict = _safe_dict(data.get('pv', {}))
            aidc_dict = _safe_dict(data.get('aidc', {}))
            if isinstance(pv_dict, dict) and 'power_mw' in pv_dict:
                summary_data['총 PV 발전량 (MWh)'] = sum(pv_dict['power_mw'])
            if isinstance(aidc_dict, dict) and 'total_power_mw' in aidc_dict:
                summary_data['총 AIDC 소비량 (MWh)'] = sum(aidc_dict['total_power_mw'])

            ems_kpi = data.get('ems_kpi')
            if ems_kpi and isinstance(ems_kpi, dict):
                summary_data['자급률 (%)'] = ems_kpi.get('self_sufficiency_ratio', 0) * 100
                summary_data['Grid 의존도 (%)'] = ems_kpi.get('grid_dependency_ratio', 0) * 100

            if summary_data:
                summary_df = pd.DataFrame(list(summary_data.items()), columns=['항목', '값'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

        st.download_button(
            label="📊 전체 데이터 다운로드 (Excel)",
            data=excel_buf.getvalue(),
            file_name="CEMS_DT5_전체데이터.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_excel_all"
        )
    except ImportError:
        st.warning("openpyxl 패키지가 필요합니다. CSV 개별 다운로드를 이용해주세요.")

    # --- 3. 전체 TXT (탭 구분) ---
    st.markdown("### 📝 전체 데이터 (TXT)")
    txt_buf = io.StringIO()
    txt_buf.write("=" * 80 + "\n")
    txt_buf.write("CEMS Digital Twin v5 — Simulation Export\n")
    txt_buf.write("=" * 80 + "\n\n")
    for name, df in download_items.items():
        txt_buf.write(f"### {name} ###\n")
        txt_buf.write(df.to_string() + "\n\n")

    st.download_button(
        label="📝 전체 데이터 다운로드 (TXT)",
        data=txt_buf.getvalue().encode('utf-8'),
        file_name="CEMS_DT5_전체데이터.txt",
        mime="text/plain",
        key="dl_txt_all"
    )


# ═══════════════════════════════════════════════════════════════
# 국제 비교 탭
# ═══════════════════════════════════════════════════════════════
def display_international_comparison(data):
    """국제 마이크로그리드 비교 — config 기반 + 사용자 override"""
    st.subheader("🌏 국제 마이크로그리드 비교")

    # config에서 벤치마크 복사 (원본 보호)
    benchmarks = copy.deepcopy(INTERNATIONAL_BENCHMARKS)

    # --- 데이터 업데이트 상태 ---
    st.caption(f"📅 기본 데이터 기준일: {BENCHMARK_LAST_UPDATED} | 출처: NREL ATB, IRENA, Fraunhofer ISE, METI, SERC")

    # --- 사용자 Override UI ---
    with st.expander("⚙️ 벤치마크 값 수동 조정 (Override)", expanded=False):
        st.markdown("최신 데이터로 직접 업데이트하거나, What-if 시나리오를 테스트할 수 있습니다.")

        override_country = st.selectbox(
            "조정할 국가", list(benchmarks.keys()),
            format_func=lambda k: benchmarks[k]['label'],
            key="override_country"
        )

        info = benchmarks[override_country]
        oc1, oc2, oc3, oc4 = st.columns(4)
        with oc1:
            new_irr = st.number_input("일사량 (kWh/m²/yr)", value=info['irradiance_kwh_m2_yr'],
                                       min_value=500, max_value=2500, step=50, key="ov_irr")
            info['irradiance_kwh_m2_yr'] = new_irr
        with oc2:
            new_elec = st.number_input("전기요금 ($/MWh)", value=info['elec_price_usd_mwh'],
                                        min_value=10, max_value=300, step=5, key="ov_elec")
            info['elec_price_usd_mwh'] = new_elec
        with oc3:
            new_ci = st.number_input("탄소강도 (gCO₂/kWh)", value=info['carbon_intensity_gco2_kwh'],
                                      min_value=50, max_value=800, step=10, key="ov_ci")
            info['carbon_intensity_gco2_kwh'] = new_ci
        with oc4:
            new_cp = st.number_input("탄소가격 ($/ton)", value=info['carbon_price_usd_ton'],
                                      min_value=0, max_value=200, step=5, key="ov_cp")
            info['carbon_price_usd_ton'] = new_cp

        oc5, oc6 = st.columns(2)
        with oc5:
            cur_lcoe = info.get('pv_lcoe_usd_mwh') or 50
            new_lcoe = st.number_input("LCOE ($/MWh)", value=int(cur_lcoe),
                                        min_value=10, max_value=200, step=5, key="ov_lcoe")
            info['pv_lcoe_usd_mwh'] = new_lcoe
        with oc6:
            cur_cf = info.get('capacity_factor') or 0.15
            new_cf = st.slider("Capacity Factor", 0.05, 0.40, float(cur_cf), 0.01, key="ov_cf")
            info['capacity_factor'] = new_cf

        # 출처 표시
        sources = info.get('sources', {})
        if sources:
            st.markdown("**데이터 출처:**")
            for field, src in sources.items():
                st.caption(f"  • {field}: {src}")

    # --- API 자동 업데이트 안내 ---
    with st.expander("🔄 자동 업데이트 파이프라인 (API 소스)", expanded=False):
        st.markdown("분기별/연간 자동 업데이트 가능한 공개 API 소스:")
        api_data = []
        for src_id, src_info in BENCHMARK_API_SOURCES.items():
            api_data.append({
                'ID': src_id,
                '설명': src_info['description'],
                '주기': src_info['update_freq'],
                '업데이트 필드': ', '.join(src_info['fields']),
                'URL': src_info['url'],
            })
        st.dataframe(pd.DataFrame(api_data), use_container_width=True, hide_index=True)
        st.info("💡 **향후 계획**: 크론잡으로 분기마다 API fetch → config.py 자동 갱신 → Git push")

    # --- 시뮬레이션 결과로 한국 데이터 자동 업데이트 ---
    if data is not None:
        kr = benchmarks['KR']
        pv_dict = _safe_dict(data.get('pv', {}))
        ems_kpi = data.get('ems_kpi', {})

        if isinstance(pv_dict, dict) and 'capacity_factor' in pv_dict:
            cf_list = pv_dict['capacity_factor']
            kr['capacity_factor'] = sum(cf_list) / len(cf_list) if cf_list else 0

        if ems_kpi and isinstance(ems_kpi, dict):
            kr['self_sufficiency'] = ems_kpi.get('self_sufficiency_ratio', 0)

        econ = data.get('modules', {}).get('economics')
        if econ:
            try:
                lcoe_result = econ.calculate_lcoe(
                    capex_total_billion_krw=10000,
                    annual_generation_mwh=sum(pv_dict.get('power_mw', [0])) * 365 / max(len(pv_dict.get('power_mw', [1])), 1) if isinstance(pv_dict, dict) else 100000,
                    opex_annual_billion_krw=200,
                    lifetime_years=25, discount_rate=0.06
                )
                kr['pv_lcoe_usd_mwh'] = lcoe_result.get('lcoe_krw_per_mwh', 136500) / 1350
            except:
                kr['pv_lcoe_usd_mwh'] = 101

    # --- 1. 비교 테이블 ---
    st.markdown("### 📋 주요 지표 비교")

    table_data = []
    for code, info in benchmarks.items():
        table_data.append({
            '국가': info['label'],
            'PV 용량 (MW)': info['capacity_mw'],
            'PV 기술': info['pv_type'],
            '일사량 (kWh/m²/yr)': info['irradiance_kwh_m2_yr'],
            '전기요금 ($/MWh)': info['elec_price_usd_mwh'],
            'LCOE ($/MWh)': f"{info['pv_lcoe_usd_mwh']:.0f}" if info['pv_lcoe_usd_mwh'] else 'DT 연동',
            'CF': f"{info['capacity_factor']:.1%}" if info['capacity_factor'] else '-',
            '자급률': f"{info['self_sufficiency']:.0%}" if info['self_sufficiency'] else '-',
            '탄소강도 (gCO₂/kWh)': info['carbon_intensity_gco2_kwh'],
            '탄소가격 ($/ton)': info['carbon_price_usd_ton'],
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # --- 2. Radar Chart ---
    st.markdown("### 🕸️ 종합 경쟁력 레이더 차트")

    fig_radar = go.Figure()
    colors = ['#ef4444', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6']
    radar_labels = ['일사량', '전기요금 경쟁력', '그리드 청정도', '탄소 규제 강도', '자급률', 'Capacity Factor']

    for idx, (code, info) in enumerate(benchmarks.items()):
        values = [
            info['irradiance_kwh_m2_yr'] / 2000,
            1 - info['elec_price_usd_mwh'] / 200,
            1 - info['carbon_intensity_gco2_kwh'] / 600,
            min(1.0, info['carbon_price_usd_ton'] / 60),
            (info.get('self_sufficiency') or 0.3),
            (info.get('capacity_factor') or 0.15) / 0.30,
        ]
        values.append(values[0])  # close polygon

        fig_radar.add_trace(go.Scatterpolar(
            r=values, theta=radar_labels + [radar_labels[0]],
            fill='toself', name=info['flag'] + ' ' + info['country'],
            line=dict(color=colors[idx % len(colors)]), opacity=0.7
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=550, template='plotly_white',
        title="국가별 마이크로그리드 경쟁력 비교"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- 3. Bar Charts ---
    st.markdown("### 📊 주요 지표 상세 비교")
    countries = [info['flag'] + ' ' + info['country'] for info in benchmarks.values()]
    bar_colors = ['#ef4444', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6']

    col1, col2 = st.columns(2)
    with col1:
        lcoe_vals = [info.get('pv_lcoe_usd_mwh') or 0 for info in benchmarks.values()]
        fig = go.Figure(go.Bar(x=countries, y=lcoe_vals, marker_color=bar_colors,
                               text=[f"${v:.0f}" for v in lcoe_vals], textposition='outside'))
        fig.update_layout(title="PV LCOE ($/MWh)", height=350, template='plotly_white', yaxis_title="$/MWh")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        elec_vals = [info['elec_price_usd_mwh'] for info in benchmarks.values()]
        fig = go.Figure(go.Bar(x=countries, y=elec_vals, marker_color=bar_colors,
                               text=[f"${v}" for v in elec_vals], textposition='outside'))
        fig.update_layout(title="산업용 전기요금 ($/MWh)", height=350, template='plotly_white', yaxis_title="$/MWh")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        ci_vals = [info['carbon_intensity_gco2_kwh'] for info in benchmarks.values()]
        fig = go.Figure(go.Bar(x=countries, y=ci_vals, marker_color=bar_colors,
                               text=[f"{v}" for v in ci_vals], textposition='outside'))
        fig.update_layout(title="그리드 탄소강도 (gCO₂/kWh)", height=350, template='plotly_white', yaxis_title="gCO₂/kWh")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        cp_vals = [info['carbon_price_usd_ton'] for info in benchmarks.values()]
        fig = go.Figure(go.Bar(x=countries, y=cp_vals, marker_color=bar_colors,
                               text=[f"${v}" for v in cp_vals], textposition='outside'))
        fig.update_layout(title="탄소가격 ($/ton CO₂)", height=350, template='plotly_white', yaxis_title="$/ton")
        st.plotly_chart(fig, use_container_width=True)

    # --- 4. 정책 환경 요약 ---
    st.markdown("### 📝 국가별 정책 환경 및 특이사항")
    for code, info in benchmarks.items():
        with st.expander(info['label']):
            st.markdown(f"""
| 항목 | 내용 |
|------|------|
| **PV 기술** | {info['pv_type']} |
| **저장 시스템** | {info['storage']} |
| **계통 연계** | {info['grid_type']} |
| **비고** | {info['notes']} |
""")
            # 출처 표시
            sources = info.get('sources', {})
            if sources:
                st.markdown("**📚 데이터 출처:**")
                for field, src in sources.items():
                    st.caption(f"  • {field}: {src}")

    # --- 5. 시사점 ---
    st.markdown("### 💡 비교 시사점")
    st.markdown("""
    **한국의 포지셔닝:**
    - 🌞 **일사량**: 중간 (1,340 kWh/m²/yr) — 독일(1,050)보다 유리, 미국 SW(1,800)보다 불리
    - 💰 **전기요금**: 중간 ($90/MWh) — 일본·독일 대비 경쟁력, 미국·중국 대비 불리
    - 🏭 **탄소강도**: 높음 (415 gCO₂/kWh) — RE 전환 필요성 큼 → **AIDC 마이크로그리드 당위성 ↑**
    - 📜 **탄소가격**: 낮음 ($20/ton) — K-ETS 강화 시 경제성 급상승 (정책 시뮬레이터 참조)

    **핵심 차별점 (본 DT5):**
    - 🔬 **Tandem Perovskite-Si**: 효율 30%+ 차세대 PV (타국 대비 기술 리드)
    - ⚡ **3-tier HESS**: Supercap(초단주기) + BESS(중주기) + H₂(장주기) → 타국은 BESS 단일
    - 🤖 **AI-EMS 3단계 최적화**: LP 기반 실시간 디스패치 → Rule-based 대비 비용 절감
    - 🏢 **AIDC 전용 설계**: GPU 워크로드 특성 반영 (타국은 범용 마이크로그리드)
    """)


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


def display_duck_curve(data):
    """🦆 Duck Curve 분석 — 한국 vs CAISO 비교"""
    st.subheader("🦆 Duck Curve 분석")
    st.markdown("""
    **Duck Curve**는 태양광 대량 도입 시 순부하(Net Load = 총수요 − PV − 풍력)가
    오리 형상을 그리는 현상입니다. CAISO(캘리포니아)에서 2013년 처음 예측되었고,
    현재 실측으로 확인되고 있습니다.
    """)

    # --- 사이드바 컨트롤 ---
    st.markdown("#### ⚙️ 시나리오 설정")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        total_demand_gw = st.slider("한국 피크 수요 (GW)", 50, 120, 90, 5, key="duck_demand")
    with sc2:
        pv_capacity_gw = st.slider("태양광 설치용량 (GW)", 10, 150, 40, 5, key="duck_pv")
    with sc3:
        wind_capacity_gw = st.slider("풍력 설치용량 (GW)", 5, 50, 20, 5, key="duck_wind")

    sc4, sc5, sc6 = st.columns(3)
    with sc4:
        storage_gw = st.slider("ESS 용량 (GW)", 0, 50, 10, 2, key="duck_storage")
    with sc5:
        season = st.selectbox("계절", ["봄 (4월)", "여름 (7월)", "가을 (10월)", "겨울 (1월)"], key="duck_season")
    with sc6:
        compare_caiso = st.checkbox("CAISO 실측 비교", value=True, key="duck_caiso")

    hours = np.arange(24)

    # --- 한국 수요 프로파일 (계절별) ---
    season_profiles = {
        "봄 (4월)": {
            'demand': [0.72, 0.68, 0.65, 0.63, 0.64, 0.70, 0.82, 0.92, 0.95, 0.96,
                       0.95, 0.93, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.95, 0.90,
                       0.85, 0.82, 0.78, 0.74],
            'solar_cf': [0, 0, 0, 0, 0, 0.02, 0.15, 0.35, 0.55, 0.70,
                         0.80, 0.85, 0.87, 0.85, 0.78, 0.65, 0.45, 0.20, 0.03, 0, 0, 0, 0, 0],
            'wind_cf': 0.25
        },
        "여름 (7월)": {
            'demand': [0.78, 0.74, 0.70, 0.68, 0.69, 0.74, 0.85, 0.93, 0.97, 1.00,
                       1.00, 0.99, 0.97, 0.98, 1.00, 1.00, 0.99, 0.97, 0.95, 0.92,
                       0.88, 0.85, 0.82, 0.80],
            'solar_cf': [0, 0, 0, 0, 0, 0.03, 0.12, 0.28, 0.45, 0.58,
                         0.65, 0.68, 0.70, 0.68, 0.62, 0.50, 0.35, 0.15, 0.02, 0, 0, 0, 0, 0],
            'wind_cf': 0.18
        },
        "가을 (10월)": {
            'demand': [0.70, 0.66, 0.63, 0.62, 0.63, 0.68, 0.80, 0.90, 0.94, 0.95,
                       0.94, 0.92, 0.89, 0.91, 0.93, 0.94, 0.95, 0.96, 0.93, 0.88,
                       0.83, 0.80, 0.76, 0.72],
            'solar_cf': [0, 0, 0, 0, 0, 0, 0.08, 0.25, 0.45, 0.62,
                         0.73, 0.78, 0.80, 0.78, 0.70, 0.55, 0.35, 0.10, 0, 0, 0, 0, 0, 0],
            'wind_cf': 0.28
        },
        "겨울 (1월)": {
            'demand': [0.75, 0.72, 0.70, 0.68, 0.70, 0.76, 0.88, 0.95, 0.97, 0.96,
                       0.94, 0.92, 0.90, 0.91, 0.93, 0.94, 0.96, 0.98, 1.00, 0.97,
                       0.92, 0.88, 0.82, 0.78],
            'solar_cf': [0, 0, 0, 0, 0, 0, 0, 0.10, 0.25, 0.40,
                         0.52, 0.58, 0.60, 0.58, 0.48, 0.30, 0.10, 0, 0, 0, 0, 0, 0, 0],
            'wind_cf': 0.32
        }
    }

    profile = season_profiles[season]
    demand = np.array(profile['demand']) * total_demand_gw
    solar_gen = np.array(profile['solar_cf']) * pv_capacity_gw
    wind_gen = np.ones(24) * wind_capacity_gw * profile['wind_cf']
    # 풍력 일변동 (새벽 강, 낮 약)
    wind_variation = np.array([1.15, 1.18, 1.20, 1.22, 1.20, 1.15, 1.05, 0.95, 0.88, 0.82,
                               0.80, 0.78, 0.78, 0.80, 0.82, 0.85, 0.90, 0.95, 1.00, 1.05,
                               1.08, 1.10, 1.12, 1.14])
    wind_gen = wind_gen * wind_variation

    net_load = demand - solar_gen - wind_gen

    # --- ESS 효과 시뮬레이션 ---
    net_load_with_storage = net_load.copy()
    storage_soc = 0.5 * storage_gw * 4  # 4시간 저장 가정, GWh
    storage_max_gwh = storage_gw * 4
    storage_profile = np.zeros(24)

    for h in range(24):
        surplus = demand[h] - net_load[h] - demand[h]  # = solar + wind
        re_gen = solar_gen[h] + wind_gen[h]

        if net_load[h] < demand[h] * 0.6 and storage_soc < storage_max_gwh * 0.95:
            # 순부하 낮을 때 충전 (belly 구간)
            charge = min(storage_gw, re_gen * 0.5, (storage_max_gwh * 0.95 - storage_soc))
            storage_soc += charge * 0.92  # 충전 효율
            storage_profile[h] = -charge  # 음수 = 충전
            net_load_with_storage[h] += charge
        elif net_load[h] > demand[h] * 0.85 and storage_soc > storage_max_gwh * 0.1:
            # 순부하 높을 때 방전 (evening ramp 구간)
            discharge = min(storage_gw, storage_soc * 0.92, net_load[h] - demand[h] * 0.7)
            discharge = max(0, discharge)
            storage_soc -= discharge / 0.92
            storage_profile[h] = discharge  # 양수 = 방전
            net_load_with_storage[h] -= discharge

    # --- CAISO 실측 데이터 (2025년 4월 기준, GridStatus.io) ---
    caiso_data = {
        'demand': [22, 21, 20, 19.5, 19.5, 20.5, 23, 25, 26.5, 27, 27.5, 28,
                   28, 27.5, 27, 27.5, 28.5, 29, 28.5, 27, 25.5, 24.5, 23.5, 22.5],
        'solar': [0, 0, 0, 0, 0, 0.2, 2, 6, 10, 13, 15, 16.5,
                  17, 16.5, 15, 12, 8, 3, 0.3, 0, 0, 0, 0, 0],
        'net_load': [22, 21, 20, 19.5, 19.5, 20.3, 21, 19, 16.5, 14, 12.5, 11.5,
                     11, 11, 12, 15.5, 20.5, 26, 28.2, 27, 25.5, 24.5, 23.5, 22.5],
        'storage': [0.5, 0.3, 0.2, 0.1, 0, -0.5, -2, -4, -5.5, -6, -5.5, -5,
                    -4, -3, -2, 0, 2, 5, 7, 6, 4, 2, 1, 0.5],
        # 발전원별 프로파일 (GW) — Chart ①
        'nuclear': np.full(24, 1.1),
        'geothermal': np.full(24, 2.8),
        'large_hydro': np.array([2.0, 1.8, 1.7, 1.6, 1.6, 1.7, 2.0, 2.2, 2.5, 2.8,
                                  3.0, 3.2, 3.3, 3.2, 3.0, 2.8, 2.5, 2.2, 2.5, 3.0,
                                  3.2, 3.0, 2.5, 2.2]),
        'gas': np.array([8.5, 8.0, 7.5, 7.0, 7.0, 7.5, 8.5, 6.5, 3.5, 2.0,
                          1.5, 1.2, 1.0, 1.2, 1.5, 2.5, 5.0, 8.0, 8.5, 8.0,
                          7.5, 7.0, 8.0, 8.5]),
        'wind': np.array([3.5, 3.8, 4.0, 4.2, 4.0, 3.5, 3.0, 2.5, 2.0, 1.8,
                           1.5, 1.3, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0, 3.2,
                           3.5, 3.5, 3.5, 3.5]),
        'imports': np.array([4.0, 4.5, 3.5, 3.0, 3.0, 4.0, 5.5, 5.5, 4.0, 2.5,
                              1.5, 1.0, 0.5, 0.5, 1.0, 2.0, 4.5, 6.5, 7.0, 7.0,
                              6.5, 5.5, 4.5, 4.0]),
        'storage_gen': np.array([0.5, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 2.0, 5.0, 7.0, 6.0,
                                  4.0, 2.0, 1.0, 0.5]),
        # 연도별 믹스 비중 (%) — Chart ②
        'yearly_mix': {
            2018: {'Gas': 21, 'Solar': 16, 'Imports': 24, 'Wind': 10, 'Hydro': 14, 'Nuclear': 9, 'Other': 4, 'Storage': 2},
            2019: {'Gas': 19, 'Solar': 18, 'Imports': 22, 'Wind': 11, 'Hydro': 13, 'Nuclear': 9, 'Other': 4, 'Storage': 4},
            2020: {'Gas': 17, 'Solar': 20, 'Imports': 20, 'Wind': 11, 'Hydro': 13, 'Nuclear': 9, 'Other': 5, 'Storage': 5},
            2021: {'Gas': 18, 'Solar': 21, 'Imports': 18, 'Wind': 12, 'Hydro': 12, 'Nuclear': 9, 'Other': 5, 'Storage': 5},
            2022: {'Gas': 17, 'Solar': 24, 'Imports': 15, 'Wind': 12, 'Hydro': 13, 'Nuclear': 8, 'Other': 6, 'Storage': 5},
            2023: {'Gas': 15, 'Solar': 28, 'Imports': 12, 'Wind': 13, 'Hydro': 13, 'Nuclear': 8, 'Other': 6, 'Storage': 5},
            2024: {'Gas': 14, 'Solar': 30, 'Imports': 11, 'Wind': 13, 'Hydro': 13, 'Nuclear': 7, 'Other': 6, 'Storage': 6},
            2025: {'Gas': 14, 'Solar': 32, 'Imports': 10, 'Wind': 13, 'Hydro': 13, 'Nuclear': 7, 'Other': 5, 'Storage': 6},
        },
        # 연도별 태양광 피크 (GW) — Chart ③
        'solar_by_year': {
            2018: [0,0,0,0,0,0,0.5,2.0,3.5,5.0,6.0,6.5,7.0,6.8,6.0,4.5,2.5,0.8,0,0,0,0,0,0],
            2019: [0,0,0,0,0,0,0.6,2.3,4.0,5.5,6.8,7.5,7.8,7.5,6.8,5.0,3.0,1.0,0,0,0,0,0,0],
            2020: [0,0,0,0,0,0,0.8,2.8,4.8,6.5,8.0,8.8,9.0,8.8,8.0,6.0,3.5,1.2,0,0,0,0,0,0],
            2021: [0,0,0,0,0,0,1.0,3.0,5.5,7.5,9.0,10.0,10.2,10.0,9.0,7.0,4.5,1.5,0,0,0,0,0,0],
            2022: [0,0,0,0,0,0,1.0,3.2,5.8,8.0,9.5,10.5,10.8,10.5,9.5,7.2,4.8,1.5,0,0,0,0,0,0],
            2023: [0,0,0,0,0,0.1,1.5,4.0,7.0,9.5,11.5,12.5,13.0,12.5,11.5,9.0,6.0,2.0,0.1,0,0,0,0,0],
            2024: [0,0,0,0,0,0.1,1.8,4.8,8.0,11.0,13.0,14.0,14.5,14.0,13.0,10.5,7.0,2.5,0.2,0,0,0,0,0],
            2025: [0,0,0,0,0,0.2,2.0,6.0,10.0,13.0,15.0,16.5,17.0,16.5,15.0,12.0,8.0,3.0,0.3,0,0,0,0,0],
        },
        # 연도별 Storage 충방전 (GW) — Chart ④
        'storage_by_year': {
            2018: [0]*24,
            2019: [0]*24,
            2020: [0]*24,
            2021: [0]*24,
            2022: [0.1,0.1,0,0,0,-0.1,-0.5,-1.0,-1.5,-1.8,-1.5,-1.2,-1.0,-0.8,-0.5,0,0.5,1.5,2.5,2.0,1.5,0.8,0.3,0.1],
            2023: [0.2,0.1,0.1,0,0,-0.2,-1.0,-2.0,-3.0,-3.5,-3.2,-2.8,-2.5,-2.0,-1.5,0,1.0,3.0,4.5,4.0,3.0,1.5,0.8,0.3],
            2024: [0.3,0.2,0.1,0.1,0,-0.3,-1.5,-3.0,-4.5,-5.0,-4.5,-4.0,-3.5,-2.5,-1.5,0,1.5,4.0,6.0,5.5,3.5,2.0,1.0,0.5],
            2025: [0.5,0.3,0.2,0.1,0,-0.5,-2.0,-4.0,-5.5,-6.0,-5.5,-5.0,-4.0,-3.0,-2.0,0,2.0,5.0,7.0,6.0,4.0,2.0,1.0,0.5],
        },
        # 연도별 Curtailment (MW) — Chart ⑤
        'curtailment_by_year': {
            2017: [0,0,0,0,0,0,0,10,30,50,60,70,65,50,30,10,0,0,0,0,0,0,0,0],
            2018: [0,0,0,0,0,0,0,20,60,100,130,150,140,120,80,30,0,0,0,0,0,0,0,0],
            2019: [0,0,0,0,0,0,0,50,150,300,400,450,420,350,200,80,0,0,0,0,0,0,0,0],
            2020: [0,0,0,0,0,0,0,80,250,500,650,750,700,600,400,150,20,0,0,0,0,0,0,0],
            2021: [0,0,0,0,0,0,0,80,250,500,650,800,750,650,400,150,20,0,0,0,0,0,0,0],
            2022: [0,0,0,0,0,0,10,200,600,1000,1300,1500,1450,1200,800,300,50,0,0,0,0,0,0,0],
            2023: [0,0,0,0,0,0,10,180,550,900,1200,1400,1350,1100,750,250,40,0,0,0,0,0,0,0],
            2024: [0,0,0,0,0,0,10,200,600,1000,1350,1550,1500,1250,850,350,60,0,0,0,0,0,0,0],
            2025: [0,0,0,0,0,0,15,220,650,1100,1400,1600,1550,1300,900,350,60,0,0,0,0,0,0,0],
        },
        # 연도별 Import/Export (MW) — Chart ⑥
        'import_export_by_year': {
            2019: [6000,6200,5800,5500,5500,6000,7000,5000,3000,2000,1500,1200,1000,1200,1500,2500,5000,6500,7500,7500,7000,6500,6200,6000],
            2020: [6000,6200,5800,5500,5500,6000,7000,4500,2500,1500,800,500,300,500,1000,2000,5000,6500,7500,7500,7000,6500,6200,6000],
            2021: [6000,6100,5700,5400,5400,5800,6800,4200,2000,1000,500,200,0,200,800,1800,4500,6200,7200,7200,6800,6200,6000,5800],
            2022: [6000,6100,5700,5400,5400,5800,6800,4000,1500,500,0,-500,-800,-500,200,1500,4500,6500,7500,7500,7000,6500,6000,5800],
            2023: [5800,6000,5600,5300,5300,5700,6500,3500,1000,-200,-1000,-1500,-1800,-1500,-800,800,4000,6500,8000,8000,7500,6500,6000,5800],
            2024: [5800,5900,5500,5200,5200,5600,6300,3200,500,-800,-1800,-2500,-2800,-2500,-1500,300,3500,6000,8000,8200,7500,6500,6000,5800],
            2025: [5500,5700,5300,5000,5000,5400,6000,3000,200,-500,-1500,-2000,-2200,-2000,-1200,500,3500,6000,8500,8500,7500,6500,6000,5800],
        }
    }

    # ==================== 차트 1: 한국 Duck Curve ====================
    fig1 = go.Figure()

    # 총 수요
    fig1.add_trace(go.Scatter(
        x=hours, y=demand, mode='lines', name='총 수요',
        line=dict(color='white', width=3, dash='dot'),
        fill=None
    ))

    # 순부하 (PV+풍력 차감)
    fig1.add_trace(go.Scatter(
        x=hours, y=net_load, mode='lines', name='순부하 (Net Load)',
        line=dict(color='#f59e0b', width=3),
        fill='tonexty', fillcolor='rgba(245,158,11,0.15)'
    ))

    if storage_gw > 0:
        fig1.add_trace(go.Scatter(
            x=hours, y=net_load_with_storage, mode='lines', name='순부하 + ESS',
            line=dict(color='#10b981', width=3, dash='dash')
        ))

    # Duck 영역 표시
    belly_min_idx = np.argmin(net_load)
    ramp_max_idx = 17  # 저녁 6시
    belly_val = net_load[belly_min_idx]
    ramp_val = net_load[ramp_max_idx] if ramp_max_idx < len(net_load) else net_load[-1]

    fig1.add_annotation(x=belly_min_idx, y=belly_val,
                        text=f"🦆 Belly<br>{belly_val:.1f} GW",
                        showarrow=True, arrowhead=2, font=dict(size=13, color='#f59e0b'))
    fig1.add_annotation(x=ramp_max_idx, y=ramp_val,
                        text=f"⚡ Evening Ramp<br>{ramp_val:.1f} GW",
                        showarrow=True, arrowhead=2, font=dict(size=13, color='#ef4444'))

    duck_depth = max(demand) - belly_val
    ramp_rate = ramp_val - belly_val

    fig1.update_layout(
        title=f"🇰🇷 한국 Duck Curve — {season} | PV {pv_capacity_gw}GW, 풍력 {wind_capacity_gw}GW",
        xaxis_title="시간 (Hour)", yaxis_title="전력 (GW)",
        template="plotly_white", height=500,
        xaxis=dict(tickmode='linear', dtick=2),
        legend=dict(orientation="h", y=-0.15)
    )

    st.plotly_chart(fig1, use_container_width=True)

    # --- KPI 카드 ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("🦆 Duck Depth", f"{duck_depth:.1f} GW",
                   help="피크 수요 대비 순부하 최저점 차이")
    with k2:
        st.metric("⚡ Evening Ramp", f"{ramp_rate:.1f} GW",
                   help="Belly → Evening Peak 상승폭 (3-4시간)")
    with k3:
        over_gen = max(0, -min(net_load))
        st.metric("⚠️ 과잉발전", f"{over_gen:.1f} GW",
                   help="순부하 < 0 구간 (curtailment 필요)")
    with k4:
        re_peak_share = (max(solar_gen) + max(wind_gen)) / max(demand) * 100
        st.metric("☀️ RE 피크 비중", f"{re_peak_share:.0f}%",
                   help="재생에너지 피크 / 수요 피크")

    # ==================== 차트 2: PV 시나리오 비교 ====================
    st.markdown("---")
    st.markdown("#### 📈 태양광 확대 시나리오별 Duck Curve 변화")

    scenarios = {
        f"현재 ({pv_capacity_gw}GW)": pv_capacity_gw,
        "2030 목표 (60GW)": 60,
        "2035 전망 (100GW)": 100,
        "극단 시나리오 (150GW)": 150
    }

    fig2 = go.Figure()
    colors = ['#6b7280', '#3b82f6', '#f59e0b', '#ef4444']

    for i, (label, pv_gw) in enumerate(scenarios.items()):
        sg = np.array(profile['solar_cf']) * pv_gw
        nl = demand - sg - wind_gen
        fig2.add_trace(go.Scatter(
            x=hours, y=nl, mode='lines', name=label,
            line=dict(color=colors[i], width=2.5 if i > 0 else 1.5,
                      dash='dot' if i == 0 else 'solid')
        ))

    fig2.add_trace(go.Scatter(
        x=hours, y=demand, mode='lines', name='총 수요',
        line=dict(color='gray', width=1, dash='dot'), opacity=0.5
    ))

    fig2.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5,
                   annotation_text="과잉발전 경계")

    fig2.update_layout(
        title=f"태양광 확대에 따른 Duck Curve 심화 — {season}",
        xaxis_title="시간 (Hour)", yaxis_title="순부하 (GW)",
        template="plotly_white", height=450,
        xaxis=dict(tickmode='linear', dtick=2),
        legend=dict(orientation="h", y=-0.15)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ==================== 차트 3: 한국 vs CAISO 비교 ====================
    if compare_caiso:
        st.markdown("---")
        st.markdown("### 🇺🇸 CAISO 실측 데이터 (GridStatus.io, 2025년 4월)")
        st.caption("캘리포니아 ISO — 세계 최초로 Duck Curve를 경험하고 관리하는 그리드")

        # --- Chart ① Stacked Area: 발전원별 Fuel Mix ---
        st.markdown("#### ① 발전원별 일중 Fuel Mix")
        fig_fuel = go.Figure()
        fuel_sources = [
            ('Nuclear', caiso_data['nuclear'], '#ef4444'),
            ('Geothermal', caiso_data['geothermal'], '#22c55e'),
            ('Large Hydro', caiso_data['large_hydro'], '#06b6d4'),
            ('Gas', caiso_data['gas'], '#f97316'),
            ('Wind', caiso_data['wind'], '#3b82f6'),
            ('Solar', np.array(caiso_data['solar']), '#eab308'),
            ('Imports', caiso_data['imports'], '#ec4899'),
            ('Storage', caiso_data['storage_gen'], '#a855f7'),
        ]
        for name, vals, color in fuel_sources:
            fig_fuel.add_trace(go.Scatter(
                x=hours, y=vals, name=name, mode='lines',
                stackgroup='one', fillcolor=color, line=dict(width=0.5, color=color)
            ))
        fig_fuel.update_layout(
            title="CAISO 발전원별 구성 — 낮에는 Solar가 지배 (2025.4)",
            xaxis_title="시간 (Hour)", yaxis_title="발전량 (GW)",
            template="plotly_white", height=450,
            xaxis=dict(tickmode='linear', dtick=2),
            yaxis=dict(range=[0, 32]),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_fuel, use_container_width=True)

        # --- Chart ② 100% Stacked Bar: 연도별 믹스 비중 변화 ---
        st.markdown("#### ② 연도별 Fuel Mix 비중 변화 (2018→2025)")
        years = list(caiso_data['yearly_mix'].keys())
        mix_sources = ['Gas', 'Solar', 'Imports', 'Wind', 'Hydro', 'Nuclear', 'Other', 'Storage']
        mix_colors = {'Gas': '#f97316', 'Solar': '#eab308', 'Imports': '#ec4899',
                      'Wind': '#3b82f6', 'Hydro': '#06b6d4', 'Nuclear': '#22c55e',
                      'Other': '#9ca3af', 'Storage': '#a855f7'}

        fig_mix = go.Figure()
        for src in mix_sources:
            vals = [caiso_data['yearly_mix'][y][src] for y in years]
            fig_mix.add_trace(go.Bar(
                x=[str(y) for y in years], y=vals, name=src,
                marker_color=mix_colors[src], text=[f"{v}%" if v >= 8 else "" for v in vals],
                textposition='inside', textfont=dict(size=11, color='white')
            ))
        fig_mix.update_layout(
            title="Solar 16%→32% (2배↑), Gas 21%→14% (↓), Storage 0%→6% (신규)",
            barmode='stack', template="plotly_white", height=420,
            yaxis=dict(title="비중 (%)", range=[0, 105]),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_mix, use_container_width=True)

        # --- Chart ③ Solar 피크 연도별 성장 ---
        st.markdown("#### ③ 태양광 발전 피크 성장 (~2.5GW/년 가속)")
        fig_solar = go.Figure()
        solar_colors = {2018: '#fed7aa', 2019: '#fdba74', 2020: '#fb923c',
                        2021: '#f97316', 2022: '#ea580c', 2023: '#c2410c',
                        2024: '#fff', 2025: '#fff'}
        for yr, vals in caiso_data['solar_by_year'].items():
            w = 3.5 if yr >= 2024 else 1.5
            fig_solar.add_trace(go.Scatter(
                x=hours, y=vals, name=str(yr), mode='lines',
                line=dict(width=w, color=solar_colors.get(yr, '#f97316'))
            ))
        fig_solar.add_annotation(x=12, y=17000/1000, text="~2.5GW ↑",
                                  showarrow=True, arrowhead=2,
                                  font=dict(size=14, color='#f59e0b'))
        fig_solar.update_layout(
            title="CAISO Solar — 2023년 이후 설치 가속",
            xaxis_title="시간 (Hour)", yaxis_title="발전량 (GW)",
            template="plotly_white", height=420,
            xaxis=dict(tickmode='linear', dtick=2),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_solar, use_container_width=True)

        # --- Chart ④ Storage 충방전 연도별 ---
        st.markdown("#### ④ ESS 충방전 스윙 확대 (22+ GW 일일 변동)")
        fig_storage = go.Figure()
        stor_colors = {2018: '#e0e0e0', 2019: '#c0c0c0', 2020: '#a0a0a0',
                       2021: '#808080', 2022: '#c4b5fd', 2023: '#a78bfa',
                       2024: '#8b5cf6', 2025: '#7c3aed'}
        for yr, vals in caiso_data['storage_by_year'].items():
            w = 3 if yr >= 2024 else 1.2
            fig_storage.add_trace(go.Scatter(
                x=hours, y=vals, name=str(yr), mode='lines',
                line=dict(width=w, color=stor_colors.get(yr, '#808080'))
            ))
        fig_storage.add_hline(y=0, line_color="gray", line_width=1, line_dash="dash")
        fig_storage.add_annotation(x=6, y=3.4, text="~3.4 GW<br>Morning",
                                    showarrow=True, font=dict(size=12))
        fig_storage.add_annotation(x=18, y=7, text="~7 GW<br>Evening Peak",
                                    showarrow=True, font=dict(size=12, color='#7c3aed'))
        fig_storage.add_annotation(x=12, y=-6, text="~6 GW<br>Charging",
                                    showarrow=True, font=dict(size=12, color='#3b82f6'))
        fig_storage.update_layout(
            title="2022년까지 거의 0 → 2025년 일일 스윙 22+GW",
            xaxis_title="시간 (Hour)", yaxis_title="전력 (GW) [+방전/-충전]",
            template="plotly_white", height=420,
            xaxis=dict(tickmode='linear', dtick=2),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_storage, use_container_width=True)

        # --- Chart ⑤ Curtailment 연도별 ---
        st.markdown("#### ⑤ Solar Curtailment — 용량 2배 ↑인데 커테일먼트는 비슷")
        fig_curt = go.Figure()
        curt_colors = {2017: '#fef3c7', 2018: '#fde68a', 2019: '#fcd34d',
                       2020: '#fbbf24', 2021: '#f59e0b', 2022: '#fff',
                       2023: '#fff', 2024: '#fff', 2025: '#fff'}
        for yr, vals in caiso_data['curtailment_by_year'].items():
            w = 3 if yr >= 2022 else 1.5
            fig_curt.add_trace(go.Scatter(
                x=list(range(6, 20)), y=vals[6:20], name=str(yr), mode='lines',
                line=dict(width=w, color=curt_colors.get(yr, '#f59e0b'))
            ))
        fig_curt.update_layout(
            title="Storage가 잉여 태양광을 흡수 → 커테일먼트 안정화",
            xaxis_title="시간 (Hour)", yaxis_title="커테일먼트 (MW)",
            template="plotly_white", height=400,
            xaxis=dict(tickmode='linear', dtick=1, range=[6, 19]),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_curt, use_container_width=True)

        # --- Chart ⑥ Import/Export 연도별 ---
        st.markdown("#### ⑥ 수출입 패턴 — 낮 수출 증가 (자체 저장 전환)")
        fig_ie = go.Figure()
        ie_colors = {2019: '#e5e7eb', 2020: '#d1d5db', 2021: '#9ca3af',
                     2022: '#ddd6fe', 2023: '#c4b5fd',
                     2024: '#a78bfa', 2025: '#7c3aed'}
        for yr, vals in caiso_data['import_export_by_year'].items():
            vals_gw = [v / 1000 for v in vals]
            w = 3 if yr >= 2024 else 1.5
            fig_ie.add_trace(go.Scatter(
                x=hours, y=vals_gw, name=str(yr), mode='lines',
                line=dict(width=w, color=ie_colors.get(yr, '#808080'))
            ))
        fig_ie.add_hline(y=0, line_color="gray", line_width=1, line_dash="dash",
                          annotation_text="Export ↓ / Import ↑")
        fig_ie.update_layout(
            title="낮 수출(음수) 확대 → Storage가 흡수하면서 점차 감소",
            xaxis_title="시간 (Hour)", yaxis_title="전력 (GW) [+수입/-수출]",
            template="plotly_white", height=420,
            xaxis=dict(tickmode='linear', dtick=2),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig_ie, use_container_width=True)

        # --- 한국 vs CAISO 비교 요약 테이블 ---
        st.markdown("---")
        st.markdown("#### 📊 한국 vs CAISO 정량 비교")

        kr_belly = min(net_load)
        kr_ramp = net_load[17] - kr_belly
        ca_belly = min(caiso_data['net_load'])
        ca_ramp = max(caiso_data['net_load']) - ca_belly

        comp_df = pd.DataFrame({
            '지표': ['피크 수요 (GW)', '태양광 설치 (GW)', '태양광 피크 발전 (GW)',
                    'Duck Belly (GW)', 'Evening Ramp (GW)', 'Belly/피크 비율',
                    'Storage 방전 피크 (GW)', 'Solar 비중', 'Storage 비중'],
            '🇰🇷 한국 (시뮬레이션)': [f"{max(demand):.0f}", f"{pv_capacity_gw}",
                       f"{max(solar_gen):.0f}", f"{kr_belly:.1f}", f"{kr_ramp:.1f}",
                       f"{kr_belly/max(demand)*100:.0f}%", f"{storage_gw}",
                       f"~{max(solar_gen)/max(demand)*100:.0f}%", f"~{storage_gw/max(demand)*100:.0f}%"],
            '🇺🇸 CAISO (실측 2025)': [f"{max(caiso_data['demand']):.0f}", "~30",
                         f"{max(caiso_data['solar']):.0f}", f"{ca_belly:.1f}", f"{ca_ramp:.1f}",
                         f"{ca_belly/max(caiso_data['demand'])*100:.0f}%", "~7",
                         "32%", "6%"]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ==================== 차트: ESS 충방전 프로파일 ====================
    if storage_gw > 0:
        st.markdown("---")
        st.markdown("#### 🔋 한국 ESS 충방전 시뮬레이션")

        fig4 = go.Figure()

        charge_vals = np.where(storage_profile < 0, storage_profile, 0)
        discharge_vals = np.where(storage_profile > 0, storage_profile, 0)

        fig4.add_trace(go.Bar(x=hours, y=charge_vals, name='충전',
                               marker_color='#3b82f6', opacity=0.8))
        fig4.add_trace(go.Bar(x=hours, y=discharge_vals, name='방전',
                               marker_color='#ef4444', opacity=0.8))

        if compare_caiso:
            ca_storage_scaled = np.array(caiso_data['storage'])
            fig4.add_trace(go.Scatter(
                x=hours, y=ca_storage_scaled, mode='lines',
                name='CAISO Storage (실측)',
                line=dict(color='#8b5cf6', width=2, dash='dash')
            ))

        fig4.add_hline(y=0, line_color="gray", line_width=1)
        fig4.update_layout(
            title="한국 ESS 일중 운영 패턴 — 낮 충전 / 저녁 방전",
            xaxis_title="시간 (Hour)", yaxis_title="전력 (GW)",
            template="plotly_white", height=400, barmode='relative',
            xaxis=dict(tickmode='linear', dtick=2),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(fig4, use_container_width=True)

    # --- 인사이트 ---
    st.markdown("---")
    st.markdown("#### 💡 핵심 인사이트")

    min_net = min(net_load)
    if min_net < 0:
        st.warning(f"⚠️ 순부하가 **{abs(min_net):.1f} GW** 음수 — 커테일먼트 또는 추가 저장 필요")
    
    st.markdown(f"""
    - **Duck Depth {duck_depth:.1f} GW**: 피크 수요의 {duck_depth/max(demand)*100:.0f}%에 해당하는 깊이
    - **Evening Ramp {ramp_rate:.1f} GW / 3~4시간**: 시간당 {ramp_rate/4:.1f} GW 급상승 → 유연성 자원 필수
    - **ESS {storage_gw} GW 투입 시**: Ramp 완화 효과 {max(0, ramp_rate - (net_load_with_storage[17] - min(net_load_with_storage))):.1f} GW 감소
    - **CAISO 교훈**: Storage 6% 비중만으로 22GW 일일 스윙 관리 중 (2025)
    
    **한국 시사점**: PV {pv_capacity_gw}GW 기준, ESS 없이는 낮 과잉/저녁 부족의 구조적 불균형 심화.
    HESS 6-layer (Li-ion + Na-ion + RFB + H₂) 조합이 최적 해법.
    
    > *CAISO 실측 데이터 출처: [GridStatus.io](https://www.gridstatus.io/live) (Data: CAISO)*
    > *BNEF 2025 Hydrogen Levelized Cost Report*
    > *Nature Reviews Materials (2025) doi:10.1038/s41578-025-00857-4*
    """)


if __name__ == "__main__":
    create_main_dashboard()