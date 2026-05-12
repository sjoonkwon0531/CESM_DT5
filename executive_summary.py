"""
Executive Summary 모듈

시뮬레이션 직후 평가위원이 30초 안에 핵심 결과를 파악하도록
6대 수치 + 한국어 해설을 제공합니다.

평가 메시지는 자동 생성하지 않습니다 — 평가위원이 직접 판단하도록
단순 수치 + 해설만 제공.
"""

import streamlit as st
import numpy as np
import pandas as pd


# === 벤치마크 데이터 (해설에 사용) ===
# 출처: KEPCO 2023, NREL ATB 2024, IEA Cost of Electricity 2024
BENCHMARKS = {
    "korea_smp_usd_per_mwh": 124.0,        # 한국 평균 SMP (2023~2024)
    "us_dc_lcoe_usd_per_mwh": 89.0,        # 미국 데이터센터 PPA 평균
    "global_dc_pue_avg": 1.58,             # 글로벌 데이터센터 평균 PUE (Uptime 2023)
    "hyperscale_pue_target": 1.10,         # 하이퍼스케일 목표 PUE
    "korea_grid_ef_kgco2_per_kwh": 0.4594, # 한국 그리드 배출계수
    "korean_re100_target_2030": 60,        # K-RE100 2030 목표
    "discount_rate": 5.0,                  # 본 시뮬레이터 디폴트 할인율
}


def _safe_dict(d):
    """Safely return dict from value (handle None/missing)."""
    return d if isinstance(d, dict) else {}


def _extract_kpis(data, cached_base):
    """시뮬레이션 데이터에서 6대 KPI 추출 — app.py의 _display_top_kpi() 로직 재사용"""
    pv_data = _safe_dict(data.get('pv', {}))
    aidc_data = _safe_dict(data.get('aidc', {}))

    pv_power = pv_data.get('power_mw', [])
    aidc_power = aidc_data.get('total_power_mw', [])

    total_pv = float(np.sum(pv_power)) if len(pv_power) > 0 else 0
    total_aidc = float(np.sum(aidc_power)) if len(aidc_power) > 0 else 0
    self_sufficiency = min(total_pv / total_aidc * 100, 100) if total_aidc > 0 else 0

    aidc_module = data['modules'].get('aidc')
    pue = aidc_module.pue_params['pue'] if aidc_module else 1.0

    carbon_df = data.get('carbon_df')
    co2_avoided = 0
    if carbon_df is not None and hasattr(carbon_df, 'sum'):
        co2_avoided = carbon_df.get('avoided_tco2', pd.Series([0])).sum()

    sim_hours = max(len(pv_power), 1)
    co2_annual = co2_avoided * (8760 / sim_hours) if sim_hours > 0 else 0

    lcoe_krw_per_mwh = cached_base.get('lcoe_krw_per_mwh', 0)
    lcoe_usd_per_mwh = lcoe_krw_per_mwh / 1350 if lcoe_krw_per_mwh else 0  # ₩/MWh → $/MWh
    irr_pct = cached_base.get('irr_pct', 0)
    payback = cached_base.get('payback_years', 0)

    return {
        "lcoe_usd": lcoe_usd_per_mwh,
        "irr_pct": irr_pct,
        "co2_annual": co2_annual,
        "pue": pue,
        "self_sufficiency": self_sufficiency,
        "payback": payback,
        "total_pv_twh": total_pv / 1e6,    # MWh → TWh
        "total_aidc_twh": total_aidc / 1e6,
    }


def _render_summary_card(kpi_label, value, unit, explanation_md):
    """단일 KPI 카드 — 큰 수치 + 해설"""
    st.markdown(
        f"""
        <div style="background: white;
                    border: 1px solid #E1E8ED;
                    border-radius: 12px;
                    padding: 1.3rem 1.4rem;
                    margin-bottom: 0.9rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
          <div style="font-size: 0.75rem;
                      color: #6B7785;
                      font-weight: 600;
                      letter-spacing: 0.05em;
                      text-transform: uppercase;
                      margin-bottom: 0.4rem;">{kpi_label}</div>
          <div style="font-size: 2.1rem;
                      font-weight: 700;
                      color: #0F1A2A;
                      line-height: 1.1;
                      letter-spacing: -0.02em;">
            {value}<span style="font-size: 1rem;
                                font-weight: 500;
                                color: #6B7785;
                                margin-left: 0.3rem;">{unit}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style="margin-top: -0.6rem;
                    margin-bottom: 1.4rem;
                    padding: 0 0.3rem;
                    font-size: 0.88rem;
                    color: #4B5A6E;
                    line-height: 1.65;">
          {explanation_md}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_executive_summary(data, cached_base):
    """Executive Summary 페이지 렌더링

    Parameters
    ----------
    data : dict
        st.session_state.simulation_data
    cached_base : dict
        _cached_base_case() 결과 — LCOE/IRR/payback 등 경제성 KPI
    """
    kpis = _extract_kpis(data, cached_base)

    # === Section 1: 시뮬레이션 한 줄 요약 ===
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0B5CAB 0%, #094A8A 100%);
                    color: white;
                    padding: 1.6rem 1.8rem;
                    border-radius: 14px;
                    margin-bottom: 1.5rem;">
          <div style="font-size: 0.72rem;
                      letter-spacing: 0.1em;
                      text-transform: uppercase;
                      opacity: 0.85;
                      margin-bottom: 0.4rem;">
            Executive Summary · 시뮬레이션 결과 요약
          </div>
          <div style="font-size: 1.45rem;
                      font-weight: 600;
                      line-height: 1.35;
                      letter-spacing: -0.01em;">
            100 MW AI 데이터센터 신재생 마이크로그리드 · 25년 운영 시뮬레이션
          </div>
          <div style="font-size: 0.92rem;
                      opacity: 0.9;
                      margin-top: 0.5rem;
                      line-height: 1.5;">
            아래 6대 핵심 수치는 본 시뮬레이션의 가장 중요한 결과입니다.
            각 수치의 의미와 벤치마크 비교를 함께 제공합니다.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # === Section 2: 6대 핵심 수치 + 해설 ===

    col1, col2 = st.columns(2, gap="medium")

    # --- LCOE ---
    with col1:
        lcoe = kpis["lcoe_usd"]
        smp = BENCHMARKS["korea_smp_usd_per_mwh"]
        us_dc = BENCHMARKS["us_dc_lcoe_usd_per_mwh"]
        smp_diff = (lcoe - smp) / smp * 100
        smp_text = f"{abs(smp_diff):.1f}% 저렴" if smp_diff < 0 else f"{abs(smp_diff):.1f}% 비쌈"

        _render_summary_card(
            "LCOE (Levelized Cost of Energy)",
            f"{lcoe:.1f}",
            "$/MWh",
            f"""
            <strong>의미</strong>: 본 시스템에서 1 MWh의 전기를 생산하는 데 드는 평균 비용 (25년 가중평균).<br/>
            <strong>벤치마크</strong>: 한국 SMP 평균 약 {smp:.0f} $/MWh ({smp_text}) ·
            미국 데이터센터 PPA 평균 약 {us_dc:.0f} $/MWh.<br/>
            <strong>가정</strong>: NREL ATB 2024 CAPEX 학습곡선, 운영 25년, 한국 금융 조건 기준.
            """
        )

    # --- IRR ---
    with col2:
        irr = kpis["irr_pct"]
        dr = BENCHMARKS["discount_rate"]
        gap = irr - dr
        gap_text = (
            f"할인율보다 {gap:.1f}%p 높음" if gap > 0
            else f"할인율보다 {abs(gap):.1f}%p 낮음"
        )

        _render_summary_card(
            "IRR (내부수익률)",
            f"{irr:.1f}",
            "%",
            f"""
            <strong>의미</strong>: 25년 현금흐름 기준 투자수익률. 양수면 투자 매력 있음.<br/>
            <strong>벤치마크</strong>: 디폴트 할인율 {dr:.0f}% ({gap_text}).<br/>
            <strong>가정</strong>: BESS 차익거래 0.1 cycle/일 (보수적 운영). 정책 인센티브
            (K-ETS, REC, RE100) 미적용 baseline. 정책 시나리오 활성화 시 정책·전략 탭 참조.
            """
        )

    col3, col4 = st.columns(2, gap="medium")

    # --- CO₂ avoided ---
    with col3:
        co2 = kpis["co2_annual"]
        # 한국 1인당 연간 CO₂ ≈ 11.6 톤 (IEA 2022)
        equiv_people = co2 / 11.6
        equiv_cars = co2 / 4.6  # 승용차 1대 연간 4.6 톤 (EPA)

        _render_summary_card(
            "CO₂ 회피 (연간)",
            f"{co2:,.0f}",
            "tCO₂/년",
            f"""
            <strong>의미</strong>: 본 시스템이 한국 그리드 전력 대비 회피하는 연간 CO₂ 배출량.<br/>
            <strong>환산</strong>: 한국 국민 {equiv_people:,.0f}명의 연간 배출량,
            또는 승용차 {equiv_cars:,.0f}대의 연간 배출량.<br/>
            <strong>가정</strong>: 한국 그리드 배출계수 {BENCHMARKS['korea_grid_ef_kgco2_per_kwh']:.4f} kgCO₂/kWh
            (KEPCO 2023 발전량 가중평균).
            """
        )

    # --- PUE ---
    with col4:
        pue = kpis["pue"]
        global_avg = BENCHMARKS["global_dc_pue_avg"]
        target = BENCHMARKS["hyperscale_pue_target"]
        improvement = (global_avg - pue) / global_avg * 100

        _render_summary_card(
            "PUE (Power Usage Effectiveness)",
            f"{pue:.2f}",
            "",
            f"""
            <strong>의미</strong>: 데이터센터 전체 전력 ÷ IT 전력. 1.0에 가까울수록 효율적.<br/>
            <strong>벤치마크</strong>: 글로벌 평균 {global_avg:.2f} ({improvement:+.0f}% 개선) ·
            하이퍼스케일 목표 {target:.2f}.<br/>
            <strong>가정</strong>: 액침냉각(immersion cooling) 적용, ASHRAE Class A1 조건.
            한국 기후 + 액침냉각 조합으로 {pue:.2f} 달성 가능.
            """
        )

    col5, col6 = st.columns(2, gap="medium")

    # --- Self-sufficiency ---
    with col5:
        ss = kpis["self_sufficiency"]
        re100_target = BENCHMARKS["korean_re100_target_2030"]
        pv_twh = kpis["total_pv_twh"]
        aidc_twh = kpis["total_aidc_twh"]

        _render_summary_card(
            "자급률 (PV / AIDC)",
            f"{ss:.1f}",
            "%",
            f"""
            <strong>의미</strong>: 데이터센터 부하 대비 PV 발전량 비율 (시간 적분).
            100%면 연간 총량 기준 PV가 부하를 모두 커버 (시간 매칭은 별개).<br/>
            <strong>벤치마크</strong>: K-RE100 2030 목표 {re100_target}% · 본 시뮬레이션은
            {'이를 초과 달성' if ss > re100_target else '미달'}.<br/>
            <strong>주의</strong>: 시간별 매칭은 HESS + H₂가 담당. 본 수치는 연간 총량 기준.
            """
        )

    # --- Payback ---
    with col6:
        pb = kpis["payback"]

        if pb > 0 and pb < 25:
            assessment = f"운영 기간(25년) 내 회수 가능 — 잔여 약 {25-pb:.1f}년간 순익 발생"
        elif pb >= 25:
            assessment = "운영 기간(25년) 내 회수 어려움 — 정책 인센티브 또는 BESS 운영 강화 필요"
        else:
            assessment = "계산 불가 (음수 IRR)"

        _render_summary_card(
            "회수 기간 (Payback)",
            f"{pb:.1f}",
            "년",
            f"""
            <strong>의미</strong>: 누적 현금흐름이 초기 투자비를 회수하는 시점 (할인 적용).<br/>
            <strong>해석</strong>: {assessment}.<br/>
            <strong>가정</strong>: WACC 5%, CAPEX 학습곡선 적용, 잔존가치 미반영 (보수).
            """
        )

    # === Section 3: 더 자세히 보기 ===
    st.markdown("---")
    st.markdown(
        """
        ### 더 자세히 보고 싶으시면

        - **경제성 상세 (NPV, 토네이도, Monte Carlo)** → 💰 경제·분석 탭
        - **PV 4종 기술 비교** → ⚡ 코어 시스템 → ☀️ PV 발전 탭
        - **3-tier HESS 분담 시각화** → 🔋 에너지 저장 → 🔋 HESS 탭
        - **정책 시나리오 (K-ETS, REC, CBAM, RE100)** → 🏛️ 전략·정책 탭
        - **국제 비교 (한미중일유럽)** → 🌏 글로벌·데이터 → 🌏 국제 비교 탭
        - **한국 그리드 컨텍스트 (OpenGridWorks)** → 🌏 글로벌·데이터 → 🇰🇷 Korea Grid 탭
        """
    )

    st.caption(
        "본 Executive Summary는 시뮬레이션 결과의 단순 요약입니다. "
        "수치의 해석과 평가는 평가위원의 판단에 따릅니다. "
        "각 수치의 상세 산출 근거는 References 탭에서 확인 가능합니다."
    )
