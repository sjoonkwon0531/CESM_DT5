"""
CEMS Digital Twin v5 — Brand Shell
====================================
Streamlit UI를 Salesforce/Tableau 스타일 클린 SaaS 느낌으로 만드는 헬퍼 모듈.

사용법:
    from branding import (
        inject_brand_css,
        render_app_header,
        render_kpi_cards,
        render_landing_page,
        render_loading_progress,
    )

    # app.py 최상단 (st.set_page_config 직후):
    inject_brand_css()

    # 기존 st.title(...) 대신:
    render_app_header()

    # 기존 _display_top_kpi(data) 대신:
    render_kpi_cards(kpi_dict)

    # 시뮬레이션 데이터 없을 때:
    render_landing_page()

설계 원칙:
- 기존 display_*() 함수들은 그대로 유지 (호환성)
- CSS는 외곽 셸만 변경 (헤더, 카드, 사이드바, 버튼 등)
- 차트/표/모듈 내부 디테일은 건드리지 않음
"""
import streamlit as st


# =============================================================================
# 브랜드 디자인 토큰
# =============================================================================
BRAND = {
    # Primary palette (Salesforce blue + slate)
    "primary": "#0B5CAB",       # 짙은 블루 — 헤더, 버튼, 액센트
    "primary_hover": "#094a8a",
    "primary_light": "#E8F1FB", # 매우 연한 블루 — 호버, 선택 상태
    "accent": "#00A1B0",         # 청록 — 보조 강조 (PV/에너지)

    # Semantic colors
    "success": "#2E7D32",
    "warning": "#ED6C02",
    "danger":  "#C62828",
    "info":    "#0288D1",

    # Neutral scale
    "bg_app":      "#F7F9FC",   # 앱 배경
    "bg_card":     "#FFFFFF",   # 카드 배경
    "bg_subtle":   "#F1F4F8",   # 보조 배경
    "border":      "#E1E8ED",   # 카드/구분선
    "border_strong":"#CBD5DD",
    "text_primary":"#0F1A2A",   # 본문
    "text_secondary":"#4B5A6E",
    "text_muted":  "#8A98AB",

    # Typography
    "font_sans": '"Inter", "Pretendard", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',

    # Brand identity
    "product_name": "Intelligent Energy Solution",
    "product_version": "v5.1",
    "product_subtitle": "by SKKU NRL — Renewable Microgrid Digital Twin for AI Data Centers",
    "company": "Sungkyunkwan University · National Research Laboratory",
}


# =============================================================================
# 1. CSS 주입 — 한 번만 호출 (set_page_config 직후)
# =============================================================================
def inject_brand_css():
    """전역 CSS 주입. Streamlit 기본 스타일을 SaaS 풍으로 덮어씀."""
    css = f"""
    <style>
    /* === 폰트 === */
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: {BRAND["font_sans"]} !important;
        color: {BRAND["text_primary"]};
    }}

    /* === 앱 배경 === */
    .stApp {{
        background: {BRAND["bg_app"]};
    }}

    /* === Streamlit 기본 헤더 숨김 === */
    header[data-testid="stHeader"] {{
        background: transparent;
        height: 0;
    }}
    #MainMenu, footer {{visibility: hidden;}}

    /* === 메인 컨테이너 패딩 조정 === */
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }}

    /* === 사이드바 === */
    section[data-testid="stSidebar"] {{
        background: {BRAND["bg_card"]} !important;
        border-right: 1px solid {BRAND["border"]};
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 1rem;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: {BRAND["text_muted"]} !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid {BRAND["border"]};
    }}

    /* === 본문 헤더 === */
    h1, h2, h3 {{
        color: {BRAND["text_primary"]} !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }}
    h1 {{ font-size: 1.6rem !important; }}
    h2 {{ font-size: 1.2rem !important; margin-top: 1.5rem !important; }}
    h3 {{ font-size: 1.0rem !important; }}

    /* === 카테고리 라디오 (탭처럼 보이게) === */
    div[role="radiogroup"] > label {{
        background: {BRAND["bg_card"]};
        border: 1px solid {BRAND["border"]};
        border-radius: 8px;
        padding: 0.5rem 1rem !important;
        margin-right: 0.4rem !important;
        margin-bottom: 0.3rem !important;
        cursor: pointer;
        transition: all 0.15s ease;
    }}
    div[role="radiogroup"] > label:hover {{
        border-color: {BRAND["primary"]};
        background: {BRAND["primary_light"]};
    }}
    div[role="radiogroup"] > label[data-checked="true"],
    div[role="radiogroup"] > label > div:first-child[aria-checked="true"] {{
        background: {BRAND["primary"]};
        color: white;
    }}

    /* === 탭 (하위 탭바) === */
    div[data-testid="stTabs"] > div[role="tablist"] {{
        gap: 0 !important;
        border-bottom: 1px solid {BRAND["border"]};
        padding-bottom: 0 !important;
    }}
    div[data-testid="stTabs"] > div[role="tablist"] button {{
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: {BRAND["text_secondary"]} !important;
        margin-right: 0.2rem !important;
        transition: all 0.15s ease;
    }}
    div[data-testid="stTabs"] > div[role="tablist"] button:hover {{
        color: {BRAND["primary"]} !important;
        background: {BRAND["primary_light"]} !important;
    }}
    div[data-testid="stTabs"] > div[role="tablist"] button[aria-selected="true"] {{
        color: {BRAND["primary"]} !important;
        border-bottom-color: {BRAND["primary"]} !important;
        font-weight: 600 !important;
    }}

    /* === 버튼 === */
    div.stButton > button {{
        background: {BRAND["primary"]};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    div.stButton > button:hover {{
        background: {BRAND["primary_hover"]};
        box-shadow: 0 2px 6px rgba(11,92,171,0.25);
        transform: translateY(-1px);
    }}
    div.stButton > button:active {{
        transform: translateY(0);
    }}

    /* === 입력 위젯 === */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextInput > div > div > input,
    .stNumberInput input {{
        border-radius: 6px !important;
        border: 1px solid {BRAND["border_strong"]} !important;
        font-size: 0.85rem !important;
    }}

    /* === Slider 트랙 === */
    div[data-baseweb="slider"] [role="slider"] {{
        background: {BRAND["primary"]} !important;
    }}

    /* === 표 === */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BRAND["border"]};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* === 커스텀 컴포넌트 === */

    /* Brand 헤더 바 */
    .brand-header {{
        background: linear-gradient(90deg, #0B5CAB 0%, #094a8a 100%);
        margin: -1rem -1rem 1.5rem -1rem;
        padding: 1.0rem 2rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255,255,255,0.15);
    }}
    .brand-header-left {{
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }}
    .brand-logo {{
        width: 36px;
        height: 36px;
        background: white;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .brand-title {{
        font-size: 1.1rem;
        font-weight: 600;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }}
    .brand-subtitle {{
        font-size: 0.78rem;
        opacity: 0.78;
        margin-top: 1px;
    }}
    .brand-header-right {{
        display: flex;
        align-items: center;
        gap: 1.2rem;
        font-size: 0.8rem;
    }}
    .brand-version-pill {{
        background: rgba(255,255,255,0.18);
        padding: 0.22rem 0.6rem;
        border-radius: 99px;
        font-weight: 500;
        font-size: 0.72rem;
    }}
    .brand-status-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4ADE80;
        margin-right: 0.4rem;
        box-shadow: 0 0 0 3px rgba(74,222,128,0.25);
    }}

    /* KPI 카드 */
    .kpi-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }}
    .kpi-card {{
        background: {BRAND["bg_card"]};
        border: 1px solid {BRAND["border"]};
        border-radius: 10px;
        padding: 0.95rem 1.1rem;
        transition: all 0.15s ease;
    }}
    .kpi-card:hover {{
        border-color: {BRAND["primary"]};
        box-shadow: 0 4px 12px rgba(11,92,171,0.08);
    }}
    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 500;
        color: {BRAND["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }}
    .kpi-value {{
        font-size: 1.55rem;
        font-weight: 600;
        color: {BRAND["text_primary"]};
        line-height: 1.1;
        letter-spacing: -0.01em;
    }}
    .kpi-unit {{
        font-size: 0.85rem;
        color: {BRAND["text_secondary"]};
        font-weight: 400;
        margin-left: 0.2rem;
    }}
    .kpi-delta {{
        font-size: 0.75rem;
        margin-top: 0.45rem;
        font-weight: 500;
    }}
    .kpi-delta.positive {{ color: {BRAND["success"]}; }}
    .kpi-delta.negative {{ color: {BRAND["danger"]}; }}
    .kpi-delta.neutral  {{ color: {BRAND["text_muted"]}; }}

    /* 랜딩 페이지 */
    .landing-hero {{
        background: linear-gradient(135deg, #0B5CAB 0%, #00A1B0 100%);
        color: white;
        padding: 3rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }}
    .landing-hero::before {{
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    }}
    .landing-hero-eyebrow {{
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.85;
        margin-bottom: 0.8rem;
    }}
    .landing-hero-title {{
        font-size: 2.3rem;
        font-weight: 700;
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin-bottom: 0.7rem;
    }}
    .landing-hero-desc {{
        font-size: 1rem;
        opacity: 0.92;
        max-width: 720px;
        line-height: 1.55;
        margin-bottom: 1.5rem;
    }}
    .landing-hero-stats {{
        display: flex;
        gap: 2.2rem;
        flex-wrap: wrap;
        margin-top: 1.5rem;
        position: relative;
    }}
    .landing-hero-stat-num {{
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1;
    }}
    .landing-hero-stat-label {{
        font-size: 0.78rem;
        opacity: 0.82;
        margin-top: 0.25rem;
    }}

    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 0.9rem;
        margin-bottom: 1.5rem;
    }}
    .feature-card {{
        background: {BRAND["bg_card"]};
        border: 1px solid {BRAND["border"]};
        border-radius: 10px;
        padding: 1.2rem 1.3rem;
        transition: all 0.15s ease;
    }}
    .feature-card:hover {{
        border-color: {BRAND["primary"]};
        box-shadow: 0 4px 14px rgba(11,92,171,0.08);
        transform: translateY(-2px);
    }}
    .feature-icon-wrap {{
        width: 38px;
        height: 38px;
        background: {BRAND["primary_light"]};
        color: {BRAND["primary"]};
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.85rem;
    }}
    .feature-title {{
        font-size: 0.97rem;
        font-weight: 600;
        color: {BRAND["text_primary"]};
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
    }}
    .feature-desc {{
        font-size: 0.83rem;
        color: {BRAND["text_secondary"]};
        line-height: 1.5;
    }}

    /* 시작 안내 박스 */
    .start-hint {{
        background: {BRAND["primary_light"]};
        border: 1px solid {BRAND["primary"]};
        border-left: 4px solid {BRAND["primary"]};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.88rem;
        color: {BRAND["text_primary"]};
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }}
    .start-hint-icon {{
        flex-shrink: 0;
        color: {BRAND["primary"]};
    }}

    /* 진행률 카드 */
    .progress-card {{
        background: {BRAND["bg_card"]};
        border: 1px solid {BRAND["border"]};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin: 1.5rem 0;
    }}
    .progress-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {BRAND["text_primary"]};
        margin-bottom: 0.3rem;
    }}
    .progress-step {{
        font-size: 0.82rem;
        color: {BRAND["text_secondary"]};
        margin-bottom: 0.7rem;
    }}
    .progress-bar-wrap {{
        background: {BRAND["bg_subtle"]};
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
    }}
    .progress-bar-fill {{
        background: linear-gradient(90deg, {BRAND["primary"]}, {BRAND["accent"]});
        height: 100%;
        transition: width 0.3s ease;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# 2. 앱 헤더 (페이지 최상단)
# =============================================================================
def render_app_header():
    """SaaS 풍 헤더 바. 좌측: 로고+제품명, 우측: 버전+상태."""
    # SVG 로고: 단순화된 마이크로그리드 노드 (3노드 + 연결)
    logo_svg = f"""
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="6" cy="6" r="2.6" fill="{BRAND['primary']}"/>
      <circle cx="18" cy="6" r="2.6" fill="{BRAND['accent']}"/>
      <circle cx="12" cy="18" r="2.6" fill="{BRAND['primary']}"/>
      <line x1="6" y1="6" x2="18" y2="6" stroke="{BRAND['primary']}" stroke-width="1.4"/>
      <line x1="6" y1="6" x2="12" y2="18" stroke="{BRAND['primary']}" stroke-width="1.4"/>
      <line x1="18" y1="6" x2="12" y2="18" stroke="{BRAND['primary']}" stroke-width="1.4"/>
    </svg>
    """

    html = f"""
    <div class="brand-header">
        <div class="brand-header-left">
            <div class="brand-logo">{logo_svg}</div>
            <div>
                <div class="brand-title">{BRAND['product_name']}</div>
                <div class="brand-subtitle">{BRAND['product_subtitle']}</div>
            </div>
        </div>
        <div class="brand-header-right">
            <span><span class="brand-status-dot"></span>System operational</span>
            <span class="brand-version-pill">{BRAND['product_version']}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 3. KPI 카드 그리드 — st.metric 대체
# =============================================================================
def render_kpi_cards(kpis):
    """
    KPI 카드 가로 그리드.

    kpis: list of dict, each with:
        - label: str (예: "LCOE")
        - value: str/number (예: "101.1" or 101.1)
        - unit: str (예: "$/MWh") — optional
        - delta: str (예: "+2.3%") — optional
        - delta_dir: "positive" | "negative" | "neutral" — optional

    예시:
        render_kpi_cards([
            {"label": "LCOE", "value": "101.1", "unit": "$/MWh", "delta": "vs. baseline", "delta_dir": "neutral"},
            {"label": "IRR", "value": "4.4", "unit": "%", "delta": "+2.3 vs. base", "delta_dir": "positive"},
            ...
        ])
    """
    cards_html = ""
    for k in kpis:
        unit = f"<span class='kpi-unit'>{k.get('unit','')}</span>" if k.get('unit') else ""
        delta_dir = k.get("delta_dir", "neutral")
        delta = (
            f"<div class='kpi-delta {delta_dir}'>{k['delta']}</div>"
            if k.get("delta") else ""
        )
        cards_html += f"""
        <div class="kpi-card">
            <div class="kpi-label">{k['label']}</div>
            <div class="kpi-value">{k['value']}{unit}</div>
            {delta}
        </div>
        """
    st.markdown(f"<div class='kpi-row'>{cards_html}</div>", unsafe_allow_html=True)


# =============================================================================
# 4. 랜딩 페이지 — 시뮬레이션 데이터 없을 때
# =============================================================================
def render_landing_page():
    """시뮬레이션 실행 전 표시되는 풀 페이지 랜딩."""

    # === Hero 섹션 ===
    hero_html = f"""
    <div class="landing-hero">
        <div class="landing-hero-eyebrow">Intelligent Energy Solution · SKKU NRL</div>
        <div class="landing-hero-title">
            Renewable microgrid digital twin<br/>
            for 100MW AI data centers
        </div>
        <div class="landing-hero-desc">
            13개 통합 모듈로 PV 발전, 6-Layer HESS 저장, H₂ 변환,
            AI-EMS 디스패치, 정책 시나리오, 경제성 평가를 동시에 시뮬레이션합니다.
            μs(전력전자) ~ 1년(경제성) 다중 시간 스케일 통합.
        </div>
        <div class="landing-hero-stats">
            <div>
                <div class="landing-hero-stat-num">100 MW</div>
                <div class="landing-hero-stat-label">AI Data Center Capacity</div>
            </div>
            <div>
                <div class="landing-hero-stat-num">9.75 GWh</div>
                <div class="landing-hero-stat-label">6-Layer HESS Storage</div>
            </div>
            <div>
                <div class="landing-hero-stat-num">68.7%</div>
                <div class="landing-hero-stat-label">Infinite-junction PV η<sub>STC</sub></div>
            </div>
            <div>
                <div class="landing-hero-stat-num">13 + 3</div>
                <div class="landing-hero-stat-label">Integrated Modules</div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # === Feature cards 섹션 ===
    st.markdown("##### Capabilities")

    features = [
        {
            "icon": "M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5",
            "title": "Physics-grade simulation",
            "desc": "NOCT 셀 온도, IEC 61215, Nernst 전압, Faraday 효율 등 표준 물리식 정확 구현. PV 4종, HESS 6레이어 통합."
        },
        {
            "icon": "M3 3v18h18M7 14l4-4 4 4 5-5",
            "title": "AI-EMS dispatch",
            "desc": "3-Tier 제어 (1ms 실시간 + 15min 예측 + 1hr 전략). LP 기반 최적화, Grid Flexibility 40% 부하감축."
        },
        {
            "icon": "M21 12c0 1-2 8-9 8s-9-7-9-8 2-8 9-8 9 7 9 8z M12 15a3 3 0 100-6 3 3 0 000 6z",
            "title": "Policy & economics",
            "desc": "K-ETS, REC, CBAM, RE100, 2026 Ratepayer Protection 시나리오. Monte Carlo 10,000회 + 토네이도 민감도."
        },
        {
            "icon": "M3 12h4l3-9 4 18 3-9h4",
            "title": "Stress & resilience",
            "desc": "3일 흐림, 폭염+피크부하, GPU 대량장애 시나리오. 99.99% 가용성 기준 BYOG 시뮬레이션."
        },
        {
            "icon": "M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12z M12 9a3 3 0 100 6 3 3 0 000-6z",
            "title": "Global benchmarks",
            "desc": "KR/US/CN/JP/EU 5개국 비교 (NREL ATB, IRENA, Fraunhofer, METI, SERC). BNEF 2025 LCOH 통합."
        },
        {
            "icon": "M9 12l2 2 4-4 M21 12c0 5-4 9-9 9s-9-4-9-9 4-9 9-9 9 4 9 9z",
            "title": "Investment dashboard",
            "desc": "NPV, IRR, LCOE, Payback 통합. Go/No-Go 매트릭스 + What-if 분석 + 보조금 민감도."
        },
    ]

    cards = ""
    for f in features:
        icon_svg = f"""
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="{f['icon']}"/>
        </svg>
        """
        cards += f"""
        <div class="feature-card">
            <div class="feature-icon-wrap">{icon_svg}</div>
            <div class="feature-title">{f['title']}</div>
            <div class="feature-desc">{f['desc']}</div>
        </div>
        """
    st.markdown(f"<div class='feature-grid'>{cards}</div>", unsafe_allow_html=True)

    # === Start hint ===
    hint_svg = """
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 5l7 7-7 7"/>
    </svg>
    """
    st.markdown(f"""
    <div class="start-hint">
        <span class="start-hint-icon">{hint_svg}</span>
        <span><strong>시작하기:</strong> 좌측 사이드바에서 시연 시나리오(A~E)를 선택하거나,
        파라미터를 직접 설정하신 후 <em>Run simulation</em> 버튼을 누르세요.</span>
    </div>
    """, unsafe_allow_html=True)

    # === References footer ===
    st.markdown("")
    st.markdown(f"""
    <div style='margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid {BRAND["border"]};
                font-size: 0.78rem; color: {BRAND["text_muted"]}; line-height: 1.6;'>
        <strong style="color: {BRAND['text_secondary']};">데이터 출처</strong> &nbsp;·&nbsp;
        NREL ATB 2024 &nbsp;·&nbsp; IRENA RENEWCOST 2024 &nbsp;·&nbsp; BloombergNEF 2025 &nbsp;·&nbsp;
        Fraunhofer ISE &nbsp;·&nbsp; METI &nbsp;·&nbsp; KEPCO &nbsp;·&nbsp; KPX &nbsp;·&nbsp;
        Nature Reviews Materials (2025)<br/>
        <span style="opacity: 0.7;">{BRAND['company']}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 5. 진행률 표시 — st.spinner 대체
# =============================================================================
def render_loading_progress(steps, current_step, label="시뮬레이션 실행 중"):
    """
    단계별 진행률 표시.

    steps: list of str — 전체 단계 라벨
    current_step: int — 현재 진행 중인 단계 인덱스 (0-based)
    label: str — 상단 제목

    예시:
        steps = ["기상 데이터 로드", "PV 시뮬레이션", "HESS 운영", "경제성 분석"]
        for i, step in enumerate(steps):
            placeholder = render_loading_progress(steps, i, "시뮬레이션 실행 중")
            ... 작업 수행 ...

    Returns:
        placeholder (st.empty) — 같은 위치에 다음 단계 표시 가능
    """
    progress_pct = int((current_step + 1) / len(steps) * 100) if steps else 0
    current_label = steps[current_step] if 0 <= current_step < len(steps) else ""

    html = f"""
    <div class="progress-card">
        <div class="progress-title">{label}</div>
        <div class="progress-step">
            Step {current_step + 1} of {len(steps)} &nbsp;·&nbsp; {current_label}
        </div>
        <div class="progress-bar-wrap">
            <div class="progress-bar-fill" style="width: {progress_pct}%;"></div>
        </div>
    </div>
    """
    return html


# =============================================================================
# 6. 카테고리 네비게이션 — st.radio 대체 옵션
# =============================================================================
def render_section_title(title, subtitle=None):
    """본문 섹션 타이틀 (이모지 없는 깔끔한 형태)."""
    st.markdown(f"""
    <div style='margin-top: 1rem; margin-bottom: 0.8rem;'>
        <div style='font-size: 1.05rem; font-weight: 600; color: {BRAND["text_primary"]};
                    letter-spacing: -0.01em;'>{title}</div>
        {f'<div style="font-size: 0.82rem; color: {BRAND["text_muted"]}; margin-top: 0.15rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 7. 사이드바 헤더 (옵션)
# =============================================================================
def render_sidebar_brand():
    """사이드바 상단에 브랜드 미니 영역 추가."""
    st.sidebar.markdown(f"""
    <div style='padding: 0.5rem 0 1rem 0; margin-bottom: 0.5rem;
                border-bottom: 1px solid {BRAND["border"]};'>
        <div style='font-size: 0.95rem; font-weight: 600; color: {BRAND["text_primary"]};'>
            Configuration
        </div>
        <div style='font-size: 0.75rem; color: {BRAND["text_muted"]}; margin-top: 0.1rem;'>
            Set parameters and run simulation
        </div>
    </div>
    """, unsafe_allow_html=True)
