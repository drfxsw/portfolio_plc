# styles.py - 공통 스타일 정의 (Bright Industrial Design)

import streamlit as st

def load_common_styles():
    """모든 페이지에 적용할 공통 CSS 스타일 - Bright Industrial Design"""
    st.markdown("""
    <style>
        /* 전체 배경 - Bright Industrial */
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            color: #1a202c;
        }
        
        .main > div {
            padding: 1.5rem;
            background: transparent;
        }
        
        /* 기본 텍스트 색상 */
        .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 {
            color: #1a202c !important;
        }
        
        /* 사이드바 - Clean Bright */
        .css-1d391kg, .css-1lcbmhc, .css-17lntkn, .stSidebar, [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%) !important;
            border-right: 1px solid #e2e8f0;
        }
        
        /* 사이드바 텍스트 */
        .stSidebar .stMarkdown, .stSidebar .stText, .stSidebar p, .stSidebar h1, .stSidebar h2, .stSidebar h3,
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stText, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label, .stSidebar label {
            color: #2d3748 !important;
        }
        
        /* 사이드바 네비게이션 */
        .css-1544g2n, [data-testid="stSidebarNav"] {
            background: rgba(102, 126, 234, 0.08) !important;
            color: #2d3748 !important;
            border-radius: 8px;
            margin: 4px 0;
            border: 1px solid rgba(102, 126, 234, 0.1);
        }
        
        .css-1544g2n:hover, [data-testid="stSidebarNav"]:hover {
            background: rgba(102, 126, 234, 0.15) !important;
        }
        
        /* Home 페이지 헤더 */
        .home-header {
            background: white;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 3rem;
            text-align: center;
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        }
        
        .home-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -2px;
        }
        
        .home-subtitle {
            font-size: 1.3rem;
            color: #4a5568;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        
        /* 페이지 헤더 */
        .page-header {
            background: white;
            padding: 2.5rem 2rem;
            border-radius: 20px;
            margin-bottom: 3rem;
            text-align: center;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        }
        
        .page-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        
        .page-subtitle {
            font-size: 1.2rem;
            color: #4a5568;
            font-weight: 500;
            margin-bottom: 1rem;
        }
        
        /* 산업용 프로젝트 카드 */
        .industrial-card {
            background: white;
            padding: 2.5rem;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        
        .industrial-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 100%);
        }
        
        .industrial-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
        }
        
        .card-ml::before {
            background: linear-gradient(90deg, #ff6b6b 0%, #ff8e8e 100%);
        }
        
        .card-dl::before {
            background: linear-gradient(90deg, #4ecdc4 0%, #6ed3d0 100%);
        }
        
        .card-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1a202c;
            margin-bottom: 0.8rem;
        }
        
        .card-type {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border: 1px solid;
        }
        
        .type-ml {
            background: rgba(255, 107, 107, 0.1);
            color: #c53030;
            border-color: rgba(255, 107, 107, 0.3);
        }
        
        .type-dl {
            background: rgba(78, 205, 196, 0.1);
            color: #2c7a7b;
            border-color: rgba(78, 205, 196, 0.3);
        }
        
        .card-description {
            color: #4a5568;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        
        .performance-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.8rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .tech-stack {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .tech-badge {
            background: #f7fafc;
            color: #2d3748;
            padding: 0.4rem 0.8rem;
            border-radius: 15px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid #e2e8f0;
            transition: all 0.2s ease;
        }
        
        .tech-badge:hover {
            background: #edf2f7;
            border-color: #cbd5e0;
        }
        
        /* 섹션 컨테이너 */
        .section-container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        }
        
        .section-title {
            font-size: 1.6rem;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        
        /* 메트릭 카드 */
        .metric-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #4a5568;
            font-weight: 500;
        }
        
        /* 차트 컨테이너 */
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        }
        
        /* Streamlit 요소 커스터마이징 */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .stSelectbox > div > div {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }
        
        .stTextInput > div > div > input {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #f7fafc;
            border-radius: 12px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: white;
            color: #4a5568;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: 1px solid #e2e8f0;
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #edf2f7;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
        }
        
        /* 메트릭 위젯 */
        .stMetric {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }
        
        /* 소형 메트릭 스타일 - 더 강력한 선택자 */
        .small-metrics [data-testid="metric-container"] {
            background: rgba(102, 126, 234, 0.05) !important;
            border-radius: 8px !important;
            padding: 0.3rem !important;
            margin: 0.1rem 0 !important;
            border: 1px solid rgba(102, 126, 234, 0.1) !important;
            transform: scale(0.8) !important;
            transform-origin: left top !important;
        }
        
        .small-metrics [data-testid="metric-container"] div {
            font-size: 0.7rem !important;
        }
        
        .small-metrics [data-testid="metric-container"] [data-testid="metric-value"] {
            font-size: 1rem !important;
            font-weight: 600 !important;
        }
        
        .small-metrics [data-testid="metric-container"] [data-testid="metric-label"] {
            font-size: 0.6rem !important;
        }
        
        /* 성공/에러/경고 메시지 */
        .stSuccess {
            background: rgba(72, 187, 120, 0.1);
            color: #2f855a;
            border: 1px solid rgba(72, 187, 120, 0.3);
            border-radius: 8px;
        }
        
        .stError {
            background: rgba(229, 62, 62, 0.1);
            color: #c53030;
            border: 1px solid rgba(229, 62, 62, 0.3);
            border-radius: 8px;
        }
        
        .stWarning {
            background: rgba(237, 137, 54, 0.1);
            color: #dd6b20;
            border: 1px solid rgba(237, 137, 54, 0.3);
            border-radius: 8px;
        }
        
        /* 반응형 */
        @media (max-width: 768px) {
            .home-title {
                font-size: 2.5rem;
            }
            .page-title {
                font-size: 2rem;
            }
            .industrial-card {
                padding: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def create_page_header(title, subtitle=""):
    """페이지 헤더 생성"""
    st.markdown(f"""
    <div class="page-header">
        <h1 class="page-title">{title}</h1>
        {f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def create_home_header():
    """Home 페이지 전용 헤더"""
    st.markdown("""
    <div class="home-header">
        <h1 class="home-title">MANUFACTURING AI</h1>
        <p class="home-subtitle">Advanced Predictive Analytics for Smart Manufacturing</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_cards(metrics_data):
    """메트릭 카드들 생성"""
    if len(metrics_data) == 3:
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
    elif len(metrics_data) == 4:
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
    else:
        cols = st.columns(len(metrics_data))
    
    for i, (value, label) in enumerate(metrics_data):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def create_section_container(title, content):
    """섹션 컨테이너 생성"""
    st.markdown(f"""
    <div class="section-container">
        <h3 class="section-title">{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# 색상 상수
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'ml': '#ff6b6b',
    'dl': '#4ecdc4',
    'success': '#48bb78',
    'warning': '#ed8936',
    'error': '#e53e3e'
}

CHART_COLORS = ['#667eea', '#764ba2', '#ff6b6b', '#4ecdc4', '#48bb78', '#ed8936']