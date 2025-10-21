# styles.py - 공통 스타일 정의 (Bright Industrial Design)

import streamlit as st

def load_common_styles():
    """모든 페이지에 적용할 공통 CSS 스타일 - Bright Industrial Design"""
    st.markdown("""
    <style>
        /* CSS 변수 정의 - config.toml 값과 연동 */
        :root {
            --primary-color: #4299e1;
            --bg-color: #ffffff;  
            --secondary-bg: #f7fafc;
            --text-color: #1a202c;
            --border-color: #e2e8f0;
            --shadow-color: rgba(0, 0, 0, 0.08);
        }
        
        /* 커스텀 레이아웃 */
        .main > div {
            padding: 1.5rem;
            background: transparent;
        }
        
        /* 사이드바 커스텀 스타일 - config.toml의 기본 색상 사용 */
        .css-1d391kg, .css-1lcbmhc, .css-17lntkn, .stSidebar, [data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
        }
        
        /* 사이드바 네비게이션 - config.toml primaryColor 기반 */
        .css-1544g2n, [data-testid="stSidebarNav"] {
            background: rgba(66, 153, 225, 0.08) !important;
            color: var(--text-color) !important;
            border-radius: 8px;
            margin: 4px 0;
            border: 1px solid rgba(66, 153, 225, 0.1);
        }
        
        .css-1544g2n:hover, [data-testid="stSidebarNav"]:hover {
            background: rgba(66, 153, 225, 0.15) !important;
        }
        
        /* Home 페이지 헤더 */
        .home-header {
            background: var(--bg-color);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 3rem;
            text-align: center;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 25px var(--shadow-color);
        }
        
        .home-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary-color) 0%, #3182ce 100%);
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
            background: var(--bg-color);
            padding: 2.5rem 2rem;
            border-radius: 20px;
            margin-bottom: 3rem;
            text-align: center;
            border: 1px solid var(--border-color);
            box-shadow: 0 8px 25px var(--shadow-color);
        }
        
        .page-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary-color) 0%, #3182ce 100%);
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
            background: var(--bg-color);
            padding: 2.5rem;
            border-radius: 20px;
            border: 1px solid var(--border-color);
            box-shadow: 0 8px 25px var(--shadow-color);
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
            background: linear-gradient(90deg, var(--primary-color) 0%, #4fd1c7 100%);
        }
        
        .industrial-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
        }
        
        .card-ml::before {
            background: linear-gradient(90deg, var(--primary-color) 0%, #63b3ed 100%);
        }
        
        .card-dl::before {
            background: linear-gradient(90deg, #4fd1c7 0%, #81e6d9 100%);
        }
        
        .card-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-color);
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
            background: rgba(66, 153, 225, 0.1);
            color: #3182ce;
            border-color: rgba(66, 153, 225, 0.3);
        }
        
        .type-dl {
            background: rgba(79, 209, 199, 0.1);
            color: #2c7a7b;
            border-color: rgba(79, 209, 199, 0.3);
        }
        
        .card-description {
            color: #4a5568;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        
        .performance-badge {
            color: white;
            padding: 0.8rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .card-ml .performance-badge {
            background: linear-gradient(135deg, var(--primary-color) 0%, #3182ce 100%);
            box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3);
        }
        
        .card-dl .performance-badge {
            background: linear-gradient(135deg, #4fd1c7 0%, #319795 100%);
            box-shadow: 0 4px 15px rgba(79, 209, 199, 0.3);
        }
        
        .tech-stack {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .tech-badge {
            padding: 0.4rem 0.8rem;
            border-radius: 15px;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .card-ml .tech-badge {
            background: rgba(66, 153, 225, 0.1);
            color: #3182ce;
            border: 1px solid rgba(66, 153, 225, 0.3);
        }
        
        .card-ml .tech-badge:hover {
            background: rgba(66, 153, 225, 0.2);
            border-color: var(--primary-color);
        }
        
        .card-dl .tech-badge {
            background: rgba(79, 209, 199, 0.1);
            color: #319795;
            border: 1px solid rgba(79, 209, 199, 0.3);
        }
        
        .card-dl .tech-badge:hover {
            background: rgba(79, 209, 199, 0.2);
            border-color: #4fd1c7;
        }
        
        /* 섹션 컨테이너 */
        .section-container {
            background: var(--bg-color);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
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
            background: var(--bg-color);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            border: 1px solid var(--border-color);
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
            background: linear-gradient(135deg, var(--primary-color) 0%, #3182ce 100%);
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
            background: var(--bg-color);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
        }
        
        /* Streamlit 요소 커스터마이징 */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color) 0%, #3182ce 100%);
            color: var(--bg-color);
            border: none;
            border-radius: 10px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(66, 153, 225, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(66, 153, 225, 0.4);
            background: linear-gradient(135deg, #3182ce 0%, #2c5282 100%);
        }
        
        .stSelectbox > div > div {
            background: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 10px;
        }
        
        .stTextInput > div > div > input {
            background: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 10px;
        }
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: var(--secondary-bg);
            border-radius: 12px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: var(--bg-color);
            color: #4a5568;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: 1px solid var(--border-color);
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #edf2f7;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary-color) 0%, #3182ce 100%);
            color: var(--bg-color);
            border-color: transparent;
        }
        
        /* 메트릭 위젯 */
        .stMetric {
            background: var(--bg-color);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
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
            background: rgba(129, 230, 217, 0.1);
            color: #2c7a7b;
            border: 1px solid rgba(129, 230, 217, 0.3);
            border-radius: 8px;
        }
        
        /* Streamlit 기본 색상 덮어쓰기 - 차가운 톤으로 */
        /* 슬라이더 */
        /* 슬라이더 조종점*/
        .stSlider > div > div > div > div {
            background-color: #4299e1 !important;
        }

        /* 슬라이더 조종점 위 숫자 */
        .stSlider > div > div > div > div > div {
            color: #3182ce !important;
        }
        
        /* 진행 표시줄 */
        .stProgress > div > div > div > div {
            background-color: #4299e1 !important;
        }
        
        /* 체크박스 */
        .stCheckbox > label > div[data-checked="true"] {
            background-color: #4299e1 !important;
            border-color: #3182ce !important;
        }
        
        /* 라디오 버튼 */
        .stRadio > div > label > div[data-checked="true"] {
            background-color: #4299e1 !important;
        }
        
        .stRadio > div > label > div[data-checked="true"]::before {
            background-color: #3182ce !important;
        }
        
        /* 선택 박스 활성화 상태 */
        .stSelectbox > div > div[data-selected="true"] {
            background-color: rgba(66, 153, 225, 0.1) !important;
            border-color: #4299e1 !important;
        }
        
        /* 스핀 박스 */
        .stNumberInput > div > div > input:focus {
            border-color: #4299e1 !important;
            box-shadow: 0 0 0 1px #4299e1 !important;
        }
        
        /* 텍스트 입력 포커스 */
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #4299e1 !important;
            box-shadow: 0 0 0 1px #4299e1 !important;
        }
        
        /* 멀티셀렉트 */
        .stMultiSelect > div > div > div {
            border-color: #4299e1 !important;
        }
        
        .stMultiSelect span[data-selected="true"] {
            background-color: #4299e1 !important;
            color: white !important;
        }
        
        /* 파일 업로더 */
        .stFileUploader > div > div > div > div {
            border-color: #4299e1 !important;
        }
        
        .stFileUploader > div > div > div > div:hover {
            background-color: rgba(66, 153, 225, 0.05) !important;
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

# 색상 상수 - config.toml과 연동된 Cool Tone Palette
COLORS = {
    'primary': '#4299e1',    # config.toml primaryColor와 동일
    'secondary': '#3182ce',  # 진한 파랑
    'ml': '#63b3ed',         # 하늘색
    'dl': '#4fd1c7',         # 청록색
    'success': '#68d391',    # 민트 그린
    'warning': '#81e6d9',    # 연한 청록
    'error': '#9f7aea'       # 보라색
}

# config.toml primaryColor 기반 차트 색상
CHART_COLORS = ['#4299e1', '#3182ce', '#63b3ed', '#4fd1c7', '#68d391', '#81e6d9']