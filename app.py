import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 页面配置 - 使用 centered 布局防止侧边栏被强制隐藏
st.set_page_config(
    page_title="时间序列平稳性分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",  # 默认展开
    menu_items={
        'Get Help': 'https://github.com/streamlit',
        'Report a bug': 'https://github.com/streamlit',
        'About': '# 时间序列平稳性分析工具'
    }
)

# 注入CSS - 修复侧边栏可见性问题
st.markdown("""
<style>
    /* 隐藏默认的footer和部署按钮 */
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* 确保侧边栏按钮始终可见 */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
        display: flex !important;
        z-index: 999999 !important;
    }
    
    /* 美化侧边栏按钮 */
    [data-testid="collapsedControl"] {
        background-color: rgba(31, 119, 180, 0.1) !important;
        border-radius: 50% !important;
        padding: 8px !important;
        margin: 10px !important;
        border: 2px solid #1f77b4 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(31, 119, 180, 0.3) !important;
        transform: scale(1.1) !important;
    }
    
    [data-testid="collapsedControl"] svg {
        color: #1f77b4 !important;
        width: 24px !important;
        height: 24px !important;
    }
    
    /* 添加一个始终可见的侧边栏切换按钮作为备用 */
    .sidebar-toggle-btn {
        position: fixed !important;
        top: 15px !important;
        left: 15px !important;
        z-index: 9999999 !important;
        background-color: #1f77b4 !important;
        color: white !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 24px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
        cursor: pointer !important;
        border: 2px solid white !important;
        opacity: 0.9 !important;
        transition: all 0.3s ease !important;
    }
    
    .sidebar-toggle-btn:hover {
        opacity: 1 !important;
        transform: scale(1.1) !important;
        background-color: #2c3e50 !important;
    }
    
    /* 侧边栏提示 */
    .sidebar-hint {
        position: fixed;
        top: 70px;
        left: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        z-index: 9998;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        animation: float 3s ease-in-out infinite;
        display: none;
        white-space: nowrap;
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }
    
    /* 主标题样式 */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding-top: 20px;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .highlight {
        background-color: #fff3cd;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-weight: bold;
    }
    
    /* 固定顶部导航 */
    .sticky-header {
        position: sticky;
        top: 0;
        background: white;
        z-index: 1000;
        padding: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>

<!-- 添加备用侧边栏切换按钮 -->
<div class="sidebar-toggle-btn" onclick="toggleSidebar()">☰</div>
<div class="sidebar-hint" id="sidebarHint">点击展开侧边栏设置面板</div>

<script>
    // 侧边栏控制函数
    function toggleSidebar() {
        const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
        const toggleBtn = parent.document.querySelector('[data-testid="collapsedControl"] button');
        if (toggleBtn) {
            toggleBtn.click();
        } else if (sidebar) {
            // 备用方法
            const width = window.getComputedStyle(sidebar).width;
            if (width === '0px' || width === '0') {
                sidebar.style.width = '300px';
                sidebar.style.minWidth = '300px';
                document.getElementById('sidebarHint').style.display = 'none';
            } else {
                sidebar.style.width = '0px';
                sidebar.style.minWidth = '0px';
                document.getElementById('sidebarHint').style.display = 'block';
            }
        }
    }
    
    // 检测侧边栏状态
    function checkSidebar() {
        const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
        const hint = parent.document.getElementById('sidebarHint');
        if (sidebar) {
            const width = window.getComputedStyle(sidebar).width;
            if (width === '0px' || width === '0') {
                hint.style.display = 'block';
            } else {
                hint.style.display = 'none';
            }
        }
    }
    
    // 定期检查侧边栏状态
    setInterval(checkSidebar, 1000);
    
    // 初始检查
    setTimeout(checkSidebar, 500);
</script>
""", unsafe_allow_html=True)

# 生成示例数据函数
def generate_sample_data():
    """生成包含趋势和季节性的示例数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='MS')
    n = len(dates)
    
    trend = np.linspace(100, 200, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = np.random.normal(0, 5, n)
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        '日期': dates,
        '销售额': values.round(2),
        '温度': (15 + 10 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 2, n)).round(1),
        '客流量': (trend * 0.5 + seasonal * 2 + np.random.normal(0, 10, n)).round(0)
    })
    return df

# ADF检验函数
def adf_test(timeseries, title=''):
    result = adfuller(timeseries.dropna(), autolag='AIC')
    output = {
        '检验统计量(ADF Statistic)': result[0],
        'p值(p-value)': result[1],
        '滞后阶数(Lags Used)': result[2],
        '观测值数量(Number of Observations)': result[3],
        '临界值(Critical Values)': result[4]
    }
    return output, result[1] <= 0.05

# 白噪声检验
def ljung_box_test(timeseries, lags=10):
    try:
        lb_test = acorr_ljungbox(timeseries.dropna(), lags=lags, return_df=True)
        return lb_test
    except:
        return None

# 主应用
def main():
    # 固定顶部导航
    st.markdown("""
    <div class="sticky-header">
        <div style="max-width: 1200px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between; padding: 0 20px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <h1 style="margin: 0; color: #1f77b4; font-size: 1.8rem;">📈 时间序列平稳性分析工具</h1>
            </div>
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="font-size: 14px; color: #666; background: #f0f8ff; padding: 5px 15px; border-radius: 20px;">
                    💡 点击左上角 <strong>☰</strong> 按钮打开/关闭设置面板
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 标题
    st.markdown('<div class="main-header">📈 时间序列平稳性分析工具</div>', unsafe_allow_html=True)
    
    # 初始化session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'using_sample' not in st.session_state:
        st.session_state.using_sample = False
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False
    
    # 侧边栏 - 改进显示和交互
    with st.sidebar:
        st.markdown("## 🧭 导航面板")
        
        # 侧边栏状态提示
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <div style="font-size: 24px; text-align: center; margin-bottom: 10px;">⚙️</div>
            <strong>控制面板</strong><br>
            在此设置分析参数
        </div>
        """, unsafe_allow_html=True)
        
        # 侧边栏关闭提示
        st.markdown("""
        <div style="font-size: 12px; color: #666; text-align: center; margin: 10px 0;">
            点击左上角 ☰ 按钮可折叠此面板
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📖 零基础使用说明", expanded=True):
            st.markdown("""
            **欢迎使用！请按以下步骤操作：**
            
            **第一步：准备数据**
            - 点击"📤 数据上传"标签
            - 选择 <span class="highlight">"使用示例数据"</span> 立即体验，或上传自己的文件
            
            **第二步：设置参数**
            - 选择时间列（日期格式）
            - 选择要分析的数据列
            - 设置季节性周期（月度数据填12，季度填4）
            
            **第三步：查看分析**
            - 原始数据可视化
            - 差分/季节差分处理
            - ADF平稳性检验结果
            - 自相关分析图表
            
            **第四步：导出结果**
            - 下载处理后的数据
            - 保存分析图表
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 显示当前数据状态
        if st.session_state.df is not None:
            st.markdown("### 📊 当前数据状态")
            df = st.session_state.df
            st.info(f"已加载数据：{len(df)} 行 × {len(df.columns)} 列")
            
            if st.session_state.using_sample:
                st.success("✅ 当前使用示例数据")
            else:
                st.success("✅ 已上传自定义数据")
        
        st.markdown("---")
        st.markdown("### ⚙️ 分析参数设置")
        
        # 这些控件会在数据上传后显示
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # 时间列选择
            date_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
            if date_cols:
                date_col = st.selectbox("选择时间列", date_cols, key='date_col')
                if df[date_col].dtype == 'object':
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        st.session_state.df = df
                    except:
                        st.error("时间列转换失败，请检查格式")
            else:
                st.error("未检测到时间列，请检查数据")
                date_col = None
            
            # 数值列选择
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols and date_cols:
                value_col = st.selectbox("选择数值列", numeric_cols, key='value_col')
                
                # 差分阶数
                st.markdown("---")
                st.markdown("### 🔧 差分设置")
                
                with st.expander("❓ 如何选择差分参数？", expanded=False):
                    st.markdown("""
                    **普通差分**：消除趋势
                    - 有明显上升趋势选1阶
                    - 有曲线趋势选2阶
                    
                    **季节差分**：消除周期性
                    - 月度数据且每年重复选12
                    - 季度数据选4
                    - 日数据且每周重复选7
                    """)
                
                diff_order = st.number_input("普通差分阶数", min_value=0, max_value=3, value=1, 
                                           help="消除趋势，1阶通常足够")
                seasonal_diff = st.number_input("季节性差分阶数", min_value=0, max_value=2, value=0,
                                              help="消除季节性，如有明显周期请设置")
                seasonal_period = st.number_input("季节性周期", min_value=2, max_value=365, value=12,
                                                help="如：月度数据=12，季度数据=4，日数据=7")
                
                # 分析按钮
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 重置参数", use_container_width=True):
                        st.session_state.analyze = False
                        st.rerun()
                
                with col2:
                    if st.button("🚀 开始分析", type="primary", use_container_width=True):
                        st.session_state.analyze = True
                        st.session_state.diff_order = diff_order
                        st.session_state.seasonal_diff = seasonal_diff
                        st.session_state.seasonal_period = seasonal_period
                        st.session_state.value_col = value_col
                        st.session_state.date_col = date_col
            else:
                st.warning("未检测到数值列")
        else:
            # 未上传数据时的提示
            st.markdown("""
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                <div style="font-size: 48px; margin-bottom: 10px;">📁</div>
                <p>请先上传数据</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📊 加载示例数据", type="secondary", use_container_width=True):
                st.session_state.df = generate_sample_data()
                st.session_state.using_sample = True
                st.rerun()

    # 主内容区 - 简化显示
    st.markdown("---")
    
    # 添加紧急侧边栏恢复按钮
    if st.button("🔄 显示/隐藏侧边栏设置面板", key="sidebar_toggle", help="如果左侧边栏不见了，点击此按钮恢复"):
        # 触发JavaScript函数
        st.markdown("""
        <script>
            parent.toggleSidebar();
        </script>
        """, unsafe_allow_html=True)
        st.rerun()
    
    # 主内容标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📤 数据上传", "📊 探索性分析", "🔍 差分与检验", "📥 结果导出"])
    
    # ... [保持原有的标签页内容不变，从你的原始代码中复制tab1到tab4的内容]
    # 由于代码长度限制，这里只显示修复部分，你需要将原始代码的tab1到tab4内容复制到这里

if __name__ == "__main__":
    main()
