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
    initial_sidebar_state="expanded"  # 默认展开
)    .toolbar-content {
        color: white;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 20px;
        width: 100%;
    }
    
    .toolbar-btn {
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.3s;
    }
    
    .toolbar-btn:hover {
        background: rgba(255,255,255,0.3);
    }
    
    /* 为顶部工具栏留出空间 */
    .main-content {
        margin-top: 60px;
    }
    
    /* 侧边栏恢复提示 */
    .sidebar-hint {
        position: fixed;
        top: 60px;
        left: 10px;
        background: #ff6b6b;
        color: white;
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 12px;
        z-index: 998;
        animation: pulse 2s infinite;
        display: none;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>

<script>
    // 检测侧边栏状态并显示提示
    function checkSidebar() {
        const sidebar = parent.document.querySelector('[data-testid="stSidebar"]');
        const hint = parent.document.querySelector('.sidebar-hint');
        if (sidebar && window.getComputedStyle(sidebar).width === '0px') {
            if (hint) hint.style.display = 'block';
        } else {
            if (hint) hint.style.display = 'none';
        }
    }
    setInterval(checkSidebar, 1000);
</script>

<div class="sidebar-hint">👈 点击左上角箭头打开设置面板</div>
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
    # 顶部工具栏（始终可见）
    st.markdown("""
    <div class="top-toolbar">
        <div class="toolbar-content">
            <span style="font-size: 20px;">📈 时间序列平稳性分析工具</span>
            <div style="margin-left: auto; display: flex; gap: 10px;">
                <span style="font-size: 12px; opacity: 0.9; align-self: center;">
                    💡 提示：点击左上角 ☰ 按钮可打开/关闭设置面板
                </span>
            </div>
        </div>
    </div>
    <div class="main-content"></div>
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
    
    # 侧边栏 - 使用更稳定的key和状态管理
    with st.sidebar:
        st.markdown("## 🧭 使用指南")
        
        # 添加侧边栏状态提示
        st.markdown("""
        <div style="background: #e3f2fd; padding: 10px; border-radius: 8px; margin-bottom: 15px; font-size: 13px;">
        <strong>💡 面板控制</strong><br>
        点击左上角的 <strong>☰</strong> 按钮可以隐藏/显示此面板<br>
        隐藏后可通过同一按钮重新打开
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
            st.info("👆 请先上传数据或点击\"使用示例数据\"")
            
            if st.button("📊 加载示例数据", type="secondary", use_container_width=True):
                st.session_state.df = generate_sample_data()
                st.session_state.using_sample = True
                st.rerun()

    # 主内容区 - 添加帮助按钮
    help_col1, help_col2 = st.columns([6, 1])
    with help_col2:
        if st.button("❓ 使用帮助", key="help_btn"):
            st.session_state.show_help = not st.session_state.show_help
    
    if st.session_state.show_help:
        st.markdown("""
        <div style="background: #f8f9fa; border: 2px solid #667eea; border-radius: 15px; padding: 20px; margin: 20px 0;">
        <h3>🆘 快速帮助</h3>
        <p><strong>Q: 侧边栏不见了怎么办？</strong><br>
        A: 点击页面左上角的 ☰ 按钮（或顶部的"❓ 使用帮助"按钮旁边的区域）可以重新打开侧边栏。</p>
        
        <p><strong>Q: 如何开始分析？</strong><br>
        A: 1) 在"📤 数据上传"标签页点击"使用示例数据"；2) 在左侧设置参数；3) 点击"开始分析"。</p>
        
        <p><strong>Q: 什么是ADF检验？</strong><br>
        A: ADF（Augmented Dickey-Fuller）检验用于判断时间序列是否平稳。p值≤0.05表示平稳。</p>
        
        <p><strong>Q: 差分后数据变少了？</strong><br>
        A: 差分操作会损失部分数据（每差分1阶损失1个数据点），这是正常现象。</p>
        </div>
        """, unsafe_allow_html=True)

    # 主内容标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📤 数据上传", "📊 探索性分析", "🔍 差分与检验", "📥 结果导出"])
    
    with tab1:
        st.markdown('<div class="section-header">📤 数据上传</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # 示例数据按钮（醒目位置）
            if st.session_state.df is None:
                st.markdown("""
                <div class="info-box">
                <h3>👋 初次使用？</h3>
                <p>我们为您准备了示例数据，包含趋势和季节性特征，帮助您快速了解功能。</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 点击使用示例数据（推荐新手）", type="primary", use_container_width=True):
                    st.session_state.df = generate_sample_data()
                    st.session_state.using_sample = True
                    st.rerun()
            
            uploaded_file = st.file_uploader("或上传您的数据文件", type=['csv', 'xlsx', 'xls'])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.df = df
                    st.session_state.using_sample = False
                    st.success(f"✅ 成功加载数据！共 {len(df)} 行，{len(df.columns)} 列")
                    
                    with st.expander("🔍 查看数据预览"):
                        st.dataframe(df.head(10), use_container_width=True)
                        st.markdown("**数据类型：**")
                        st.write(df.dtypes)
                        
                except Exception as e:
                    st.error(f"❌ 读取文件失败：{str(e)}")
            
            # 显示当前数据状态
            if st.session_state.df is not None:
                if st.session_state.using_sample:
                    st.markdown("""
                    <div class="success-box">
                    ✅ 当前正在使用<span class="highlight">示例数据</span>（模拟月度销售数据，2020-2023年）<br>
                    您可以切换到其他标签页直接查看分析，或上传自己的数据替换。
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("查看示例数据结构"):
                        st.write("**时间范围：** 2020-01 至 2023-12（共48个月）")
                        st.write("**包含字段：**")
                        st.write("- 日期：月度时间戳")
                        st.write("- 销售额：含上升趋势+季节性的模拟数据")
                        st.write("- 温度：季节性温度数据")
                        st.write("- 客流量：与销售额相关的模拟数据")
                        st.dataframe(st.session_state.df.head())
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>📋 数据格式要求</h4>
            <ul>
                <li>支持 CSV/Excel 格式</li>
                <li>必须包含时间列</li>
                <li>必须包含数值列</li>
                <li>建议无缺失值</li>
            </ul>
            <h4>💡 示例格式</h4>
            <table style='font-size:12px; width:100%; border-collapse: collapse;'>
                <tr style='background-color:#f0f0f0'>
                    <th style='border:1px solid #ddd; padding:4px'>日期</th>
                    <th style='border:1px solid #ddd; padding:4px'>销售额</th>
                </tr>
                <tr>
                    <td style='border:1px solid #ddd; padding:4px'>2023-01-01</td>
                    <td style='border:1px solid #ddd; padding:4px'>120.5</td>
                </tr>
                <tr>
                    <td style='border:1px solid #ddd; padding:4px'>2023-02-01</td>
                    <td style='border:1px solid #ddd; padding:4px'>135.2</td>
                </tr>
                <tr>
                    <td style='border:1px solid #ddd; padding:4px'>2023-03-01</td>
                    <td style='border:1px solid #ddd; padding:4px'>142.8</td>
                </tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        if st.session_state.df is None:
            st.info("👈 请先在左侧上传数据或加载示例数据")
            # 展示示例预览图
            st.markdown("---")
            st.markdown("### 🖼️ 功能预览（使用示例数据）")
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.markdown("""
                <div style="border:2px dashed #ccc; padding:20px; text-align:center; border-radius:10px;">
                <h4>📈 时间序列图</h4>
                <p style="color:#666;">展示原始数据的趋势和季节性</p>
                <p style="font-size:40px;">📉</p>
                </div>
                """, unsafe_allow_html=True)
            with preview_col2:
                st.markdown("""
                <div style="border:2px dashed #ccc; padding:20px; text-align:center; border-radius:10px;">
                <h4>📊 统计分布</h4>
                <p style="color:#666;">直方图和箱线图分析</p>
                <p style="font-size:40px;">📉</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-header">📊 原始数据探索</div>', unsafe_allow_html=True)
            
            df = st.session_state.df
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("选择要可视化的列", numeric_cols, key='explore_col')
                
                # 时间序列图
                date_col = st.session_state.date_col if 'date_col' in st.session_state else df.columns[0]
                try:
                    df_plot = df.copy()
                    df_plot[date_col] = pd.to_datetime(df_plot[date_col])
                    df_plot = df_plot.sort_values(date_col)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_plot[date_col],
                        y=df_plot[selected_col],
                        mode='lines+markers',
                        name='原始数据',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # 添加趋势线
                    z = np.polyfit(range(len(df_plot)), df_plot[selected_col], 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=df_plot[date_col],
                        y=p(range(len(df_plot))),
                        mode='lines',
                        name='趋势线',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_col} 时间序列图',
                        xaxis_title='时间',
                        yaxis_title='数值',
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 统计摘要
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("均值", f"{df[selected_col].mean():.2f}")
                    with col2:
                        st.metric("标准差", f"{df[selected_col].std():.2f}")
                    with col3:
                        st.metric("最小值", f"{df[selected_col].min():.2f}")
                    with col4:
                        st.metric("最大值", f"{df[selected_col].max():.2f}")
                    
                    # 分布图
                    col_left, col_right = st.columns(2)
                    with col_left:
                        fig_hist = px.histogram(df, x=selected_col, nbins=30, 
                                              title=f'{selected_col} 分布直方图',
                                              color_discrete_sequence=['#3498db'])
                        fig_hist.update_layout(template='plotly_white')
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col_right:
                        fig_box = px.box(df, y=selected_col, title=f'{selected_col} 箱线图',
                                       color_discrete_sequence=['#e74c3c'])
                        fig_box.update_layout(template='plotly_white')
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # 季节性分解（如果设置了周期）
                    if 'seasonal_period' in st.session_state:
                        try:
                            series = df_plot.set_index(date_col)[selected_col].dropna()
                            period = st.session_state.seasonal_period
                            if len(series) >= 2 * period:
                                decomposition = seasonal_decompose(series, model='additive', period=period)
                                
                                fig_decomp = make_subplots(rows=4, cols=1, 
                                                         subplot_titles=('原始序列', '趋势', '季节性', '残差'),
                                                         vertical_spacing=0.08)
                                
                                fig_decomp.add_trace(go.Scatter(x=series.index, y=series, 
                                                               line=dict(color='#1f77b4'), name='原始'), row=1, col=1)
                                fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, 
                                                               line=dict(color='#ff7f0e'), name='趋势'), row=2, col=1)
                                fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, 
                                                               line=dict(color='#2ca02c'), name='季节'), row=3, col=1)
                                fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, 
                                                               line=dict(color='#d62728'), name='残差'), row=4, col=1)
                                
                                fig_decomp.update_layout(height=800, showlegend=False, template='plotly_white',
                                                       title_text=f"季节性分解 (周期={period})")
                                st.plotly_chart(fig_decomp, use_container_width=True)
                                
                                # 添加说明
                                st.markdown("""
                                <div class="info-box">
                                <strong>图表说明：</strong><br>
                                - <strong>原始序列</strong>：实际观测值<br>
                                - <strong>趋势</strong>：长期走势（去除季节和随机因素）<br>
                                - <strong>季节性</strong>：周期性重复的模式<br>
                                - <strong>残差</strong>：去除趋势和季节后的随机波动
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"季节性分解失败：{str(e)}")
                except Exception as e:
                    st.error(f"绘图失败：{str(e)}")
            else:
                st.error("未找到数值列")
    
    with tab3:
        if 'analyze' not in st.session_state:
            st.info("👈 请在左侧设置参数并点击'开始分析'")
            if st.session_state.df is not None:
                st.markdown("---")
                st.markdown("### 📝 分析步骤提示")
                st.markdown("""
                1. 在左侧边栏确认<strong>时间列</strong>和<strong>数值列</strong>选择正确
                2. 设置<strong>差分阶数</strong>（建议从1阶开始）
                3. 如有明显季节性，设置<strong>季节差分</strong>和<strong>周期</strong>
                4. 点击<strong>开始分析</strong>查看结果
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-header">🔍 差分处理与平稳性检验</div>', unsafe_allow_html=True)
            
            df = st.session_state.df.copy()
            value_col = st.session_state.value_col
            date_col = st.session_state.date_col
            diff_order = st.session_state.diff_order
            seasonal_diff = st.session_state.seasonal_diff
            seasonal_period = st.session_state.seasonal_period
            
            # 准备数据
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            series = df.set_index(date_col)[value_col].dropna()
            
            # 执行差分
            diff_series = series.copy()
            diff_steps = []
            
            if seasonal_diff > 0:
                diff_series = diff_series.diff(seasonal_period * seasonal_diff)
                diff_steps.append(f"季节差分({seasonal_diff}阶, 周期{seasonal_period})")
            
            if diff_order > 0:
                diff_series = diff_series.diff(diff_order)
                diff_steps.append(f"普通差分({diff_order}阶)")
            
            diff_series = diff_series.dropna()
            diff_type = " → ".join(diff_steps) if diff_steps else "无差分（原始序列）"
            
            # 显示处理信息
            st.markdown(f"""
            <div class="info-box">
            <strong>处理流程：</strong>{diff_type}<br>
            <strong>原始数据长度：</strong>{len(series)} 条<br>
            <strong>差分后长度：</strong>{len(diff_series)} 条<br>
            <strong>数据损失：</strong>{len(series) - len(diff_series)} 条（差分导致的NaN）
            </div>
            """, unsafe_allow_html=True)
            
            # 对比图
            fig_compare = make_subplots(rows=2, cols=1, subplot_titles=('原始序列', f'差分后序列'),
                                      vertical_spacing=0.12)
            
            fig_compare.add_trace(go.Scatter(x=series.index, y=series, line=dict(color='#3498db', width=2),
                                           name='原始'), row=1, col=1)
            fig_compare.add_trace(go.Scatter(x=diff_series.index, y=diff_series, line=dict(color='#e74c3c', width=2),
                                           name='差分后'), row=2, col=1)
            
            # 添加零线到差分图
            fig_compare.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            fig_compare.update_layout(height=600, template='plotly_white', showlegend=False,
                                    title_text="差分效果对比")
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # ADF检验
            st.markdown('<div class="section-header">📋 ADF平稳性检验结果</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**原始序列检验**")
                adf_orig, is_stationary_orig = adf_test(series)
                
                for key, value in adf_orig.items():
                    if key != '临界值(Critical Values)':
                        st.write(f"**{key}:** {value:.6f}" if isinstance(value, float) else f"**{key}:** {value}")
                    else:
                        st.write(f"**{key}:**")
                        for k, v in value.items():
                            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;{k}: {v:.4f}")
                
                if is_stationary_orig:
                    st.markdown('<div class="success-box">✅ 原始序列已通过平稳性检验 (p ≤ 0.05)<br>序列平稳，无需差分</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">⚠️ 原始序列非平稳 (p > 0.05)<br>建议进行差分处理</div>', 
                               unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**差分后序列检验**")
                if len(diff_series) > 0:
                    adf_diff, is_stationary_diff = adf_test(diff_series)
                    
                    for key, value in adf_diff.items():
                        if key != '临界值(Critical Values)':
                            st.write(f"**{key}:** {value:.6f}" if isinstance(value, float) else f"**{key}:** {value}")
                        else:
                            st.write(f"**{key}:**")
                            for k, v in value.items():
                                st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;{k}: {v:.4f}")
                    
                    if is_stationary_diff:
                        st.markdown('<div class="success-box">✅ 差分后序列已通过平稳性检验 (p ≤ 0.05)<br>差分有效，序列已平稳</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ 差分后序列仍非平稳<br>建议增加差分阶数或检查周期设置</div>', 
                                   unsafe_allow_html=True)
                else:
                    st.error("差分后数据不足，无法检验")
            
            # ACF/PACF分析
            st.markdown('<div class="section-header">📈 自相关与偏自相关分析</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>如何解读：</strong><br>
            - <strong>ACF</strong>（自相关）：序列与自身滞后期的相关性，拖尾表示需要差分<br>
            - <strong>PACF</strong>（偏自相关）：排除中间滞后期影响后的相关性，截尾提示AR阶数<br>
            - 蓝色阴影为95%置信区间，超出说明相关性显著
            </div>
            """, unsafe_allow_html=True)
            
            col_acf1, col_acf2 = st.columns(2)
            
            with col_acf1:
                st.markdown("**原始序列 ACF/PACF**")
                try:
                    fig_acf, axes = plt.subplots(2, 1, figsize=(10, 6))
                    plot_acf(series.dropna(), ax=axes[0], lags=min(40, len(series)//2-1), title='自相关函数(ACF)')
                    plot_pacf(series.dropna(), ax=axes[1], lags=min(40, len(series)//2-1), title='偏自相关函数(PACF)', method='ywm')
                    plt.tight_layout()
                    st.pyplot(fig_acf)
                except Exception as e:
                    st.error(f"绘图失败：{str(e)}")
            
            with col_acf2:
                st.markdown("**差分后序列 ACF/PACF**")
                if len(diff_series) > 5:
                    try:
                        fig_acf_diff, axes_diff = plt.subplots(2, 1, figsize=(10, 6))
                        plot_acf(diff_series, ax=axes_diff[0], lags=min(40, len(diff_series)//2-1), 
                                title='自相关函数(ACF)')
                        plot_pacf(diff_series, ax=axes_diff[1], lags=min(40, len(diff_series)//2-1), 
                                 title='偏自相关函数(PACF)', method='ywm')
                        plt.tight_layout()
                        st.pyplot(fig_acf_diff)
                    except Exception as e:
                        st.error(f"绘图失败：{str(e)}")
                else:
                    st.warning("差分后数据不足，无法绘制ACF/PACF")
            
            # 白噪声检验
            st.markdown('<div class="section-header">🎲 白噪声检验 (Ljung-Box)</div>', unsafe_allow_html=True)
            st.markdown("检验序列是否为白噪声（随机波动），白噪声序列无预测价值")
            
            col_lb1, col_lb2 = st.columns(2)
            
            with col_lb1:
                st.markdown("**原始序列**")
                lb_orig = ljung_box_test(series)
                if lb_orig is not None:
                    st.dataframe(lb_orig.head(10), use_container_width=True)
                    p_val = lb_orig['lb_pvalue'].iloc[0]
                    if p_val < 0.05:
                        st.write(f"滞后1期p值: {p_val:.4f} - 非白噪声（有自相关性）")
                    else:
                        st.write(f"滞后1期p值: {p_val:.4f} - 可能是白噪声")
            
            with col_lb2:
                st.markdown("**差分后序列**")
                if len(diff_series) > 0:
                    lb_diff = ljung_box_test(diff_series)
                    if lb_diff is not None:
                        st.dataframe(lb_diff.head(10), use_container_width=True)
                        p_val = lb_diff['lb_pvalue'].iloc[0]
                        if p_val < 0.05:
                            st.write(f"滞后1期p值: {p_val:.4f} - 非白噪声（有自相关性）")
                        else:
                            st.write(f"滞后1期p值: {p_val:.4f} - 可能是白噪声")
            
            # 保存结果到session state
            st.session_state.diff_series = diff_series
            st.session_state.diff_type = diff_type
    
    with tab4:
        if 'diff_series' not in st.session_state:
            st.info("👈 请先完成差分分析")
            st.markdown("---")
            st.markdown("### 💡 提示")
            st.markdown("完成分析后，您可以在此下载：")
            st.markdown("- 处理后的时间序列数据（CSV/Excel）")
            st.markdown("- 包含原始值和差分值的对比表")
            st.markdown("- 分析参数摘要")
        else:
            st.markdown('<div class="section-header">📥 结果导出</div>', unsafe_allow_html=True)
            
            # 准备导出数据
            df_export = st.session_state.df.copy()
            date_col = st.session_state.date_col
            df_export[date_col] = pd.to_datetime(df_export[date_col])
            df_export = df_export.set_index(date_col)
            
            export_df = pd.DataFrame({
                '原始值': df_export[st.session_state.value_col],
            })
            export_df['差分值'] = st.session_state.diff_series
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**数据预览（前20行）**")
                st.dataframe(export_df.head(20), use_container_width=True)
                
                # 统计信息
                st.markdown("**差分效果统计**")
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("原始序列均值", f"{export_df['原始值'].mean():.2f}")
                    st.metric("原始序列标准差", f"{export_df['原始值'].std():.2f}")
                with col_stat2:
                    st.metric("差分后均值", f"{export_df['差分值'].mean():.2f}")
                    st.metric("差分后标准差", f"{export_df['差分值'].std():.2f}")
                
                # CSV下载
                csv = export_df.to_csv().encode('utf-8')
                st.download_button(
                    label="⬇️ 下载CSV格式",
                    data=csv,
                    file_name=f"差分数据_{st.session_state.value_col}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**分析摘要**")
                summary_text = f"""分析时间序列平稳性报告
================================
分析列：{st.session_state.value_col}
时间列：{st.session_state.date_col}
处理流程：{st.session_state.diff_type}
原始数据量：{len(export_df)} 条
有效差分数据量：{export_df['差分值'].notna().sum()} 条
数据时间范围：{export_df.index.min()} 至 {export_df.index.max()}
分析执行时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

统计对比：
- 原始序列均值：{export_df['原始值'].mean():.4f}
- 差分后均值：{export_df['差分值'].mean():.4f}
- 原始序列标准差：{export_df['原始值'].std():.4f}
- 差分后标准差：{export_df['差分值'].std():.4f}
"""
                st.text_area("分析摘要（可复制）", summary_text, height=300)
                
                # Excel下载
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='差分数据')
                    # 添加摘要sheet
                    summary_df = pd.DataFrame({
                        '项目': ['分析列', '时间列', '差分类型', '原始数据量', '有效差分数据量', 
                                '时间范围起点', '时间范围终点', '分析时间'],
                        '值': [st.session_state.value_col, st.session_state.date_col, 
                              st.session_state.diff_type, len(export_df), 
                              export_df['差分值'].notna().sum(),
                              str(export_df.index.min()), str(export_df.index.max()),
                              pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    summary_df.to_excel(writer, sheet_name='分析摘要', index=False)
                    
                    # 添加统计sheet
                    stats_df = pd.DataFrame({
                        '指标': ['均值', '标准差', '最小值', '最大值'],
                        '原始序列': [export_df['原始值'].mean(), export_df['原始值'].std(),
                                   export_df['原始值'].min(), export_df['原始值'].max()],
                        '差分后序列': [export_df['差分值'].mean(), export_df['差分值'].std(),
                                    export_df['差分值'].min(), export_df['差分值'].max()]
                    })
                    stats_df.to_excel(writer, sheet_name='统计对比', index=False)
                
                st.download_button(
                    label="⬇️ 下载Excel格式（含3个工作表）",
                    data=buffer.getvalue(),
                    file_name=f"差分数据_{st.session_state.value_col}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()

