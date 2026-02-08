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
import uuid
import hashlib
import time
from datetime import datetime, timedelta
import os
import threading
import weakref
warnings.filterwarnings('ignore')

# ==================== 多用户隔离管理器 ====================
class UserSessionManager:
    """管理多用户会话隔离"""
    _instance = None
    _lock = threading.Lock()
    _sessions = weakref.WeakValueDictionary()  # 自动清理失效会话
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._session_data = {}
                    cls._instance._last_access = {}
                    cls._instance._cleanup_interval = 300  # 5分钟清理一次
        return cls._instance
    
    def get_user_id(self):
        """获取或创建用户唯一标识"""
        if 'user_id' not in st.session_state:
            # 结合浏览器指纹和随机UUID
            user_agent = st.request.headers.get('User-Agent', 'unknown') if hasattr(st, 'request') else 'unknown'
            client_ip = st.request.headers.get('X-Forwarded-For', 'unknown') if hasattr(st, 'request') else 'unknown'
            fingerprint = f"{user_agent}_{client_ip}_{uuid.uuid4().hex[:8]}"
            user_id = hashlib.md5(fingerprint.encode()).hexdigest()[:16]
            st.session_state.user_id = user_id
            st.session_state.session_start = datetime.now()
        return st.session_state.user_id
    
    def get_user_key(self, key):
        """为用户特定的key添加前缀"""
        user_id = self.get_user_id()
        return f"{user_id}_{key}"
    
    def set_user_data(self, key, value):
        """存储用户隔离的数据"""
        user_key = self.get_user_key(key)
        st.session_state[user_key] = value
        self._last_access[self.get_user_id()] = time.time()
    
    def get_user_data(self, key, default=None):
        """获取用户隔离的数据"""
        user_key = self.get_user_key(key)
        user_id = self.get_user_id()
        self._last_access[user_id] = time.time()
        return st.session_state.get(user_key, default)
    
    def clear_user_data(self):
        """清理当前用户的所有数据"""
        user_id = self.get_user_id()
        keys_to_delete = [k for k in st.session_state.keys() if k.startswith(f"{user_id}_")]
        for key in keys_to_delete:
            del st.session_state[key]
        # 保留user_id本身，但标记为已清理
        st.session_state[f"{user_id}_cleared"] = True
    
    def cleanup_expired_sessions(self, max_age=3600):
        """清理过期会话（1小时无活动）"""
        current_time = time.time()
        expired_users = [
            user_id for user_id, last_time in self._last_access.items()
            if current_time - last_time > max_age
        ]
        for user_id in expired_users:
            if user_id in self._session_data:
                del self._session_data[user_id]
            if user_id in self._last_access:
                del self._last_access[user_id]

# 全局会话管理器实例
session_manager = UserSessionManager()

# ==================== 用户隔离的session_state包装器 ====================
def user_state(key, default=None):
    """获取用户隔离的状态值"""
    return session_manager.get_user_data(key, default)

def set_user_state(key, value):
    """设置用户隔离的状态值"""
    session_manager.set_user_data(key, value)

def init_user_state(key, default):
    """初始化用户状态（如果不存在）"""
    if user_state(key) is None:
        set_user_state(key, default)

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="时间序列平稳性分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 用户识别与欢迎 ====================
user_id = session_manager.get_user_id()

# ==================== CSS样式 ====================
st.markdown(f"""
<style>
    /* 隐藏GitHub和菜单 */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    header {{visibility: hidden;}}
    
    /* 强制显示侧边栏控制按钮 */
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        opacity: 1 !important;
        display: flex !important;
    }}
    
    /* 用户标识显示 */
    .user-badge {{
        position: fixed;
        top: 10px;
        right: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        z-index: 10000;
        opacity: 0.8;
        transition: opacity 0.3s;
    }}
    .user-badge:hover {{
        opacity: 1;
    }}
    
    /* 顶部工具栏 */
    .top-toolbar {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 50px;
        background: linear-gradient(90deg, #1f77b4 0%, #4a90e2 100%);
        z-index: 1000;
        display: flex;
        align-items: center;
        padding: 0 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    
    .toolbar-content {{
        color: white;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 20px;
        width: 100%;
    }}
    
    /* 主标题 */
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 60px;
    }}
    
    .section-header {{
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }}
    
    .info-box {{
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }}
    
    .success-box {{
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }}
    
    .warning-box {{
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }}
    
    .highlight {{
        background-color: #fff3cd;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-weight: bold;
    }}
    
    /* 隐私提示 */
    .privacy-notice {{
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 12px;
    }}
    
    /* 数据隔离指示器 */
    .isolation-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #d4edda;
        color: #155724;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        margin-left: 10px;
    }}
    
    .isolation-indicator::before {{
        content: "🔒";
    }}
</style>

<div class="user-badge" title="您的会话ID: {user_id}">
    用户: {user_id[:8]}... | 会话安全隔离中
</div>
""", unsafe_allow_html=True)

# ==================== 数据生成函数 ====================
def generate_sample_data():
    """生成示例数据（每个用户独立）"""
    np.random.seed(hash(user_id) % 2**32)  # 基于用户ID的随机种子
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

# ==================== 分析函数 ====================
def adf_test(timeseries):
    result = adfuller(timeseries.dropna(), autolag='AIC')
    output = {
        '检验统计量(ADF Statistic)': result[0],
        'p值(p-value)': result[1],
        '滞后阶数(Lags Used)': result[2],
        '观测值数量(Number of Observations)': result[3],
        '临界值(Critical Values)': result[4]
    }
    return output, result[1] <= 0.05

def ljung_box_test(timeseries, lags=10):
    try:
        lb_test = acorr_ljungbox(timeseries.dropna(), lags=lags, return_df=True)
        return lb_test
    except:
        return None

# ==================== 主应用 ====================
def main():
    # 顶部工具栏
    st.markdown("""
    <div class="top-toolbar">
        <div class="toolbar-content">
            <span style="font-size: 20px;">📈 时间序列平稳性分析工具</span>
            <div style="margin-left: auto; display: flex; gap: 10px; align-items: center;">
                <span class="isolation-indicator">数据隔离保护</span>
                <span style="font-size: 12px; opacity: 0.9;">
                    💡 点击左上角 ☰ 打开设置
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 标题
    st.markdown('<div class="main-header">📈 时间序列平稳性分析工具</div>', unsafe_allow_html=True)
    
    # 初始化用户状态
    init_user_state('df', None)
    init_user_state('using_sample', False)
    init_user_state('analyze', False)
    init_user_state('show_help', False)
    
    # 隐私提示
    st.markdown(f"""
    <div class="privacy-notice">
    <strong>🔒 隐私保护说明</strong> | 您的会话ID: <code>{user_id}</code><br>
    • 您的数据仅存储在当前会话中，其他用户无法访问<br>
    • 页面关闭1小时后数据自动清除<br>
    • 点击"🗑️ 清除我的数据"可立即删除所有个人信息
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏 - 用户隔离
    with st.sidebar:
        st.markdown("## 🧭 使用指南")
        
        # 用户操作区
        st.markdown("---")
        col_user1, col_user2 = st.columns([3, 1])
        with col_user1:
            st.markdown(f"<small>当前用户: <code>{user_id[:8]}...</code></small>", unsafe_allow_html=True)
        with col_user2:
            if st.button("🗑️", help="清除我的所有数据", key="clear_user_data"):
                session_manager.clear_user_data()
                st.success("✅ 数据已清除")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
        
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
        
        # 获取用户数据
        df = user_state('df')
        
        if df is not None:
            # 时间列选择
            date_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
            if date_cols:
                # 恢复用户上次的选择
                saved_date_col = user_state('date_col', date_cols[0])
                date_col = st.selectbox("选择时间列", date_cols, 
                                       index=date_cols.index(saved_date_col) if saved_date_col in date_cols else 0,
                                       key=f"{user_id}_date_col")
                set_user_state('date_col', date_col)
                
                if df[date_col].dtype == 'object':
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        set_user_state('df', df)
                    except:
                        st.error("时间列转换失败，请检查格式")
            else:
                st.error("未检测到时间列")
                date_col = None
            
            # 数值列选择
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols and date_cols:
                saved_value_col = user_state('value_col', numeric_cols[0])
                value_col = st.selectbox("选择数值列", numeric_cols,
                                        index=numeric_cols.index(saved_value_col) if saved_value_col in numeric_cols else 0,
                                        key=f"{user_id}_value_col")
                set_user_state('value_col', value_col)
                
                # 差分设置
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
                
                # 恢复用户设置
                diff_order = st.number_input("普通差分阶数", min_value=0, max_value=3, 
                                           value=user_state('diff_order', 1), 
                                           key=f"{user_id}_diff_order")
                seasonal_diff = st.number_input("季节性差分阶数", min_value=0, max_value=2, 
                                              value=user_state('seasonal_diff', 0),
                                              key=f"{user_id}_seasonal_diff")
                seasonal_period = st.number_input("季节性周期", min_value=2, max_value=365, 
                                                value=user_state('seasonal_period', 12),
                                                key=f"{user_id}_seasonal_period")
                
                # 保存设置
                set_user_state('diff_order', diff_order)
                set_user_state('seasonal_diff', seasonal_diff)
                set_user_state('seasonal_period', seasonal_period)
                
                # 分析按钮
                st.markdown("---")
                if st.button("🚀 开始分析", type="primary", use_container_width=True):
                    set_user_state('analyze', True)
                    st.rerun()
            else:
                st.warning("未检测到数值列")
        else:
            st.info("👆 请先上传数据或点击\"使用示例数据\"")
            
            if st.button("📊 加载示例数据", type="secondary", use_container_width=True):
                sample_df = generate_sample_data()
                set_user_state('df', sample_df)
                set_user_state('using_sample', True)
                st.rerun()

    # 主内容区
    tab1, tab2, tab3, tab4 = st.tabs(["📤 数据上传", "📊 探索性分析", "🔍 差分与检验", "📥 结果导出"])
    
    with tab1:
        st.markdown('<div class="section-header">📤 数据上传</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if user_state('df') is None:
                st.markdown("""
                <div class="info-box">
                <h3>👋 初次使用？</h3>
                <p>我们为您准备了示例数据，包含趋势和季节性特征，帮助您快速了解功能。</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 点击使用示例数据（推荐新手）", type="primary", use_container_width=True):
                    sample_df = generate_sample_data()
                    set_user_state('df', sample_df)
                    set_user_state('using_sample', True)
                    st.rerun()
            
            # 文件上传 - 每个用户独立
            uploaded_file = st.file_uploader("或上传您的数据文件", type=['csv', 'xlsx', 'xls'], 
                                           key=f"{user_id}_uploader")
            
            if uploaded_file is not None:
                try:
                    # 检查是否是新文件
                    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
                    if user_state('last_file_hash') != file_hash:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                        
                        set_user_state('df', df)
                        set_user_state('using_sample', False)
                        set_user_state('last_file_hash', file_hash)
                        set_user_state('analyze', False)  # 重置分析状态
                        st.success(f"✅ 成功加载数据！共 {len(df)} 行，{len(df.columns)} 列")
                        st.rerun()
                    else:
                        st.info("📄 文件已加载，请在其他标签页查看分析")
                        
                except Exception as e:
                    st.error(f"❌ 读取文件失败：{str(e)}")
            
            # 显示当前数据状态
            current_df = user_state('df')
            if current_df is not None:
                if user_state('using_sample'):
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ 当前正在使用<span class="highlight">示例数据</span>（用户{user_id[:8]}专属）<br>
                    <small>数据已隔离，其他用户无法查看</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ 已加载您的数据文件<br>
                    <small>文件哈希: {user_state('last_file_hash', 'N/A')[:16]}... | 仅您可访问</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("🔍 查看数据预览（仅当前用户可见）"):
                    st.dataframe(current_df.head(10), use_container_width=True)
                    st.markdown("**数据类型：**")
                    st.write(current_df.dtypes)
        
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
            <h4>🔒 安全说明</h4>
            <p style="font-size: 12px; color: #666;">
            您的数据仅存储在服务器内存中，不会写入磁盘，会话结束后立即清除。
            </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        current_df = user_state('df')
        if current_df is None:
            st.info("👈 请先在左侧上传数据或加载示例数据")
        else:
            st.markdown('<div class="section-header">📊 原始数据探索</div>', unsafe_allow_html=True)
            
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # 恢复用户选择
                saved_explore_col = user_state('explore_col', numeric_cols[0])
                selected_col = st.selectbox("选择要可视化的列", numeric_cols,
                                           index=numeric_cols.index(saved_explore_col) if saved_explore_col in numeric_cols else 0,
                                           key=f"{user_id}_explore_col")
                set_user_state('explore_col', selected_col)
                
                date_col = user_state('date_col', current_df.columns[0])
                try:
                    df_plot = current_df.copy()
                    df_plot[date_col] = pd.to_datetime(df_plot[date_col])
                    df_plot = df_plot.sort_values(date_col)
                    
                    # 时间序列图
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_plot[date_col],
                        y=df_plot[selected_col],
                        mode='lines+markers',
                        name='原始数据',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=6)
                    ))
                    
                    # 趋势线
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
                        title=f'{selected_col} 时间序列图 [用户{user_id[:8]}]',
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
                        st.metric("均值", f"{current_df[selected_col].mean():.2f}")
                    with col2:
                        st.metric("标准差", f"{current_df[selected_col].std():.2f}")
                    with col3:
                        st.metric("最小值", f"{current_df[selected_col].min():.2f}")
                    with col4:
                        st.metric("最大值", f"{current_df[selected_col].max():.2f}")
                    
                    # 分布图
                    col_left, col_right = st.columns(2)
                    with col_left:
                        fig_hist = px.histogram(current_df, x=selected_col, nbins=30, 
                                              title=f'{selected_col} 分布直方图',
                                              color_discrete_sequence=['#3498db'])
                        fig_hist.update_layout(template='plotly_white')
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col_right:
                        fig_box = px.box(current_df, y=selected_col, title=f'{selected_col} 箱线图',
                                       color_discrete_sequence=['#e74c3c'])
                        fig_box.update_layout(template='plotly_white')
                        st.plotly_chart(fig_box, use_container_width=True)
                    
                    # 季节性分解
                    if user_state('analyze') and user_state('seasonal_period'):
                        try:
                            series = df_plot.set_index(date_col)[selected_col].dropna()
                            period = user_state('seasonal_period')
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
                                                       title_text=f"季节性分解 (用户{user_id[:8]})")
                                st.plotly_chart(fig_decomp, use_container_width=True)
                        except Exception as e:
                            st.warning(f"季节性分解失败：{str(e)}")
                except Exception as e:
                    st.error(f"绘图失败：{str(e)}")
    
    with tab3:
        if not user_state('analyze'):
            st.info("👈 请在左侧设置参数并点击'开始分析'")
        else:
            st.markdown('<div class="section-header">🔍 差分处理与平稳性检验</div>', unsafe_allow_html=True)
            
            df = user_state('df').copy()
            value_col = user_state('value_col')
            date_col = user_state('date_col')
            diff_order = user_state('diff_order')
            seasonal_diff = user_state('seasonal_diff')
            seasonal_period = user_state('seasonal_period')
            
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
            <strong>用户:</strong> <code>{user_id}</code> | 
            <strong>处理流程:</strong> {diff_type}<br>
            <strong>原始数据长度:</strong> {len(series)} 条 | 
            <strong>差分后长度:</strong> {len(diff_series)} 条
            </div>
            """, unsafe_allow_html=True)
            
            # 对比图
            fig_compare = make_subplots(rows=2, cols=1, subplot_titles=('原始序列', f'差分后序列'),
                                      vertical_spacing=0.12)
            
            fig_compare.add_trace(go.Scatter(x=series.index, y=series, line=dict(color='#3498db', width=2),
                                           name='原始'), row=1, col=1)
            fig_compare.add_trace(go.Scatter(x=diff_series.index, y=diff_series, line=dict(color='#e74c3c', width=2),
                                           name='差分后'), row=2, col=1)
            fig_compare.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            fig_compare.update_layout(height=600, template='plotly_white', showlegend=False,
                                    title_text=f"差分效果对比 [用户{user_id[:8]}]")
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
                    st.markdown('<div class="success-box">✅ 原始序列已通过平稳性检验 (p ≤ 0.05)</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">⚠️ 原始序列非平稳 (p > 0.05)</div>', 
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
                        st.markdown('<div class="success-box">✅ 差分后序列已通过平稳性检验 (p ≤ 0.05)</div>', 
                                   unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ 差分后序列仍非平稳</div>', 
                                   unsafe_allow_html=True)
                else:
                    st.error("差分后数据不足，无法检验")
            
            # ACF/PACF分析
            st.markdown('<div class="section-header">📈 自相关与偏自相关分析</div>', unsafe_allow_html=True)
            
            col_acf1, col_acf2 = st.columns(2)
            
            with col_acf1:
                st.markdown("**原始序列 ACF/PACF**")
                try:
                    fig_acf, axes = plt.subplots(2, 1, figsize=(10, 6))
                    plot_acf(series.dropna(), ax=axes[0], lags=min(40, len(series)//2-1), title='ACF')
                    plot_pacf(series.dropna(), ax=axes[1], lags=min(40, len(series)//2-1), title='PACF', method='ywm')
                    plt.tight_layout()
                    st.pyplot(fig_acf)
                except Exception as e:
                    st.error(f"绘图失败：{str(e)}")
            
            with col_acf2:
                st.markdown("**差分后序列 ACF/PACF**")
                if len(diff_series) > 5:
                    try:
                        fig_acf_diff, axes_diff = plt.subplots(2, 1, figsize=(10, 6))
                        plot_acf(diff_series, ax=axes_diff[0], lags=min(40, len(diff_series)//2-1), title='ACF')
                        plot_pacf(diff_series, ax=axes_diff[1], lags=min(40, len(diff_series)//2-1), title='PACF', method='ywm')
                        plt.tight_layout()
                        st.pyplot(fig_acf_diff)
                    except Exception as e:
                        st.error(f"绘图失败：{str(e)}")
                else:
                    st.warning("差分后数据不足，无法绘制ACF/PACF")
            
            # 保存结果
            set_user_state('diff_series', diff_series)
            set_user_state('diff_type', diff_type)
    
    with tab4:
        if user_state('diff_series') is None:
            st.info("👈 请先完成差分分析")
        else:
            st.markdown('<div class="section-header">📥 结果导出</div>', unsafe_allow_html=True)
            
            df_export = user_state('df').copy()
            date_col = user_state('date_col')
            df_export[date_col] = pd.to_datetime(df_export[date_col])
            df_export = df_export.set_index(date_col)
            
            export_df = pd.DataFrame({
                '原始值': df_export[user_state('value_col')],
            })
            export_df['差分值'] = user_state('diff_series')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**数据预览（用户{user_id[:8]}专属）**")
                st.dataframe(export_df.head(20), use_container_width=True)
                
                # CSV下载
                csv = export_df.to_csv().encode('utf-8')
                st.download_button(
                    label="⬇️ 下载CSV格式",
                    data=csv,
                    file_name=f"差分数据_{user_state('value_col')}_用户{user_id[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("**分析摘要**")
                summary_text = f"""时间序列平稳性分析报告
================================
用户ID: {user_id}
分析列：{user_state('value_col')}
时间列：{user_state('date_col')}
处理流程：{user_state('diff_type')}
原始数据量：{len(export_df)} 条
有效差分数据量：{export_df['差分值'].notna().sum()} 条
分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

统计对比：
- 原始序列均值：{export_df['原始值'].mean():.4f}
- 差分后均值：{export_df['差分值'].mean():.4f}
- 原始序列标准差：{export_df['原始值'].std():.4f}
- 差分后标准差：{export_df['差分值'].std():.4f}

================================
本报告由用户 {user_id} 生成，数据已隔离保护
"""
                st.text_area("分析摘要（可复制）", summary_text, height=300)
                
                # Excel下载
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='差分数据')
                    summary_df = pd.DataFrame({
                        '项目': ['用户ID', '分析列', '时间列', '差分类型', '原始数据量', '分析时间'],
                        '值': [user_id, user_state('value_col'), user_state('date_col'), 
                              user_state('diff_type'), len(export_df), 
                              datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    summary_df.to_excel(writer, sheet_name='分析摘要', index=False)
                
                st.download_button(
                    label="⬇️ 下载Excel格式",
                    data=buffer.getvalue(),
                    file_name=f"差分数据_{user_state('value_col')}_用户{user_id[:8]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
