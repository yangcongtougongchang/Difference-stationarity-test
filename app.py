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
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== 修复版：多用户隔离管理器 ====================
class UserSessionManager:
    """管理多用户会话隔离 - 每个会话独立"""
    
    def __init__(self):
        # 每个实例都强制初始化用户ID，不依赖缓存
        self._ensure_user_id()
    
    def _ensure_user_id(self):
        """确保当前会话有用户ID"""
        # 直接操作session_state，不使用属性访问
        if 'user_id' not in st.session_state:
            # 生成唯一用户ID
            new_id = uuid.uuid4().hex[:16]
            st.session_state['user_id'] = new_id
            st.session_state['_session_initialized'] = datetime.now().isoformat()
    
    @property
    def user_id(self):
        """安全获取用户ID"""
        self._ensure_user_id()  # 双重保险
        return st.session_state['user_id']
    
    def get_widget_key(self, base_key):
        """生成widget专用的key"""
        return f"widget_{self.user_id}_{base_key}"
    
    def get_data_key(self, base_key):
        """生成数据存储专用的key"""
        return f"data_{self.user_id}_{base_key}"
    
    def save_data(self, base_key, value):
        """保存数据"""
        data_key = self.get_data_key(base_key)
        st.session_state[data_key] = value
    
    def get_data(self, base_key, default=None):
        """获取数据"""
        data_key = self.get_data_key(base_key)
        return st.session_state.get(data_key, default)
    
    def clear_all_data(self):
        """清除当前用户的所有数据"""
        user_id = self.user_id
        user_prefix_data = f"data_{user_id}_"
        user_prefix_widget = f"widget_{user_id}_"
        
        keys_to_delete = []
        for key in list(st.session_state.keys()):
            if key.startswith(user_prefix_data) or key.startswith(user_prefix_widget):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        
        # 保留user_id和标记
        st.session_state[f"data_{user_id}_cleared"] = True
        st.session_state[f"data_{user_id}_clear_time"] = datetime.now().isoformat()

# ==================== 关键修复：不使用缓存，每个会话创建独立实例 ====================
def get_session_manager():
    """获取当前会话的SessionManager（不缓存）"""
    # 使用session_state存储manager实例，确保每个会话独立
    if '_session_mgr' not in st.session_state:
        st.session_state['_session_mgr'] = UserSessionManager()
    return st.session_state['_session_mgr']

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="时间序列平稳性分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 获取当前会话管理器（必须在所有UI操作之前） ====================
try:
    session_mgr = get_session_manager()
    user_id = session_mgr.user_id
except Exception as e:
    st.error(f"初始化失败: {str(e)}")
    st.stop()

# ==================== CSS样式 ====================
st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    # header {{visibility: hidden;}}
    
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        opacity: 1 !important;
        display: flex !important;
    }}
    
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
    }}
    
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
    
    .privacy-notice {{
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 12px;
    }}
</style>

<div class="user-badge" title="您的会话ID: {user_id}">
    用户: {user_id[:8]}... | 数据隔离中
</div>
""", unsafe_allow_html=True)

# ==================== 数据生成函数 ====================
def generate_sample_data():
    """生成示例数据（每个用户独立）"""
    np.random.seed(hash(user_id) % 2**32)
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
    <div style="position: fixed; top: 0; left: 0; right: 0; height: 50px; 
                background: linear-gradient(90deg, #1f77b4 0%, #4a90e2 100%);
                z-index: 1000; display: flex; align-items: center; padding: 0 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <div style="color: white; font-weight: bold; display: flex; align-items: center; gap: 20px; width: 100%;">
            <span style="font-size: 20px;">📈 时间序列平稳性分析工具</span>
            <div style="margin-left: auto; display: flex; gap: 10px; align-items: center;">
                <span style="background: #d4edda; color: #155724; padding: 2px 10px; 
                      border-radius: 12px; font-size: 11px;">🔒 数据隔离保护</span>
                <span style="font-size: 12px; opacity: 0.9;">💡 点击左上角 ☰ 打开设置</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📈 时间序列平稳性分析工具</div>', unsafe_allow_html=True)
    
    # 隐私提示
    st.markdown(f"""
    <div class="privacy-notice">
    <strong>🔒 隐私保护说明</strong> | 会话ID: <code>{user_id}</code>
    <span style="float: right;">
        <a href="#" onclick="window.location.reload(); return false;" style="color: #856404;">[刷新页面清除数据]</a>
    </span><br>
    • 数据仅存储在当前会话内存中 • 其他用户无法访问 • 页面关闭后自动清除
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown("## 🧭 使用指南")
        
        # 用户操作
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<small>用户: <code>{user_id[:8]}...</code></small>", unsafe_allow_html=True)
        with col2:
            if st.button("🗑️", help="清除我的所有数据", key=session_mgr.get_widget_key("btn_clear")):
                session_mgr.clear_all_data()
                st.success("✅ 已清除")
                st.rerun()
        
        st.markdown("---")
        
        with st.expander("📖 零基础使用说明", expanded=True):
            st.markdown("""
            **欢迎使用！请按以下步骤操作：**
            
            **第一步：准备数据**
            - 点击"📤 数据上传"标签
            - 选择"使用示例数据"立即体验，或上传自己的文件
            
            **第二步：设置参数**
            - 选择时间列和数值列
            - 设置差分阶数（通常1阶）和季节周期（月度=12）
            
            **第三步：开始分析**
            - 点击"🚀 开始分析"按钮
            - 查看差分效果和平稳性检验结果
            
            **第四步：导出结果**
            - 下载CSV或Excel格式的处理结果
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ 分析参数设置")
        
        # 获取当前数据
        current_df = session_mgr.get_data('df')
        
        if current_df is not None:
            # 时间列选择
            date_cols = current_df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
            if date_cols:
                saved_date_col = session_mgr.get_data('date_col', date_cols[0])
                default_index = date_cols.index(saved_date_col) if saved_date_col in date_cols else 0
                
                date_col = st.selectbox(
                    "选择时间列", 
                    date_cols, 
                    index=default_index,
                    key=session_mgr.get_widget_key("sel_date_col")
                )
                session_mgr.save_data('date_col', date_col)
                
                # 尝试转换时间格式
                if current_df[date_col].dtype == 'object':
                    try:
                        current_df[date_col] = pd.to_datetime(current_df[date_col])
                        session_mgr.save_data('df', current_df)
                    except:
                        st.error("时间列转换失败，请检查格式")
            else:
                st.error("未检测到时间列")
                date_col = None
            
            # 数值列选择
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols and date_cols:
                saved_value_col = session_mgr.get_data('value_col', numeric_cols[0])
                default_index = numeric_cols.index(saved_value_col) if saved_value_col in numeric_cols else 0
                
                value_col = st.selectbox(
                    "选择数值列", 
                    numeric_cols,
                    index=default_index,
                    key=session_mgr.get_widget_key("sel_value_col")
                )
                session_mgr.save_data('value_col', value_col)
                
                # 差分设置
                st.markdown("---")
                st.markdown("### 🔧 差分设置")
                
                with st.expander("❓ 如何选择参数？", expanded=False):
                    st.markdown("""
                    **普通差分**：消除趋势（1阶通常足够）
                    **季节差分**：消除周期性（月度=12，季度=4，日度=7）
                    """)
                
                diff_order = st.number_input(
                    "普通差分阶数", 
                    min_value=0, max_value=3, 
                    value=session_mgr.get_data('diff_order', 1),
                    key=session_mgr.get_widget_key("num_diff_order")
                )
                session_mgr.save_data('diff_order', diff_order)
                
                seasonal_diff = st.number_input(
                    "季节性差分阶数", 
                    min_value=0, max_value=2, 
                    value=session_mgr.get_data('seasonal_diff', 0),
                    key=session_mgr.get_widget_key("num_seasonal_diff")
                )
                session_mgr.save_data('seasonal_diff', seasonal_diff)
                
                seasonal_period = st.number_input(
                    "季节性周期", 
                    min_value=2, max_value=365, 
                    value=session_mgr.get_data('seasonal_period', 12),
                    key=session_mgr.get_widget_key("num_seasonal_period")
                )
                session_mgr.save_data('seasonal_period', seasonal_period)
                
                # 分析按钮
                st.markdown("---")
                if st.button("🚀 开始分析", type="primary", use_container_width=True, 
                           key=session_mgr.get_widget_key("btn_analyze")):
                    session_mgr.save_data('analyze', True)
                    st.rerun()
            else:
                st.warning("未检测到数值列")
        else:
            st.info("👆 请先上传数据")
            
            if st.button("📊 加载示例数据", type="secondary", use_container_width=True,
                       key=session_mgr.get_widget_key("btn_sample")):
                sample_df = generate_sample_data()
                session_mgr.save_data('df', sample_df)
                session_mgr.save_data('using_sample', True)
                st.rerun()

    # 主内容区 - 标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📤 数据上传", "📊 探索性分析", "🔍 差分与检验", "📥 结果导出"])
    
    with tab1:
        st.markdown('<div class="section-header">📤 数据上传</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if session_mgr.get_data('df') is None:
                st.markdown("""
                <div class="info-box">
                <h3>👋 初次使用？</h3>
                <p>点击"加载示例数据"立即体验功能，或上传您自己的数据文件。</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📊 使用示例数据（推荐新手）", type="primary", use_container_width=True,
                           key=session_mgr.get_widget_key("btn_sample_main")):
                    sample_df = generate_sample_data()
                    session_mgr.save_data('df', sample_df)
                    session_mgr.save_data('using_sample', True)
                    st.rerun()
            
            # 文件上传
            uploaded_file = st.file_uploader(
                "上传CSV或Excel文件", 
                type=['csv', 'xlsx', 'xls'],
                key=session_mgr.get_widget_key("file_uploader")
            )
            
            # 处理上传的文件
            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()
                last_hash = session_mgr.get_data('last_file_hash')
                
                if file_hash != last_hash:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            new_df = pd.read_csv(uploaded_file)
                        else:
                            new_df = pd.read_excel(uploaded_file)
                        
                        session_mgr.save_data('df', new_df)
                        session_mgr.save_data('using_sample', False)
                        session_mgr.save_data('last_file_hash', file_hash)
                        session_mgr.save_data('analyze', False)
                        
                        st.success(f"✅ 成功加载！共 {len(new_df)} 行，{len(new_df.columns)} 列")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ 读取失败：{str(e)}")
                else:
                    st.info("📄 文件已加载，请在其他标签页查看")
            
            # 显示当前数据状态
            current_df = session_mgr.get_data('df')
            if current_df is not None:
                if session_mgr.get_data('using_sample'):
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ 正在使用示例数据（用户{user_id[:8]}专属）<br>
                    <small>数据已隔离，仅当前会话可访问</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    file_hash = session_mgr.get_data('last_file_hash', 'N/A')[:16]
                    st.markdown(f"""
                    <div class="success-box">
                    ✅ 已加载您的数据文件<br>
                    <small>文件指纹: {file_hash}... | 仅当前用户可访问</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("🔍 数据预览（仅您可见）"):
                    st.dataframe(current_df.head(10), use_container_width=True)
                    st.caption(f"数据维度: {current_df.shape[0]} 行 × {current_df.shape[1]} 列")
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>📋 数据格式要求</h4>
            <ul>
                <li>支持 CSV、Excel 格式</li>
                <li>必须包含时间列（日期格式）</li>
                <li>必须包含数值列（用于分析）</li>
                <li>建议数据量 > 24 条（用于季节性分析）</li>
            </ul>
            <h4>🔒 安全说明</h4>
            <p style="font-size: 12px; color: #666;">
            数据仅保存在服务器内存中，不会写入磁盘，会话结束后立即清除。
            </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        current_df = session_mgr.get_data('df')
        if current_df is None:
            st.info("👈 请先上传数据或加载示例数据")
        else:
            st.markdown('<div class="section-header">📊 原始数据探索</div>', unsafe_allow_html=True)
            
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                saved_col = session_mgr.get_data('explore_col', numeric_cols[0])
                default_idx = numeric_cols.index(saved_col) if saved_col in numeric_cols else 0
                
                selected_col = st.selectbox(
                    "选择要可视化的列", 
                    numeric_cols,
                    index=default_idx,
                    key=session_mgr.get_widget_key("sel_explore_col")
                )
                session_mgr.save_data('explore_col', selected_col)
                
                date_col = session_mgr.get_data('date_col')
                if date_col and date_col in current_df.columns:
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
                        if len(df_plot) > 1:
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
                        
                        # 统计指标
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
                        
                    except Exception as e:
                        st.error(f"绘图失败：{str(e)}")
                else:
                    st.warning("请先选择正确的时间列")
            else:
                st.error("未找到数值列")
    
    with tab3:
        if not session_mgr.get_data('analyze'):
            st.info("👈 请在左侧设置参数并点击'开始分析'")
        else:
            st.markdown('<div class="section-header">🔍 差分处理与平稳性检验</div>', unsafe_allow_html=True)
            
            df = session_mgr.get_data('df').copy()
            value_col = session_mgr.get_data('value_col')
            date_col = session_mgr.get_data('date_col')
            diff_order = session_mgr.get_data('diff_order', 1)
            seasonal_diff = session_mgr.get_data('seasonal_diff', 0)
            seasonal_period = session_mgr.get_data('seasonal_period', 12)
            
            if not all([value_col, date_col]) or value_col not in df.columns or date_col not in df.columns:
                st.error("参数错误，请重新选择数据列")
                session_mgr.save_data('analyze', False)
                st.rerun()
            
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
            <strong>用户:</strong> <code>{user_id[:8]}</code> | 
            <strong>处理:</strong> {diff_type}<br>
            <strong>原始长度:</strong> {len(series)} 条 | 
            <strong>差分后:</strong> {len(diff_series)} 条 | 
            <strong>损失:</strong> {len(series) - len(diff_series)} 条
            </div>
            """, unsafe_allow_html=True)
            
            # 对比图
            fig_compare = make_subplots(rows=2, cols=1, 
                                      subplot_titles=('原始序列', f'差分后序列'),
                                      vertical_spacing=0.12)
            
            fig_compare.add_trace(go.Scatter(x=series.index, y=series, 
                                           line=dict(color='#3498db', width=2), name='原始'), row=1, col=1)
            fig_compare.add_trace(go.Scatter(x=diff_series.index, y=diff_series, 
                                           line=dict(color='#e74c3c', width=2), name='差分后'), row=2, col=1)
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
                    st.markdown('<div class="success-box">✅ 原始序列已平稳 (p ≤ 0.05)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">⚠️ 原始序列非平稳 (p > 0.05)</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("**差分后序列检验**")
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
                        st.markdown('<div class="success-box">✅ 差分后序列已平稳 (p ≤ 0.05)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ 差分后仍非平稳，建议调整参数</div>', unsafe_allow_html=True)
                else:
                    st.error("差分后数据不足")
            
            # ACF/PACF
            st.markdown('<div class="section-header">📈 自相关与偏自相关分析</div>', unsafe_allow_html=True)
            
            col_acf1, col_acf2 = st.columns(2)
            
            with col_acf1:
                st.markdown("**原始序列**")
                try:
                    fig_acf, axes = plt.subplots(2, 1, figsize=(10, 6))
                    plot_acf(series.dropna(), ax=axes[0], lags=min(40, len(series)//2-1), title='ACF')
                    plot_pacf(series.dropna(), ax=axes[1], lags=min(40, len(series)//2-1), title='PACF', method='ywm')
                    plt.tight_layout()
                    st.pyplot(fig_acf)
                except Exception as e:
                    st.error(f"ACF/PACF绘制失败：{str(e)}")
            
            with col_acf2:
                st.markdown("**差分后序列**")
                if len(diff_series) > 5:
                    try:
                        fig_acf_diff, axes_diff = plt.subplots(2, 1, figsize=(10, 6))
                        plot_acf(diff_series, ax=axes_diff[0], lags=min(40, len(diff_series)//2-1), title='ACF')
                        plot_pacf(diff_series, ax=axes_diff[1], lags=min(40, len(diff_series)//2-1), title='PACF', method='ywm')
                        plt.tight_layout()
                        st.pyplot(fig_acf_diff)
                    except Exception as e:
                        st.error(f"ACF/PACF绘制失败：{str(e)}")
                else:
                    st.warning("差分后数据不足")
            
            # 保存结果
            session_mgr.save_data('diff_series', diff_series)
            session_mgr.save_data('diff_type', diff_type)
    
    with tab4:
        if session_mgr.get_data('diff_series') is None:
            st.info("👈 请先完成差分分析")
        else:
            st.markdown('<div class="section-header">📥 结果导出</div>', unsafe_allow_html=True)
            
            df_export = session_mgr.get_data('df').copy()
            date_col = session_mgr.get_data('date_col')
            value_col = session_mgr.get_data('value_col')
            
            df_export[date_col] = pd.to_datetime(df_export[date_col])
            df_export = df_export.set_index(date_col)
            
            export_df = pd.DataFrame({
                '原始值': df_export[value_col],
            })
            export_df['差分值'] = session_mgr.get_data('diff_series')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**数据预览（用户{user_id[:8]}专属）**")
                st.dataframe(export_df.head(20), use_container_width=True)
                
                # CSV下载
                csv = export_df.to_csv().encode('utf-8')
                st.download_button(
                    label="⬇️ 下载CSV格式",
                    data=csv,
                    file_name=f"差分数据_{value_col}_用户{user_id[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=session_mgr.get_widget_key("btn_download_csv")
                )
            
            with col2:
                st.markdown("**分析摘要**")
                summary_text = f"""时间序列平稳性分析报告
================================
用户ID: {user_id}
分析列：{value_col}
时间列：{date_col}
处理流程：{session_mgr.get_data('diff_type')}
原始数据量：{len(export_df)} 条
有效差分数据量：{export_df['差分值'].notna().sum()} 条
分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

统计对比：
- 原始序列均值：{export_df['原始值'].mean():.4f}
- 差分后均值：{export_df['差分值'].mean():.4f}
- 原始序列标准差：{export_df['原始值'].std():.4f}
- 差分后标准差：{export_df['差分值'].std():.4f}
"""
                st.text_area("分析摘要", summary_text, height=250)
                
                # Excel下载
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='差分数据')
                    summary_df = pd.DataFrame({
                        '项目': ['用户ID', '分析列', '时间列', '差分类型', '原始数据量', '分析时间'],
                        '值': [user_id, value_col, date_col, 
                              session_mgr.get_data('diff_type'), len(export_df), 
                              datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    summary_df.to_excel(writer, sheet_name='分析摘要', index=False)
                
                st.download_button(
                    label="⬇️ 下载Excel格式",
                    data=buffer.getvalue(),
                    file_name=f"差分数据_{value_col}_用户{user_id[:8]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key=session_mgr.get_widget_key("btn_download_excel")
                )

if __name__ == "__main__":
    main()
