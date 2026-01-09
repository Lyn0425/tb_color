# modules/ui_components.py

import streamlit as st
from typing import Dict, Any, List, Optional
from PIL import Image
from io import BytesIO
import pandas as pd
from .config import APP_CONFIG

class UIComponents:
    """UI组件类"""
    
    @staticmethod
    def setup_page_config():
        """设置页面配置"""
        st.set_page_config(
            page_title="胸片增强系统",
            page_icon="🩺",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @staticmethod
    def create_sidebar() -> Dict[str, Any]:
        """创建侧边栏"""
        st.sidebar.title("⚙️ 控制面板")
        st.sidebar.markdown("---")
        
        # 颜色方案选择
        color_scheme = st.sidebar.selectbox(
            "🎨 选择颜色方案",
            ["标准", "高对比度", "柔和"],
            index=0,
            help="不同的颜色方案适用于不同的组织显示"
        )
        
        # 增强选项
        st.sidebar.markdown("### 增强选项")
        apply_clahe = st.sidebar.checkbox("应用CLAHE增强", value=True)
        contrast = st.sidebar.slider("对比度增强", 0.5, 2.0, 1.0, 0.1)
        brightness = st.sidebar.slider("亮度调节", -50, 50, 0, 5)
        
        # 保存选项
        save_to_history = st.sidebar.checkbox("保存到历史记录", value=True)
        
        # 其他选项
        st.sidebar.markdown("### 其他选项")
        show_stats = st.sidebar.checkbox("显示详细统计", value=True)
        st.sidebar.markdown("### 数据库设置")
        db_enabled = st.sidebar.checkbox("启用数据库持久化", value=APP_CONFIG.get("db_enabled", False))
        db_type = st.sidebar.selectbox("数据库类型", ["sqlite", "mysql"], index=0 if APP_CONFIG.get("db_type", "sqlite") == "sqlite" else 1)

        db_path = APP_CONFIG.get("db_path", "medical_images.db")
        if db_type == "sqlite":
            db_path = st.sidebar.text_input("sqlite文件路径", value=db_path)

        return {
            "color_scheme": color_scheme,
            "apply_clahe": apply_clahe,
            "contrast": contrast,
            "brightness": brightness,
            "save_to_history": save_to_history,
            "show_stats": show_stats,
            "db_enabled": db_enabled,
            "db_type": db_type,
            "db_path": db_path,

        }
    
    @staticmethod
    def create_header(history_count: int = 0):
        """创建页面头部"""
        col_title1, col_title2, col_title3 = st.columns([2, 1, 1])
        
        with col_title1:
            st.title("🩺 胸片灰度分层伪色彩增强系统")
            st.markdown("### 智能医学影像处理与教学演示平台")
        
        with col_title3:
            st.metric("历史记录数", history_count)
    
    @staticmethod
    def show_warning():
        """显示警告信息"""
        with st.container():
            st.warning("""
            ⚠️ **重要医学声明**: 
            本系统仅用于教学演示和学术研究目的，**不能替代专业医疗诊断**！
            临床诊断必须使用经认证的医疗设备和专业医疗人员的判断。
            """)
    
    @staticmethod
    def show_image_comparison(original_img, enhanced_img, 
                            original_stats: Optional[Dict] = None,
                            enhanced_stats: Optional[Dict] = None):
        """显示图像对比"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 原始胸片")
            st.image(original_img, caption=f"尺寸: {original_img.shape[1]}x{original_img.shape[0]}", 
                    use_column_width=True)
            
            if original_stats:
                with st.expander("📊 原始图像统计"):
                    st.json(original_stats)
        
        with col2:
            st.markdown("#### 🎨 增强图像")
            st.image(enhanced_img, caption="伪彩色增强处理", use_column_width=True)

    @staticmethod
    def show_histogram(counts: List[int]):
        df = pd.DataFrame({"intensity": list(range(256)), "count": list(counts)})
        st.markdown("#### 📈 灰度直方图")
        st.bar_chart(df.set_index("intensity"), use_container_width=True)

    @staticmethod
    def show_legend(legend_img):
        st.markdown("#### 🎨 颜色图例")
        st.image(legend_img, caption="强度分段颜色映射", use_column_width=True)
    
    @staticmethod
    def create_download_button(image: Image.Image, filename: str = "enhanced_image.jpg") -> BytesIO:
        """创建下载按钮数据"""
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf
    
    @staticmethod
    def show_history_table(history_list: List[Dict], max_entries: int = 10):
        """显示历史记录表格"""
        if not history_list:
            st.info("📭 暂无历史记录。上传并处理图像后，记录将显示在这里。")
            return
        
        # 显示最近记录
        with st.expander("📋 最近处理记录", expanded=True):
            recent_entries = history_list[:5]
            df = pd.DataFrame(recent_entries)
            
            if not df.empty:
                # 重命名列名
                column_names = {
                    "timestamp": "时间",
                    "filename": "文件名",
                    "color_scheme": "颜色方案",
                    "original_shape": "原始尺寸",
                    "enhanced_shape": "增强后尺寸"
                }
                df_display = df.rename(columns=column_names)
                
                # 只显示需要的列
                display_cols = ["时间", "文件名", "颜色方案", "原始尺寸"]
                df_display = df_display[display_cols] if all(col in df_display.columns for col in display_cols) else df_display
                
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # 详细历史记录（分页显示）
        st.markdown("### 📊 历史记录详情")
        
        # 分页设置
        page_size = 5
        total_pages = (len(history_list) + page_size - 1) // page_size
        current_page = st.selectbox(
            "选择页码",
            range(1, total_pages + 1),
            index=0
        )
        
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_records = history_list[start_idx:end_idx]
        
        # 显示当前页记录
        for i, record in enumerate(paginated_records):
            with st.container():
                cols = st.columns([2, 1, 1, 1])
                cols[0].write(f"**{record.get('filename', 'N/A')}**")
                cols[1].write(f"📅 {record.get('timestamp', 'N/A')}")
                cols[2].write(f"🎨 {record.get('color_scheme', 'N/A')}")
                cols[3].write(f"📏 {record.get('original_shape', (0, 0))[1]}x{record.get('original_shape', (0, 0))[0]}")
                
                # 使用唯一的键
                unique_key = f"stats_{start_idx + i}"
                if cols[3].button("📈 详情", key=unique_key):
                    with st.expander("记录详情", expanded=True):
                        st.json({
                            "基本": {
                                "时间": record.get('timestamp', 'N/A'),
                                "文件名": record.get('filename', 'N/A'),
                                "颜色方案": record.get('color_scheme', 'N/A'),
                                "原始尺寸": record.get('original_shape', (0, 0)),
                                "增强尺寸": record.get('enhanced_shape', (0, 0)),
                            },
                            "统计": record.get('stats', {}),
                        })
                
                st.markdown("---")
        
        # 显示分页信息
        st.caption(f"显示第 {current_page} 页，共 {total_pages} 页，总计 {len(history_list)} 条记录")
    
    @staticmethod
    def create_footer(history_count: int = 0):
        """创建页脚"""
        st.markdown("---")
        footer_col1, footer_col2, footer_col3 = st.columns(3)
        
        with footer_col1:
            st.caption("🩺 医学影像处理系统 v2.0")
        with footer_col2:
            st.caption("仅供教学演示使用")
        with footer_col3:
            st.caption(f"© 2024 医学影像实验室 | 已处理: {history_count} 张图像")

    @staticmethod
    def show_history_query_filters() -> Dict[str, Any]:
        st.markdown("### 🔎 查询筛选")
        filename_contains = st.text_input("文件名包含")
        color_scheme = st.selectbox("颜色方案", ["全部", "standard", "high_contrast", "soft"], index=0)
        date_cols = st.columns(2)
        with date_cols[0]:
            start_date = st.date_input("起始日期", value=None)
        with date_cols[1]:
            end_date = st.date_input("结束日期", value=None)
        return {
            "filename_contains": filename_contains.strip() if filename_contains else None,
            "color_scheme": color_scheme,
            "start_date": start_date,
            "end_date": end_date,
        }

    @staticmethod
    def show_history_query_results(records: List[Dict]):
        if not records:
            st.info("无匹配记录")
            return
        
        # 显示查询结果表格
        df = pd.DataFrame(records)
        if not df.empty:
            st.dataframe(
                df[["timestamp", "filename", "color_scheme", "original_shape"]], 
                use_container_width=True,
                hide_index=True
            )
        
        st.markdown("### 记录列表")
        
        # 分页设置
        page_size = 5
        total_pages = (len(records) + page_size - 1) // page_size
        current_page = st.selectbox(
            "选择页码",
            range(1, total_pages + 1),
            index=0
        )
        
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_records = records[start_idx:end_idx]
        
        # 显示当前页记录
        for i, record in enumerate(paginated_records):
            with st.container():
                cols = st.columns([2, 1, 1, 1])
                cols[0].write(f"**{record.get('filename', 'N/A')}**")
                cols[1].write(f"📅 {record.get('timestamp', 'N/A')}")
                cols[2].write(f"🎨 {record.get('color_scheme', 'N/A')}")
                cols[3].write(f"📏 {record.get('original_shape', (0, 0))[1]}x{record.get('original_shape', (0, 0))[0]}")
                
                # 使用唯一的键
                unique_key = f"query_stats_{start_idx + i}"
                if cols[3].button("📈 详情", key=unique_key):
                    with st.expander("记录详情", expanded=True):
                        st.json({
                            "基本": {
                                "时间": record.get('timestamp', 'N/A'),
                                "文件名": record.get('filename', 'N/A'),
                                "颜色方案": record.get('color_scheme', 'N/A'),
                                "原始尺寸": record.get('original_shape', (0, 0)),
                                "增强尺寸": record.get('enhanced_shape', (0, 0)),
                            },
                            "统计": record.get('stats', {}),
                        })
                
                st.markdown("---")
        
        # 显示分页信息
        st.caption(f"显示第 {current_page} 页，共 {total_pages} 页，总计 {len(records)} 条记录")
