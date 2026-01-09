# app.py

import streamlit as st
import cv2
import numpy as np
import logging
import datetime
from PIL import Image
import tqdm
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("medical_image_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入自定义模块
from modules.ui_components import UIComponents
from modules.image_processor import MedicalImageProcessor
from modules.history_manager import HistoryManager
from modules.config import APP_CONFIG

# 初始化组件
ui = UIComponents()
processor = MedicalImageProcessor()

# 统一MySQL配置
mysql_config = {
    "host": APP_CONFIG.get("mysql_host", "localhost"),
    "port": APP_CONFIG.get("mysql_port", 3306),
    "user": APP_CONFIG.get("mysql_user", "root"),
    "password": APP_CONFIG.get("mysql_password", "liu123"),
    "database": APP_CONFIG.get("mysql_database", "medical_images"),
}

history_manager = HistoryManager(
    max_entries=APP_CONFIG["max_history_entries"],
    db_enabled=APP_CONFIG.get("db_enabled", False),
    db_path=APP_CONFIG.get("db_path", "medical_images.db"),
    db_type=APP_CONFIG.get("db_type", "sqlite"),
    mysql_config=mysql_config
)

# 初始化session_state
def init_session_state():
    """初始化session_state"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'enhanced_image' not in st.session_state:
        st.session_state.enhanced_image = None
    if 'image_stats' not in st.session_state:
        st.session_state.image_stats = None

# 核心图像处理函数
def process_single_image(uploaded_file, controls):
    """处理单个图像的核心函数"""
    try:
        logger.info(f"开始处理文件: {uploaded_file.name}")
    
        # 读取文件
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        logger.info(f"文件读取成功，大小: {len(file_bytes)} bytes")
    
        # 解码图像
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("无法解码图像文件")
        logger.info(f"图像解码成功，尺寸: {img.shape}")
    
        # 预处理图像
        preprocessed_img = processor.preprocess_image(
            img,
            apply_clahe=controls["apply_clahe"],
            contrast=controls["contrast"],
            brightness=controls["brightness"]
        )
        logger.info("图像预处理完成")
    
        # 伪彩色增强
        enhanced_img = processor.enhance_pseudocolor(
            preprocessed_img,
            controls["color_scheme"]
        )
        logger.info(f"伪彩色增强完成，使用颜色方案: {controls['color_scheme']}")
    
        # 计算图像统计
        stats = processor.calculate_image_stats(img)
        logger.info(f"图像统计计算完成: {stats}")
    
        return {
            "filename": uploaded_file.name,
            "original_image": img,
            "enhanced_image": enhanced_img,
            "stats": stats,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except ValueError as ve:
        logger.error(f"值错误: {str(ve)}", exc_info=True)
        raise
    except cv2.error as cv_err:
        logger.error(f"OpenCV错误: {str(cv_err)}", exc_info=True)
        raise
    except MemoryError:
        logger.error("内存错误", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"未知错误: {str(e)}", exc_info=True)
        raise

# 主应用
def main():
    # 设置页面配置
    ui.setup_page_config()
    
    # 初始化session_state
    init_session_state()
    
    # 创建页面头部
    ui.create_header(len(st.session_state.history))
    
    # 显示警告信息
    ui.show_warning()
    
    st.markdown("---")
    controls = ui.create_sidebar()
    
    history_manager.set_db_config(
        controls["db_enabled"],
        controls.get("db_type", "sqlite"),
        controls["db_path"],
        mysql_config
    )
    
    # 侧边栏其他控制
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 清空历史记录", type="secondary"):
        st.session_state.history = history_manager.clear_history()
        st.sidebar.success("历史记录已清空！")
        st.rerun()
    
    processing_tab, history_tab, db_tab, help_tab = st.tabs(["处理", "历史", "数据库", "帮助"])

    with processing_tab:
        # 批量处理选项
        batch_mode = st.checkbox("开启批量处理模式", value=False)
        
        if batch_mode:
            st.subheader("📤 批量上传胸片图像")
            uploaded_files = st.file_uploader(
                "选择多个胸片图像文件",
                type=APP_CONFIG["allowed_file_types"],
                accept_multiple_files=True,
                help="支持常见的医学图像格式，建议使用标准胸片图像"
            )
            
            if uploaded_files:
                st.info(f"已上传 {len(uploaded_files)} 个文件，准备开始批量处理...")
                
                if st.button("🚀 开始批量处理", type="primary"):
                    from tqdm import tqdm
                    import time
                    
                    batch_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        for idx, uploaded_file in enumerate(uploaded_files):
                            logger.info(f"批量处理文件 {idx+1}/{len(uploaded_files)}: {uploaded_file.name}")
                            status_text.text(f"处理中: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
                            
                            # 使用统一的单图像处理函数
                            result = process_single_image(uploaded_file, controls)
                            batch_results.append(result)
                            
                            # 更新进度
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                            time.sleep(0.1)  # 给UI一点刷新时间
                        
                        # 处理完成
                        status_text.success(f"✅ 批量处理完成！共处理 {len(batch_results)} 个文件")
                        
                        # 显示结果预览
                        if batch_results:
                            st.markdown("### 📊 批量处理结果预览")
                            
                            # 保存到历史记录
                            if controls["save_to_history"]:
                                # 批量处理历史记录
                                batch_history_entries = []
                                for result in batch_results:
                                    entry = {
                                        "timestamp": result["timestamp"],
                                        "filename": result["filename"],
                                        "color_scheme": controls["color_scheme"],
                                        "stats": result["stats"],
                                        "original_shape": result["original_image"].shape,
                                        "enhanced_shape": result["enhanced_image"].shape
                                    }
                                    batch_history_entries.append(entry)
                                    
                                # 更新内存中的历史记录
                                st.session_state.history = batch_history_entries + st.session_state.history
                                if len(st.session_state.history) > APP_CONFIG["max_history_entries"]:
                                    st.session_state.history = st.session_state.history[:APP_CONFIG["max_history_entries"]]
                                
                                # 批量保存到数据库
                                if history_manager.db_enabled:
                                    from modules.history_manager import HistoryEntry
                                    history_objects = [HistoryEntry(**entry) for entry in batch_history_entries]
                                    history_manager.save_entries_to_db(history_objects)
                                
                                st.success(f"✅ 已保存到历史记录，共 {len(batch_history_entries)} 条")
                            
                            # 提供打包下载
                            if st.button("📦 打包下载所有增强图像", type="secondary"):
                                import zipfile
                                import io
                                
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                    for result in batch_results:
                                        pil_img = processor.convert_to_pil(result["enhanced_image"])
                                        img_buffer = io.BytesIO()
                                        pil_img.save(img_buffer, format="JPEG", quality=95)
                                        img_buffer.seek(0)
                                        zf.writestr(f"enhanced_{result['filename']}", img_buffer.read())
                                
                                zip_buffer.seek(0)
                                st.download_button(
                                    label="📥 下载ZIP包",
                                    data=zip_buffer,
                                    file_name="enhanced_images_batch.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )
                            
                            # 显示前5个结果
                            for i, result in enumerate(batch_results[:5]):
                                with st.expander(f"结果 {i+1}: {result['filename']}", expanded=False):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**原始图像**")
                                        st.image(result["original_image"], caption=f"原始尺寸: {result['original_image'].shape[1]}x{result['original_image'].shape[0]}", use_column_width=True)
                                    with col2:
                                        st.markdown("**增强图像**")
                                        st.image(result["enhanced_image"], caption=f"增强尺寸: {result['enhanced_image'].shape[1]}x{result['enhanced_image'].shape[0]}", use_column_width=True)
                                
                            if len(batch_results) > 5:
                                st.info(f"共 {len(batch_results)} 个结果，仅显示前5个。请使用打包下载功能获取所有结果。")
                        
                    except Exception as e:
                        logger.error(f"批量处理错误: {str(e)}", exc_info=True)
                        st.error(f"❌ 批量处理失败: {str(e)}")
                        status_text.error("处理失败")
        else:
            # 单文件处理模式（原代码）
            st.subheader("📤 上传胸片图像")
            uploaded_file = st.file_uploader(
                "选择胸片图像文件",
                type=APP_CONFIG["allowed_file_types"],
                help="支持常见的医学图像格式，建议使用标准胸片图像"
            )

            if uploaded_file:
                try:
                    logger.info(f"开始处理文件: {uploaded_file.name}")
                
                    # 读取文件
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    logger.info(f"文件读取成功，大小: {len(file_bytes)} bytes")
                
                    # 解码图像
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError("无法解码图像文件")
                    logger.info(f"图像解码成功，尺寸: {img.shape}")
                
                    # 预处理图像
                    with st.spinner("🔄 正在预处理图像..."):
                        preprocessed_img = processor.preprocess_image(
                            img,
                            apply_clahe=controls["apply_clahe"],
                            contrast=controls["contrast"],
                            brightness=controls["brightness"]
                        )
                    logger.info("图像预处理完成")
                
                    # 伪彩色增强
                    with st.spinner("🎨 正在应用伪彩色增强..."):
                        enhanced_img = processor.enhance_pseudocolor(
                            preprocessed_img,
                            controls["color_scheme"]
                        )
                    logger.info(f"伪彩色增强完成，使用颜色方案: {controls['color_scheme']}")
                
                    # 保存到session_state
                    st.session_state.current_image = img
                    st.session_state.enhanced_image = enhanced_img
                
                    # 计算图像统计
                    stats = processor.calculate_image_stats(img)
                    st.session_state.image_stats = stats
                    logger.info(f"图像统计计算完成: {stats}")
                
                    # 显示结果
                    st.markdown("---")
                    st.subheader("🔍 处理结果对比")
                    ui.show_image_comparison(
                        img,
                        enhanced_img,
                        original_stats=stats if controls["show_stats"] else None
                    )
                
                    # 显示图例
                    legend_img = processor.generate_legend(controls["color_scheme"])
                    ui.show_legend(legend_img)
                
                    # 显示直方图
                    counts = processor.compute_histogram(preprocessed_img)
                    ui.show_histogram(counts)
                
                    # 输出选项
                    st.markdown("### 📥 输出选项")
                    col1, col2 = st.columns(2)
                
                    with col1:
                        # 下载按钮
                        pil_img = processor.convert_to_pil(enhanced_img)
                        download_data = ui.create_download_button(
                            pil_img,
                            filename=f"enhanced_{uploaded_file.name}"
                        )
                        st.download_button(
                            label="📥 下载增强图像",
                            data=download_data,
                            file_name=f"enhanced_{uploaded_file.name}",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                        logger.info(f"生成下载数据: enhanced_{uploaded_file.name}")
                
                    with col2:
                        # 保存到历史
                        if controls["save_to_history"]:
                            if st.button("💾 保存到历史", use_container_width=True):
                                entry_data = {
                                    "filename": uploaded_file.name,
                                    "color_scheme": controls["color_scheme"],
                                    "stats": stats,
                                    "original_shape": img.shape,
                                    "enhanced_shape": enhanced_img.shape
                                }
                                st.session_state.history = history_manager.add_entry(
                                    st.session_state.history,
                                    entry_data
                                )
                                logger.info(f"保存到历史记录: {uploaded_file.name}")
                                st.success(f"✅ 已保存到历史记录！当前记录数: {len(st.session_state.history)}")
                                st.rerun()
                        else:
                            st.info("🔒 历史记录保存已禁用")
                
                except ValueError as ve:
                    logger.error(f"值错误: {str(ve)}", exc_info=True)
                    st.error(f"❌ 图像格式错误: {str(ve)}")
                except cv2.error as cv_err:
                    logger.error(f"OpenCV错误: {str(cv_err)}", exc_info=True)
                    st.error(f"❌ 图像处理错误: OpenCV操作失败")
                except MemoryError:
                    logger.error("内存错误", exc_info=True)
                    st.error("❌ 内存不足: 图像过大，无法处理")
                except Exception as e:
                    logger.error(f"未知错误: {str(e)}", exc_info=True)
                    st.error(f"❌ 处理图像时出错: {str(e)}")

    with history_tab:
        st.subheader("📜 处理历史")
        ui.show_history_table(st.session_state.history)

    with db_tab:
        st.subheader("🗄️ 历史数据库")
        col_db1, col_db2, col_db3 = st.columns(3)
        with col_db1:
            if st.button("初始化数据库", use_container_width=True):
                history_manager.init_db()
                st.success("数据库已初始化")
        with col_db2:
            if st.button("从数据库加载历史", use_container_width=True):
                st.session_state.history = history_manager.load_history_from_db(APP_CONFIG["max_history_entries"])
                st.success("已从数据库加载历史")
                st.rerun()
        with col_db3:
            if st.button("清空数据库历史", use_container_width=True):
                history_manager.clear_history_db()
                st.success("数据库历史已清空")

        st.markdown("---")
        filters = ui.show_history_query_filters()
        start_ts = None
        end_ts = None
        if filters.get("start_date"):
            start_ts = f"{filters['start_date'].strftime('%Y-%m-%d')} 00:00:00"
        if filters.get("end_date"):
            end_ts = f"{filters['end_date'].strftime('%Y-%m-%d')} 23:59:59"
        do_query = st.button("查询", type="primary")
        if do_query:
            records = history_manager.load_history_from_db(
                APP_CONFIG["max_history_entries"],
                filters={
                    "filename_contains": filters.get("filename_contains"),
                    "color_scheme": filters.get("color_scheme"),
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                }
            )
            ui.show_history_query_results(records)

    with help_tab:
        with st.expander("❓ 使用说明与医学解释", expanded=False):
            st.markdown("""
            ### 🎯 使用说明
            
            1. **上传图像**: 在侧边栏选择或拖拽上传胸片图像
            2. **调整参数**: 在侧边栏调整颜色方案、对比度和亮度
            3. **查看结果**: 系统自动显示增强前后的对比图像
            4. **保存记录**: 可选是否保存处理记录到历史
            
            ### 🩺 医学解释
            
            **颜色编码说明**:
            - 🔵 **蓝色区域 (50-100)**: 正常肺组织，肺泡和支气管
            - 🟢 **绿色区域 (100-150)**: 实变区域，可能提示肺炎、肺水肿
            - 🟠 **橙色区域 (150-200)**: 血管和中等密度组织
            - 🔴 **红色区域 (200-255)**: 骨骼结构、钙化灶
            
            ### ⚠️ 重要提醒
            
            本系统仅用于教学演示和学术研究目的，**不能替代专业医疗诊断**！
            """)

    ui.create_footer(len(st.session_state.history))
    
    # 应用自定义样式
    st.markdown("""
    <style>
        .stButton > button {
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
