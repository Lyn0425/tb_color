# modules/history_manager.py

import datetime
import json
import sqlite3
import numpy as np
import cv2
import logging
from typing import Optional
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from io import BytesIO

logger = logging.getLogger(__name__)



@dataclass
class HistoryEntry:
    """历史记录条目数据类"""
    timestamp: str
    filename: str
    color_scheme: str
    stats: Dict[str, Any]
    original_shape: tuple
    enhanced_shape: tuple
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoryEntry':
        """从字典创建实例"""
        return cls(**data)


class HistoryManager:
    """历史记录管理器"""
    
    def __init__(self, max_entries: int = 20, db_enabled: bool = False, db_path: str = "medical_images.db", db_type: str = "sqlite", mysql_config: Optional[Dict[str, Any]] = None):
        self.max_entries = max_entries
        self.db_enabled = db_enabled
        self.db_path = db_path
        self.db_type = db_type
        self.mysql_config = mysql_config or {}
        self._mysql_pool = None  # MySQL连接池
        # 不再使用持久化的SQLite连接，每次操作都创建新连接

    def set_db_config(self, enabled: bool, db_type: str, path: str, mysql_config: Optional[Dict[str, Any]] = None):
        self.db_enabled = enabled
        self.db_type = db_type
        self.db_path = path
        if mysql_config:
            self.mysql_config = mysql_config
        # 重置MySQL连接池
        self._mysql_pool = None

    def _get_conn(self, db_name: Optional[str] = None):
        if self.db_type == "mysql":
            import mysql.connector.pooling
            if self._mysql_pool is None:
                # 创建MySQL连接池
                cfg = {
                    "host": self.mysql_config.get("host", "localhost"),
                    "port": int(self.mysql_config.get("port", 3306)),
                    "user": self.mysql_config.get("user", "root"),
                    "password": self.mysql_config.get("password", "liu123"),
                    "charset": "utf8mb4",
                    "connect_timeout": 10
                }
                if not db_name and self.mysql_config.get("database"):
                    cfg["database"] = self.mysql_config.get("database")
                elif db_name:
                    cfg["database"] = db_name
                
                try:
                    self._mysql_pool = mysql.connector.pooling.MySQLConnectionPool(
                        pool_name="medical_images_pool",
                        pool_size=5,
                        pool_reset_session=True,
                        **cfg
                    )
                    logger.info("MySQL连接池创建成功")
                except Exception as e:
                    logger.error(f"创建MySQL连接池失败: {e}")
                    # 失败时返回单个连接
                    try:
                        return mysql.connector.connect(**cfg)
                    except Exception as ex:
                        logger.error(f"创建单个MySQL连接失败: {ex}")
                        raise
            
            try:
                conn = self._mysql_pool.get_connection()
                logger.debug("从连接池获取MySQL连接成功")
                return conn
            except Exception as e:
                logger.error(f"从连接池获取MySQL连接失败: {e}")
                # 连接池获取失败时创建新连接
                cfg = {
                    "host": self.mysql_config.get("host", "localhost"),
                    "port": int(self.mysql_config.get("port", 3306)),
                    "user": self.mysql_config.get("user", "root"),
                    "password": self.mysql_config.get("password", "liu123"),
                    "charset": "utf8mb4",
                    "connect_timeout": 10
                }
                if db_name:
                    cfg["database"] = db_name
                elif self.mysql_config.get("database"):
                    cfg["database"] = self.mysql_config.get("database")
                
                try:
                    return mysql.connector.connect(**cfg)
                except Exception as ex:
                    logger.error(f"创建单个MySQL连接失败: {ex}")
                    raise
        
        # SQLite连接 - 每次都创建新连接，不再使用持久化连接
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.debug("创建SQLite连接成功")
            return conn
        except Exception as e:
            logger.error(f"创建SQLite连接失败: {e}")
            raise

    def init_db(self):
        if self.db_type == "mysql":
            try:
                #连接不带数据库
                conn_no_db =self._get_conn(db_name="")
                cur_no_db = conn_no_db.cursor() 
                target_db = self.mysql_config.get("database", "medical_images")
                #创建
                cur_no_db.execute(
                    f"CREATE DATABASE IF NOT EXISTS {target_db}"
                )
                conn_no_db.commit()
                cur_no_db.close()
                conn_no_db.close()
            except Exception as e:
                print(f"Error connecting to MySQL: {e}")    
        
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            if self.db_type == "mysql":
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp VARCHAR(32),
                        filename VARCHAR(255),
                        color_scheme VARCHAR(64),
                        stats_json TEXT,
                        original_width INT,
                        original_height INT,
                        enhanced_width INT,
                        enhanced_height INT
                    )
                    """
                )
            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        filename TEXT,
                        color_scheme TEXT,
                        stats_json TEXT,
                        original_width INTEGER,
                        original_height INTEGER,
                        enhanced_width INTEGER,
                        enhanced_height INTEGER
                    )
                    """
                )
            conn.commit()
        finally:
            conn.close()
    
    def add_entry(self, history_list: List[Dict], entry_data: Dict) -> List[Dict]:
        """添加新的历史记录"""
        # 创建历史记录条目
        entry = HistoryEntry(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename=entry_data.get("filename", "unknown"),
            color_scheme=entry_data.get("color_scheme", "标准"),
            stats=entry_data.get("stats", {}),
            original_shape=entry_data.get("original_shape", (0, 0)),
            enhanced_shape=entry_data.get("enhanced_shape", (0, 0))
        )
        
        # 添加到列表开头
        history_list.insert(0, entry.to_dict())
        
        if self.db_enabled:
            self.save_entry_to_db(entry)
        
        # 限制记录数量
        if len(history_list) > self.max_entries:
            history_list = history_list[:self.max_entries]
        
        return history_list
    
    def save_entries_to_db(self, entries: List[HistoryEntry]):
        """批量保存历史记录到数据库"""
        if not entries:
            return
        
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            placeholder = "%s" if self.db_type == "mysql" else "?"
            sql = f"INSERT INTO history (timestamp, filename, color_scheme, stats_json, original_width, original_height, enhanced_width, enhanced_height) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})"
            
            # 准备批量插入的数据
            data = []
            for entry in entries:
                stats_json = json.dumps(entry.stats, ensure_ascii=False)
                oh, ow = entry.original_shape[:2] if len(entry.original_shape) >= 2 else (0, 0)
                eh, ew = entry.enhanced_shape[:2] if len(entry.enhanced_shape) >= 2 else (0, 0)
                data.append((entry.timestamp, entry.filename, entry.color_scheme, stats_json, int(ow), int(oh), int(ew), int(eh)))
            
            # 执行批量插入
            cur.executemany(sql, data)
            conn.commit()
        finally:
            conn.close()
    
    def clear_history(self) -> List:
        """清空历史记录"""
        return []
    
    def get_recent_entries(self, history_list: List[Dict], count: int = 5) -> List[Dict]:
        """获取最近的记录"""
        return history_list[:count]
    
    def export_to_json(self, history_list: List[Dict]) -> str:
        """导出历史记录为JSON"""
        return json.dumps(history_list, indent=2, ensure_ascii=False)
    
    def import_from_json(self, json_str: str) -> List[Dict]:
        """从JSON导入历史记录"""
        return json.loads(json_str)

    def save_entry_to_db(self, entry: HistoryEntry):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            stats_json = json.dumps(entry.stats, ensure_ascii=False)
            ow = 0
            oh = 0
            ew = 0
            eh = 0
            if isinstance(entry.original_shape, tuple):
                if len(entry.original_shape) >= 2:
                    oh = int(entry.original_shape[0])
                    ow = int(entry.original_shape[1])
            if isinstance(entry.enhanced_shape, tuple):
                if len(entry.enhanced_shape) >= 2:
                    eh = int(entry.enhanced_shape[0])
                    ew = int(entry.enhanced_shape[1])
            placeholder = "%s" if self.db_type == "mysql" else "?"
            sql = f"INSERT INTO history (timestamp, filename, color_scheme, stats_json, original_width, original_height, enhanced_width, enhanced_height) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})"
            cur.execute(sql, (
                entry.timestamp,
                entry.filename,
                entry.color_scheme,
                stats_json,
                ow,
                oh,
                ew,
                eh,
            ))
            conn.commit()
        finally:
            conn.close()

    def load_history_from_db(self, limit: Optional[int] = None, filters: Optional[Dict[str, Any]] = None, page: int = 1, page_size: int = 10) -> List[Dict]:
        """从数据库加载历史记录，支持分页"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            placeholder = "%s" if self.db_type == "mysql" else "?"
            q = "SELECT timestamp, filename, color_scheme, stats_json, original_width, original_height, enhanced_width, enhanced_height FROM history"
            where = []
            params: List[Any] = []
            if filters:
                if filters.get("filename_contains"):
                    where.append(f"filename LIKE {placeholder}")
                    params.append(f"%{filters['filename_contains']}%")
                if filters.get("color_scheme") and filters["color_scheme"] != "全部":
                    where.append(f"color_scheme = {placeholder}")
                    params.append(filters["color_scheme"])
                if filters.get("start_ts"):
                    where.append(f"timestamp >= {placeholder}")
                    params.append(filters["start_ts"])
                if filters.get("end_ts"):
                    where.append(f"timestamp <= {placeholder}")
                    params.append(filters["end_ts"])
            if where:
                q += " WHERE " + " AND ".join(where)
            q += " ORDER BY id DESC"
            
            # 实现分页
            if page and page_size:
                offset = (page - 1) * page_size
                if self.db_type == "mysql":
                    q += f" LIMIT {page_size} OFFSET {offset}"
                else:  # SQLite
                    q += f" LIMIT {page_size} OFFSET {offset}"
            elif limit and isinstance(limit, int):
                q += f" LIMIT {limit}"
            
            cur.execute(q, tuple(params))
            rows = cur.fetchall()
            result: List[Dict[str, Any]] = []
            for r in rows:
                stats = {}
                try:
                    stats = json.loads(r[3]) if r[3] else {}
                except Exception:
                    stats = {}
                result.append(
                    HistoryEntry(
                        timestamp=r[0],
                        filename=r[1],
                        color_scheme=r[2],
                        stats=stats,
                        original_shape=(int(r[5]), int(r[4])),
                        enhanced_shape=(int(r[7]), int(r[6])),
                    ).to_dict()
                )
            return result
        finally:
            conn.close()
    
    def count_history_entries(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """统计符合条件的历史记录数量"""
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            placeholder = "%s" if self.db_type == "mysql" else "?"
            q = "SELECT COUNT(*) FROM history"
            where = []
            params: List[Any] = []
            if filters:
                if filters.get("filename_contains"):
                    where.append(f"filename LIKE {placeholder}")
                    params.append(f"%{filters['filename_contains']}%")
                if filters.get("color_scheme") and filters["color_scheme"] != "全部":
                    where.append(f"color_scheme = {placeholder}")
                    params.append(filters["color_scheme"])
                if filters.get("start_ts"):
                    where.append(f"timestamp >= {placeholder}")
                    params.append(filters["start_ts"])
                if filters.get("end_ts"):
                    where.append(f"timestamp <= {placeholder}")
                    params.append(filters["end_ts"])
            if where:
                q += " WHERE " + " AND ".join(where)
            
            cur.execute(q, tuple(params))
            count = cur.fetchone()[0]
            return int(count) if count else 0
        finally:
            conn.close()

    def clear_history_db(self):
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM history")
            conn.commit()
        finally:
            conn.close()
