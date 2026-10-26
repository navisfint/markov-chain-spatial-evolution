# -*- coding: utf-8 -*-
"""
utils.py —— 公共工具
================================================
日志、配置加载、Excel/CSV 写出、中文表头整理。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 项目根目录（src 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------------------
_LOGGER_NAME = "markov"


def setup_logger(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """初始化日志。同时输出到控制台和文件。"""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(fmt)
    # Windows 控制台编码兜底
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def banner(logger: logging.Logger, title: str, char: str = "=", width: int = 74) -> None:
    """打一个醒目分隔块，方便用户在滚动窗口里定位阶段。"""
    logger.info(char * width)
    logger.info(f"  {title}")
    logger.info(char * width)


# ------------------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    """读 YAML 配置。没装 PyYAML 就直接报错提示怎么装。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "缺少 PyYAML，无法读取配置文件。请先执行： pip install PyYAML"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式不对，应该是一个字典：{path}")
    return cfg


def cfg_get(cfg: dict, *keys: str, default: Any = None) -> Any:
    """安全地按路径取配置值：cfg_get(cfg, 'spatial_weights', 'adjacency', 'mode')"""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


# ------------------------------------------------------------------------------
# 输出目录
# ------------------------------------------------------------------------------
class OutputManager:
    """统一管理输出路径，按类别分目录，避免上万文件堆一起。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.tables = self.root / "tables"
        self.figures = self.root / "figures"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

    def table_path(self, name: str, ext: str = "csv") -> Path:
        safe = _safe_filename(name)
        return self.tables / f"{safe}.{ext}"

    def figure_path(self, name: str, ext: str = "png") -> Path:
        safe = _safe_filename(name)
        return self.figures / f"{safe}.{ext}"


def _safe_filename(name: str) -> str:
    """把表格名里的非法字符替换掉，保留中文。"""
    bad = '<>:"/\\|?*'
    out = "".join(("_" if c in bad else c) for c in str(name))
    return out.strip() or "unnamed"


# ------------------------------------------------------------------------------
# 写出
# ------------------------------------------------------------------------------
def save_table(
    df: pd.DataFrame | pd.Series,
    path: str | Path,
    logger: logging.Logger | None = None,
    index: bool = True,
) -> Path:
    """保存一张表。按后缀自动选 csv / xlsx。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = df.to_frame() if isinstance(df, pd.Series) else df
    if path.suffix.lower() in (".xlsx", ".xls"):
        obj.to_excel(path, index=index)
    else:
        obj.to_csv(path, index=index, encoding="utf-8-sig")
    if logger:
        logger.debug(f"  已保存：{path}")
    return path


def save_excel_workbook(
    sheets: dict[str, pd.DataFrame],
    path: str | Path,
    logger: logging.Logger | None = None,
) -> Path | None:
    """
    把多张表写进一个 Excel 的多 sheet。
    sheet 名超过 31 字符或含非法字符会被截断/替换（Excel 限制）。
    写失败不致命，返回 None。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not sheets:
        return None
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                if df is None or (hasattr(df, "empty") and df.empty):
                    continue
                obj = df.to_frame() if isinstance(df, pd.Series) else df
                sheet = _safe_sheet_name(name)
                obj.to_excel(writer, sheet_name=sheet)
        if logger:
            logger.info(f"  汇总工作簿：{path}  （{len(sheets)} 个 sheet）")
        return path
    except Exception as e:
        if logger:
            logger.warning(f"  Excel 汇总写出失败（不影响 CSV 结果）：{e}")
        return None


def _safe_sheet_name(name: str) -> str:
    bad = "[]:*?/\\"
    out = "".join(("_" if c in bad else c) for c in str(name))
    return out[:31] or "Sheet"


# ------------------------------------------------------------------------------
# 数值格式化
# ------------------------------------------------------------------------------
def fmt_pct(x: float, digits: int = 2) -> str:
    """概率转百分数字符串。"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


def describe_array(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return "空"
    return (f"n={x.size}  min={x.min():.4f}  p25={np.percentile(x, 25):.4f}  "
            f"median={np.median(x):.4f}  p75={np.percentile(x, 75):.4f}  max={x.max():.4f}")


def class_label(i: int) -> str:
    """0→Ⅰ  1→Ⅱ  2→Ⅲ  3→Ⅳ"""
    roman = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ"]
    return roman[i] if 0 <= i < len(roman) else f"C{i + 1}"


def class_labels(n: int) -> list[str]:
    return [class_label(i) for i in range(n)]


def class_display_names(n: int) -> list[str]:
    """四档时的中文名（论文口径）。非四档用罗马数字。"""
    if n == 4:
        return ["低水平", "中低水平", "中高水平", "高水平"]
    return class_labels(n)


def class_full_labels(n: int) -> list[str]:
    """形如 'Ⅰ (低水平)' """
    if n == 4:
        names = ["低水平", "中低水平", "中高水平", "高水平"]
        return [f"{class_label(i)} ({names[i]})" for i in range(n)]
    return class_labels(n)
