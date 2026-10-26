# -*- coding: utf-8 -*-
"""
loader.py —— 数据读入与形态识别
================================================
职责：把用户随手丢进来的数据文件，统一成内部标准格式：

    DataFrame，列 = [province, year, value]
    - province : 标准省份名（已归一化）
    - year     : int
    - value    : 共同富裕指数（float）

支持四种输入形态（config 的 data.shape）：
  auto  —— 自动判断（推荐）
  long  —— 一行一个"省-年"，已有算好的指数列
  wide  —— 行=年份，列=省份，单元格是已算好的指数
  raw   —— 一行一个"省-年"，但只有原始指标，需要熵权-TOPSIS 合成指数

另外还提供：
  load_raw_panel()     读原始指标面板（供经济距离矩阵取人均GDP等变量用）
  entropy_topsis()     熵权-TOPSIS 合成指数

设计原则：任何一步失败都给"人话"报错，指明该改 config 哪一项。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from province_data import PROVINCES, normalize_province

logger = logging.getLogger("markov")

# 内部标准列名
C_PROV, C_YEAR, C_VALUE = "province", "year", "value"

# 省份列候选名（自动识别用）
_PROV_KEYS = ["province", "省份", "省", "地区", "region", "area", "省域", "省份名称", "地区名称", "id"]
# 年份列候选名
_YEAR_KEYS = ["year", "年份", "年", "时间", "yr", "date", "统计年份"]
# 指数列候选名
_VALUE_KEYS = ["cwt_index", "index", "value", "共同富裕", "共同富裕水平", "共同富裕指数",
               "得分", "score", "综合指数", "水平", "指数", "cfy_index", "gtfy"]


# ------------------------------------------------------------------------------
# 文件读取
# ------------------------------------------------------------------------------
def read_table(path: str | Path) -> pd.DataFrame:
    """读 csv / xlsx / xls。自动尝试多种编码。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{path}\n"
            f"  → 请把数据文件放到项目的 data/ 目录下，并在 config.yaml 的 data.file 里填写正确的文件名。"
        )
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            return pd.read_excel(path)
        except Exception as e:
            raise RuntimeError(f"读取 Excel 失败：{path}\n  原因：{e}") from e
    if suffix in (".csv", ".txt", ".tsv"):
        sep = "\t" if suffix == ".tsv" else None   # sep=None 让 pandas 自己嗅探逗号/分号/制表符
        last_err: Exception | None = None
        for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep, engine="python")
            except UnicodeDecodeError as e:
                last_err = e
                continue
            except Exception as e:
                raise RuntimeError(f"读取 CSV 失败：{path}\n  原因：{e}") from e
        raise RuntimeError(
            f"CSV 编码识别失败：{path}\n  已尝试 utf-8-sig / utf-8 / gbk / gb18030 / latin-1\n"
            f"  → 建议用 Excel 另存为「CSV UTF-8」格式后重试。原始错误：{last_err}"
        )
    raise ValueError(
        f"不支持的文件类型：{suffix}\n  → 请提供 .csv / .xlsx / .xls 之一。"
    )


# ------------------------------------------------------------------------------
# 列名识别
# ------------------------------------------------------------------------------
def _norm(s: object) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def find_col(df: pd.DataFrame, keys: list[str]) -> str | None:
    """在 df 列名里找最匹配 keys 中的一个列。先精确后模糊。"""
    nmap = {_norm(c): c for c in df.columns}
    for k in keys:
        if _norm(k) in nmap:
            return nmap[_norm(k)]
    for k in keys:
        nk = _norm(k)
        for ncol, col in nmap.items():
            if nk and nk in ncol:
                return col
    return None


def _looks_like_year(series: pd.Series) -> bool:
    """判断一列是不是年份。"""
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return False
    # 年份应在 1900-2200 之间，且为整数为主
    ok = ((valid >= 1900) & (valid <= 2200)).mean()
    if ok < 0.9:
        return False
    # 排除明显是"数值指标"的列：值域不应太宽且不该有小数
    return True


def _looks_like_province(series: pd.Series) -> bool:
    """判断一列是不是省份名：至少有 60% 的值能归一化成标准省份名。"""
    vals = series.dropna().astype(str).head(200)
    if vals.empty:
        return False
    hit = sum(1 for v in vals if normalize_province(v) is not None)
    return hit / len(vals) >= 0.6


def detect_shape(df: pd.DataFrame) -> str:
    """
    判断数据形态：long / raw / wide / wide_transposed。

    判定顺序（先看行，再看列，因为横竖两种宽表长得很像）：
      1. 有省份列 且有年份列
           → 有独立指数列 → long
           → 没有指数列   → raw（只有原始指标）
      2. 没有省份列：
           a. 行索引（或第一列）是省份名，且列名多为年份 → wide_transposed
           b. 列名里能认出 ≥3 个省份名，且行索引是第一列是年份 → wide
           c. 反过来再确认一次，避免宽/长两种排列误判
    """
    prov_col = find_col(df, _PROV_KEYS)
    year_col = find_col(df, _YEAR_KEYS)

    has_prov = prov_col is not None and _looks_like_province(df[prov_col])
    has_year = year_col is not None and _looks_like_year(df[year_col])

    if has_prov and has_year:
        val_col = find_col(df, _VALUE_KEYS)
        if val_col is not None and val_col not in (prov_col, year_col):
            return "long"
        return "raw"

    # ---- 没有明确的省份列：判断是宽表还是转置宽表 ----
    idx_vals = pd.Series(df.index.astype(str))
    first_col_vals: pd.Series | None = None
    if len(df.columns):
        try:
            first_col_vals = df[df.columns[0]].astype(str)
        except Exception:
            first_col_vals = None

    # 行方向特征
    idx_is_prov = _looks_like_province(idx_vals)
    idx_is_year = _looks_like_year(pd.to_numeric(idx_vals, errors="coerce"))
    first_is_prov = (first_col_vals is not None
                     and _looks_like_province(first_col_vals)
                     and not idx_is_prov)
    first_is_year = (first_col_vals is not None
                     and _looks_like_year(pd.to_numeric(first_col_vals,
                                                        errors="coerce")))

    # 列方向特征
    n_prov_cols = sum(1 for c in df.columns if normalize_province(c) is not None)
    n_year_cols = sum(1 for c in df.columns if _is_yearish(c))

    # a. 行是省份 + 列是年份 → 转置宽表
    if (idx_is_prov or first_is_prov) and n_year_cols >= 3:
        return "wide_transposed"
    if idx_is_prov and n_prov_cols < 3:
        # 行是省份，列名又不全是年份（可能是 "2000" 被读成 int 但列名丢了）
        return "wide_transposed"

    # b. 列是省份 → 常规宽表（行=年份）
    if n_prov_cols >= 3:
        if idx_is_year or first_is_year or True:   # 行是年份是最常见情形
            return "wide"

    # c. 再兜底判断一次
    if idx_is_year and n_prov_cols >= 3:
        return "wide"
    if idx_is_prov and n_year_cols >= 3:
        return "wide_transposed"

    return "unknown"
    return "unknown"


def _maybe_int(x: Any) -> Any:
    return x


def _is_yearish(x: Any) -> bool:
    try:
        v = float(str(x).strip())
    except Exception:
        return False
    return 1900 <= v <= 2200 and float(v).is_integer()


# ------------------------------------------------------------------------------
# 形态转换
# ------------------------------------------------------------------------------
def _to_long_from_long(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    col_map = cfg.get("col_map") or {}
    prov_col = col_map.get("province") or find_col(df, _PROV_KEYS)
    year_col = col_map.get("year") or find_col(df, _YEAR_KEYS)
    val_col = col_map.get("index_value") or find_col(df, _VALUE_KEYS)

    missing = [n for n, c in [("省份", prov_col), ("年份", year_col), ("指数", val_col)] if c is None]
    if missing:
        raise ValueError(
            f"长表模式识别列失败，缺：{'、'.join(missing)}\n"
            f"  → 请在 config.yaml 的 data.col_map 里显式指定列名。\n"
            f"  当前文件列名：{list(df.columns)}"
        )
    out = pd.DataFrame({
        C_PROV: df[prov_col].map(normalize_province),
        C_YEAR: pd.to_numeric(df[year_col], errors="coerce").astype("Int64"),
        C_VALUE: pd.to_numeric(df[val_col], errors="coerce"),
    })
    return out


def _to_long_from_wide(df: pd.DataFrame, transposed_hint: bool = False) -> pd.DataFrame:
    """宽表 → 长表。行=年份、列=省份；若 transposed_hint 则行=省份、列=年份。"""
    work = df.copy()
    # 若第一列/索引是年份或省份名，先收拾干净
    if transposed_hint:
        # 行是省份、列是年份
        first_col = find_col(work, _PROV_KEYS)
        if first_col is None:
            # 省份在索引上
            work.index.name = "__prov"
            work = work.reset_index()
            first_col = "__prov"
        else:
            # 把省份列改名为固定名，避免与列名冲突
            work = work.rename(columns={first_col: "__prov"})
            first_col = "__prov"
        long = work.melt(id_vars=[first_col], var_name="year",
                         value_name="value_raw")
        long = long.rename(columns={first_col: "province_raw"})
        long = long.dropna(subset=["value_raw"])
    else:
        # 行是年份
        year_col = find_col(work, _YEAR_KEYS)
        if year_col is not None:
            work = work.set_index(year_col)
        else:
            # 尝试用索引当年份
            idx = pd.to_numeric(pd.Series(work.index), errors="coerce")
            if idx.notna().mean() > 0.8 and ((idx >= 1900) & (idx <= 2200)).mean() > 0.8:
                pass  # 索引就是年份，直接用
            else:
                work = work.reset_index(drop=True)

        # ★ 关键：只保留能归一化成省份的列。
        #   注意 set_index 之后索引可能是"未命名"或"year"，
        #   展开时列名会跟着变，所以不能硬写 "index"。
        prov_cols = [c for c in work.columns if normalize_province(c) is not None]
        if len(prov_cols) < 3:
            raise ValueError(
                "宽表模式下，至少要有 3 列能识别为省份名。\n"
                f"  当前识别的省份列：{prov_cols}\n"
                f"  当前文件列名：{list(df.columns)[:20]}\n"
                "  → 请检查列名是否是省份名，或改用长表（config: data.shape='long'）。"
            )
        work = work[prov_cols]

        # 年份放在索引上，明确命名为 __year，避免与已存在的列名冲突
        work.index = pd.to_numeric(pd.Series(work.index), errors="coerce")
        work = work[work.index.notna()]
        work.index.name = "__year"

        long = (work.reset_index()
                    .melt(id_vars=["__year"], var_name="province_raw",
                          value_name="value_raw")
                    .rename(columns={"__year": "year"}))
        long = long.dropna(subset=["value_raw"])

    out = pd.DataFrame({
        C_PROV: long["province_raw"].map(lambda v: normalize_province(v) or str(v)),
        C_YEAR: pd.to_numeric(long["year"], errors="coerce").astype("Int64"),
        C_VALUE: pd.to_numeric(long["value_raw"], errors="coerce"),
    })
    return out


def _guess_year_row_labels(df: pd.DataFrame) -> bool:
    """行索引（或第一列）是不是年份。"""
    cand = df.index.astype(str).tolist()
    try:
        first_col = df.columns[0]
        cand += [str(v) for v in df[first_col].astype(str).tolist()]
    except Exception:
        pass
    n_yearish = sum(1 for v in cand if _is_yearish(v))
    return n_yearish >= 3


# ------------------------------------------------------------------------------
# 缺失值处理
# ------------------------------------------------------------------------------
def _handle_missing(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """
    缺值处理。

    ★ 契约：输入整表，返回整表（数据行可能减少，列不变）。
      不要返回 Series —— 早期版本在"无缺值"时直接 return df（整表），
      而调用方按 Series 用，于是在 pandas 3.0 下报
      「Columns must be same length as key」。这里统一返回整表。

    ★ pandas 3.0 兼容性说明：分组插值用 transform，不用 groupby().apply() ——
      apply 在 pandas 3.0 下返回 DataFrame 而非 Series，赋值会失败。
    """
    n_before = len(df)
    n_na = int(df[C_VALUE].isna().sum())
    if n_na == 0:
        return df

    how = str(how).lower()
    out = df.sort_values([C_PROV, C_YEAR]).copy()
    g = out.groupby(C_PROV, sort=False)[C_VALUE]

    if how == "drop":
        out = out.dropna(subset=[C_VALUE])
    elif how == "fill_mean":
        out[C_VALUE] = g.transform("mean")
        out[C_VALUE] = out[C_VALUE].fillna(df[C_VALUE].mean())
    else:  # interpolate（默认）
        # 组内线性插值：transform 返回与 out 等长的 Series
        out[C_VALUE] = g.transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        # 某省全为 NaN 时（组内无可插值信息），用该省均值 → 再退到全体均值
        out[C_VALUE] = out[C_VALUE].fillna(g.transform("mean"))
        out[C_VALUE] = out[C_VALUE].fillna(df[C_VALUE].mean())

    out = out.dropna(subset=[C_VALUE])
    logger.warning(f"  缺值处理（{how}）：原始 {n_na} 个缺失，样本 {n_before} → {len(out)}")
    return out


# ------------------------------------------------------------------------------
# 数据质量检查
# ------------------------------------------------------------------------------
def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """检查并报告面板数据的完整性（是否平衡、年份是否连续、省份是否齐全）。"""
    dup = df.duplicated(subset=[C_PROV, C_YEAR]).sum()
    if dup:
        logger.warning(f"  发现 {dup} 条重复的「省份-年份」记录，已保留最后一条")
        df = df.drop_duplicates(subset=[C_PROV, C_YEAR], keep="last")

    provinces = sorted(df[C_PROV].unique().tolist())
    years = sorted(df[C_YEAR].dropna().unique().tolist())
    n_expect = len(provinces) * len(years)

    extra = set(provinces) - set(PROVINCES)
    missing_p = set(PROVINCES) - set(provinces)

    logger.info(f"  省份数：{len(provinces)}  |  年份：{int(years[0])}–{int(years[-1])}（{len(years)} 年）")
    logger.info(f"  样本量：{len(df)}（平衡面板应为 {n_expect}）")

    if missing_p:
        logger.warning(f"  缺少省份（占位为空）：{'、'.join(sorted(missing_p))}")
        logger.warning("    → 马尔科夫链可以只跑已有省份，但空间权重矩阵会退化为子矩阵。"
                       "建议补齐 31 省数据。")
    if extra:
        logger.warning(f"  以下名称不在 31 省标准名单内，将被保留但视为额外个体："
                       f"{'、'.join(sorted(extra))}")

    # 年份连续性
    yarr = np.array(years, dtype=int)
    gaps = np.diff(yarr)
    if len(gaps) and gaps.max() > 1:
        gap_positions = [int(yarr[i]) for i in range(len(gaps)) if gaps[i] > 1]
        logger.warning(f"  年份不连续，缺失中间年份：{gap_positions}（缺失年之后的那一年）")
        logger.warning("    → 马尔科夫链按「相邻观测」转移，年份断裂会把跨年转移算成 1 步转移。"
                       "若断裂较大，建议在 config 用 data.shape 修正数据或补全年份。")

    # 不平衡面板
    if len(df) != n_expect:
        logger.warning("  面板不平衡，马尔科夫链仍可运行（按每个省份自身的年份序列转移）")

    return df.sort_values([C_PROV, C_YEAR]).reset_index(drop=True)


# ------------------------------------------------------------------------------
# 熵权 - TOPSIS
# ------------------------------------------------------------------------------
def entropy_topsis(
    panel: pd.DataFrame,
    indicator_cols: list[str],
    directions: dict[str, str] | None = None,
    default_direction: str = "positive",
) -> pd.Series:
    """
    熵权-TOPSIS 法合成综合指数。

    参数
    ----
    panel : DataFrame，需含 province / year 两列，以及 indicator_cols 里的指标列。
            本函数按"全样本混合"求权重（论文口径：面板整体赋权）。
    indicator_cols : 参与合成的指标列名列表
    directions : {列名: 'positive'|'negative'}，未指定的按 default_direction
    default_direction : 'positive' 或 'negative'

    返回
    ----
    Series，索引与 panel 一致，值为 TOPSIS 相对贴近度 C（0~1，越大越好）

    步骤
    ----
    1. 负向指标逆向化（x' = 1 - 归一化后的 x，在归一化之后处理）
    2. 归一化到 [0,1]（min-max），消除量纲
    3. 计算各指标信息熵 e_j 与权重 w_j = (1-e_j) / Σ(1-e_j)
    4. 加权归一化矩阵 → 正理想解 V+ / 负理想解 V-
    5. 计算到正负理想解的欧氏距离 D+、D-，贴近度 C = D- / (D+ + D-)
    """
    directions = directions or {}
    missing = [c for c in indicator_cols if c not in panel.columns]
    if missing:
        raise ValueError(
            f"熵权-TOPSIS 需要的指标列不存在：{missing}\n"
            f"  → 请检查 config.yaml 的 data.raw_indicators.indicator_cols，"
            f"或改用 mode: all_numeric。\n"
            f"  当前文件列名：{list(panel.columns)}"
        )
    if len(indicator_cols) < 2:
        raise ValueError(
            "熵权-TOPSIS 至少需要 2 个指标。\n"
            "  → 若你只有 1 个指标（如人均GDP），直接用该列作为指数即可，"
            "把 config 的 data.shape 设为 long 并指定 index_value。"
        )

    n = len(panel)
    X = panel[indicator_cols].astype(float).copy()

    if X.isna().any().any():
        n_na = int(X.isna().sum().sum())
        logger.warning(f"  指标矩阵有 {n_na} 个缺失值，按列中位数填充后再合成")
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    # --- 步骤 1+2：归一化，负向指标逆向化 ---
    Xn = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in indicator_cols:
        col = X[c].astype(float)
        lo, hi = float(col.min()), float(col.max())
        rng = hi - lo
        if rng <= 1e-12:
            # 该指标所有值相同，无信息量，给个 0.5 的中性值
            logger.warning(f"  指标「{c}」在样本内无变异（全为同一值），权重会趋近 0")
            Xn[c] = 0.5
            continue
        z = (col - lo) / rng                      # min-max 归一化到 [0,1]
        direction = str(directions.get(c, default_direction)).lower()
        if direction.startswith("neg"):
            z = 1.0 - z                            # 逆向化
        Xn[c] = z

    # 平移一个极小量，避免 log(0)
    eps = 1e-6
    Xs = Xn + eps

    # --- 步骤 3：信息熵与权重 ---
    P = Xs / Xs.sum(axis=0)                      # 列归一化，Σp_ij = 1
    k = 1.0 / np.log(n) if n > 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    e = -k * plogp.sum(axis=0)                   # 各指标信息熵，0<=e<=1
    e = pd.Series(np.clip(e, 0.0, 1.0), index=indicator_cols)
    d = 1.0 - e                                  # 差异系数
    if d.sum() <= 1e-12:
        logger.warning("  所有指标的差异系数之和为 0，退化为等权重")
        w = pd.Series(np.ones(len(indicator_cols)) / len(indicator_cols), index=indicator_cols)
    else:
        w = d / d.sum()

    # --- 步骤 4：加权归一化决策矩阵 ---
    V = Xn * w

    # --- 步骤 5：TOPSIS 贴近度 ---
    v_pos = V.max(axis=0)
    v_neg = V.min(axis=0)
    d_pos = np.sqrt(((V - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - v_neg) ** 2).sum(axis=1))
    denom = d_pos + d_neg
    c = np.where(denom > 1e-12, d_neg / denom, 0.5)
    closeness = pd.Series(c, index=panel.index, name="value")

    # 把权重和熵值记在 Series 上，方便主流程输出
    closeness.attrs["entropy_weights"] = w.to_dict()
    closeness.attrs["entropies"] = e.to_dict()
    return closeness


def build_raw_indicator_panel(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """
    从原始数据里整理出指标面板，返回 (panel, indicator_cols, directions)。
    供 entropy_topsis 使用。
    """
    ri = (cfg.get("data") or {}).get("raw_indicators") or {}
    mode = str(ri.get("mode", "explicit")).lower()

    col_map = cfg.get("col_map") or {}
    prov_col = col_map.get("province") or find_col(df, _PROV_KEYS)
    year_col = col_map.get("year") or find_col(df, _YEAR_KEYS)
    if prov_col is None or year_col is None:
        raise ValueError(
            "原始指标模式需要明确的省份列和年份列。\n"
            f"  → 请在 config.yaml 的 data.col_map 里指定 province / year。\n"
            f"  当前文件列名：{list(df.columns)}"
        )

    panel = df.copy()
    panel["__prov"] = panel[prov_col].map(normalize_province)
    panel["__year"] = pd.to_numeric(panel[year_col], errors="coerce")
    panel = panel[panel["__prov"].notna() & panel["__year"].notna()].copy()
    panel["__year"] = panel["__year"].astype(int)

    if mode == "all_numeric":
        exclude = {prov_col, year_col, "__prov", "__year"}
        cols = [c for c in panel.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(panel[c])]
    else:
        cols = list(ri.get("indicator_cols") or [])
        if not cols:
            raise ValueError(
                "data.raw_indicators.mode = 'explicit' 时必须给出 indicator_cols 列表。\n"
                "  → 或者在 config 里把 mode 改为 'all_numeric'（自动用全部数值列）。"
            )

    directions = {str(k): str(v) for k, v in (ri.get("directions") or {}).items()}
    default_dir = str(ri.get("default_direction", "positive"))
    return panel, cols, directions, default_dir  # type: ignore[return-value]


# ------------------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------------------
def load_panel(
    path: str | Path,
    cfg: dict,
    shape_override: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    读入并标准化数据。

    返回
    ----
    panel : DataFrame，列 = [province, year, value]
    info  : dict，含 shape / entropies / weights / n_provinces / years 等元信息
    """
    df = read_table(path)
    if df.empty:
        raise ValueError(f"数据文件是空的：{path}")

    # 去掉全空行/列
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"去掉空行空列后数据为空：{path}")

    dcfg = (cfg.get("data") or {})
    shape = (shape_override or dcfg.get("shape") or "auto").lower()
    if shape == "auto":
        shape = detect_shape(df)
        logger.info(f"  自动识别数据形态：{shape}")
    else:
        logger.info(f"  使用配置指定的数据形态：{shape}")

    info: dict[str, Any] = {"shape": shape, "source_file": str(path)}

    # ---- raw：先用熵权-TOPSIS 合成指数 ----
    if shape == "raw":
        panel_raw, cols, directions, default_dir = build_raw_indicator_panel(df, cfg)  # type: ignore[misc]
        logger.info(f"  熵权-TOPSIS 合成指数：{len(cols)} 个指标，"
                    f"方向默认「{default_dir}」")
        series = entropy_topsis(panel_raw, cols, directions, default_dir)
        panel = pd.DataFrame({
            C_PROV: panel_raw["__prov"].values,
            C_YEAR: panel_raw["__year"].values,
            C_VALUE: series.values,
        })
        info["entropies"] = series.attrs.get("entropies", {})
        info["weights"] = series.attrs.get("entropy_weights", {})
        w = info["weights"]
        if w:
            top = sorted(w.items(), key=lambda kv: -kv[1])[:8]
            logger.info("  权重最高的 8 个指标：" +
                        "、".join(f"{k}({v:.3f})" for k, v in top))

    # ---- long：已有指数 ----
    elif shape == "long":
        panel = _to_long_from_long(df, dcfg)

    # ---- wide：行=年、列=省 ----
    elif shape == "wide":
        panel = _to_long_from_wide(df, transposed_hint=False)

    # ---- wide_transposed：行=省、列=年 ----
    elif shape in ("wide_transposed", "wide_T"):
        panel = _to_long_from_wide(df, transposed_hint=True)

    else:
        raise ValueError(
            f"无法识别数据形态（判定结果：{shape}）。\n"
            f"  当前文件列名：{list(df.columns)[:30]}\n"
            f"  行数：{len(df)}\n"
            "  → 请在 config.yaml 的 data.shape 里手动指定：long / wide / raw。\n"
            "     long = 一行一个「省-年」，有指数列\n"
            "     wide = 行是年份、列是省份，格子里是指数值\n"
            "     raw  = 一行一个「省-年」，只有原始指标，需自动合成指数"
        )

    if panel[C_PROV].isna().any():
        n_bad = int(panel[C_PROV].isna().sum())
        logger.warning(f"  有 {n_bad} 行省份名无法识别，已剔除（可在日志 DEBUG 级别查看原始名称）")
        bad_names = df.head(50).to_string()
        logger.debug(f"  原始数据前 50 行：\n{bad_names}")
        panel = panel.dropna(subset=[C_PROV])

    panel[C_YEAR] = pd.to_numeric(panel[C_YEAR], errors="coerce")
    panel = panel.dropna(subset=[C_YEAR, C_VALUE])
    panel[C_YEAR] = panel[C_YEAR].astype(int)

    if panel.empty:
        raise ValueError(
            "清理后没有有效样本。\n"
            "  → 常见原因：省份名/年份列识别错、指数列全是空值。\n"
            "  → 请在 config.yaml 的 data.col_map 里显式指定列名后重试。"
        )

    # _handle_missing 输入整表、返回整表（可能是剔除缺值后的子集）
    panel = _handle_missing(panel, str(dcfg.get("missing", "interpolate")))
    panel = validate_panel(panel)

    info["n_provinces"] = int(panel[C_PROV].nunique())
    info["n_years"] = int(panel[C_YEAR].nunique())
    info["year_min"] = int(panel[C_YEAR].min())
    info["year_max"] = int(panel[C_YEAR].max())
    info["n_obs"] = int(len(panel))
    info["provinces"] = sorted(panel[C_PROV].unique().tolist())
    info["years"] = sorted(panel[C_YEAR].unique().tolist())

    logger.info(f"  指数取值范围：min={panel[C_VALUE].min():.4f}  "
                f"max={panel[C_VALUE].max():.4f}  "
                f"mean={panel[C_VALUE].mean():.4f}  "
                f"sd={panel[C_VALUE].std():.4f}")

    return panel, info


def to_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """面板 → 矩阵：行=年份，列=省份，值=指数。缺失格留 NaN。"""
    m = panel.pivot_table(index=C_YEAR, columns=C_PROV, values=C_VALUE, aggfunc="mean")
    m.index.name = "year"
    return m
