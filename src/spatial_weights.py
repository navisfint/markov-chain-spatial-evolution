# -*- coding: utf-8 -*-
"""
spatial_weights.py —— 空间权重矩阵（可插拔）
================================================
这一块的设计目标：**换矩阵只改配置，不动代码**。

支持的矩阵
-----------
  adjacency       邻接矩阵      W_ij = 1 若 i、j 相邻，否则 0
  geographic      地理距离矩阵   W_ij = 1 / d_ij^power（d 为省会大圆距离，km）
  economic        经济距离矩阵   W_ij = 1 / |econ_i - econ_j|（econ 为指定经济变量）
  eco_geographic  经济地理嵌套   W = 地理权重 ⊙ 经济权重（元素相乘，最常用）
  custom          自定义矩阵     直接从 CSV 读用户手搓的矩阵

统一处理（对所有矩阵生效，由 config 控制）
------------------------------------------
  standardize_rows : 行标准化 W_ij / Σ_j W_ij  ← 空间马尔科夫链必须做
  diagonal_zero    : 对角线置 0
  symmetry         : auto 保持 / average 对称化 / max 取大

输出统一为 numpy 矩阵 + 省份顺序列表，方便与 classifier 的标签对齐。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from province_data import (
    PROVINCE_COORDS,
    PROVINCES,
    build_adjacency_edges,
    normalize_province,
)

logger = logging.getLogger("markov")

_EARTH_RADIUS_KM = 6371.0088


# ------------------------------------------------------------------------------
# 结果容器
# ------------------------------------------------------------------------------
@dataclass
class WeightMatrix:
    name: str
    provinces: list[str]           # 行/列顺序
    W: np.ndarray                  # 原始（未标准化）矩阵
    W_norm: np.ndarray             # 行标准化后的矩阵（下游用这个）
    meta: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_frame(self, normalized: bool = False) -> pd.DataFrame:
        M = self.W_norm if normalized else self.W
        return pd.DataFrame(M, index=self.provinces, columns=self.provinces)

    @property
    def n(self) -> int:
        return len(self.provinces)

    def density(self) -> float:
        """非零权重占比（邻接矩阵里反映连通密度）。"""
        M = self.W
        n = M.shape[0]
        if n <= 1:
            return 0.0
        return float((M > 1e-12).sum() / (n * (n - 1)))

    def neighbor_count(self) -> pd.Series:
        """每个省份的邻居数（非零权重的列数）。"""
        return pd.Series((self.W > 1e-12).sum(axis=1), index=self.provinces, name="邻居数")

    def summary(self) -> str:
        lines = [
            f"矩阵：{self.name}",
            f"  维度：{self.n} × {self.n}",
            f"  非零密度：{self.density() * 100:.1f}%",
        ]
        nc = self.neighbor_count()
        lines.append(f"  邻居数：min={nc.min()}  median={int(nc.median())}  max={nc.max()}")
        isolate = sorted(nc[nc == 0].index.tolist())
        if isolate:
            lines.append(f"  孤立省份（无邻居）：{'、'.join(isolate)}")
        for n in self.notes:
            lines.append(f"  提示：{n}")
        return "\n".join(lines)


# ------------------------------------------------------------------------------
# 地理距离
# ------------------------------------------------------------------------------
def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """两点大圆距离（km）。"""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * _EARTH_RADIUS_KM * np.arcsin(min(1.0, np.sqrt(a))))


def distance_matrix(provinces: list[str]) -> pd.DataFrame:
    """省会两两距离矩阵（km），对称，对角线为 1e-9（避免除零，后面会被置 0）。"""
    n = len(provinces)
    D = np.zeros((n, n), dtype=float)
    missing = [p for p in provinces if p not in PROVINCE_COORDS]
    if missing:
        raise KeyError(
            f"以下省份缺少经纬度，无法算地理距离：{missing}\n"
            "  → 请检查省份名是否为标准 31 省名称；若是港澳台或直辖市下属区县，"
            "本项目不覆盖。"
        )
    for i in range(n):
        for j in range(i + 1, n):
            lon1, lat1 = PROVINCE_COORDS[provinces[i]]
            lon2, lat2 = PROVINCE_COORDS[provinces[j]]
            d = haversine_km(lon1, lat1, lon2, lat2)
            D[i, j] = D[j, i] = d
    np.fill_diagonal(D, 0.0)
    return pd.DataFrame(D, index=provinces, columns=provinces)


# ------------------------------------------------------------------------------
# 各矩阵构造
# ------------------------------------------------------------------------------
def build_adjacency(
    provinces: list[str],
    mode: str = "rook",
    isolate_fix: str = "nearest",
    manual_edges: list | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    邻接矩阵。W_ij = 1 若相邻。

    mode:
      rook   —— 有共同陆地边界（默认）
      bishop —— 更宽容：rook 基础上，再补充"距离 800km 内"视为相邻（能连上岛屿与近邻）
      manual —— 只用 manual_edges 里手写的边
    isolate_fix:
      对没有邻居的省份（典型是海南）：
        nearest —— 连地理距离最近的省份
        none    —— 保留为孤立点（此时空间矩阵里该省的行为等价于"无邻居"）
    """
    n = len(provinces)
    idx = {p: i for i, p in enumerate(provinces)}
    W = np.zeros((n, n), dtype=float)

    if mode == "manual":
        edges = []
        for e in (manual_edges or []):
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                a, b = normalize_province(e[0]), normalize_province(e[1])
                if a and b and a in idx and b in idx:
                    edges.append((a, b))
        if not edges:
            raise ValueError(
                "邻接矩阵 mode='manual' 但 manual_edges 为空或全部无法识别。\n"
                "  → 请按 [['北京','天津'], ['广东','海南']] 的格式填写。"
            )
    else:
        edges = build_adjacency_edges(only=provinces)

    for a, b in edges:
        if a in idx and b in idx:
            W[idx[a], idx[b]] = 1.0
            W[idx[b], idx[a]] = 1.0

    notes: list[str] = []

    if mode == "bishop":
        # 补充：距离 < 800km 也视为相邻（把岛屿、隔海相望的省份连上）
        D = distance_matrix(provinces).to_numpy()
        extra = 0
        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] == 0 and 0 < D[i, j] <= 800:
                    W[i, j] = W[j, i] = 1.0
                    extra += 1
        notes.append(f"bishop 模式：在 rook 基础上补充了 {extra} 对「距离≤800km」的邻接关系")

    # 孤岛处理
    deg = (W > 0).sum(axis=1)
    isolates = [provinces[i] for i in range(n) if deg[i] == 0]
    if isolates:
        if isolate_fix == "nearest":
            D = distance_matrix(provinces).to_numpy()
            for i in np.where(deg == 0)[0]:
                # 排除自身找最近
                d = D[i].copy()
                d[i] = np.inf
                j = int(np.argmin(d))
                W[i, j] = W[j, i] = 1.0
                notes.append(f"孤立省份「{provinces[i]}」已连接最近省份「{provinces[j]}」"
                             f"（距离 {D[i, j]:.0f} km）")
        else:
            notes.append(f"以下省份无邻居且未做处理，将在空间分析中被视为孤立点："
                         f"{'、'.join(isolates)}")

    if mode == "rook" and not notes:
        notes.append("rook 邻接：有共同陆地边界的省份互为邻居")
    return W, notes


def build_geographic(
    provinces: list[str],
    power: float = 1.0,
    use_threshold: bool = False,
    threshold_km: float = 3000.0,
) -> tuple[np.ndarray, list[str]]:
    """
    地理距离权重：W_ij = 1 / d_ij^power。距离越近权重越大。
    可选阈值：超过 threshold_km 直接置 0，避免远距离省份产生微弱但存在的连接。
    """
    D = distance_matrix(provinces).to_numpy().copy()
    notes = [f"地理距离反距离衰减：W = 1 / d^{power:g}（d 为省会大圆距离，单位 km）"]
    np.fill_diagonal(D, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = 1.0 / np.power(D, float(power))
    W = np.nan_to_num(W, nan=0.0, posinf=0.0)
    # 若两省省会重合（如直辖市距离极小），权重会极大 → 上限截断防止单点主导
    cap = np.percentile(W[W > 0], 99.5) if (W > 0).any() else 1.0
    n_cap = int((W > cap).sum())
    if n_cap:
        W = np.minimum(W, cap)
        notes.append(f"{n_cap} 个权重值超过 99.5% 分位，已截断到 {cap:.6f}（防止单点主导）")
    if use_threshold:
        thr = float(threshold_km)
        # 注意：此时 D 已是 nan 对角线，用原始距离重算
        D2 = distance_matrix(provinces).to_numpy()
        mask_far = D2 > thr
        n_cut = int(((W > 0) & mask_far).sum())
        W[mask_far] = 0.0
        notes.append(f"已设距离阈值 {thr:.0f} km，截断 {n_cut} 条远距离连接")
    np.fill_diagonal(W, 0.0)
    return W, notes


def build_economic(
    provinces: list[str],
    econ_values: pd.Series | dict[str, float] | np.ndarray,
    inverse: bool = True,
    variable_name: str = "经济变量",
) -> tuple[np.ndarray, list[str]]:
    """
    经济距离权重：W_ij = 1 / |econ_i - econ_j|。经济水平越接近，权重越大。

    ★ 注意方向性：经济距离矩阵有"趋同"含义 —— 经济水平相近的省份互相影响更大。
      这与地理距离的作用机制不同，做空间马尔科夫链时结论可能不一样，这正是
      需要换矩阵对比的原因。

    econ_values 可为 Series（索引=省份名）/ dict / 与 provinces 等长的数组。
    """
    n = len(provinces)
    if isinstance(econ_values, pd.Series):
        v = np.array([float(econ_values.get(p, np.nan)) for p in provinces], dtype=float)
    elif isinstance(econ_values, dict):
        v = np.array([float(econ_values.get(p, np.nan)) for p in provinces], dtype=float)
    else:
        v = np.asarray(econ_values, dtype=float).ravel()
        if v.size != n:
            raise ValueError(f"econ_values 长度（{v.size}）与省份数（{n}）不一致")

    if np.isnan(v).any():
        miss = [provinces[i] for i in range(n) if np.isnan(v[i])]
        raise ValueError(
            f"经济距离矩阵需要每个省份都有「{variable_name}」取值，但以下省份缺失：{miss}\n"
            "  → 请检查 config.yaml 的 spatial_weights.economic.variable 是否写对了列名，"
            "并确认数据里该变量覆盖了所有省份。"
        )

    notes = []
    if not inverse:
        # 不取倒数：直接用绝对差作为"距离"，但权重应该是距离的减函数，
        # 所以这里用 max - |diff| 的形式翻转
        D = np.abs(v[:, None] - v[None, :])
        W = D.max() - D
        np.fill_diagonal(W, 0.0)
        notes.append(f"经济距离（未取倒数）：W = max|diff| - |{variable_name}_i - {variable_name}_j|")
    else:
        D = np.abs(v[:, None] - v[None, :])
        np.fill_diagonal(D, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            W = 1.0 / D
        W = np.nan_to_num(W, nan=0.0, posinf=0.0)
        # 经济水平完全相同的两省 → 权重无穷，截断
        cap = np.percentile(W[W > 0], 99.5) if (W > 0).any() else 1.0
        n_cap = int((W > cap).sum())
        if n_cap:
            W = np.minimum(W, cap)
            notes.append(f"{n_cap} 个权重值超限已截断到 {cap:.6f}")
        np.fill_diagonal(W, 0.0)
        notes.append(f"经济距离反距离衰减：W = 1 / |{variable_name}_i - {variable_name}_j|"
                     "（经济水平越接近，相互影响越大）")

    notes.append(f"经济变量「{variable_name}」取值区间：{v.min():.4f} – {v.max():.4f}")
    return W, notes


def build_eco_geographic(
    W_geo: np.ndarray,
    W_eco: np.ndarray,
    combine: str = "multiply",
    geo_weight: float = 0.5,
    eco_weight: float = 0.5,
) -> tuple[np.ndarray, list[str]]:
    """
    经济地理嵌套矩阵。
      multiply     : W = W_geo ⊙ W_eco（元素相乘）——论文最常用
      weighted_sum : W = a*W_geo + b*W_eco ——要求两个矩阵已同量纲
    """
    if W_geo.shape != W_eco.shape:
        raise ValueError(
            f"地理矩阵与经济矩阵维度不一致：{W_geo.shape} vs {W_eco.shape}。"
            "两者必须基于同一省份顺序构造。"
        )
    combine = str(combine).lower()
    if combine == "weighted_sum":
        a, b = float(geo_weight), float(eco_weight)
        if a + b <= 0:
            raise ValueError("geo_weight 与 eco_weight 不能同时为 0")
        # 先各自行标准化，再加权，避免量纲差异
        Wg = _row_normalize(W_geo.copy())
        We = _row_normalize(W_eco.copy())
        W = a * Wg + b * We
        notes = [f"经济地理嵌套（加权求和）：W = {a:g}*W_geo_norm + {b:g}*W_eco_norm"]
    else:
        W = W_geo * W_eco
        notes = ["经济地理嵌套（元素相乘）：W = W_geo ⊙ W_eco",
                 "含义：既要地理位置邻近，经济社会发展水平也要相近，才算强关联"]
    np.fill_diagonal(W, 0.0)
    return W, notes


def build_custom(
    provinces: list[str],
    path: str | Path,
    province_col: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """从 CSV 读自定义矩阵。行=省份、列=省份；顺序会被自动对齐到 provinces。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"自定义空间矩阵文件不存在：{path}\n"
            "  → 请把矩阵 CSV 放到 data/ 目录，或在 config.yaml 的 "
            "spatial_weights.custom.file 里填写正确路径。\n"
            "  → 格式要求：第一列是省份名，其余列名也是省份名，格子填权重。"
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    if province_col and province_col in df.columns:
        df = df.set_index(province_col)
    else:
        # 第一列若不是省份名，就用行索引
        first = df.columns[0]
        if df[first].map(lambda x: normalize_province(x) is not None).mean() > 0.6:
            df = df.set_index(first)
        else:
            df = df.set_index(df.columns[0])

    df.index = [normalize_province(x) or str(x).strip() for x in df.index]
    df.columns = [normalize_province(x) or str(x).strip() for x in df.columns]

    missing_rows = [p for p in provinces if p not in df.index]
    missing_cols = [p for p in provinces if p not in df.columns]
    if missing_rows or missing_cols:
        raise ValueError(
            "自定义矩阵缺少以下省份：\n"
            f"  缺少行：{missing_rows}\n"
            f"  缺少列：{missing_cols}\n"
            "  → 自定义矩阵必须覆盖全部 31 个省份（或与数据中省份完全一致）。"
        )
    W = df.loc[provinces, provinces].to_numpy(dtype=float)
    W = np.nan_to_num(W, nan=0.0)
    np.fill_diagonal(W, 0.0)
    return W, ["从 CSV 载入的自定义矩阵（省份顺序已自动对齐）"]


# ------------------------------------------------------------------------------
# 通用后处理
# ------------------------------------------------------------------------------
def _resolve_econ_var(requested: str, econ_vars: dict[str, Any]) -> str:
    """
    确定经济距离矩阵要用的变量。

    优先级：
      1. 配置里指定的变量存在 → 直接用
      2. 按常见别名表匹配（人均GDP、人均可支配收入等的中英文各种写法）
      3. 退化为"取第一个数值变量"，但这在语义上不严谨，会打警告让用户确认

    这样处理的原因：实际数据里列名千差万别（per_gdp / 人均GDP / GDP_per_capita），
    直接报错会让人卡住；自动兜底 + 明确提示，用户一眼就能看到需不需要改配置。
    """
    if not econ_vars:
        raise ValueError(
            "经济距离矩阵需要「人均GDP」等经济变量，但数据里没有任何可用的数值变量。\n"
            "  → 若你的数据是「已算好的指数宽表」（只有省份列和指数值），"
            "没有任何经济变量，无法构造经济距离矩阵。\n"
            "     解决办法：① 把 spatial_weights.active 里的 economic / eco_geographic "
            "去掉，改用 adjacency 或 geographic；\n"
            "             ② 或在数据里补一列人均GDP，并把 data.shape 设为 raw。"
        )
    if requested in econ_vars:
        return requested

    # 常见别名匹配
    aliases = {
        "per_gdp": ["per_gdp", "pergdp", "gdp_per_capita", "人均gdp", "人均生产总值",
                    "人均地区生产总值", "pgdp", "gdppc"],
        "income": ["income", "per_capita_income", "人均可支配收入", "居民收入",
                   "人均收入", "可支配收入"],
        "gdp": ["gdp", "gdp_total", "地区生产总值", "生产总值"],
    }
    norm = {_norm_name(k): k for k in econ_vars.keys()}
    for key in [requested] + aliases.get(requested, []):
        nk = _norm_name(key)
        if nk in norm:
            return norm[nk]
    # 反向：配置名是别名之一
    for primary, alist in aliases.items():
        if _norm_name(requested) in [_norm_name(a) for a in alist]:
            if primary in econ_vars:
                return primary

    # 兜底：取第一个数值变量
    first = sorted(econ_vars.keys())[0]
    logger.warning(
        f"  经济距离矩阵配置的变量「{requested}」在数据里不存在，"
        f"已退化为使用「{first}」。\n"
        f"    数据里可用的数值变量：{sorted(econ_vars.keys())[:20]}\n"
        "    → 如果这个变量不是经济水平指标，请修改 config.yaml 的 "
        "spatial_weights.economic.variable"
    )
    return first


def _norm_name(s: object) -> str:
    return (str(s).strip().lower()
            .replace(" ", "").replace("_", "").replace("-", "")
            .replace("（", "(").replace("）", ")"))


def _row_normalize(W: np.ndarray) -> np.ndarray:
    """行标准化：每行除以行和。行和为 0 的行保持原样（全 0）。"""
    rs = W.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(W, rs, out=np.zeros_like(W, dtype=float), where=rs > 1e-12)
    return out


def _symmetrize(W: np.ndarray, how: str) -> np.ndarray:
    how = str(how).lower()
    if how == "average":
        return (W + W.T) / 2.0
    if how == "max":
        return np.maximum(W, W.T)
    return W


def postprocess(
    W: np.ndarray,
    provinces: list[str],
    standardize_rows: bool = True,
    diagonal_zero: bool = True,
    symmetry: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """统一的矩阵后处理。返回 (W_原始后处理, W_行标准化)。"""
    W = np.asarray(W, dtype=float).copy()
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"权重矩阵不是方阵：{W.shape}")
    if W.shape[0] != len(provinces):
        raise ValueError(
            f"权重矩阵维度（{W.shape[0]}）与省份数（{len(provinces)}）不一致"
        )
    if diagonal_zero:
        np.fill_diagonal(W, 0.0)
    W = _symmetrize(W, symmetry)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    W = np.maximum(W, 0.0)          # 负权重无解释意义，截为 0

    if standardize_rows:
        W_norm = _row_normalize(W)
    else:
        # 不标准化时，下游用未标准化的矩阵按"权重比例"处理会有量纲问题，
        # 这里给个提示但仍按用户要求返回原矩阵
        W_norm = W.copy()

    # 检查全零行
    allzero = [provinces[i] for i in range(W.shape[0]) if W[i].sum() <= 1e-12]
    return W, W_norm


# ------------------------------------------------------------------------------
# 主入口：按配置构建
# ------------------------------------------------------------------------------
def build_matrix(
    name: str,
    provinces: list[str],
    cfg: dict,
    context: dict[str, Any] | None = None,
) -> WeightMatrix:
    """
    构建指定名称的空间权重矩阵。

    context 需要提供经济距离矩阵所需的变量：
        {'econ_vars': {'per_gdp': pd.Series(index=省份, ...), ...},
         'data_dir': Path}
    """
    ctx = context or {}
    sw = cfg or {}
    common = {
        "standardize_rows": bool(sw.get("standardize_rows", True)),
        "diagonal_zero": bool(sw.get("diagonal_zero", True)),
        "symmetry": str(sw.get("symmetry", "auto")),
    }
    name = str(name).lower().strip()
    meta: dict[str, Any] = {}

    # ---------- 邻接矩阵 ----------
    if name in ("adjacency", "邻接", "邻接矩阵", "rook", "contiguity"):
        acfg = sw.get("adjacency") or {}
        W, notes = build_adjacency(
            provinces,
            mode=str(acfg.get("mode", "rook")),
            isolate_fix=str(acfg.get("isolate_fix", "nearest")),
            manual_edges=acfg.get("manual_edges"),
        )
        display = "邻接矩阵"
        meta["mode"] = acfg.get("mode", "rook")

    # ---------- 地理距离 ----------
    elif name in ("geographic", "geography", "地理", "地理距离", "地理距离矩阵", "distance"):
        gcfg = sw.get("geographic") or {}
        W, notes = build_geographic(
            provinces,
            power=float(gcfg.get("power", 1.0)),
            use_threshold=bool(gcfg.get("use_threshold", False)),
            threshold_km=float(gcfg.get("threshold_km", 3000)),
        )
        display = "地理距离矩阵"
        meta["power"] = gcfg.get("power", 1.0)

    # ---------- 经济距离 ----------
    elif name in ("economic", "economy", "经济", "经济距离", "经济距离矩阵"):
        ecfg = sw.get("economic") or {}
        varname = str(ecfg.get("variable", "per_gdp"))
        econ_vars = ctx.get("econ_vars") or {}
        varname = _resolve_econ_var(varname, econ_vars)
        W, notes = build_economic(
            provinces, econ_vars[varname],
            inverse=bool(ecfg.get("inverse", True)),
            variable_name=varname,
        )
        display = f"经济距离矩阵（{varname}）"
        meta["variable"] = varname
        if varname != str(ecfg.get("variable", "per_gdp")):
            notes.append(
                f"配置里写的是「{ecfg.get('variable')}」，但数据里没有这个变量，"
                f"已自动改用「{varname}」。"
                "如需固定用某个变量，请修改 config.yaml 的 "
                "spatial_weights.economic.variable"
            )

    # ---------- 经济地理嵌套 ----------
    elif name in ("eco_geographic", "eco-geographic", "经济地理", "经济地理嵌套",
                  "经济地理嵌套矩阵", "nested"):
        egcfg = sw.get("eco_geographic") or {}
        # 先造出两个子矩阵，再做嵌套
        gcfg = sw.get("geographic") or {}
        W_geo, _ = build_geographic(
            provinces,
            power=float(gcfg.get("power", 1.0)),
            use_threshold=bool(gcfg.get("use_threshold", False)),
            threshold_km=float(gcfg.get("threshold_km", 3000)),
        )
        ecfg = sw.get("economic") or {}
        varname = _resolve_econ_var(str(ecfg.get("variable", "per_gdp")),
                                    ctx.get("econ_vars") or {})
        W_eco, _ = build_economic(
            provinces, (ctx.get("econ_vars") or {})[varname],
            inverse=bool(ecfg.get("inverse", True)),
            variable_name=varname,
        )
        W, notes = build_eco_geographic(
            W_geo, W_eco,
            combine=str(egcfg.get("combine", "multiply")),
            geo_weight=float(egcfg.get("geo_weight", 0.5)),
            eco_weight=float(egcfg.get("eco_weight", 0.5)),
        )
        display = f"经济地理嵌套矩阵（地理 ⊗ {varname}）"
        meta["combine"] = egcfg.get("combine", "multiply")
        meta["variable"] = varname

    # ---------- 自定义 ----------
    elif name in ("custom", "自定义", "自定义矩阵"):
        ccfg = sw.get("custom") or {}
        fname = ccfg.get("file")
        if not fname:
            raise ValueError(
                "自定义矩阵未指定文件。\n"
                "  → 请在 config.yaml 的 spatial_weights.custom.file 里填写 CSV 文件名。"
            )
        data_dir = Path(ctx.get("data_dir") or "data")
        fpath = Path(fname)
        if not fpath.is_absolute():
            fpath = data_dir / fpath
        W, notes = build_custom(provinces, fpath, ccfg.get("province_col"))
        display = "自定义矩阵"

    else:
        raise ValueError(
            f"不认识的空间权重矩阵：{name}\n"
            "  → 可选：adjacency / geographic / economic / eco_geographic / custom\n"
            "  请检查 config.yaml 的 spatial_weights.active。"
        )

    Wp, Wn = postprocess(W, provinces, **common)  # type: ignore[arg-type]
    if not common["standardize_rows"]:
        notes.append("⚠ 行标准化已关闭，空间滞后量按原始权重加权，"
                     "不同省份的邻居权重缩放不一致，结果解释需谨慎")
        logger.warning("  spatial_weights.standardize_rows = false，"
                       "空间马尔科夫链建议开启行标准化")

    return WeightMatrix(
        name=display, provinces=provinces, W=Wp, W_norm=Wn,
        meta=meta, notes=notes,
    )


# ------------------------------------------------------------------------------
# 空间滞后
# ------------------------------------------------------------------------------
def spatial_lag(
    values: pd.Series | np.ndarray,
    W_norm: np.ndarray,
    provinces: list[str],
    neighbor_stat: str = "weighted_mean",
    W_raw: np.ndarray | None = None,
) -> pd.Series:
    """
    计算空间滞后量。

    neighbor_stat:
      weighted_mean —— 用 W（默认行标准化后的 W_norm）加权平均（论文口径）
      simple_mean   —— 邻居简单平均（用 0/1 邻接关系，不看权重大小）
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.size != len(provinces):
        raise ValueError(f"values 长度（{v.size}）与省份数（{len(provinces)}）不一致")

    stat = str(neighbor_stat).lower()
    if stat == "simple_mean":
        # 从 W 里还原 0/1 邻接关系
        A = (np.asarray(W_raw if W_raw is not None else W_norm, dtype=float) > 1e-12).astype(float)
        deg = A.sum(axis=1)
        lag = np.divide(A @ v, deg, out=np.full_like(v, np.nan), where=deg > 1e-12)
    else:
        W = np.asarray(W_norm, dtype=float)
        lag = W @ v                      # 行标准化后即加权平均
        # 全零行 → NaN（该省无邻居）
        allzero = W.sum(axis=1) <= 1e-12
        lag = np.where(allzero, np.nan, lag)

    return pd.Series(lag, index=provinces, name="spatial_lag")


def neighbor_contribution(
    values: pd.Series | np.ndarray,
    W_norm: np.ndarray,
    provinces: list[str],
    target: str,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    拆解某省的空间滞后是由哪些邻居贡献的（解答"为什么它的邻居组是这个等级"）。
    """
    v = np.asarray(values, dtype=float).ravel()
    if target not in provinces:
        raise ValueError(f"目标省份「{target}」不在省份列表里")
    i = provinces.index(target)
    w = np.asarray(W_norm, dtype=float)[i]
    total = w.sum()
    rows = []
    order = np.argsort(-w)
    for j in order:
        if w[j] <= 1e-12:
            continue
        rows.append({
            "邻居省份": provinces[j],
            "权重": w[j],
            "权重占比": w[j] / total if total > 1e-12 else 0.0,
            "邻居指数值": v[j],
            "贡献": w[j] * v[j],
        })
    df = pd.DataFrame(rows).head(top_k)
    if not df.empty:
        df["贡献占比"] = df["贡献"] / df["贡献"].sum()
    return df
