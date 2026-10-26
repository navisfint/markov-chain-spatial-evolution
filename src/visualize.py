# -*- coding: utf-8 -*-
"""
visualize.py —— 出图
================================================
产出四类图：
  1. 转移概率矩阵热力图（传统 + 空间各邻居组）
  2. 核密度演进图（看分布右移、拖尾、极化）
  3. 等级构成演进堆叠图（逐年各等级省份数）
  4. 省份指数轨迹图（挑几个代表省份）

中文显示坑：matplotlib 默认字体不支持中文，必须显式指定。
Linux 服务器上常见缺失 SimHei，代码会依次尝试多个候选字体，
全失败时降级为英文标签（不报错、不中断流程）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("markov")

C_PROV, C_YEAR, C_VALUE = "province", "year", "value"

# 候选中文字体（Windows 优先）
_CJK_FONTS = [
    "SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS",
]


def setup_matplotlib(font_family: str = "SimHei") -> tuple[Any, bool]:
    """
    初始化 matplotlib，返回 (plt, 中文是否可用)。
    """
    import matplotlib
    matplotlib.use("Agg")           # 无界面后端，保证服务器/沙箱也能出图
    import matplotlib.pyplot as plt

    candidates = [font_family] + [f for f in _CJK_FONTS if f != font_family]
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    chosen = None
    for f in candidates:
        if f in available:
            chosen = f
            break

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + _CJK_FONTS
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = 11
        return plt, True

    logger.warning(
        f"  未找到可用的中文字体（尝试过：{candidates[:4]}…），图内文字将改用英文标签。\n"
        "    → 如需中文图，请安装字体后在 config.yaml 的 output.font_family 指定，"
        "或 Windows 下直接用默认的 SimHei。"
    )
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    return plt, False


def _label(i: int, n: int, cjk: bool) -> str:
    if cjk:
        from utils import class_full_labels
        return class_full_labels(n)[i]
    roman = ["I", "II", "III", "IV", "V", "VI"][:n]
    names = {"I": "Low", "II": "Mid-Low", "III": "Mid-High", "IV": "High"}
    return f"{roman[i]} ({names.get(roman[i], '')})" if n == 4 else roman[i]


# ------------------------------------------------------------------------------
# 1. 转移矩阵热力图
# ------------------------------------------------------------------------------
def plot_transition_heatmap(
    P: np.ndarray,
    n_classes: int,
    out_path: Path,
    title: str,
    plt: Any,
    cjk: bool = True,
    row_totals: np.ndarray | None = None,
    cmap: str = "YlOrRd",
    annotate: bool = True,
) -> Path | None:
    """画转移概率矩阵热力图。格子内标百分数，颜色深浅表示概率大小。"""
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    M = np.asarray(P, dtype=float) * 100
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=max(100.0, np.nanmax(M)))

    labels = [_label(i, n_classes, cjk) for i in range(n_classes)]
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("目标等级" if cjk else "To class")
    ax.set_ylabel("起始等级" if cjk else "From class")
    ax.set_title(title, pad=12, fontsize=12)

    if annotate:
        for i in range(n_classes):
            for j in range(n_classes):
                v = M[i, j]
                txt = f"{v:.1f}" if np.isfinite(v) else "-"
                color = "white" if np.isfinite(v) and v > (np.nanmax(M) * 0.6) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    if row_totals is not None:
        for i in range(n_classes):
            n = int(row_totals[i])
            tag = f"n={n}" if n > 0 else ("无样本" if cjk else "n=0")
            ax.text(n_classes - 0.35, i, tag, ha="left", va="center",
                    fontsize=8, color="#666666")

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("转移概率 (%)" if cjk else "Transition prob. (%)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------------
# 2. 核密度演进图
# ------------------------------------------------------------------------------
def plot_kernel_density(
    panel: pd.DataFrame,
    years: list[int],
    out_path: Path,
    plt: Any,
    cjk: bool = True,
    bw_method: str | float = "scott",
) -> Path | None:
    """
    核密度估计演进图。横轴=指数值，纵轴=密度，每条线一个年份。
    能直观看到：曲线右移（水平提升）、右拖尾延长（绝对差距拉大）、
    峰数变化（极化程度）。
    """
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        logger.warning("  缺少 scipy，跳过核密度图")
        return None

    data = panel[[C_YEAR, C_VALUE]].dropna()
    avail_years = sorted(data[C_YEAR].unique().tolist())
    use_years = [y for y in years if y in avail_years]
    if len(use_years) < 2:
        use_years = avail_years[::max(1, len(avail_years) // 6)][:7]
    if len(use_years) < 2:
        logger.warning("  可用于核密度图的年份少于 2 个，跳过")
        return None

    grid = np.linspace(data[C_VALUE].min() - 0.02, data[C_VALUE].max() + 0.02, 400)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(use_years) - 1)) for i in range(len(use_years))]

    for c, y in zip(colors, use_years):
        v = data.loc[data[C_YEAR] == y, C_VALUE].to_numpy(dtype=float)
        if v.size < 3 or np.std(v) <= 1e-12:
            continue
        try:
            kde = gaussian_kde(v, bw_method=bw_method)
            density = kde(grid)
        except Exception:
            continue
        ax.plot(grid, density, color=c, lw=1.9, label=str(int(y)))

    ax.set_xlabel("共同富裕指数" if cjk else "Index value")
    ax.set_ylabel("核密度" if cjk else "Kernel density")
    ax.set_title("共同富裕水平核密度演进" if cjk else "Kernel density evolution",
                 pad=12)
    ax.legend(title="年份" if cjk else "Year", frameon=False, fontsize=9, ncol=2)
    ax.grid(alpha=0.25, ls="--", lw=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------------
# 3. 等级构成演进堆叠图
# ------------------------------------------------------------------------------
def plot_class_composition(
    panel: pd.DataFrame,
    n_classes: int,
    out_path: Path,
    plt: Any,
    cjk: bool = True,
    class_col: str = "class",
) -> Path | None:
    """逐年各等级省份数堆叠面积图。"""
    tab = pd.crosstab(panel[C_YEAR], panel[class_col]).reindex(
        columns=range(n_classes), fill_value=0)
    if tab.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    colors = plt.get_cmap("RdYlGn_r")(np.linspace(0.12, 0.88, n_classes))
    ax.stackplot(tab.index, [tab[k].to_numpy() for k in range(n_classes)],
                 labels=[_label(i, n_classes, cjk) for i in range(n_classes)],
                 colors=colors, alpha=0.9, edgecolor="white", lw=0.5)
    ax.set_xlabel("年份" if cjk else "Year")
    ax.set_ylabel("省份数" if cjk else "Number of provinces")
    ax.set_title("各省共同富裕等级构成演进" if cjk else "Composition of classes over time",
                 pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9,
              title="等级" if cjk else "Class")
    ax.set_xlim(tab.index.min(), tab.index.max())
    ax.grid(alpha=0.2, ls="--", lw=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------------
# 4. 省份轨迹图
# ------------------------------------------------------------------------------
def plot_province_trajectories(
    panel: pd.DataFrame,
    out_path: Path,
    plt: Any,
    cjk: bool = True,
    highlight: list[str] | None = None,
    max_lines: int = 8,
) -> Path | None:
    """
    各省指数轨迹。默认只高亮几条代表线（均值最高/最低/中位的省份），
    其余用淡灰色背景线，避免 31 条线糊成一团。
    """
    mat = panel.pivot_table(index=C_YEAR, columns=C_PROV, values=C_VALUE, aggfunc="mean")
    if mat.empty or mat.shape[1] == 0:
        return None

    mean_by_prov = mat.mean(axis=0).sort_values()
    if highlight is None:
        picks = []
        if len(mean_by_prov) >= 2:
            picks += [mean_by_prov.index[0], mean_by_prov.index[-1]]
        mid = mean_by_prov.index[len(mean_by_prov) // 2]
        picks.append(mid)
        # 再补几个分位代表
        for q in (0.25, 0.75):
            i = int(q * (len(mean_by_prov) - 1))
            picks.append(mean_by_prov.index[i])
        # 去重且保持顺序
        seen = set()
        highlight = []
        for p in picks:
            if p not in seen:
                seen.add(p)
                highlight.append(p)
        highlight = highlight[:max_lines]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for col in mat.columns:
        if col in highlight:
            continue
        ax.plot(mat.index, mat[col], color="#d5d5d5", lw=0.9, zorder=1)

    cmap = plt.get_cmap("tab10")
    for k, col in enumerate(highlight):
        if col in mat.columns:
            ax.plot(mat.index, mat[col], lw=2.0, color=cmap(k % 10),
                    label=col, zorder=3, marker="o", ms=3)

    ax.set_xlabel("年份" if cjk else "Year")
    ax.set_ylabel("共同富裕指数" if cjk else "Index value")
    ax.set_title("各省共同富裕水平演进（灰线为其他省份）"
                 if cjk else "Provincial trajectories (grey = others)", pad=12)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(alpha=0.22, ls="--", lw=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------------------
# 5. 空间滞后地图（可选，用省份中心点散点代替真地图）
# ------------------------------------------------------------------------------
def plot_spatial_lag_scatter(
    lag_panel: pd.DataFrame,
    year: int,
    out_path: Path,
    plt: Any,
    cjk: bool = True,
) -> Path | None:
    """
    用经纬度散点画某年的"省份指数 vs 邻居均值"分布，
    颜色=自身等级，大小=指数值。这是无 GIS 环境下最直观的空间视图。
    """
    from province_data import PROVINCE_COORDS

    sub = lag_panel[lag_panel[C_YEAR] == year].dropna(subset=["lag_value"])
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    xs, ys = [], []
    for p in sub[C_PROV]:
        lon, lat = PROVINCE_COORDS.get(p, (np.nan, np.nan))
        xs.append(lon)
        ys.append(lat)
    sub = sub.assign(lon=xs, lat=ys).dropna(subset=["lon"])

    sc = ax.scatter(
        sub["lon"], sub["lat"],
        c=sub[C_VALUE], s=40 + 520 * (sub[C_VALUE] - sub[C_VALUE].min()) /
        max(1e-9, (sub[C_VALUE].max() - sub[C_VALUE].min())),
        cmap="RdYlGn_r", edgecolor="#333333", lw=0.6, alpha=0.9, zorder=3,
    )
    for _, r in sub.iterrows():
        ax.annotate(r[C_PROV], (r["lon"], r["lat"]),
                    fontsize=7.5, ha="center", va="center",
                    xytext=(0, 0), textcoords="offset points", zorder=4)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
    cbar.set_label("指数值" if cjk else "Index")

    ax.set_xlabel("经度" if cjk else "Longitude")
    ax.set_ylabel("纬度" if cjk else "Latitude")
    ax.set_title(f"{int(year)} 年各省共同富裕水平空间分布"
                 if cjk else f"Spatial distribution, {int(year)}", pad=12)
    ax.grid(alpha=0.2, ls="--", lw=0.5)
    ax.set_aspect(1.15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
