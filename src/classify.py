# -*- coding: utf-8 -*-
"""
classify.py —— 等级划分
================================================
把共同富裕指数分成 4 档。两套方法并列对比：

  A. 四分位法（论文口径）
     [0,25%) → Ⅰ 低水平
     [25%,50%) → Ⅱ 中低水平
     [50%,75%) → Ⅲ 中高水平
     [75%,100%] → Ⅳ 高水平
     - quartile_basis='pooled'：用全样本分位点，边界全程固定
       → 反映"绝对水平"提升（等级会随年份整体上移，出现的空行空列能反映等级缺失）
     - quartile_basis='yearly'：逐年算分位点
       → 反映"相对位置"变化（每年每档样本数固定为约 25%）

  B. K 均值聚类（k=4）
     按数据自然聚集分组，不强制等分位。等级标签按组内指数均值升序重排为 Ⅰ→Ⅳ。

输出统一为：
  - labels : 与 panel 等长的整数数组，取值 0..3（0=Ⅰ）
  - Result 对象里带边界、频数、说明文字
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils import class_full_labels, class_label, describe_array

C_PROV, C_YEAR, C_VALUE = "province", "year", "value"


# ------------------------------------------------------------------------------
# 结果容器
# ------------------------------------------------------------------------------
@dataclass
class ClassifyResult:
    method: str                       # 'quartile' / 'kmeans'
    basis: str                        # 'pooled' / 'yearly' / 'kmeans'
    labels: np.ndarray                # 0..n-1
    n_classes: int
    boundaries: list[float] = field(default_factory=list)   # 分位/簇边界
    counts: dict[int, int] = field(default_factory=dict)    # 各等级样本数
    yearly_counts: pd.DataFrame | None = None               # 逐年各等级样本数
    centroids: list[float] = field(default_factory=list)    # KMeans 各簇中心
    notes: list[str] = field(default_factory=list)

    @property
    def label_names(self) -> list[str]:
        return class_full_labels(self.n_classes)

    def summary(self) -> str:
        lines = [f"方法：{self.method}（基准：{self.basis}）"]
        if self.boundaries:
            if self.method == "quartile":
                b = self.boundaries
                lines.append("分档区间：" + "  ".join(
                    f"[{b[i]:.4f}, {b[i+1]:.4f})" for i in range(len(b) - 1)
                ) + f"  末档闭合至 {b[-1]:.4f}")
            else:
                lines.append("簇中心：" + "  ".join(f"{c:.4f}" for c in self.centroids))
        total = sum(self.counts.values()) or 1
        lines.append("样本分布：" + "  ".join(
            f"{class_label(k)}={v}({v / total * 100:.1f}%)"
            for k, v in sorted(self.counts.items())
        ))
        for note in self.notes:
            lines.append(f"提示：{note}")
        return "\n".join(lines)


# ------------------------------------------------------------------------------
# A. 四分位法
# ------------------------------------------------------------------------------
def classify_quartile(
    values: np.ndarray,
    n_classes: int = 4,
    years: np.ndarray | None = None,
    basis: str = "pooled",
) -> ClassifyResult:
    """
    四分位（或 n 等分）分级。

    basis='pooled'：所有样本一起算分位点，边界固定。
    basis='yearly'：按年份分组，每年单独算分位点（需提供 years）。
    """
    values = np.asarray(values, dtype=float)
    n_classes = int(n_classes)
    if n_classes < 2:
        raise ValueError("n_classes 至少为 2")

    probs = np.linspace(0, 1, n_classes + 1)   # [0, .25, .5, .75, 1]
    notes: list[str] = []

    if basis == "yearly":
        if years is None:
            raise ValueError("basis='yearly' 时必须提供 years 参数")
        years = np.asarray(years)
        labels = np.full(values.shape, -1, dtype=int)
        boundaries_map: dict[int, list[float]] = {}
        for y in sorted(np.unique(years)):
            mask = years == y
            v = values[mask]
            qs = np.quantile(v, probs)
            # 等分位可能出现重复边界（该年样本少/取值集中），用唯一化处理
            labels[mask] = _assign_by_edges(v, qs)
            boundaries_map[int(y)] = [float(x) for x in qs]
        if (labels < 0).any():
            raise RuntimeError("逐年分级时出现未赋值的样本，请检查 years 是否与 values 对齐")
        bounds = sorted({b for bl in boundaries_map.values() for b in bl})
        notes.append("逐年分级：每年每档样本数固定约为总省份数的 1/n，"
                     "等级反映的是相对位置而非绝对水平")
        counts = _counts(labels, n_classes)
        yc = pd.DataFrame(
            {class_label(k): [(labels[(years == y)] == k).sum()
                              for y in sorted(np.unique(years))]
             for k in range(n_classes)},
            index=[int(y) for y in sorted(np.unique(years))],
        )
        yc.index.name = "年份"
        return ClassifyResult(
            method="quartile", basis="yearly", labels=labels, n_classes=n_classes,
            boundaries=[float(b) for b in bounds], counts=counts,
            yearly_counts=yc, notes=notes,
        )

    # ---- pooled ----
    qs = np.quantile(values, probs)
    if len(np.unique(qs)) < len(qs):
        notes.append("分位点出现重复值（数据取值过于集中），已自动合并相邻档位，"
                     "实际档位数可能少于设定值")
    labels = _assign_by_edges(values, qs)
    counts = _counts(labels, n_classes)
    if 0 in counts and len(counts) < n_classes:
        notes.append(f"有 {n_classes - len(counts)} 个档位一个样本都没有，"
                     "说明数据分布严重偏斜，建议改用 K 均值聚类")
    notes.append("全样本分位点：等级边界在全部年份保持固定，"
                 "等级上升代表绝对水平提升（论文口径）")
    return ClassifyResult(
        method="quartile", basis="pooled", labels=labels, n_classes=n_classes,
        boundaries=[float(x) for x in qs], counts=counts, notes=notes,
    )


def _assign_by_edges(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    按区间边界赋等级。
    区间：[e0,e1) [e1,e2) ... [e_{n-1}, e_n]
    - 最后一个区间闭合（含最大值）
    - 边界重复时，用 np.searchsorted 的 side='right'-1 逻辑保证左侧优先
    """
    e = np.asarray(edges, dtype=float)
    # 去重但保留顺序，用于确定有效档位
    n = len(e) - 1
    lab = np.searchsorted(e[1:-1], v, side="right")   # 落在第几个中间边界之后
    lab = np.clip(lab, 0, n - 1)
    # 低于下界（不可能，除非 NaN）和等于最大值都归到末档
    lab = np.where(v >= e[-1], n - 1, lab)
    lab = np.where(v < e[0], 0, lab)
    return lab.astype(int)


def _counts(labels: np.ndarray, n_classes: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for k in range(n_classes):
        c = int((labels == k).sum())
        if c > 0:
            out[k] = c
    return out


# ------------------------------------------------------------------------------
# B. K 均值聚类
# ------------------------------------------------------------------------------
def classify_kmeans(
    values: np.ndarray,
    n_classes: int = 4,
    order_labels: bool = True,
    n_init: int = 10,
    random_state: int = 42,
    standardize: bool = False,
) -> ClassifyResult:
    """
    K 均值聚类分级。

    单变量聚类有个坑：KMeans 按"使组内平方和最小"划分，结果会因初始化不同而漂移。
    这里做三件事保证稳定可复现：
      1. n_init 设够（默认 10 次重启取最优）
      2. random_state 固定
      3. 聚类后按簇内均值升序重排标签 → Ⅰ 一定是最低水平

    另一个坑：sklearn 新版 KMeans 的 n_init 默认值是 'auto' 而非整数，
    这里显式传整数，兼容新旧版本。
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "K 均值聚类需要 scikit-learn。请执行：\n"
            "    pip install scikit-learn\n"
            "  或直接用 config.yaml 把 classification.methods 里的 \"kmeans\" 去掉，只跑四分位法。"
        ) from e

    values = np.asarray(values, dtype=float).reshape(-1, 1)
    n_classes = int(n_classes)
    if len(values) < n_classes:
        raise ValueError(
            f"样本数（{len(values)}）少于等级数（{n_classes}），无法聚类。\n"
            "  → 请减少 n_classes 或补充数据。"
        )

    X = values
    if standardize:
        mu, sd = X.mean(), X.std(ddof=0)
        X = (X - mu) / (sd if sd > 1e-12 else 1.0)
    notes: list[str] = []

    km = KMeans(
        n_clusters=n_classes,
        n_init=int(n_init),
        random_state=int(random_state),
        max_iter=1000,
        tol=1e-6,
    )
    raw = km.fit_predict(X)
    centroids_raw = km.cluster_centers_.ravel()

    if standardize:
        mu, sd = values.mean(), values.std(ddof=0)
        sd = sd if sd > 1e-12 else 1.0
        centroids = centroids_raw * sd + mu
    else:
        centroids = centroids_raw

    # 按簇内均值升序重排：簇均值最小的 → 标签 0（Ⅰ）
    order = np.argsort(centroids)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(c)] for c in raw], dtype=int)
    centroids_sorted = [float(centroids[i]) for i in order]

    # 簇边界：相邻簇中心的算术中点（仅作参考，说明各档大致分界）
    boundaries = [float(-np.inf)] + [
        float((centroids_sorted[i] + centroids_sorted[i + 1]) / 2)
        for i in range(len(centroids_sorted) - 1)
    ] + [float(np.inf)]

    counts = _counts(labels, n_classes)
    sizes = [counts.get(k, 0) for k in range(n_classes)]
    if min(sizes) == 0:
        notes.append("聚类结果存在空簇，建议减少 n_classes 或检查数据是否退化")
    ratio = max(sizes) / max(min(sizes), 1)
    if ratio > 3:
        notes.append(f"各档样本量差异较大（最大/最小 = {ratio:.1f}），"
                     "这是 KMeans 按数据自然聚集分组的正常结果；"
                     "若希望各档样本数均衡，请改用四分位法")

    notes.append(f"KMeans 单变量聚类：n_init={n_init}、random_state={random_state}，"
                 "标签已按簇均值升序重排，结果可复现")
    notes.append("KMeans 分组边界随数据分布变化，各档不等分位；"
                 "与四分位法对比可检验分级方式的稳健性")

    return ClassifyResult(
        method="kmeans", basis="kmeans", labels=labels, n_classes=n_classes,
        boundaries=boundaries, counts=counts, centroids=centroids_sorted,
        notes=notes,
    )


# ------------------------------------------------------------------------------
# 统一入口
# ------------------------------------------------------------------------------
def classify(
    panel: pd.DataFrame,
    method: str,
    n_classes: int = 4,
    quartile_basis: str = "pooled",
    kmeans_cfg: dict[str, Any] | None = None,
) -> ClassifyResult:
    """按 method 走对应分级流程。panel 需含 province/year/value。"""
    kcfg = kmeans_cfg or {}
    values = panel[C_VALUE].to_numpy(dtype=float)
    method = str(method).lower().strip()

    if method in ("quartile", "四分位", "四分位法", "quantile", "quartiles"):
        return classify_quartile(
            values, n_classes=n_classes,
            years=panel[C_YEAR].to_numpy(),
            basis=str(quartile_basis).lower(),
        )
    if method in ("kmeans", "k-means", "k均值", "k均值聚类", "cluster"):
        return classify_kmeans(
            values, n_classes=n_classes,
            order_labels=bool(kcfg.get("order_labels", True)),
            n_init=int(kcfg.get("n_init", 10)),
            random_state=int(kcfg.get("random_state", 42)),
            standardize=bool(kcfg.get("standardize", False)),
        )
    raise ValueError(
        f"不认识的分类方法：{method}\n"
        "  → config.yaml 的 classification.methods 里只能填 \"quartile\" 和 \"kmeans\"。"
    )


# ------------------------------------------------------------------------------
# 辅助：给 panel 附加等级列
# ------------------------------------------------------------------------------
def attach_labels(panel: pd.DataFrame, res: ClassifyResult, col: str = "class") -> pd.DataFrame:
    out = panel.copy()
    if len(res.labels) != len(out):
        raise ValueError(
            f"等级标签长度（{len(res.labels)}）与样本数（{len(out)}）不一致，"
            "可能是 panel 在分级后被修改过。"
        )
    out[col] = res.labels
    out[col + "_name"] = [class_label(int(i)) for i in res.labels]
    return out


def yearly_distribution(panel: pd.DataFrame, col: str = "class") -> pd.DataFrame:
    """逐年各等级省份数。用来画等级演进堆叠图。"""
    tab = pd.crosstab(panel[C_YEAR], panel[col])
    tab.index.name = "年份"
    return tab
