# -*- coding: utf-8 -*-
"""
markov.py —— 马尔科夫链核心
================================================
两大块：

一、传统马尔科夫链
    用等级序列 {Ⅰ,Ⅱ,Ⅲ,Ⅳ} 构造转移频数矩阵 N，极大似然估计转移概率 P：
        P_ij = N_ij / Σ_j N_ij
    并给出：
        · 稳态分布 π（πP = π，长期极限分布）
        · 期望持续时间 D_i = 1 / (1 - P_ii)
        · 卡方独立性检验（转移是否随机、是否显著路径依赖）
        · 遍历性检验（是否不可约 + 非周期，是否收敛到唯一稳态）

二、空间马尔科夫链
    核心思想：一个省份向上一级转移的概率，取决于它"邻居的平均水平"。
    做法：
      1. 对每期每省算空间滞后值 x_lag_i = Σ_j W_ij · x_j   （W 行标准化后即加权平均）
      2. 按 x_lag 分成 K 组（K = 4 时即"邻居的 4 个等级组"）
      3. 在每组内分别统计转移频数 → 得到 K 个 4×4 条件转移矩阵
      4. 对比条件矩阵与全样本矩阵，看"邻居等级"是否系统性地改变转移概率
         → 这就是空间溢出效应

关键实现要点（容易踩坑的地方）
------------------------------
· 转移必须是**同一省份的相邻两期**。所以先按 province 分组、按 year 排序，
  用 shift(-lag) 配对，绝不能让不同省份之间串行。
· 频数矩阵的行和可能为 0（该等级从未出现过，或该省从未在该等级停留）。
  此时概率行应全为 0，并**明确标注**，不要用 uniform 填充（会污染稳态分布）。
· 稳态分布要用线性方程组求，不要用 P^n 迭代（n 大时慢且数值抖动）。
· 论文表 4 的频数（109/109/97/88）与行概率不对应 —— 那是按"省份最终等级"
  统计的，不是按转移样本统计。本代码按标准做法实现，并在输出里给出说明。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from classify import ClassifyResult
from utils import class_full_labels, class_label

logger = logging.getLogger("markov")

C_PROV, C_YEAR, C_VALUE = "province", "year", "value"


# ==============================================================================
# 一、传统马尔科夫链
# ==============================================================================
@dataclass
class MarkovResult:
    name: str                              # 结果名，如 "四分位法 / 传统马尔科夫链"
    n_classes: int
    labels: list[str]                      # ['Ⅰ (低水平)', ...]
    counts: np.ndarray                     # 频数矩阵 N (n×n)
    P: np.ndarray                          # 转移概率矩阵 (n×n)
    row_totals: np.ndarray                 # 每行样本数
    lag: int = 1
    estimator: str = "mle"
    steady_state: np.ndarray | None = None
    expected_duration: np.ndarray | None = None
    chi2: dict[str, Any] | None = None
    ergodicity: dict[str, Any] | None = None
    raw_transitions: pd.DataFrame | None = None    # 明细：省-年-起等级-终等级
    notes: list[str] = field(default_factory=list)

    # ---- 格式化输出 ----
    def prob_frame(self, digits: int = 2) -> pd.DataFrame:
        """转移概率矩阵（百分数），带 'Ⅰ' 等标签，用于出表和画热力图。"""
        display = class_full_labels(self.n_classes)
        df = pd.DataFrame(
            np.round(self.P * 100, digits),
            index=[f"{d}" for d in display],
            columns=[f"{d}" for d in display],
        )
        df.index.name = "起始等级 \\ 目标等级"
        return df

    def count_frame(self) -> pd.DataFrame:
        display = class_full_labels(self.n_classes)
        df = pd.DataFrame(
            self.counts.astype(int),
            index=[f"{d}" for d in display],
            columns=[f"{d}" for d in display],
        )
        df["行合计"] = self.row_totals.astype(int)
        df.index.name = "起始等级 \\ 目标等级"
        return df

    def summary(self) -> str:
        n = self.n_classes
        lines = [f"【{self.name}】",
                 f"  转移期数 lag = {self.lag}，估计方法 = {self.estimator}",
                 f"  有效转移样本 = {int(self.row_totals.sum())}"]

        lines.append("  转移概率矩阵（行=起始等级，列=目标等级，单位 %）：")
        hdr = "        " + "".join(f"{class_label(j):>10}" for j in range(n))
        lines.append(hdr)
        for i in range(n):
            row = "".join(f"{self.P[i, j] * 100:>10.2f}" for j in range(n))
            lines.append(f"    {class_label(i):<4}{row}   (n={int(self.row_totals[i])})")

        # 对角线观察
        diag = np.diag(self.P)
        valid = self.row_totals > 0
        if valid.any():
            d = diag[valid]
            lines.append(f"  对角线（维持原等级概率）："
                         f"min={d.min() * 100:.2f}%  max={d.max() * 100:.2f}%")
            if d.max() >= 0.999:
                k = int(np.argmax(np.where(valid, diag, -1)))
                if self.P[k, k] >= 0.999:
                    lines.append(f"  ⚠ 等级 {class_label(k)} 的存在概率为 100%"
                                 "（该等级从未有任何省份离开过）"
                                 "→ 这是一个吸收态，长期看会吸收全部省份")
            if d.min() < 0.5:
                k = int(np.argmin(np.where(valid, diag, 2)))
                lines.append(f"  提示：等级 {class_label(k)} 的维持概率仅 "
                             f"{self.P[k, k] * 100:.2f}%，属于不稳定阶段，"
                             "是政策需要重点关注的「脆弱区」")
            lines.append(f"  路径依赖判断：维持原等级的平均概率 "
                         f"{d.mean() * 100:.2f}% "
                         f"{'（强路径依赖）' if d.mean() > 0.5 else '（路径依赖不明显）'}")

        # 跌落风险
        for i in range(n):
            if self.row_totals[i] == 0:
                continue
            down = self.P[i, :i].sum() if i > 0 else 0.0
            if down > 1e-9:
                lines.append(f"  等级 {class_label(i)} 向下跌落概率合计 "
                             f"{down * 100:.2f}%")

        if self.steady_state is not None:
            lines.append("  稳态分布 π：" + "  ".join(
                f"{class_label(i)}={self.steady_state[i] * 100:.2f}%"
                for i in range(n)))
        if self.expected_duration is not None:
            lines.append("  期望持续时间（年）：" + "  ".join(
                f"{class_label(i)}={self.expected_duration[i]:.2f}"
                for i in range(n)))
        if self.chi2:
            c = self.chi2
            if c.get("ok"):
                lines.append(f"  卡方独立性检验：χ²={c['chi2']:.3f}  df={c['dof']}  "
                             f"p={c['p']:.4g}  → "
                             f"{'拒绝独立性，转移存在显著方向性（路径依赖成立）' if c['p'] < 0.05 else '不能拒绝独立性，转移近似随机'}")
                if c.get("n_small_cells"):
                    lines.append(f"  ⚠ 有 {c['n_small_cells']} 个格子的期望频数 < 5，"
                                 "卡方近似可能不可靠，结果仅作参考")
            else:
                lines.append(f"  卡方检验未执行：{c.get('reason', '')}")
        if self.ergodicity:
            e = self.ergodicity
            lines.append(
                f"  遍历性：{'不可约（全部等级互通）' if e['irreducible'] else '可约（存在吸收态/封闭子集）'}"
                f"；{'非周期' if e['aperiodic'] else '周期'}；"
                f"稳态{'唯一' if e['unique_steady'] else '不唯一'}"
            )
            if e["absorbing"]:
                lines.append(f"  ⚠ 吸收态等级：{'、'.join(class_label(k) for k in e['absorbing'])}")

        for note in self.notes:
            lines.append(f"  ▸ {note}")
        return "\n".join(lines)


# ------------------------------------------------------------------------------
# 转移对构造
# ------------------------------------------------------------------------------
def build_transitions(
    panel: pd.DataFrame,
    class_col: str = "class",
    lag: int = 1,
    value_col: str = C_VALUE,
) -> pd.DataFrame:
    """
    构造「起等级 → 终等级」配对样本。

    ★ 关键：必须按省份分组、按年份排序后再 shift。否则会把 A 省的末期
      与 B 省的初期配成一对，结果完全错。这里用 groupby().shift() 保证
      只在同一省份内部配对。

    ★ 另一处坑：如果年份有断裂（如缺 2010 年），shift(1) 会把 2009 和 2011
      当成相邻两期。这里显式检查年份差是否等于 lag，不等的样本剔除并警告。
    """
    df = panel.sort_values([C_PROV, C_YEAR]).copy()
    g = df.groupby(C_PROV, sort=False)
    df["_next_class"] = g[class_col].shift(-lag)
    df["_next_year"] = g[C_YEAR].shift(-lag)
    df["_next_value"] = g[value_col].shift(-lag)

    total = len(df)
    tr = df.dropna(subset=["_next_class", "_next_year"]).copy()

    # 年份间隔校验
    tr["_gap"] = (tr["_next_year"] - tr[C_YEAR]).astype(float)
    bad = tr[tr["_gap"] != float(lag)]
    if len(bad):
        logger.warning(
            f"  有 {len(bad)} 组「相邻观测」的实际年份间隔不等于 lag={lag}"
            f"（多为数据年份断裂导致），已剔除。"
            f"涉及年份实例：{sorted(bad[C_YEAR].astype(int).unique().tolist())[:10]}"
        )
        tr = tr[tr["_gap"] == float(lag)]

    tr["_from"] = tr[class_col].astype(int)
    tr["_to"] = tr["_next_class"].astype(int)
    tr = tr.rename(columns={
        C_PROV: "省份", C_YEAR: "起始年份", "_next_year": "目标年份",
        "_from": "起始等级", "_to": "目标等级",
        value_col: "起始指数值", "_next_value": "目标指数值",
    })
    keep = ["省份", "起始年份", "目标年份", "起始等级", "目标等级",
            "起始指数值", "目标指数值"]
    out = tr[keep].reset_index(drop=True)
    out["等级变化"] = out["目标等级"] - out["起始等级"]

    logger.info(f"  构造转移对：面板样本 {total} 条 → 有效转移 {len(out)} 条"
                f"（每个省份减少 {lag} 条末期观测）")
    return out


# ------------------------------------------------------------------------------
# 频数与概率
# ------------------------------------------------------------------------------
def transition_counts(transitions: pd.DataFrame, n_classes: int) -> np.ndarray:
    """统计频数矩阵 N，N[i,j] = 从等级 i 转到等级 j 的次数。"""
    N = np.zeros((n_classes, n_classes), dtype=float)
    for f, t in zip(transitions["起始等级"].to_numpy(),
                    transitions["目标等级"].to_numpy()):
        if 0 <= f < n_classes and 0 <= t < n_classes:
            N[int(f), int(t)] += 1.0
    return N


def transition_probs(
    N: np.ndarray,
    estimator: str = "mle",
    bayes_prior: float = 1.0,
) -> np.ndarray:
    """
    频数 → 概率。

    mle   : P_ij = N_ij / N_i·    （极大似然，论文口径）
    bayes : P_ij = (N_ij + α) / (N_i· + k·α)   （拉普拉斯/Bayes 平滑）
            小样本时避免出现大量 0，但会引入先验偏误，需在结论里说明

    行和为 0 时（该等级从未作为起点出现），返回全 0 行，**不填充均匀分布**。
    """
    N = np.asarray(N, dtype=float)
    k = N.shape[0]
    est = str(estimator).lower()

    if est == "bayes":
        a = float(bayes_prior)
        if a < 0:
            raise ValueError("bayes_prior 不能为负")
        P = (N + a) / (N.sum(axis=1, keepdims=True) + k * a)
        # 全零行仍是均匀分布（因为有先验），这是 Bayes 平滑的正常行为
        return P

    rs = N.sum(axis=1, keepdims=True)
    P = np.divide(N, rs, out=np.zeros_like(N), where=rs > 0)
    return P


# ------------------------------------------------------------------------------
# 稳态分布
# ------------------------------------------------------------------------------
def steady_state(P: np.ndarray) -> np.ndarray | None:
    """
    求稳态分布 π（满足 πP = π, Σπ = 1）。

    用线性方程组解：
        (P^T - I) π = 0  加上归一化约束 Σπ = 1
    即把最后一行的方程替换为 Σπ = 1，解出唯一解。

    对可约链（存在吸收态），可能有多个稳态解，此时返回使残差最小的解，
    并在调用方通过 ergodicity 提示用户。
    """
    P = np.asarray(P, dtype=float)
    k = P.shape[0]
    if k == 0:
        return None
    A = P.T - np.eye(k)
    A[-1, :] = 1.0                      # 用归一化条件替换一行
    b = np.zeros(k)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 奇异 → 用最小二乘兜底
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    # 数值修正：去掉极小负值、重新归一化
    pi = np.where(np.abs(pi) < 1e-10, 0.0, pi)
    if pi.sum() <= 1e-12:
        return None
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()
    return pi


def expected_durations(P: np.ndarray) -> np.ndarray:
    """各等级期望持续时间 D_i = 1/(1-P_ii)。P_ii=1（吸收态）时返回 inf。"""
    P = np.asarray(P, dtype=float)
    d = np.diag(P)
    with np.errstate(divide="ignore"):
        out = 1.0 / (1.0 - d)
    out = np.where(d >= 1.0 - 1e-12, np.inf, out)
    return out


# ------------------------------------------------------------------------------
# 检验
# ------------------------------------------------------------------------------
def chi_square_independence(N: np.ndarray, n_classes: int) -> dict[str, Any]:
    """
    对转移频数矩阵做卡方独立性检验。
    H0：下一等级与当前等级独立（转移是"随机分配"，不存在路径依赖）
    """
    N = np.asarray(N, dtype=float)
    # 剔除全零行列，否则卡方统计量无定义
    keep_r = N.sum(axis=1) > 0
    keep_c = N.sum(axis=0) > 0
    if keep_r.sum() < 2 or keep_c.sum() < 2:
        return {"ok": False, "reason": "有效行列不足 2，无法做卡方检验（样本太少或等级过于集中）"}

    obs = N[np.ix_(keep_r, keep_c)]
    try:
        chi2, p, dof, expected = stats.chi2_contingency(obs, correction=False)
    except Exception as e:
        return {"ok": False, "reason": f"卡方检验失败：{e}"}

    n_small = int((expected < 5).sum())
    return {
        "ok": True,
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "n_small_cells": n_small,
        "n_cells": int(expected.size),
        "cramers_v": float(np.sqrt(chi2 / (obs.sum() * (min(obs.shape) - 1))))
        if obs.sum() > 0 and min(obs.shape) > 1 else float("nan"),
    }


def ergodicity_check(P: np.ndarray) -> dict[str, Any]:
    """
    遍历性三件事：
      1. 不可约性：从任一等级能否到达任一其他等级（转移图强连通 + 所有等级可达）
      2. 非周期性：是否存在周期 > 1 的循环
      3. 稳态唯一性

    实现：对 P > 0 的支撑做图的可达性分析（Floyd-Warshall 式闭包）。
    """
    P = np.asarray(P, dtype=float)
    k = P.shape[0]
    reach = (P > 1e-12)

    # 可达闭包
    R = reach.copy()
    for m in range(k):
        R = R | (R[:, [m]] & R[[m], :])

    # 只有被视为"存在"的状态才参与（行和为 0 的状态不存在）
    alive = P.sum(axis=1) > 1e-12
    if alive.sum() == 0:
        return {"irreducible": False, "aperiodic": False, "unique_steady": False,
                "absorbing": [], "n_alive": 0}

    idx_alive = np.where(alive)[0]
    # 不可约：任意两个存活状态互相可达
    irreducible = True
    for i in idx_alive:
        for j in idx_alive:
            if not R[i, j]:
                irreducible = False
                break
        if not irreducible:
            break

    # 周期：若幂零图里存在长度 > 1 的环且 gcd > 1
    # 简化判定：用 P 的支撑图，检查每个状态到自身的回路长度集合的 gcd
    aperiodic = _is_aperiodic(reach, idx_alive)

    absorbing = [int(i) for i in idx_alive if P[i, i] >= 1.0 - 1e-12]

    return {
        "irreducible": bool(irreducible),
        "aperiodic": bool(aperiodic),
        "unique_steady": bool(irreducible and aperiodic),
        "absorbing": absorbing,
        "n_alive": int(alive.sum()),
    }


def _is_aperiodic(reach: np.ndarray, alive_idx: np.ndarray) -> bool:
    """
    周期判定：对每个存活状态求"回到自身的最短回路长度"集合的 gcd。
    这里用标准的 BFS 分层法：若状态 i 到自身存在长度 d 的回路，
    则图中存在周期 d。整体周期 = gcd(所有回路长度)。
    简化为：只要存在 P[i,i] > 0（自环长度 1），则周期必为 1 → 非周期。
    否则查是否有长度为 2、3 的回路，若 gcd 为 1 则非周期。
    """
    k = reach.shape[0]
    # 自环直接判定非周期
    for i in alive_idx:
        if reach[i, i]:
            return True
    # 无自环：求各状态回路长度的 gcd
    from math import gcd
    g = 0
    for i in alive_idx:
        # 找 i 出发回到 i 的最短路径长度
        dist = _bfs_cycle_length(reach, int(i))
        for d in dist:
            g = gcd(g, d)
            if g == 1:
                return True
    return g == 1


def _bfs_cycle_length(reach: np.ndarray, start: int) -> list[int]:
    """BFS 求 start 到自身的最短回路长度（限制在较小范围，够判定周期用）。"""
    k = reach.shape[0]
    seen = {start: 0}
    frontier = [start]
    lengths: list[int] = []
    d = 0
    while frontier and d < k + 1:
        d += 1
        nxt = []
        for u in frontier:
            for v in np.where(reach[u])[0]:
                if v == start:
                    lengths.append(d)
                elif v not in seen:
                    seen[int(v)] = d
                    nxt.append(int(v))
        frontier = nxt
    return lengths


# ------------------------------------------------------------------------------
# 主入口：传统马尔科夫链
# ------------------------------------------------------------------------------
def run_traditional(
    panel: pd.DataFrame,
    cres: ClassifyResult,
    lag: int = 1,
    estimator: str = "mle",
    bayes_prior: float = 1.0,
    extras: dict[str, Any] | None = None,
    name: str | None = None,
) -> MarkovResult:
    """跑传统马尔科夫链。panel 需已附加 class 列（列名 'class'）。"""
    ex = extras or {}
    n = cres.n_classes

    transitions = build_transitions(panel, class_col="class", lag=lag)
    if transitions.empty:
        raise RuntimeError(
            "没有构造出任何转移样本。\n"
            "  → 常见原因：每个省份只有 1 期数据（无法形成 t→t+1 转移）。\n"
            "  → 马尔科夫链至少需要 2 期数据。"
        )

    N = transition_counts(transitions, n)
    P = transition_probs(N, estimator=estimator, bayes_prior=bayes_prior)
    row_totals = N.sum(axis=1)

    res = MarkovResult(
        name=name or f"{cres.method} / 传统马尔科夫链",
        n_classes=n,
        labels=class_full_labels(n),
        counts=N,
        P=P,
        row_totals=row_totals,
        lag=lag,
        estimator=estimator,
        raw_transitions=transitions,
    )

    # 频数稀疏性检查
    empty_rows = np.where(row_totals == 0)[0]
    if len(empty_rows):
        res.notes.append(
            f"等级 {'、'.join(class_label(int(i)) for i in empty_rows)} "
            "从未作为转移起点出现（该等级样本未发生转移，或该等级无样本），"
            "对应概率行全为 0，不参与稳态分布计算"
        )
    n_zero = int(((N == 0)).sum())
    if n_zero > 0:
        res.notes.append(
            f"频数矩阵中有 {n_zero}/{n * n} 个格子为 0。"
            + ("样本量偏少时属正常，但需注意估计精度；"
               "可把 estimator 改为 'bayes' 做平滑。"
               if row_totals.sum() < 10 * n * n else
               "说明大量转移路径在实际中未发生（如等级跳级转移）")
        )

    # 附加指标
    if ex.get("steady_state", True):
        res.steady_state = steady_state(P)
    if ex.get("expected_duration", True):
        res.expected_duration = expected_durations(P)
    if ex.get("chi_square_test", True):
        res.chi2 = chi_square_independence(N, n)
    if ex.get("ergodicity_test", True):
        res.ergodicity = ergodicity_check(P)

    if estimator == "bayes":
        res.notes.append(
            f"使用 Bayes 平滑（伪计数={bayes_prior}），概率被轻微拉向均匀分布，"
            "行概率不会再出现严格的 0，与论文的极大似然结果会有细微差异"
        )
    return res


# ==============================================================================
# 二、空间马尔科夫链
# ==============================================================================
@dataclass
class SpatialMarkovResult:
    name: str
    n_classes: int
    group_labels: list[str]                 # 邻居组名称，如 ['Ⅰ','Ⅱ','Ⅲ','Ⅳ']
    group_display: list[str]
    # 三维：group × from × to
    counts: np.ndarray
    probs: np.ndarray
    row_totals: np.ndarray                  # (group, from)
    full_P: np.ndarray                      # 全样本（不分组）的转移矩阵，用于对比
    lag_values: pd.DataFrame | None = None  # 各省各期的空间滞后值明细
    notes: list[str] = field(default_factory=list)

    # ---- 输出 ----
    def prob_frames(self, digits: int = 2) -> dict[str, pd.DataFrame]:
        """每个邻居组一个转移概率 DataFrame。"""
        out: dict[str, pd.DataFrame] = {}
        for g in range(len(self.group_labels)):
            display = class_full_labels(self.n_classes)
            df = pd.DataFrame(
                np.round(self.probs[g] * 100, digits),
                index=[f"{d}" for d in display],
                columns=[f"{d}" for d in display],
            )
            df.index.name = "起始等级 \\ 目标等级"
            out[f"邻居为{self.group_display[g]}"] = df
        return out

    def count_frames(self) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for g in range(len(self.group_labels)):
            display = class_full_labels(self.n_classes)
            df = pd.DataFrame(
                self.counts[g].astype(int),
                index=[f"{d}" for d in display],
                columns=[f"{d}" for d in display],
            )
            df["行合计"] = self.row_totals[g].astype(int)
            df.index.name = "起始等级 \\ 目标等级"
            out[f"邻居为{self.group_display[g]}"] = df
        return out

    def spillover_table(self) -> pd.DataFrame:
        """
        空间溢出效应表：对每个起始等级，列出在不同邻居组下的转移概率分布。
        这是空间马尔科夫链最核心的一张表 —— 看同一个起始等级，
        邻居强弱不同时，向上转移/向下转移的概率怎么变。
        """
        n = self.n_classes
        rows = []
        for i in range(n):
            for g in range(len(self.group_labels)):
                tot = self.row_totals[g, i]
                if tot == 0:
                    continue
                p = self.probs[g, i]
                rows.append({
                    "起始等级": class_label(i),
                    "邻居组": self.group_display[g],
                    "样本数": int(tot),
                    "维持原等级": p[i],
                    "向上转移": p[i + 1:].sum() if i < n - 1 else 0.0,
                    "向下转移": p[:i].sum() if i > 0 else 0.0,
                    **{f"→{class_label(j)}": p[j] for j in range(n)},
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["起始等级", "邻居组"]).reset_index(drop=True)
        return df

    def upward_prob_matrix(self) -> pd.DataFrame:
        """起始等级 × 邻居组 的「向上转移概率」矩阵 —— 最直观的溢出效应视图。"""
        n = self.n_classes
        M = np.full((n, len(self.group_labels)), np.nan)
        for i in range(n):
            for g in range(len(self.group_labels)):
                if self.row_totals[g, i] > 0:
                    p = self.probs[g, i]
                    M[i, g] = p[i + 1:].sum() if i < n - 1 else 0.0
        return pd.DataFrame(
            M,
            index=[f"{class_label(i)} (n={int(self.row_totals[:, i].sum())})" for i in range(n)],
            columns=[f"邻居为{d}" for d in self.group_display],
        ).T

    def summary(self, max_rows: int = 40) -> str:
        n = self.n_classes
        lines = [f"【{self.name}】",
                 f"  邻居组划分：{len(self.group_labels)} 组 "
                 f"（{'、'.join(self.group_display)}）",
                 f"  有效转移样本 = {int(self.row_totals.sum())}"]

        lines.append("  ── 各邻居组下的转移概率矩阵（%）──")
        for g in range(len(self.group_labels)):
            lines.append(f"  ▶ 邻居为 {self.group_display[g]}"
                         f"（样本 {int(self.row_totals[g].sum())}）")
            hdr = "        " + "".join(f"{class_label(j):>10}" for j in range(n))
            lines.append(hdr)
            for i in range(n):
                row = "".join(f"{self.probs[g, i, j] * 100:>10.2f}" for j in range(n))
                lines.append(f"    {class_label(i):<4}{row}"
                             f"   (n={int(self.row_totals[g, i])})")

        # 溢出效应对比
        lines.append("  ── 空间溢出效应：同一起始等级在不同邻居环境下的向上转移概率 ──")
        for i in range(n):
            if self.row_totals[:, i].sum() == 0:
                continue
            parts = []
            for g in range(len(self.group_labels)):
                if self.row_totals[g, i] > 0:
                    up = self.probs[g, i, i + 1:].sum() if i < n - 1 else 0.0
                    parts.append(f"{self.group_display[g]}={up * 100:.1f}%")
            if parts:
                lines.append(f"    起始 {class_label(i)}：  " + "   ".join(parts))
        lines.append("  → 若「邻居越强、向上转移概率越高」，说明存在正向空间溢出；")
        lines.append("     反之（邻居强反而更难上升）说明存在竞争/虹吸效应。")

        for note in self.notes:
            lines.append(f"  ▸ {note}")
        return "\n".join(lines)


# ------------------------------------------------------------------------------
# 空间滞后值
# ------------------------------------------------------------------------------
def compute_lag_panel(
    panel: pd.DataFrame,
    W_norm: np.ndarray,
    W_raw: np.ndarray,
    provinces: list[str],
    value_col: str = C_VALUE,
    class_col: str = "class",
    neighbor_stat: str = "weighted_mean",
) -> pd.DataFrame:
    """
    逐期计算各省的空间滞后值。
    返回带 'lag_value' 和 'lag_class' 的 panel。
    """
    from spatial_weights import spatial_lag

    frames = []
    for y, sub in panel.groupby(C_YEAR, sort=True):
        s = sub.set_index(C_PROV)[value_col].reindex(provinces)

        # ---- 缺省值处理：某省某年没有数据时，空间滞后无法算 ----
        n_missing = int(s.isna().sum())
        if n_missing:
            # 用该省在全样本的均值临时补位，避免整年滞后全废
            fill = panel.groupby(C_PROV)[value_col].mean().reindex(provinces)
            s = s.fillna(fill)
            logger.debug(f"  {int(y)} 年有 {n_missing} 个省份缺值，"
                         "已用该省全样本均值临时填充以计算空间滞后")

        lag = spatial_lag(s, W_norm, provinces,
                          neighbor_stat=neighbor_stat, W_raw=W_raw)
        tmp = pd.DataFrame({
            C_PROV: provinces,
            C_YEAR: int(y),
            "value": s.to_numpy(),
            "lag_value": lag.to_numpy(),
        })
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)
    # 合并回 class
    lab = panel.set_index([C_PROV, C_YEAR])[class_col]
    out["class"] = [lab.get((p, int(y)), -1) for p, y in
                    zip(out[C_PROV], out[C_YEAR])]
    out["class"] = out["class"].astype(int)
    # ★ 统一列名为内部标准名 province / year，避免下游按不同列名取数时 KeyError
    out = out.rename(columns={C_PROV: "province", C_YEAR: "year"})
    return out


def lag_groups_by_class(
    lag_panel: pd.DataFrame,
    n_quantiles: int,
    boundaries: list[float],
    group_kind: str = "class",
    n_bins: int = 2,
) -> pd.DataFrame:
    """
    给空间滞后值分组。

    group_kind='class'      : 用与传统矩阵相同的分位边界把滞后值分 n_quantiles 组
                              （论文口径，产出 4×4×4 三维矩阵）
    group_kind='value_block': 按中位数（或 n_bins 分位）分 2 组（低/高）
    """
    df = lag_panel.copy()
    v = df["lag_value"].to_numpy(dtype=float)

    if group_kind == "value_block":
        vv = v[~np.isnan(v)]
        if vv.size == 0:
            raise RuntimeError("空间滞后值全为空，无法分组。请检查空间权重矩阵是否有孤立省份。")
        qs = np.quantile(vv, np.linspace(0, 1, int(n_bins) + 1))
        edges = np.unique(qs)
        lab = np.searchsorted(edges[1:-1], np.nan_to_num(v, nan=-1e18), side="right")
        lab = np.clip(lab, 0, len(edges) - 2)
        lab = np.where(np.isnan(v), -1, lab)
        df["lag_group"] = lab.astype(int)
        if int(n_bins) == 2:
            df["lag_group_name"] = df["lag_group"].map({0: "低", 1: "高", -1: "无邻居"})
        else:
            df["lag_group_name"] = df["lag_group"].map(
                lambda k: f"第{k + 1}档" if k >= 0 else "无邻居")
        return df

    # 用给定的分位边界分组
    b = np.asarray(boundaries, dtype=float)
    nb = len(b) - 1
    lab = np.searchsorted(b[1:-1], np.nan_to_num(v, nan=-1e18), side="right")
    lab = np.clip(lab, 0, nb - 1)
    lab = np.where(np.isnan(v), -1, lab)
    df["lag_group"] = lab.astype(int)
    df["lag_group_name"] = df["lag_group"].map(
        lambda k: class_label(k) if k >= 0 else "无邻居")
    return df


# ------------------------------------------------------------------------------
# 主入口：空间马尔科夫链
# ------------------------------------------------------------------------------
def run_spatial(
    panel: pd.DataFrame,
    cres: ClassifyResult,
    W_norm: np.ndarray,
    W_raw: np.ndarray,
    provinces: list[str],
    lag: int = 1,
    neighbor_stat: str = "weighted_mean",
    neighbor_grouping: str = "class",
    value_block_bins: int = 2,
    name: str | None = None,
) -> SpatialMarkovResult:
    """跑空间马尔科夫链。"""
    ex_notes: list[str] = []
    n = cres.n_classes

    # ---- 1. 算空间滞后 ----
    logger.info("  计算空间滞后值（每期每省的邻居加权平均）…")
    lag_panel_raw = compute_lag_panel(
        panel, W_norm, W_raw, provinces,
        neighbor_stat=neighbor_stat,
    )

    n_no_neighbor = int(lag_panel_raw["lag_value"].isna().sum())
    if n_no_neighbor:
        ex_notes.append(
            f"有 {n_no_neighbor} 个「省-年」观测没有邻居（空间权重矩阵中该省全零行），"
            "已从空间转移统计中剔除。若数量较多，请检查矩阵的 isolate_fix 设置"
        )

    # ---- 2. 给滞后值分组 ----
    boundaries: list[float] = []
    if neighbor_grouping == "class":
        # 邻居组要与"自身等级组"用同一套边界，这样三维矩阵的每一层才有可比性。
        # cres.boundaries 的长度：四分位法是 n+1（[p0,p25,p50,p75,p100]），
        # KMeans 也是 n+1（[-inf, b1, b2, b3, inf]）。两者都能直接用。
        boundaries = [float(x) for x in (cres.boundaries or [])]
        if len(boundaries) != n + 1:
            logger.warning(
                f"  分级边界长度为 {len(boundaries)}，与等级数 {n} 不匹配"
                f"（应为 {n + 1}），邻居分组改为按邻居滞后值的分位数重新计算。"
                "此时「邻居组」与「自身等级」的边界不一致，对比时需注意口径差异"
            )
            vv = lag_panel_raw["lag_value"].dropna().to_numpy()
            if vv.size == 0:
                raise RuntimeError(
                    "所有观测的空间滞后值都为空，无法分组。\n"
                    "  → 请检查空间权重矩阵是否每个省份都有邻居"
                    "（config 里 spatial_weights.adjacency.isolate_fix 设为 'nearest'）。"
                )
            boundaries = [float(x) for x in np.quantile(vv, np.linspace(0, 1, n + 1))]

        # 把 ±inf 边界换成有限值（KMeans 的边界首尾是 ±inf）
        lag_min = float(lag_panel_raw["lag_value"].min())
        lag_max = float(lag_panel_raw["lag_value"].max())
        if not np.isfinite(boundaries[0]):
            boundaries[0] = lag_min - 1.0
        if not np.isfinite(boundaries[-1]):
            boundaries[-1] = lag_max + 1.0

        # 关键：KMeans 的簇边界是基于"指标值"算的，而邻居滞后值的
        # 数值分布与指标值不同（滞后是加权平均，方差更小）。
        # 若直接用指标边界切滞后值，可能出现绝大多数观测挤在一档的情况。
        # 这里做一个适配性检查：若某档样本占比 > 90%，改用滞后值自身分位点。
        test_lab = lag_groups_by_class(lag_panel_raw, n, boundaries,
                                       group_kind="class")
        valid = test_lab["lag_group"] >= 0
        if valid.sum() > 0:
            vc = test_lab.loc[valid, "lag_group"].value_counts(normalize=True)
            if vc.max() > 0.90:
                logger.warning(
                    f"  用自身等级边界切邻居滞后值时，有 {vc.max() * 100:.1f}% 的观测"
                    "落在同一档（滞后值方差小于指标值导致），"
                    "已自动改用邻居滞后值自身的分位数作为分组边界。"
                    "此时邻居组描述的是「邻居相对强弱」，与自身等级边界不完全一致"
                )
                vv = lag_panel_raw["lag_value"].dropna().to_numpy()
                boundaries = [float(x) for x in np.quantile(vv, np.linspace(0, 1, n + 1))]

    lag_panel = lag_groups_by_class(
        lag_panel_raw, n, boundaries,
        group_kind=neighbor_grouping,
        n_bins=value_block_bins,
    )

    n_groups = n if neighbor_grouping == "class" else int(value_block_bins)
    if neighbor_grouping == "class":
        group_display = [f"{class_full_labels(n)[g]}" for g in range(n_groups)]
    else:
        group_display = (["低水平邻居", "高水平邻居"] if n_groups == 2
                         else [f"第{g + 1}档邻居" for g in range(n_groups)])

    # ---- 3. 三维转移统计 ----
    # 把滞后组信息挂到面板上（同一省同一年的起始状态用当年的邻居环境）
    merged = lag_panel[["province", "year", "lag_group"]].rename(
        columns={"province": C_PROV, "year": C_YEAR})
    panel2 = panel.merge(merged, on=[C_PROV, C_YEAR], how="left")

    transitions = build_transitions(panel2, class_col="class", lag=lag)
    # 把起始年的邻居组挂到转移对上。
    # ★ 注意：build_transitions 输出的是中文列名（省份 / 起始年份），
    #   这里必须按输出列名合并，不能用 C_PROV，否则 KeyError: 'province'。
    tg = merged.rename(columns={C_PROV: "省份", C_YEAR: "起始年份",
                                "lag_group": "起始邻居组"})
    transitions = transitions.merge(tg, on=["省份", "起始年份"], how="left")

    n_drop = int(transitions["起始邻居组"].isna().sum())
    if n_drop:
        ex_notes.append(f"{n_drop} 条转移因起始年缺少邻居组信息被剔除")
        transitions = transitions.dropna(subset=["起始邻居组"])

    counts = np.zeros((n_groups, n, n), dtype=float)
    for g, f, t in zip(transitions["起始邻居组"].to_numpy(dtype=float),
                       transitions["起始等级"].to_numpy(),
                       transitions["目标等级"].to_numpy()):
        gi = int(g)
        if 0 <= gi < n_groups and 0 <= f < n and 0 <= t < n:
            counts[gi, int(f), int(t)] += 1.0

    probs = np.zeros_like(counts)
    row_totals = np.zeros((n_groups, n), dtype=float)
    for g in range(n_groups):
        Pg = transition_probs(counts[g], estimator="mle")
        probs[g] = Pg
        row_totals[g] = counts[g].sum(axis=1)

    # 全样本矩阵（作为对比基准）
    full_N = transition_counts(transitions, n)
    full_P = transition_probs(full_N)

    # ---- 4. 稀疏性提示 ----
    empty_cells = int((row_totals == 0).sum())
    if empty_cells:
        ex_notes.append(
            f"三维矩阵中有 {empty_cells}/{n_groups * n} 个「邻居组×起始等级」格子没有样本，"
            "对应概率为 0（不是真的 0 概率，而是没有观测到）。"
            "解读时不要把这些空格当成「绝不可能发生」"
        )
    if row_totals.sum() < 5 * n_groups * n:
        ex_notes.append(
            f"空间转移样本偏少（{int(row_totals.sum())} 条 / {n_groups * n * n} 个格子），"
            "个别条件概率的抽样误差较大。建议：① 合并邻居组（把 neighbor_grouping "
            "改为 'value_block' 且 value_block_bins=2）；② 或增大样本年份跨度"
        )

    res = SpatialMarkovResult(
        name=name or f"{cres.method} / 空间马尔科夫链",
        n_classes=n,
        group_labels=[class_label(g) if neighbor_grouping == "class" else f"G{g + 1}"
                      for g in range(n_groups)],
        group_display=group_display,
        counts=counts,
        probs=probs,
        row_totals=row_totals,
        full_P=full_P,
        lag_values=lag_panel,
        notes=ex_notes,
    )
    return res


# ------------------------------------------------------------------------------
# 空间溢出显著性：把"有邻居组"和"无邻居组"的转移分布做对比
# ------------------------------------------------------------------------------
def spillover_test(res: SpatialMarkovResult, base_level: int) -> dict[str, Any]:
    """
    对指定起始等级，检验"邻居环境"是否显著影响转移方向。

    做法：把 K 个邻居组的转移去向做成 K×n 列联表，做卡方检验。
    H0：邻居环境不影响转移方向。
    """
    n = res.n_classes
    if base_level < 0 or base_level >= n:
        raise ValueError(f"起始等级 {base_level} 超出范围")
    rows = []
    for g in range(res.row_totals.shape[0]):
        if res.row_totals[g, base_level] > 0:
            rows.append(res.counts[g, base_level])
    if len(rows) < 2:
        return {"ok": False, "reason": "有样本的邻居组少于 2 个，无法做对比检验"}
    obs = np.array(rows, dtype=float)
    # 去掉全零列
    keep = obs.sum(axis=0) > 0
    obs = obs[:, keep]
    if obs.shape[1] < 2:
        return {"ok": False, "reason": "所有邻居组下的转移去向都集中在同一等级，无变异"}
    try:
        chi2, p, dof, expected = stats.chi2_contingency(obs, correction=False)
    except Exception as e:
        return {"ok": False, "reason": f"检验失败：{e}"}
    return {
        "ok": True,
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "n_small_cells": int((expected < 5).sum()),
        "obs": obs,
        "conclusion": ("邻居环境对转移方向有显著影响（空间溢出成立）"
                       if p < 0.05 else
                       "邻居环境对转移方向影响不显著，空间溢出效应较弱"),
    }
