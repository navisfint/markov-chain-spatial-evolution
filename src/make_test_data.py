# -*- coding: utf-8 -*-
"""
make_test_data.py —— 生成测试数据
================================================
用途：在拿到真实数据之前，验证整套代码能不能跑通。

生成三份测试数据（模拟 2000—2024 年 31 省共同/富裕指数）：

  test_long.csv   长表：省份 / 年份 / cwt_index（已有算好的指数）
  test_wide.csv   宽表：行=年份，列=31 个省份
  test_raw.csv    原始指标表：省份 / 年份 / 8 个模拟指标（含正向与负向）

数据生成逻辑（不是随便造，要让马尔科夫链有「有意义」的结构）：
  · 各省有一个持久的基础水平（模拟东部高、西部低）
  · 逐年缓慢上行（模拟共同富裕水平的整体提升）
  · 叠加省份固定效应 + 随机扰动（模拟路径依赖与偶发跃迁）
  · 少数省份在中后期明显加速（模拟政策冲击导致向上跃迁）

运行：
    python src/make_test_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from province_data import PROVINCES, REGION_MAP  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

YEARS = list(range(2000, 2025))     # 2000—2024
SEED = 20260923


def generate_index_panel(seed: int = SEED) -> pd.DataFrame:
    """
    生成"共同富裕指数"面板。

    结构：value_{i,t} = base_i + trend_t + persist_{i,t} + noise
      base_i    : 省份基础水平（东部高、中部中、西部低）
      trend_t   : 全国共同的时间趋势（缓慢上行）
      persist   : 一阶自回归项，制造路径依赖（这是马尔科夫链显著的来源）
      noise     : 白噪声
    另加：若干省份在特定年份获得"政策性跃升"
    """
    rng = np.random.default_rng(seed)

    # ---- 省份基础水平 ----
    region_base = {"东部": 0.30, "中部": 0.20, "西部": 0.15}
    base = {}
    for p in PROVINCES:
        r = REGION_MAP.get(p, "西部")
        # 同区域内再拉开差距
        base[p] = region_base[r] + rng.normal(0, 0.045)
    # 手动让几个典型省份位置符合常识（便于结果看起来"合理"）
    for p, v in {"北京": 0.42, "上海": 0.45, "天津": 0.34, "江苏": 0.34,
                 "浙江": 0.36, "广东": 0.33, "海南": 0.16, "西藏": 0.10,
                 "甘肃": 0.12, "贵州": 0.13}.items():
        base[p] = v

    # ---- 时间趋势：2000—2024 缓慢上行 ----
    n_y = len(YEARS)
    trend = np.linspace(0.0, 0.26, n_y) + 0.012 * np.sin(np.arange(n_y) / 2.6)

    # ---- 逐省逐期生成 ----
    rows = []
    for p in PROVINCES:
        # 一阶自回归：persist_t = rho * persist_{t-1} + shock
        rho = 0.85
        persist = 0.0
        # 该省是否在某个年份获得跃升（随机挑约 1/3 的省份）
        jump_year = None
        jump_size = 0.0
        if rng.random() < 0.34:
            jump_year = int(rng.integers(2008, 2022))
            jump_size = float(rng.uniform(0.06, 0.16))

        for k, y in enumerate(YEARS):
            persist = rho * persist + rng.normal(0, 0.030)
            v = base[p] + trend[k] + persist + rng.normal(0, 0.012)
            if jump_year is not None and y >= jump_year:
                # 跃升后逐步释放，而不是一步到位
                v += jump_size * min(1.0, (y - jump_year + 1) / 2.0)
            # 压缩到 0~1，与文献里的指数区间一致
            v = float(np.clip(v, 0.02, 0.98))
            rows.append({"province": p, "year": y, "cwt_index": round(v, 4)})

    return pd.DataFrame(rows)


def generate_raw_indicators(index_df: pd.DataFrame, seed: int = SEED + 1) -> pd.DataFrame:
    """
    反推一套"原始指标"，使其熵权-TOPSIS 合成结果与 index_df 大致同向。
    只是为了测试 raw 模式能跑通，不追求指标间真实的经济逻辑。
    """
    rng = np.random.default_rng(seed)
    df = index_df.copy()
    n = len(df)

    # 8 个模拟指标：6 正向 + 2 负向
    out = pd.DataFrame({
        "province": df["province"],
        "year": df["year"],
        # 正向指标：与指数同向 + 噪声（不同尺度，测试量纲处理）
        "per_gdp": (df["cwt_index"] * 90000 + rng.normal(0, 4000, n)).round(1),
        "third_industry_ratio": np.clip(
            df["cwt_index"] * 80 + rng.normal(0, 4, n), 20, 95).round(2),
        "urban_income": (df["cwt_index"] * 60000 + rng.normal(0, 2500, n)).round(1),
        "rnd_ratio": np.clip(df["cwt_index"] * 5 + rng.normal(0, 0.4, n),
                             0.2, 8).round(3),
        "beds_per_1k": np.clip(df["cwt_index"] * 10 + rng.normal(0, 0.8, n),
                               1.5, 12).round(2),
        "library_books_per_capita": np.clip(
            df["cwt_index"] * 2.2 + rng.normal(0, 0.3, n), 0.1, 3.5).round(3),
        # 负向指标：与指数反向（指数越高，值越低）
        "urban_engel": np.clip(48 - df["cwt_index"] * 40 + rng.normal(0, 1.6, n),
                               18, 55).round(2),
        "urban_rural_income_gap": np.clip(
            4.2 - df["cwt_index"] * 3.2 + rng.normal(0, 0.14, n), 1.4, 4.5).round(3),
    })
    return out


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel = generate_index_panel()

    # ---- 1. 长表 ----
    p_long = DATA_DIR / "test_long.csv"
    panel.to_csv(p_long, index=False, encoding="utf-8-sig")
    print(f"[OK] 长表      {p_long}   {panel.shape}")

    # ---- 2. 宽表 ----
    wide = panel.pivot(index="year", columns="province", values="cwt_index")
    wide.index.name = "year"
    p_wide = DATA_DIR / "test_wide.csv"
    wide.to_csv(p_wide, encoding="utf-8-sig")
    print(f"[OK] 宽表      {p_wide}   {wide.shape}")

    # ---- 3. 原始指标表 ----
    raw = generate_raw_indicators(panel)
    p_raw = DATA_DIR / "test_raw.csv"
    raw.to_csv(p_raw, index=False, encoding="utf-8-sig")
    print(f"[OK] 原始指标  {p_raw}   {raw.shape}")

    # ---- 4. 自定义空间矩阵示例（经济地理嵌套的手写版）----
    from spatial_weights import build_adjacency, build_economic
    W, _ = build_adjacency(PROVINCES, mode="rook", isolate_fix="nearest")
    econ = panel.groupby("province")["cwt_index"].mean().to_dict()
    W2, _ = build_economic(PROVINCES, econ, inverse=True, variable_name="均值指数")
    Wn = W * W2
    from spatial_weights import _row_normalize
    Wn = _row_normalize(Wn)
    cdf = pd.DataFrame(Wn, index=PROVINCES, columns=PROVINCES)
    cdf.index.name = "province"
    p_custom = DATA_DIR / "spatial_weight_custom.csv"
    cdf.round(6).to_csv(p_custom, encoding="utf-8-sig")
    print(f"[OK] 自定义矩阵 {p_custom}   {cdf.shape}")

    # ---- 概览 ----
    print()
    print("=" * 66)
    print("测试数据概览")
    print("=" * 66)
    print(f"  年份：{panel['year'].min()} – {panel['year'].max()}"
          f"（{panel['year'].nunique()} 年）")
    print(f"  省份：{panel['province'].nunique()}")
    print(f"  观测：{len(panel)}")
    print(f"  指数：min={panel['cwt_index'].min():.4f}  "
          f"max={panel['cwt_index'].max():.4f}  "
          f"mean={panel['cwt_index'].mean():.4f}")
    print()
    print("  各年指数均值（前 3 / 后 3 年）：")
    ym = panel.groupby("year")["cwt_index"].mean()
    for y in list(ym.index[:3]) + list(ym.index[-3:]):
        print(f"    {y}: {ym[y]:.4f}")
    print()
    print("  提示：这些是**模拟数据**，仅用于验证代码流程。")
    print("       请把 config.yaml 的 data.file 换成你的真实数据文件名。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
