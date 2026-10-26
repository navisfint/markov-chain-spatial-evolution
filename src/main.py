# -*- coding: utf-8 -*-
"""
main.py —— 主流程
================================================
把各模块串起来：

  读数据 → 分级（四分位 / KMeans 两套） → 逐个空间权重矩阵
  → 传统马尔科夫链 + 空间马尔科夫链 → 出表 + 出图 + 汇总

运行：
    python src/main.py
    python src/main.py --config config.yaml
    python src/main.py --shape long          # 强制指定数据形态
    python src/main.py --weights adjacency   # 只跑指定矩阵（逗号分隔）
    python src/main.py --data path/to/x.xlsx # 临时换数据文件
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# 让 src 目录下的模块能互相 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import attach_labels, classify, yearly_distribution                # noqa: E402
from loader import C_PROV, C_YEAR, C_VALUE, load_panel, to_matrix, read_table     # noqa: E402
from markov import (run_spatial, run_traditional, spillover_test,               # noqa: E402
                    build_transitions)
from spatial_weights import build_matrix, neighbor_contribution                  # noqa: E402
from utils import (OutputManager, banner, cfg_get, get_logger, save_excel_workbook,  # noqa: E402
                   save_table, setup_logger, class_full_labels, class_label)
import visualize as viz                                                          # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# 原始变量提取（供经济距离矩阵使用）
# ==============================================================================
def extract_econ_vars(raw_df: pd.DataFrame, cfg: dict) -> dict[str, pd.Series]:
    """
    从原始数据里抽取"跨年求均值"的省份级数值变量，供经济距离矩阵用。

    返回 {'per_gdp': Series(index=省份), ...}
    """
    from loader import find_col, _PROV_KEYS, _YEAR_KEYS
    from province_data import normalize_province

    col_map = cfg.get("col_map") or {}
    prov_col = col_map.get("province") or find_col(raw_df, _PROV_KEYS)

    if prov_col is None:
        # 宽表：列是省份，没法抽经济变量
        return {}

    df = raw_df.copy()
    df["__prov"] = df[prov_col].map(normalize_province)
    df = df[df["__prov"].notna()]
    if df.empty:
        return {}

    exclude = {prov_col, "__prov"}
    year_col = col_map.get("year") or find_col(raw_df, _YEAR_KEYS)
    if year_col:
        exclude.add(year_col)

    out: dict[str, pd.Series] = {}
    for c in df.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        # 跳过明显是"指数结果"的列和年份类整数列，它们不是经济水平变量
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        cn = str(c).strip().lower()
        if cn in ("index", "cwt_index", "value", "score", "得分", "指数", "共同富裕"):
            continue
        # 全为整数且取值范围像年份 → 跳过
        if s.between(1900, 2200).mean() > 0.9:
            continue
        agg = df.groupby("__prov")[c].mean()
        if agg.notna().sum() >= 3:
            out[str(c)] = agg
    return out


# ==============================================================================
# 结果收集
# ==============================================================================
class ResultBundle:
    """收集所有结果表，最后统一写 CSV + 汇总 Excel。"""

    def __init__(self) -> None:
        self.sheets: dict[str, pd.DataFrame] = {}
        self.files: list[Path] = []
        self.text_sections: list[str] = []

    def add(self, name: str, df: pd.DataFrame | None, om: OutputManager,
            save_csv: bool = True) -> None:
        if df is None:
            return
        obj = df.to_frame() if isinstance(df, pd.Series) else df
        if obj.empty:
            return
        # sheet 名去重
        key = name
        i = 2
        while key in self.sheets:
            key = f"{name}_{i}"
            i += 1
        self.sheets[key] = obj
        if save_csv:
            p = om.table_path(name)
            save_table(obj, p, get_logger())
            self.files.append(p)

    def add_text(self, title: str, text: str) -> None:
        self.text_sections.append(f"\n{'=' * 74}\n{title}\n{'=' * 74}\n{text}\n")


# ==============================================================================
# 主流程
# ==============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="马尔科夫链动态演进分析（共同富裕水平 / 任意指数）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"),
                    help="配置文件路径，默认 config.yaml")
    ap.add_argument("--data", default=None, help="临时指定数据文件，覆盖配置里的 data.file")
    ap.add_argument("--shape", default=None,
                    choices=["auto", "long", "wide", "raw", "wide_transposed"],
                    help="临时指定数据形态")
    ap.add_argument("--weights", default=None,
                    help="只跑指定空间矩阵（逗号分隔），如 adjacency,economic")
    ap.add_argument("--methods", default=None,
                    help="只跑指定分级方法（逗号分隔），如 quartile")
    ap.add_argument("--outdir", default=None, help="临时指定输出目录")
    args = ap.parse_args(argv)

    # ---------------- 配置与日志 ----------------
    cfg_path = Path(args.config)
    from utils import load_config
    cfg = load_config(cfg_path)

    out_dir = Path(args.outdir) if args.outdir else Path(
        cfg_get(cfg, "output", "dir", default="output"))
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    om = OutputManager(out_dir)

    log_cfg = cfg.get("log") or {}
    log_file = om.root / "run.log" if log_cfg.get("to_file", True) else None
    logger = setup_logger(str(log_cfg.get("level", "INFO")), log_file)

    t_start = pd.Timestamp.now()
    banner(logger, "马尔科夫链动态演进分析  ——  开始")
    logger.info(f"  配置文件：{cfg_path}")
    logger.info(f"  输出目录：{om.root}")
    logger.info(f"  启动时间：{t_start:%Y-%m-%d %H:%M:%S}")

    bundle = ResultBundle()

    # ---------------- 1. 读数据 ----------------
    banner(logger, "STEP 1 / 6  读取数据")
    data_dir = PROJECT_ROOT / "data"
    data_name = args.data or cfg_get(cfg, "data", "file", default="")
    if not data_name:
        logger.error(
            "配置里没有指定数据文件（data.file）。\n"
            f"  → 请把你的数据文件放到 {data_dir} 目录，\n"
            "     然后在 config.yaml 的 data.file 里填写文件名。"
        )
        return 2
    data_path = Path(data_name)
    if not data_path.is_absolute():
        data_path = data_dir / data_path

    try:
        panel, info = load_panel(data_path, cfg, shape_override=args.shape)
    except Exception as e:
        logger.error(f"数据读取失败：{e}")
        logger.debug(traceback.format_exc())
        _write_error(om, "数据读取失败", e)
        return 2

    # 保存清洗后的标准面板，方便核对
    bundle.add("00_标准化面板", panel.rename(columns={
        C_PROV: "省份", C_YEAR: "年份", C_VALUE: "指数值"}), om)

    if info.get("shape") == "raw":
        w = info.get("weights") or {}
        e = info.get("entropies") or {}
        if w:
            wdf = pd.DataFrame({
                "指标": list(w.keys()),
                "信息熵": [e.get(k, np.nan) for k in w.keys()],
                "权重": list(w.values()),
            }).sort_values("权重", ascending=False).reset_index(drop=True)
            bundle.add("01_熵权TOPSIS权重", wdf.set_index("指标"), om)

    matrix = to_matrix(panel)
    logger.info(f"  指数矩阵（年×省）：{matrix.shape[0]} 行 × {matrix.shape[1]} 列")

    # ---------------- 2. 分级 ----------------
    banner(logger, "STEP 2 / 6  等级划分")
    ccfg = cfg.get("classification") or {}
    n_classes = int(ccfg.get("n_classes", 4))
    methods = args.methods.split(",") if args.methods else list(
        ccfg.get("methods") or ["quartile", "kmeans"])
    methods = [m.strip() for m in methods if m.strip()]
    if not methods:
        logger.error("classification.methods 为空，至少填一个：quartile / kmeans")
        return 2

    results: list[tuple[str, object, pd.DataFrame]] = []   # (key, ClassifyResult, panel_with_class)

    for m in methods:
        try:
            cres = classify(
                panel, m, n_classes=n_classes,
                quartile_basis=str(ccfg.get("quartile_basis", "pooled")),
                kmeans_cfg=ccfg.get("kmeans"),
            )
        except Exception as e:
            logger.error(f"分级方法「{m}」失败：{e}")
            logger.debug(traceback.format_exc())
            continue

        key = cres.method if cres.method == "quartile" else "kmeans"
        pl = attach_labels(panel, cres)
        results.append((key, cres, pl))

        logger.info(f"  ▸ {m}")
        for line in cres.summary().split("\n"):
            logger.info(f"    {line}")

        # 输出边界、频数、逐年构成
        cname = f"10_{key}_分档边界"
        if cres.method == "quartile":
            bdf = pd.DataFrame({
                "档位": class_full_labels(n_classes),
                "下界": cres.boundaries[:-1],
                "上界": cres.boundaries[1:],
            })
            bundle.add(cname, bdf, om)
        else:
            bundle.add(cname, pd.DataFrame({
                "档位": class_full_labels(n_classes),
                "簇中心": cres.centroids,
            }), om)

        cnt = pd.Series(cres.counts).rename("样本数").to_frame()
        cnt.index = [class_label(int(i)) for i in cnt.index]
        cnt["占比"] = cnt["样本数"] / cnt["样本数"].sum()
        bundle.add(f"11_{key}_等级分布", cnt, om)

        yd = yearly_distribution(pl)
        yd.columns = [class_label(int(c)) for c in yd.columns]
        bundle.add(f"12_{key}_逐年等级构成", yd, om)
        bundle.add_text(f"等级划分 · {m}", cres.summary())

    if not results:
        logger.error("所有分级方法都失败了，流程终止。请检查上面的报错信息。")
        return 1

    # ---------------- 3. 空间权重矩阵 ----------------
    banner(logger, "STEP 3 / 6  空间权重矩阵")
    sw_cfg = cfg.get("spatial_weights") or {}
    active = args.weights.split(",") if args.weights else list(sw_cfg.get("active") or [])
    active = [a.strip() for a in active if a.strip()]

    # 经济变量（供 economic / eco_geographic 使用）
    econ_vars: dict[str, pd.Series] = {}
    if any(a.lower() in ("economic", "eco_geographic", "eco-geographic", "nested")
           for a in active):
        try:
            raw_df = read_table(data_path)
            econ_vars = extract_econ_vars(raw_df, cfg)
            if econ_vars:
                logger.info(f"  可用作经济距离的经济变量：{sorted(econ_vars.keys())[:20]}")
            else:
                logger.warning("  未能从数据里提取任何省份级经济变量")
        except Exception as e:
            logger.warning(f"  提取经济变量失败：{e}")

    # 省份顺序：优先用数据里出现的省份；若数据只有部分省份，矩阵按数据里的省份构造
    provinces = info["provinces"]
    logger.info(f"  空间矩阵维度按数据中的 {len(provinces)} 个省份构造")
    if len(provinces) != 31:
        logger.warning(
            f"  数据里只有 {len(provinces)} 个省份（标准应为 31）。"
            "空间权重矩阵会用这 %d 个省份重新构造，结果是「这 %d 省内部」的空间关系，"
            "与全国 31 省的结论不可直接比较。" % (len(provinces), len(provinces))
        )

    weight_matrices = []
    for wname in active:
        try:
            wm = build_matrix(wname, provinces, sw_cfg,
                              context={"econ_vars": econ_vars, "data_dir": data_dir})
        except Exception as e:
            logger.error(f"  构建空间矩阵「{wname}」失败：{e}")
            logger.debug(traceback.format_exc())
            continue
        weight_matrices.append(wm)
        logger.info(f"  ▸ {wm.name}")
        for line in wm.summary().split("\n"):
            logger.info(f"    {line}")
        bundle.add(f"20_空间矩阵_{wname}", wm.to_frame(normalized=False), om)
        bundle.add(f"21_空间矩阵_{wname}_行标准化", wm.to_frame(normalized=True), om)
        bundle.add(f"22_空间矩阵_{wname}_邻居数",
                   wm.neighbor_count().to_frame().T, om)
        bundle.add_text(f"空间权重矩阵 · {wm.name}", wm.summary())

    # ---------------- 4. 传统马尔科夫链 ----------------
    banner(logger, "STEP 4 / 6  传统马尔科夫链")
    mcfg = cfg.get("markov") or {}
    tcfg = mcfg.get("traditional") or {}
    scfg = mcfg.get("spatial") or {}
    ecfg = mcfg.get("extras") or {}
    lag = int(tcfg.get("lag", 1))
    estimator = str(tcfg.get("estimator", "mle"))

    trad_results: list[tuple[str, object]] = []   # (key, MarkovResult)
    if tcfg.get("enabled", True):
        for key, cres, pl in results:
            try:
                res = run_traditional(
                    pl, cres, lag=lag, estimator=estimator,
                    bayes_prior=float(tcfg.get("bayes_prior", 1.0)),
                    extras=ecfg,
                    name=f"{cres.method}分级 / 传统马尔科夫链",
                )
            except Exception as e:
                logger.error(f"  传统马尔科夫链（{key}）失败：{e}")
                logger.debug(traceback.format_exc())
                continue
            trad_results.append((key, res))
            logger.info(f"  ▸ {res.name}")
            for line in res.summary().split("\n"):
                logger.info(f"    {line}")

            bundle.add(f"30_{key}_传统_概率矩阵", res.prob_frame(), om)
            bundle.add(f"31_{key}_传统_频数矩阵", res.count_frame(), om)
            if res.steady_state is not None:
                bundle.add(f"32_{key}_传统_稳态分布", pd.DataFrame({
                    "等级": class_full_labels(n_classes),
                    "稳态概率": res.steady_state,
                    "期望持续时间(年)": res.expected_duration
                    if res.expected_duration is not None else np.nan,
                }).set_index("等级"), om)
            if res.raw_transitions is not None:
                bundle.add(f"33_{key}_传统_转移明细", res.raw_transitions, om)
            if res.chi2 and res.chi2.get("ok"):
                bundle.add(f"34_{key}_传统_卡方检验",
                           pd.DataFrame([{k: v for k, v in res.chi2.items()
                                          if k != "ok"}]), om)
            bundle.add_text(f"传统马尔科夫链 · {res.name}", res.summary())
    else:
        logger.info("  配置里 traditional.enabled = false，跳过")

    # ---------------- 5. 空间马尔科夫链 ----------------
    banner(logger, "STEP 5 / 6  空间马尔科夫链")
    spatial_results: list[tuple[str, str, object]] = []   # (method_key, wname, SpatialMarkovResult)

    if scfg.get("enabled", True):
        if not weight_matrices:
            logger.warning("  没有任何可用的空间权重矩阵，空间马尔科夫链跳过。\n"
                           "    → 请检查 config.yaml 的 spatial_weights.active。")
        for key, cres, pl in results:
            for wm in weight_matrices:
                try:
                    sres = run_spatial(
                        pl, cres,
                        W_norm=wm.W_norm, W_raw=wm.W,
                        provinces=provinces,
                        lag=lag,
                        neighbor_stat=str(scfg.get("neighbor_stat", "weighted_mean")),
                        neighbor_grouping=str(scfg.get("neighbor_grouping", "class")),
                        value_block_bins=int(scfg.get("value_block_bins", 2)),
                        name=f"{cres.method}分级 / 空间马尔科夫链（{wm.name}）",
                    )
                except Exception as e:
                    logger.error(f"  空间马尔科夫链（{key} / {wm.name}）失败：{e}")
                    logger.debug(traceback.format_exc())
                    continue
                spatial_results.append((key, wm.name, sres))
                logger.info(f"  ▸ {sres.name}")
                for line in sres.summary().split("\n"):
                    logger.info(f"    {line}")

                prefix = f"40_{key}_{_short(wm.name)}"
                for gname, df in sres.prob_frames().items():
                    bundle.add(f"{prefix}_概率_{gname}", df, om)
                for gname, df in sres.count_frames().items():
                    bundle.add(f"{prefix}_频数_{gname}", df, om)
                bundle.add(f"{prefix}_空间溢出对比", sres.spillover_table(), om)
                bundle.add(f"{prefix}_向上转移概率矩阵", sres.upward_prob_matrix(), om)
                if sres.lag_values is not None:
                    bundle.add(f"{prefix}_空间滞后值明细",
                               sres.lag_values.rename(columns={
                                   "province": "省份", "year": "年份",
                                   "value": "指数值", "lag_value": "邻居加权均值",
                                   "class": "等级", "lag_group": "邻居组"}),
                               om, save_csv=False)   # 明细太长，只进 Excel

                # 空间溢出显著性检验
                if ecfg.get("mpr_comparison", True):
                    rows = []
                    for lv in range(n_classes):
                        st = spillover_test(sres, lv)
                        if st.get("ok"):
                            rows.append({
                                "起始等级": class_label(lv),
                                "卡方值": st["chi2"], "p值": st["p"],
                                "自由度": st["dof"],
                                "期望频数<5的格子数": st["n_small_cells"],
                                "结论": st["conclusion"],
                            })
                        else:
                            rows.append({"起始等级": class_label(lv),
                                         "卡方值": np.nan, "p值": np.nan,
                                         "自由度": np.nan,
                                         "期望频数<5的格子数": np.nan,
                                         "结论": st.get("reason", "无法检验")})
                    bundle.add(f"{prefix}_空间溢出显著性",
                               pd.DataFrame(rows).set_index("起始等级"), om)

                bundle.add_text(f"空间马尔科夫链 · {sres.name}", sres.summary())
    else:
        logger.info("  配置里 spatial.enabled = false，跳过")

    # ---------------- 6. 出图 ----------------
    banner(logger, "STEP 6 / 6  生成图表")
    ocfg = cfg.get("output") or {}
    if ocfg.get("save_figures", True):
        fig_ext = str(ocfg.get("fig_format", "png"))
        plt, cjk = viz.setup_matplotlib(str(ocfg.get("font_family", "SimHei")))

        # 6.1 传统转移矩阵热力图
        for key, res in trad_results:
            p = om.figure_path(f"传统转移矩阵_{key}", fig_ext)
            viz.plot_transition_heatmap(
                res.P, n_classes, p,
                f"传统马尔科夫链转移概率矩阵（{key}分级）" if cjk
                else f"Transition matrix ({key})",
                plt, cjk, row_totals=res.row_totals)
            bundle.files.append(p)

        # 6.2 空间转移矩阵热力图（每个邻居组一张）
        for key, wname, sres in spatial_results:
            for g in range(len(sres.group_display)):
                if sres.row_totals[g].sum() <= 0:
                    continue
                p = om.figure_path(
                    f"空间转移矩阵_{key}_{_short(wname)}_邻居{sres.group_display[g]}",
                    fig_ext)
                viz.plot_transition_heatmap(
                    sres.probs[g], n_classes, p,
                    (f"空间马尔科夫链转移概率（{key} / {wname} / "
                     f"邻居为{sres.group_display[g]}）" if cjk
                     else f"Spatial transition ({key}/{wname}/neighbor {g + 1})"),
                    plt, cjk, row_totals=sres.row_totals[g], cmap="PuBuGn")
                bundle.files.append(p)

        # 6.3 核密度演进
        if ocfg.get("kernel_density", True):
            yrs = [int(y) for y in (ocfg.get("kernel_density_years") or [])]
            p = om.figure_path("核密度演进", fig_ext)
            viz.plot_kernel_density(panel, yrs, p, plt, cjk)
            bundle.files.append(p)

        # 6.4 等级构成演进（每种分级方法一张）
        for key, cres, pl in results:
            p = om.figure_path(f"等级构成演进_{key}", fig_ext)
            viz.plot_class_composition(pl, n_classes, p, plt, cjk)
            bundle.files.append(p)

        # 6.5 省份轨迹
        p = om.figure_path("省份轨迹", fig_ext)
        viz.plot_province_trajectories(panel, p, plt, cjk)
        bundle.files.append(p)

        # 6.6 空间分布（选几个年份）
        if spatial_results:
            key, wname, sres = spatial_results[0]
            if sres.lag_values is not None:
                avail = sorted(sres.lag_values[C_YEAR].unique().tolist())
                picks = [avail[0], avail[len(avail) // 2], avail[-1]]
                for y in picks:
                    p = om.figure_path(f"空间分布_{int(y)}", fig_ext)
                    viz.plot_spatial_lag_scatter(sres.lag_values, int(y), p, plt, cjk)
                    bundle.files.append(p)
    else:
        logger.info("  配置里 output.save_figures = false，跳过出图")

    # ---------------- 汇总写出 ----------------
    banner(logger, "汇总输出")
    if ocfg.get("save_excel", True):
        save_excel_workbook(bundle.sheets, om.root / "结果汇总.xlsx", logger)

    # 文本汇总
    txt = _build_report(panel, info, results, trad_results, spatial_results,
                        weight_matrices, cfg, t_start)
    (om.root / "结果说明.txt").write_text(txt, encoding="utf-8")
    bundle.add_text("总览", txt)

    n_tables = len(list(om.tables.glob("*.csv")))
    n_figs = len(list(om.figures.glob(f"*.{str(ocfg.get('fig_format', 'png'))}")))
    dur = (pd.Timestamp.now() - t_start).total_seconds()

    banner(logger, "完成")
    logger.info(f"  结果表：{n_tables} 个  → {om.tables}")
    logger.info(f"  图表：{n_figs} 张  → {om.figures}")
    logger.info(f"  汇总工作簿：{om.root / '结果汇总.xlsx'}")
    logger.info(f"  文字说明：{om.root / '结果说明.txt'}")
    logger.info(f"  总耗时：{dur:.1f} 秒")
    logger.info("")
    logger.info("  ★ 下一步：打开「结果说明.txt」看结论摘要；")
    logger.info("    打开 output/tables/ 里的 CSV 看具体矩阵；")
    logger.info("    图在 output/figures/ 里。")

    return 0


def _short(name: str) -> str:
    """矩阵显示名 → 短标签，用于文件名。"""
    import re
    s = re.sub(r"[（(].*?[)）]", "", str(name)).strip()
    s = re.sub(r"\s+", "", s)
    return s or "W"


def _write_error(om: OutputManager, title: str, err: Exception) -> None:
    p = om.root / "运行失败说明.txt"
    p.write_text(
        f"{title}\n{'=' * 60}\n\n{type(err).__name__}: {err}\n\n"
        "排查顺序：\n"
        "  1. 数据文件是否放在 data/ 目录，config.yaml 的 data.file 是否写对\n"
        "  2. data.shape 是否与你的数据形态匹配（long / wide / raw）\n"
        "  3. data.col_map 里的列名是否与文件里的列名一字不差\n"
        "  4. 数据里省份名是否是 31 省标准名称（如「内蒙古」而非「内蒙」以外的变体）\n"
        "  5. 日志文件 output/run.log 里有完整堆栈\n",
        encoding="utf-8",
    )


def _build_report(
    panel: pd.DataFrame,
    info: dict,
    results: list,
    trad_results: list,
    spatial_results: list,
    weight_matrices: list,
    cfg: dict,
    t_start: pd.Timestamp,
) -> str:
    """生成人类可读的结果说明。"""
    from utils import class_full_labels
    lines: list[str] = []
    A = lines.append

    A("马尔科夫链动态演进分析 · 结果说明")
    A("=" * 74)
    A(f"运行时间：{t_start:%Y-%m-%d %H:%M:%S}")
    A("")

    # --- 数据概况 ---
    A("一、数据概况")
    A("-" * 74)
    A(f"  数据形态识别：{info.get('shape')}")
    A(f"  省份数：{info.get('n_provinces')}")
    A(f"  年份范围：{info.get('year_min')} – {info.get('year_max')}"
      f"（{info.get('n_years')} 年）")
    A(f"  有效样本：{info.get('n_obs')}")
    A(f"  指数区间：{panel[C_VALUE].min():.4f} – {panel[C_VALUE].max():.4f}"
      f"（均值 {panel[C_VALUE].mean():.4f}，标准差 {panel[C_VALUE].std():.4f}）")
    A("")

    # --- 分级 ---
    A("二、等级划分")
    A("-" * 74)
    for key, cres, pl in results:
        A(f"  ▸ {key}")
        for ln in cres.summary().split("\n"):
            A(f"    {ln}")
        A("")
    if len(results) >= 2:
        A("  ★ 两套分级方法的稳健性对比：")
        k0, r0, _ = results[0]
        k1, r1, _ = results[1]
        A(f"    {k0} 各等级样本占比："
          + "、".join(f"{class_label(k)}={v / sum(r0.counts.values()) * 100:.1f}%"
                      for k, v in sorted(r0.counts.items())))
        A(f"    {k1} 各等级样本占比："
          + "、".join(f"{class_label(k)}={v / sum(r1.counts.values()) * 100:.1f}%"
                      for k, v in sorted(r1.counts.items())))
        A("    → 若两套方法的转移矩阵结论方向一致（如都显示高等级维持概率最高），"
          "说明结论对分级方式不敏感，更可靠。")
        A("")

    # --- 空间矩阵 ---
    A("三、空间权重矩阵")
    A("-" * 74)
    for wm in weight_matrices:
        for ln in wm.summary().split("\n"):
            A(f"  {ln}")
        A("")

    # --- 传统马尔科夫链 ---
    A("四、传统马尔科夫链")
    A("-" * 74)
    for key, res in trad_results:
        for ln in res.summary().split("\n"):
            A(f"  {ln}")
        A("")

    # --- 空间马尔科夫链 ---
    A("五、空间马尔科夫链")
    A("-" * 74)
    if not spatial_results:
        A("  未运行（配置关闭或无可用空间矩阵）")
    for key, wname, sres in spatial_results:
        A(f"  【{key} 分级 / {wname}】")
        A(f"    邻居组：{'、'.join(sres.group_display)}")
        A(f"    有效转移样本：{int(sres.row_totals.sum())}")
        A("    空间溢出效应（同一起始等级，不同邻居环境下的向上转移概率）：")
        for i in range(sres.n_classes):
            parts = []
            for g in range(len(sres.group_display)):
                if sres.row_totals[g, i] > 0:
                    up = (sres.probs[g, i, i + 1:].sum()
                          if i < sres.n_classes - 1 else 0.0)
                    parts.append(f"{sres.group_display[g]}={up * 100:.1f}%")
            if parts:
                A(f"      起始 {class_label(i)}：  " + "   ".join(parts))
        A("")

    # --- 关键结论提示 ---
    A("六、怎么读这些结果（速查）")
    A("-" * 74)
    A("  1. 转移概率矩阵 P 的第 i 行第 j 列 = 「本期处于等级 i 的省份，")
    A("     下一期落到等级 j 的概率」。行和恒为 1（除空行）。")
    A("  2. 对角线 P_ii 越大 → 路径依赖越强，等级越难改变。")
    A("     若 P_ii 接近 1（如 100%），该等级是「吸收态」，长期会把所有省份吸进去。")
    A("  3. 非对角线右上（P_ij, j>i）= 向上跃升概率；左下 = 跌落概率。")
    A("     论文的重要发现是「中高水平存在小概率跌回中低水平的风险」。")
    A("  4. 稳态分布 π = 长期极限下各省落在各等级的比例。与当前分布对比，")
    A("     就能看出「按现有转移规律，未来会演变成什么格局」。")
    A("  5. 期望持续时间 D_i = 1/(1-P_ii)，是「停留在等级 i 的平均年数」。")
    A("  6. 空间马尔科夫链：同一行（同一起始等级）在不同邻居组下的概率分布不一样，")
    A("     就说明邻居环境有影响。邻居越强、该等级向上转移概率越高 → 正向溢出。")
    A("  7. 换空间矩阵后结论不稳定 → 说明结论对「空间关系怎么定义」敏感，")
    A("     需要在论文里讨论；结论稳定 → 说明发现是稳健的。")
    A("")

    # --- 与论文的差异说明 ---
    A("七、与参考文献（统计与决策 2022 年第 23 期）的对应关系")
    A("-" * 74)
    A("  本代码按该论文的方法框架实现，但有两点刻意的差异，需要注意：")
    A("  (1) 频数口径。论文表 4 报告的频数为 109/109/97/88（合计 403），")
    A("      但该表的每一行概率与这些频数并不对应（那一列是按「省份最终等级」")
    A("      统计的省份数，不是按转移样本统计的）。本代码按标准做法统计：")
    A("      N_ij = 「一个省份、相邻两期」从等级 i 转到等级 j 的次数，")
    A("      因此频数是长条形的，与论文表 4 的数字不会一致。这是更规范的口径。")
    A("  (2) 样本期间。论文为 2007—2020 年，本代码不限定年份，")
    A("      按你提供的数据年份自动适配（例如 2000—2024 年）。")
    A("      期间不同，等级边界与转移概率都会不同，属正常现象。")
    A("")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
