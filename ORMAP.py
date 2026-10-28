#!/usr/bin/env python
"""COUNT 单文件完整流程：单细胞 → 小鼠均值 → 共表达模块 → 基因评分与排名。

CMD：
    python run_count_coexpression_d060_m10_k2_l2_v0_c0.py
已有小鼠均值表时跳过聚合：
    python run_count_coexpression_d060_m10_k2_l2_v0_c0.py --sample-mean mouse_level_mean_expression.csv
依赖：python -m pip install numpy pandas scipy matplotlib

输入：ML_mOSN_Olfr_COUNTS_cell_by_gene_with_labels.csv
字段：cell_id、month、group（AD/WT）、sample，其他列均为基因。
流程：
1. 分块读取 COUNT 单细胞表，按 sample 累加表达和细胞数。
   每个基因的小鼠均值 = 该小鼠所有细胞的表达总和 / n_cells（包含零表达细胞）。
2. 均值表达做 log1p，再沿样本对每个基因做 Z-score；剔除零方差基因。
3. Pearson 相关距离 1-r，average-linkage 层次聚类，在距离 0.60 处切树。
4. 只评分至少含 10 个基因的模块；每个模块保留前 2 个 SVD 成分。
5. 每个成分拟合 OLS：PC ~ 1 + AD + z(month) + AD*z(month) + z(log1p(n_cells))。
   对疾病、年龄、交互项的绝对 t 值，在各合格模块的同编号成分间转为百分位，
   三者平均得到表型分数 P。基因成员度 kME = abs(corr(基因, PC))。
6. 基因分数 = sqrt(mean((kME_j * P_j)^2))，j=1,2；按分数降序排名。
7. 输出均值表、全部基因排名、目标基因排名、运行摘要和 COUNT 排名曲线。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent
CONFIG = "d0.60_m10_k2_l2_v0_c0"
METADATA = ["sample", "month", "group", "n_cells"]
TARGET_GENES = ["Olfr180", "Olfr173"]


def aggregate_count(input_path: Path, chunksize: int) -> pd.DataFrame:
    """跨分块累加，再除以每个 sample 的全部细胞数，直接生成均值表。"""
    columns = pd.read_csv(input_path, nrows=0).columns.tolist()
    cell_metadata = ["cell_id", "month", "group", "sample"]
    missing = set(cell_metadata) - set(columns)
    if missing:
        raise ValueError(f"Missing cell metadata: {sorted(missing)}")
    genes = [c for c in columns if c not in cell_metadata]
    if not genes:
        raise ValueError("No gene columns")
    sums, counts, metadata = {}, {}, {}
    total = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        if chunk[cell_metadata[1:]].isna().any().any():
            raise ValueError("Missing sample/month/group metadata")
        for sample, cells in chunk.groupby("sample", sort=False):
            if cells["month"].nunique() != 1 or cells["group"].nunique() != 1:
                raise ValueError(f"Inconsistent metadata in sample {sample}")
            meta = (cells["month"].iloc[0], cells["group"].iloc[0])
            if sample in metadata and metadata[sample] != meta:
                raise ValueError(f"Metadata changed across chunks: {sample}")
            metadata[sample] = meta
            values = cells[genes].apply(pd.to_numeric, errors="raise")
            sums[sample] = sums.get(sample, np.zeros(len(genes))) + values.sum().to_numpy(dtype=float)
            counts[sample] = counts.get(sample, 0) + len(cells)
        total += len(chunk)
        print(f"Aggregated {total:,} cells", flush=True)
    if not sums:
        raise ValueError("Input contains no cells")
    rows = []
    for sample in sorted(sums):
        month, group = metadata[sample]
        row = dict(sample=sample, month=month, group=group, n_cells=counts[sample])
        row.update(zip(genes, sums[sample] / counts[sample]))
        rows.append(row)
    return pd.DataFrame(rows, columns=METADATA + genes)


def zscore(values: np.ndarray) -> np.ndarray:
    sd = values.std(ddof=0)
    return np.zeros_like(values) if sd == 0 or not np.isfinite(sd) else (values - values.mean()) / sd


def phenotype_design(frame: pd.DataFrame) -> np.ndarray:
    def month_number(value: object) -> float:
        match = re.search(r"[-+]?\d*\.?\d+", str(value))
        if match is None:
            raise ValueError(f"Cannot parse month: {value}")
        return float(match.group(0))

    if not frame["group"].isin(["AD", "WT"]).all():
        raise ValueError("group must contain only AD or WT")
    disease = frame["group"].eq("AD").to_numpy(dtype=float)
    age = zscore(frame["month"].map(month_number).to_numpy(dtype=float))
    cells = zscore(np.log1p(pd.to_numeric(frame["n_cells"]).to_numpy(dtype=float)))
    design = np.column_stack([np.ones(len(frame)), disease, age, disease * age, cells])
    if not np.isfinite(design).all() or len(frame) <= np.linalg.matrix_rank(design):
        raise ValueError("Invalid phenotype values or insufficient samples for OLS")
    return design


def ols_t_values(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    variance = np.square(residual).sum(axis=0) / (len(x) - np.linalg.matrix_rank(x))
    se = np.sqrt(np.outer(np.diag(inverse), variance))
    return np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)


def percentile_high(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(ascending=False, method="min")
    return ((len(ranks) - ranks + 1) / len(ranks)).to_numpy()


def rank_count(frame: pd.DataFrame) -> pd.DataFrame:
    """固定配置评分；数值运算沿用原有 COUNT 实现。"""
    missing = set(METADATA) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing sample metadata: {sorted(missing)}")
    genes = np.asarray([c for c in frame.columns if c not in METADATA], dtype=object)
    expression = frame[list(genes)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    log_expression = np.log1p(expression.to_numpy(dtype=float))
    sd = log_expression.std(axis=0, ddof=0)
    valid = np.isfinite(sd) & (sd > 0)
    if valid.sum() < 2:
        raise ValueError("At least two variable genes are required")
    z = (log_expression[:, valid] - log_expression[:, valid].mean(axis=0)) / sd[valid]
    corr = np.clip(np.nan_to_num(np.corrcoef(z.T), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    distance = 1.0 - corr
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(tree, t=0.60, criterion="distance")
    design = phenotype_design(frame)
    modules = []
    for module_id in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == module_id)
        if len(indices) < 10:
            continue
        module_z = z[:, indices]
        u, singular, _ = np.linalg.svd(module_z, full_matrices=False)
        pcs = u[:, :2] * singular[:2]
        pcs = np.column_stack([zscore(pcs[:, j]) for j in range(2)])
        membership = np.abs(np.corrcoef(module_z.T, pcs.T)[:len(indices), len(indices):])
        membership = np.nan_to_num(membership, nan=0.0)
        within_corr = np.maximum(corr[np.ix_(indices, indices)], 0.0)
        connectivity = (within_corr.sum(axis=1) - 1.0) / (len(indices) - 1)
        modules.append(dict(module_id=module_id, indices=indices, membership=membership,
                            connectivity=connectivity, t=np.abs(ols_t_values(pcs, design)[1:4])))

    # 同编号 PC 在所有合格模块间比较疾病、年龄、交互项的 |t|。
    phenotype = np.zeros((len(modules), 2))
    for component in range(2):
        for term in range(3):
            values = np.asarray([m["t"][term, component] for m in modules])
            if len(values):
                phenotype[:, component] += percentile_high(values) / 3.0

    size = int(valid.sum())
    scores = np.full(size, np.nan)
    module_ids = np.full(size, np.nan)
    module_sizes = np.zeros(size, dtype=int)
    dominant = np.full(size, np.nan)
    core_scores = np.full(size, np.nan)
    connectivity_values = np.full(size, np.nan)
    for i, module in enumerate(modules):
        indices = module["indices"]
        contributions = module["membership"] * phenotype[i]
        core = np.sqrt(np.square(contributions).mean(axis=1))
        scores[indices] = np.maximum(core, 1e-12)
        core_scores[indices] = core
        module_ids[indices] = module["module_id"]
        module_sizes[indices] = len(indices)
        dominant[indices] = contributions.argmax(axis=1) + 1
        connectivity_values[indices] = module["connectivity"]
    result = pd.DataFrame(dict(
        gene=genes[valid], module_id=module_ids, module_size=module_sizes,
        dominant_component=dominant, core_component_score=core_scores,
        positive_connectivity=connectivity_values, coexpression_score=scores,
        coexpression_rank=pd.Series(scores).fillna(-np.inf).rank(ascending=False, method="min").astype(int),
    ))
    if (~valid).any():
        invalid = pd.DataFrame(dict(gene=genes[~valid], module_size=0,
                                    coexpression_rank=np.arange(size + 1, len(genes) + 1)))
        result = pd.concat([result, invalid], ignore_index=True)
    result = result.sort_values(["coexpression_rank", "gene"]).reset_index(drop=True)
    result.insert(0, "dataset", "COUNT")
    result.insert(1, "config", CONFIG)
    return result


def plot_ranking(ranking: pd.DataFrame, path: Path) -> None:
    scored = ranking[ranking["coexpression_score"].notna()]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(scored["coexpression_rank"], scored["coexpression_score"], color="#8B949E", linewidth=1.8)
    for gene, color, offset in [("Olfr180", "#D1495B", (10, 12)), ("Olfr173", "#2A9D8F", (10, -20))]:
        rows = scored[scored["gene"].eq(gene)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        point = (row["coexpression_rank"], row["coexpression_score"])
        ax.scatter(*point, color=color, s=65, zorder=3)
        ax.annotate(f"{gene}: {int(point[0])}", point, xytext=offset,
                    textcoords="offset points", color=color, fontweight="bold")
    if len(scored) < len(ranking):
        ax.axvspan(len(scored) + 0.5, len(ranking), color="#F2F4F7")
        ax.axvline(len(scored) + 0.5, color="#98A2B3", linestyle="--", linewidth=1)
    ax.set(xlim=(1, max(2, len(ranking))), ylim=(0, None), xlabel="Gene rank (smaller is better)",
           ylabel="Coexpression gene score", title=f"COUNT: {CONFIG}")
    ax.text(0.98, 0.95, f"Scored: {len(scored)}/{len(ranking)}", transform=ax.transAxes, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, help="COUNT cell-level CSV; defaults to the project COUNT input")
    inputs.add_argument("--sample-mean", type=Path, help="Use an existing COUNT sample-mean CSV and skip aggregation")
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "count_coexpression_d060_m10_k2_l2_v0_c0")
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    source = args.sample_mean or args.input or ROOT / "ML_mOSN_Olfr_COUNTS_cell_by_gene_with_labels.csv"
    frame = pd.read_csv(source) if args.sample_mean else aggregate_count(source, args.chunksize)
    print(f"Ranking COUNT: {len(frame)} samples, {len(frame.columns) - len(METADATA)} genes", flush=True)
    ranking = rank_count(frame)
    targets = ranking[ranking["gene"].isin(TARGET_GENES)].copy()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "mouse_level_mean_expression.csv", index=False)
    ranking.to_csv(output / "gene_rankings.csv", index=False)
    targets.to_csv(output / "target_gene_ranks.csv", index=False)
    plot_ranking(ranking, output / "coexpression_rank_curves.png")
    scored = ranking["coexpression_score"].notna().sum()
    lines = [f"Config: {CONFIG}", "Dataset: COUNT", f"Input: {source.resolve()}",
             f"Aggregation: {'existing sample means' if args.sample_mean else 'sum of cell expression / n_cells'}",
             f"Samples: {len(frame)}", f"Genes: {len(ranking)}",
             "distance_threshold=0.6; min_module_size=10; n_components=2",
             "aggregation=l2 (RMS); variance_power=0; connectivity_weight=0",
             f"Coverage: {scored}/{len(ranking)}", "", "Target ranks:"]
    for row in targets.itertuples():
        lines.append(f"- {row.gene}: rank={row.coexpression_rank}/{len(ranking)}, "
                     f"score={row.coexpression_score:.8f}, module_size={row.module_size}")
    lines += ["", "Unscored genes have NaN scores; tail ranks are not meaningful relative rankings.",
              "This fixed configuration was previously selected using target ranks across COUNT/NORMALIZED.",
              "This COUNT-only run is a reproduction, not an independent validation."]
    (output / "run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(targets[["gene", "coexpression_rank", "coexpression_score", "module_size"]].to_string(index=False))
    print(f"Saved outputs to: {output.resolve()}")


if __name__ == "__main__":
    main()
