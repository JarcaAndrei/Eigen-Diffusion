import os
import gc
import json
import numpy as np
import torch
from sklearn.decomposition import PCA


def stratified_quantile_sampling(
    combined_score,
    var,
    var_max,
    var_ratio,
    n_strat_first,
    n_strat_second,
    n_bins,
    seed,
    stratify,
    variance_weighted,
):
    rng = np.random.default_rng(seed)
    logs = []
    used = set()

    n_samples = combined_score.shape[1]
    n_pcs1 = var

    def allocate(n_total, var_slice):
        v = np.asarray(var_slice)
        if not variance_weighted:
            v = np.ones_like(v)
        v = v / (v.sum() + 1e-8)
        k = np.floor(v * n_total).astype(int)
        diff = n_total - k.sum()
        if diff > 0:
            for i in np.argsort(-v)[:diff]:
                k[i] += 1
        return k

    var_ratio1 = np.asarray(var_ratio[:n_pcs1])
    var_ratio2 = np.asarray(var_ratio[n_pcs1:var_max])
    k_stage1 = allocate(n_strat_first, var_ratio1)
    k_stage2 = allocate(n_strat_second, var_ratio2)

    for pc in range(n_pcs1):
        k_pc = int(k_stage1[pc])
        if k_pc <= 0:
            continue

        scores_full = combined_score[pc]
        idx = np.fromiter((i for i in range(n_samples) if i not in used),
                          dtype=np.int64)
        if len(idx) == 0:
            continue

        local_scores = scores_full[idx]
        top_k = min(k_pc, len(idx))
        order = np.argpartition(-local_scores, top_k - 1)[:top_k]
        order = order[np.argsort(-local_scores[order])]

        for j in order:
            i = int(idx[j])
            used.add(i)
            logs.append({"index": i, "pc": int(pc),
                         "score": float(scores_full[i]), "stage": 1})

    for local_pc in range(var_max - n_pcs1):
        pc = local_pc + n_pcs1
        k_pc = int(k_stage2[local_pc])
        if k_pc <= 0:
            continue

        scores_full = combined_score[pc]
        keep = np.fromiter((i for i in range(n_samples) if i not in used),
                           dtype=np.int64)
        if len(keep) == 0:
            continue

        if not stratify:
            take = min(k_pc, len(keep))
            for i in rng.choice(keep, size=take, replace=False):
                i = int(i)
                used.add(i)
                logs.append({"index": i, "pc": int(pc),
                             "score": float(scores_full[i]), "stage": 2})
            continue

        rank_scores = scores_full[keep]
        edges = np.quantile(rank_scores, np.linspace(0.0, 1.0, n_bins + 1))

        per_band_base = k_pc // n_bins
        per_band_extra = k_pc - per_band_base * n_bins  # leftovers to top bands

        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            if b == n_bins - 1:
                mask = (rank_scores >= lo) & (rank_scores <= hi)
            else:
                mask = (rank_scores >= lo) & (rank_scores < hi)
            band_pool = keep[mask]
            if len(band_pool) == 0:
                continue

            target = per_band_base + (1 if (n_bins - 1 - b) < per_band_extra else 0)
            take = min(target, len(band_pool))  # accept shortfall
            if take == 0:
                continue

            for i in rng.choice(band_pool, size=take, replace=False):
                i = int(i)
                used.add(i)
                logs.append({
                    "index": i, "pc": int(pc),
                    "score": float(scores_full[i]), "stage": 2,
                    "band": int(b),
                    "band_quantile": [float(b) / n_bins, float(b + 1) / n_bins],
                    "band_edges": [float(lo), float(hi)],
                })

    return logs


def get_strategy_indexes(
    dataset_name,
    fixed_seed,
    threshold,
    thresh_stage2, 
    n_strat_first,
    ratio,
    n_bins,
    variance_weighted,
    stratify, 
    sft,
    r_cap=None,
    n_components=1000,
    base_path="all_embeddings",
    strategy="eigen",
    indices_path="baseline_indices",
):
    path = os.path.join(base_path, f"embeddings_{dataset_name}_{fixed_seed}.npz")
    data = np.load(path)
    X = data["X"]
    print("X shape (N=pairs, D):", X.shape)

    if strategy != "eigen":
        # precomputed by dataset_creation/select_baselines.py
        sel_path = os.path.join(indices_path, f"{strategy}_{dataset_name}_{fixed_seed}.json")
        with open(sel_path) as f:
            sel = json.load(f)
        k = n_strat_first + int(n_strat_first * ratio)
        if (sel["n"], sel["k"]) != (X.shape[0], k):
            raise ValueError(f"{sel_path} was selected with n={sel['n']}, k={sel['k']}; this run has n={X.shape[0]}, k={k}")
        return sel["indices"]

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=fixed_seed)
    pca.fit(X)
    var_ratio = pca.explained_variance_ratio_
    eigenvectors = pca.components_

    if thresh_stage2 <= threshold:
        raise ValueError(f"thresh_stage2 ({thresh_stage2}) must be > threshold ({threshold}).")

    cumvar = np.cumsum(var_ratio)

    def pcs_for(t):
        reached = np.argmax(cumvar >= t)
        return (int(reached) + 1) if cumvar[reached] >= t else len(cumvar)

    var = pcs_for(threshold)
    if r_cap is not None:
        var = min(var, r_cap)
    var_max = pcs_for(thresh_stage2)

    if sft:
        combined_score = (eigenvectors @ X.T).astype(np.float32)   
    else:
        Y = data["Y"]
        combined_score = (eigenvectors @ X.T - eigenvectors @ Y.T).astype(np.float32)  

    del pca, eigenvectors, X, data
    if not sft:
        del Y
    gc.collect()
    torch.cuda.empty_cache()

    logs = stratified_quantile_sampling(
        combined_score=combined_score,
        var=var,
        var_max=var_max,
        var_ratio=var_ratio,
        n_strat_first=n_strat_first,
        n_strat_second=int(n_strat_first * ratio),
        n_bins=n_bins,
        seed=fixed_seed,
        variance_weighted=variance_weighted,
        stratify=stratify,
    )

    print("stage 1 picks:", sum(1 for l in logs if l["stage"] == 1))
    print("stage 2 picks:", sum(1 for l in logs if l["stage"] == 2))
    print("total unique: ", len(set(l["index"] for l in logs)))

    return [x["index"] for x in logs]