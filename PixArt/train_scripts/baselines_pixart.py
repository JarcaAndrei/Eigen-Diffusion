import time

import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus
from threadpoolctl import threadpool_limits


def kcenter(F, k, seed):    
    # Greedy K-Center (Gonzalez 1985)
    first = int(np.random.default_rng(seed).integers(F.shape[0]))
    selected = [first]
    mind = 1.0 - F @ F[first]
    mind[first] = -np.inf
    for _ in range(k - 1):
        nxt = int(np.argmax(mind))
        selected.append(nxt)
        np.minimum(mind, 1.0 - F @ F[nxt], out=mind)
        mind[nxt] = -np.inf
    return selected


def facility_location(F, k, seed, eps=0.1):
    # Stochastic greedy (Mirzasoleiman et al., AAAI 2015, Alg. 1) & FL (Cornuejols et al., 1977)
    # adaptation from https://apricot-select.readthedocs.io/en/latest/optimizers/stochastic.html
    n = F.shape[0]
    rng = np.random.default_rng(seed)
    s = int(n / k * np.log(1 / eps))
    print(f"facility_location: n={n} k={k} s={s}")

    maxsim = np.full(n, -1.0, dtype=np.float32)
    used = np.zeros(n, dtype=bool)
    selected = []
    with threadpool_limits(4, user_api="blas"):
        for _ in range(k):
            pool = np.flatnonzero(~used)
            cand = pool if pool.size <= s else rng.choice(pool, size=s, replace=False)
            gains = np.maximum(F[cand] @ F.T - maxsim, 0.0).sum(axis=1)
            best = int(cand[np.argmax(gains)])
            selected.append(best)
            used[best] = True
            np.maximum(maxsim, F[best] @ F.T, out=maxsim)
    return selected


def kmeans_pp(F, k, seed):
    # k-means++ seeding (Arthur & Vassilvitskii, SODA 2007)
    return [int(i) for i in kmeans_plusplus(F, n_clusters=k, random_state=seed, n_local_trials=1)[1]]


def semdedup(F, k, seed, min_eps=0.00095):
    # SemDeDup (Abbas et al. 2023) from https://github.com/facebookresearch/SemDeDup/.
    # The official code uses 50,000 clusters (8,748 points per cluster on average)
    import faiss

    faiss.omp_set_num_threads(4)
    ncentroids = max(1, round(F.shape[0] / 8748))
    kmeans = faiss.Kmeans(F.shape[1], ncentroids, niter=100, seed=seed, spherical=True)
    kmeans.train(F)
    dist_to_cent, nearest_cent = kmeans.index.search(F, 1)
    dist_to_cent, nearest_cent = dist_to_cent.squeeze(1), nearest_cent.squeeze(1)

    order, M = [], []
    for cluster_c in range(ncentroids):
        cluster_items = list(np.flatnonzero(nearest_cent == cluster_c))
        if not cluster_items:
            continue
        cluster_dists_to_cent = list(1 - dist_to_cent[cluster_items])
        cluster_sorted = sorted(zip(cluster_items, cluster_dists_to_cent), key=lambda x: x[1], reverse=True)
        cluster_ids = [int(x[0]) for x in cluster_sorted]
        cluster_reps = torch.tensor(F[cluster_ids])
        pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
        pair_w_sim_matrix.fill_diagonal_(0.0)
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)
        M.append(torch.max(triu_sim_mat, dim=0)[0].numpy())
        order += cluster_ids
    M = np.concatenate(M)

    eps = max(1 - float(np.sort(M)[k - 1]), min_eps)
    keep = np.flatnonzero(M <= 1 - eps)
    assert len(keep) <= k, "points tied at the k-th similarity straddle the cut: no eps keeps exactly k"
    print(f"semdedup: n={len(M)} ncentroids={ncentroids} eps={eps:.9g} kept {len(keep)} of k={k}")
    return [order[i] for i in keep]


def get_baseline_indexes(strategy, F, k, seed):
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    norms[norms == 0] = 1
    F = F / norms
    select = {"kcenter": kcenter, "facility_location": facility_location, "kmeans++": kmeans_pp, "semdedup": semdedup}
    t0 = time.perf_counter()
    idx = select[strategy](F, k, seed)
    print(f"selection_seconds={time.perf_counter() - t0:.2f} strategy={strategy} n={F.shape[0]} k={k}")
    return idx
