from __future__ import annotations

import numba
import numpy as np
import torch
from termcolor import colored

from ttc.data.datasets import imagenet_a_class_number, imagenet_r_class_number

from .fusion import feature_fusion


@numba.njit(cache=True, fastmath=True)
def _refresh_invs(classes, mem_features, valid_count, inv_cache,
                  cache_valid, ridge, mem_size):
    K = classes.shape[0]
    for k in range(K):
        c = classes[k]
        if cache_valid[c]:
            continue
        n = valid_count[c]
        if n == 0:
            continue
        feats = mem_features[c]
        gram = feats @ feats.T
        a = np.zeros((mem_size, mem_size), dtype=np.float32)
        for i in range(mem_size):
            for j in range(mem_size):
                if i < n and j < n:
                    a[i, j] = gram[i, j]
            a[i, i] += ridge
        inv_cache[c] = np.linalg.inv(a)
        cache_valid[c] = True


@numba.njit(cache=True, fastmath=True)
def _gather_K(top_n, mem_features, inv_cache, valid_count, mem_size, dim):
    K = top_n.shape[0]
    feats_K = np.empty((K, mem_size, dim), dtype=np.float32)
    inv_K = np.empty((K, mem_size, mem_size), dtype=np.float32)
    valid_K = np.empty(K, dtype=np.int64)
    for k in range(K):
        c = top_n[k]
        feats_K[k] = mem_features[c]
        inv_K[k] = inv_cache[c]
        valid_K[k] = valid_count[c]
    return feats_K, inv_K, valid_K


@numba.njit(cache=True, fastmath=True)
def _augment_inv_batched(m_K, inv_K, x, ridge, mem_size):
    K = m_K.shape[0]
    n1 = mem_size + 1
    dim = x.shape[0]

    inv_aug = np.zeros((K, n1, n1), dtype=np.float32)
    m_aug = np.empty((K, n1, dim), dtype=np.float32)

    xx = 0.0
    for d in range(dim):
        xx += x[d] * x[d]
    c_const = xx + ridge

    for k in range(K):
        m = m_K[k]
        inv = inv_K[k]
        b = m @ x
        inv_b = inv @ b
        s = c_const
        for i in range(mem_size):
            s -= b[i] * inv_b[i]
        s_safe = s if abs(s) > 1e-12 else 1e-12
        inv_s = np.float32(1.0 / s_safe)

        for i in range(mem_size):
            for j in range(mem_size):
                inv_aug[k, i, j] = inv[i, j] + inv_b[i] * inv_b[j] * inv_s
        for i in range(mem_size):
            inv_aug[k, i, mem_size] = -inv_b[i] * inv_s
            inv_aug[k, mem_size, i] = -inv_b[i] * inv_s
        inv_aug[k, mem_size, mem_size] = inv_s

        for i in range(mem_size):
            for d in range(dim):
                m_aug[k, i, d] = m[i, d]
        for d in range(dim):
            m_aug[k, mem_size, d] = x[d]

    return m_aug, inv_aug


@numba.njit(cache=True, fastmath=True)
def _batched_nse(m_i, inv_i, m_j, valid_j):
    K_i = m_i.shape[0]
    K_j = m_j.shape[0]
    m_j_max = m_j.shape[1]
    dim = m_i.shape[2]

    err = np.zeros((K_i, K_j), dtype=np.float32)
    for ki in range(K_i):
        for kj in range(K_j):
            n_j = valid_j[kj]
            if n_j == 0:
                continue
            proj = m_j[kj] @ m_i[ki].T
            coef = proj @ inv_i[ki]
            recon = coef @ m_i[ki]

            sum_norms = 0.0
            for r in range(m_j_max):
                if r < n_j:
                    nrm = 0.0
                    for d in range(dim):
                        diff = recon[r, d] - m_j[kj, r, d]
                        nrm += diff * diff
                    sum_norms += np.sqrt(nrm)
            err[ki, kj] = np.float32(sum_norms / n_j)
    return err


@numba.njit(cache=True, fastmath=True)
def _update_memory(x, conf, target, mem_features, mem_confidence,
                   valid_count, cache_valid, mem_size, fixed_size):
    # the few-shot slots [0, fixed_size) are never evicted
    ptr = valid_count[target]
    dim = x.shape[0]

    if ptr >= mem_size:
        if fixed_size >= mem_size:
            return
        j = fixed_size
        min_v = mem_confidence[target, fixed_size]
        for i in range(fixed_size + 1, mem_size):
            if mem_confidence[target, i] < min_v:
                min_v = mem_confidence[target, i]
                j = i
        if conf <= min_v:
            return
        for d in range(dim):
            mem_features[target, j, d] = x[d]
        mem_confidence[target, j] = conf
    else:
        for d in range(dim):
            mem_features[target, ptr, d] = x[d]
        mem_confidence[target, ptr] = conf
        valid_count[target] = ptr + 1

    cache_valid[target] = False


@numba.njit(cache=True, fastmath=True)
def _update_topk(x, scaled_logit, top_n, mem_features, mem_confidence,
                 valid_count, cache_valid, mem_size, fixed_size):
    K = top_n.shape[0]
    for rank in range(K):
        c = top_n[rank]
        conf = scaled_logit[c] / np.float32(rank + 1)
        _update_memory(x, conf, c, mem_features, mem_confidence,
                       valid_count, cache_valid, mem_size, fixed_size)


@numba.njit(cache=True, fastmath=True)
def _predict_one(x, logit, top_n,
                 mem_features, mem_confidence, valid_count,
                 inv_cache, cache_valid,
                 attempts, lam, ridge, anchor_weight, mem_size, fixed_size,
                 confidence_skip, min_valid_k):
    if confidence_skip < 1.0:
        max_logit = logit[0]
        for i in range(1, logit.shape[0]):
            if logit[i] > max_logit:
                max_logit = logit[i]
        sum_exp = 0.0
        for i in range(logit.shape[0]):
            sum_exp += np.exp(logit[i] - max_logit)
        p_top1 = np.exp(logit[top_n[0]] - max_logit) / sum_exp
        if p_top1 >= confidence_skip:
            pred = top_n[0]
            _update_memory(x, np.float32(p_top1 * lam), pred,
                           mem_features, mem_confidence,
                           valid_count, cache_valid, mem_size, fixed_size)
            return pred, 1

    cold = False
    for k in range(attempts):
        if valid_count[top_n[k]] < min_valid_k:
            cold = True
            break
    if cold:
        _update_topk(x, logit * lam, top_n, mem_features, mem_confidence,
                     valid_count, cache_valid, mem_size, fixed_size)
        return top_n[0], 0

    _refresh_invs(top_n, mem_features, valid_count, inv_cache,
                  cache_valid, ridge, mem_size)

    feats_K, inv_K, valid_K = _gather_K(
        top_n, mem_features, inv_cache, valid_count, mem_size, x.shape[0])

    m_aug_K, inv_aug_K = _augment_inv_batched(
        feats_K, inv_K, x, ridge, mem_size)

    x_stack = np.empty((1, 1, x.shape[0]), dtype=np.float32)
    for d in range(x.shape[0]):
        x_stack[0, 0, d] = x[d]
    x_valid = np.ones(1, dtype=np.int64)
    anchor_err = _batched_nse(feats_K, inv_K, x_stack, x_valid)
    anchor = np.empty(attempts, dtype=np.float32)
    for k in range(attempts):
        anchor[k] = anchor_err[k, 0]

    pos1 = _batched_nse(feats_K, inv_K, feats_K, valid_K)
    pos2 = _batched_nse(m_aug_K, inv_aug_K, feats_K, valid_K)

    margin = np.empty(attempts, dtype=np.float32)
    for k in range(attempts):
        s = anchor_weight * anchor[k]
        for p in range(attempts):
            if p != k:
                s += (pos1[k, p] - pos2[k, p]) - (pos1[p, k] - pos2[p, k])
        s -= np.float32(attempts - 1) * lam * logit[top_n[k]]
        margin[k] = s

    best = 0
    for k in range(1, attempts):
        if margin[k] < margin[best]:
            best = k
    pred = top_n[best]
    _update_memory(x, -margin[best], pred,
                   mem_features, mem_confidence,
                   valid_count, cache_valid, mem_size, fixed_size)
    return pred, 0


@numba.njit(cache=True, fastmath=True, parallel=False)
def _run_loop(features, logits, top_n,
              mem_features, mem_confidence, valid_count,
              inv_cache, cache_valid,
              attempts, lam, ridge, anchor_weight, mem_size, fixed_size,
              confidence_skip, min_valid_k):
    n = features.shape[0]
    preds = np.empty(n, dtype=np.int64)
    n_skipped = 0
    for i in range(n):
        pred, skipped = _predict_one(
            features[i], logits[i], top_n[i],
            mem_features, mem_confidence, valid_count,
            inv_cache, cache_valid,
            attempts, lam, ridge, anchor_weight, mem_size, fixed_size,
            confidence_skip, min_valid_k)
        preds[i] = pred
        n_skipped += skipped
    return preds, n_skipped


def _alloc_memory(num_classes, dim, total_mem_size):
    mem_features = np.zeros((num_classes, total_mem_size, dim), dtype=np.float32)
    mem_confidence = np.full((num_classes, total_mem_size), -np.inf,
                             dtype=np.float32)
    valid_count = np.zeros(num_classes, dtype=np.int64)
    cache_valid = np.zeros(num_classes, dtype=np.bool_)
    inv_cache = np.zeros((num_classes, total_mem_size, total_mem_size),
                         dtype=np.float32)
    return mem_features, mem_confidence, valid_count, inv_cache, cache_valid


def _install_fixed(mem_features, mem_confidence, valid_count, cache_valid,
                   train_features, fixed_size):
    if fixed_size == 0:
        return
    C, S, D = train_features.shape
    assert S == fixed_size
    mem_features[:, :fixed_size, :] = train_features.astype(np.float32, copy=False)
    mem_confidence[:, :fixed_size] = np.inf
    valid_count[:] = fixed_size
    cache_valid[:] = False


def _initialize_dynamic(mem_features, mem_confidence, valid_count, cache_valid,
                        features, scaled_logit, top_n, fixed_size, total_mem_size,
                        num_classes):
    dyn_capacity = total_mem_size - fixed_size
    if dyn_capacity == 0:
        return

    N, K = top_n.shape
    flat_class = top_n.reshape(-1)
    rank_div = np.arange(1, K + 1, dtype=np.float32)[None, :]
    flat_conf = (scaled_logit[np.arange(N)[:, None], top_n]
                 / rank_div).reshape(-1).astype(np.float32)
    flat_src = np.repeat(np.arange(N), K)

    order = np.argsort(flat_class, kind='stable')
    sc = flat_class[order]
    scc = flat_conf[order]
    sf = flat_src[order]
    bounds = np.searchsorted(sc, np.arange(num_classes + 1))

    for c in range(num_classes):
        s, e = int(bounds[c]), int(bounds[c + 1])
        if s == e:
            continue
        n = e - s
        confs_c = scc[s:e]
        srcs_c = sf[s:e]
        if n > dyn_capacity:
            top = np.argpartition(-confs_c, dyn_capacity)[:dyn_capacity]
        else:
            top = np.arange(n)
        m = top.shape[0]
        mem_features[c, fixed_size:fixed_size + m] = features[srcs_c[top]]
        mem_confidence[c, fixed_size:fixed_size + m] = confs_c[top]
        valid_count[c] = fixed_size + m

    cache_valid[:] = False


def _to_np(t, dtype=np.float32):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=dtype)


def _zca_fit(X, eps_ratio=1e-1):
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    Xc = X - mu
    n = Xc.shape[0]
    Sigma = (Xc.T @ Xc) / max(n - 1, 1)
    w, V = np.linalg.eigh(Sigma)
    eps = eps_ratio * float(w.max())
    w_inv_sqrt = 1.0 / np.sqrt(np.maximum(w, eps))
    W = ((V * w_inv_sqrt) @ V.T).astype(np.float32)
    return mu, W


def _zca_apply(X, mu, W):
    out = (X - mu) @ W
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
    return out.astype(np.float32)


def _normalize_train_shape(train_features, train_labels, num_classes):
    tf = _to_np(train_features)
    if tf.ndim == 3:
        assert tf.shape[0] == num_classes
        return tf

    assert tf.ndim == 2 and train_labels is not None
    tl = _to_np(train_labels, dtype=np.int64)
    if tl.ndim == 2:
        tl = tl.argmax(axis=1).astype(np.int64)
    n = tl.shape[0]
    if tf.shape[0] == n:
        sf = tf
    elif tf.shape[1] == n:
        sf = tf.T
    else:
        raise ValueError(
            f"train_features {tf.shape} does not match label count {n}")

    counts = np.bincount(tl, minlength=num_classes)
    if counts[:num_classes].min() != counts[:num_classes].max():
        raise ValueError(
            f"Per-class sample count must be uniform; got "
            f"min={counts.min()} max={counts.max()}")
    S = int(counts[0])
    D = sf.shape[1]

    grouped = np.zeros((num_classes, S, D), dtype=np.float32)
    ptr = np.zeros(num_classes, dtype=np.int64)
    for i in range(n):
        c = int(tl[i])
        if ptr[c] < S:
            grouped[c, ptr[c]] = sf[i]
            ptr[c] += 1
    return grouped


def TTC(cfg, train_features, train_labels, test_features, test_labels,
           clip_weights):
    if cfg['dataset'] == 'ImageNetA' and imagenet_a_class_number is not None:
        clip_weights = clip_weights[imagenet_a_class_number, :]
    elif cfg['dataset'] == 'ImageNetR' and imagenet_r_class_number is not None:
        clip_weights = clip_weights[imagenet_r_class_number, :]

    if (feature_fusion is not None
            and hasattr(test_features, 'dim')
            and test_features.dim() == 3):
        alpha = 0.0 if 'imagenet' in cfg['dataset'].lower() else 0.5
        test_features = feature_fusion(test_features, clip_weights, alpha)

    test_features = _to_np(test_features)
    test_labels = _to_np(test_labels, dtype=np.int64)
    clip_weights = _to_np(clip_weights)

    lam = np.float32(cfg.get('lam', 0.03))
    attempts = cfg.get('attempts', 3)
    memory_size = cfg.get('memory_size', 16)
    ridge = np.float32(cfg.get('ridge', 0.3))
    confidence_skip = np.float32(cfg.get('confidence_skip', 1.01))
    use_whiten = cfg.get('whiten', True)
    anchor_weight = np.float32(0.375 * (attempts - 1))

    num_classes = clip_weights.shape[0]

    train_grouped = None
    if train_features is not None:
        train_grouped = _normalize_train_shape(train_features, train_labels,
                                                num_classes)
        print(colored(
            f"Few-shot: {train_grouped.shape[1]} train samples per class "
            f"installed in fixed memory region.", 'cyan'))

    logits = (test_features @ clip_weights.T * 100.0).astype(np.float32)
    print(f"CLIP Accuracy: "
          f"{(np.argmax(logits, -1) == test_labels).mean() * 100:.2f}")

    if use_whiten:
        print(colored("ZCA-whitening test features...", 'cyan'))
        mu, W = _zca_fit(test_features)
        feats_mem = _zca_apply(test_features, mu, W)
        if train_grouped is not None:
            C, S, D = train_grouped.shape
            train_feats = _zca_apply(
                train_grouped.reshape(-1, D), mu, W).reshape(C, S, D)
        else:
            train_feats = None
    else:
        feats_mem = test_features
        train_feats = train_grouped

    fixed_size = train_feats.shape[1] if train_feats is not None else 0
    total_mem_size = fixed_size + memory_size
    min_valid_k = min(8, total_mem_size)

    print(colored(
        f"TTC (numba): K={attempts}, m={memory_size}, fixed={fixed_size}, "
        f"total={total_mem_size}, ridge={ridge}, lam={lam}", 'cyan'))

    top_n = np.argsort(logits, axis=-1)[:, -attempts:][:, ::-1].copy()
    top_n = np.ascontiguousarray(top_n.astype(np.int64))

    mem_features, mem_confidence, valid_count, inv_cache, cache_valid = \
        _alloc_memory(num_classes, feats_mem.shape[-1], total_mem_size)

    if train_feats is not None:
        _install_fixed(mem_features, mem_confidence, valid_count, cache_valid,
                       train_feats, fixed_size)

    _initialize_dynamic(mem_features, mem_confidence, valid_count, cache_valid,
                        feats_mem, logits * lam, top_n,
                        fixed_size, total_mem_size, num_classes)

    preds, n_skipped = _run_loop(
        feats_mem, logits, top_n,
        mem_features, mem_confidence, valid_count,
        inv_cache, cache_valid,
        attempts, lam, ridge, anchor_weight, total_mem_size, fixed_size,
        confidence_skip, min_valid_k)

    accuracy = (preds == test_labels).mean() * 100
    if confidence_skip < 1.0:
        print(colored(
            f"  Fast-path skipped: {n_skipped} "
            f"({100 * n_skipped / preds.shape[0]:.1f}%)", 'cyan'))
    print(f'TTC Accuracy: {accuracy:.2f}')

    succ = int(((preds == test_labels) & (top_n[:, 0] != preds)).sum())
    fail = int(((preds != test_labels) & (top_n[:, 0] == test_labels)).sum())
    return accuracy, succ, fail