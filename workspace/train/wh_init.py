"""Wiener-Hopf deterministic initialisation pipeline for HNSTPP.

Phases
------
1. Cross-covariance (FFT) + Block-Toeplitz solve  → R_effective (D×D)
2. Reverse mean-field mapping                      → R_exc, R_inh, b
3. Overcomplete dictionary generation              → K_s + K_m candidate rules
4. Dual-track feature caching                      → P_event, P_int
5. BCD + FISTA sparse optimisation                 → surviving rules
6. Debiased refit + Drop-one ΔNLL + Top-R truncation → final parameter injection

References
----------
- Bacry et al., 2012. Non-parametric kernel estimation for symmetric
  Hawkes processes.
- van Krieken et al., 2022. Analyzing Differentiable Fuzzy Logic Operators.
"""

from __future__ import annotations

from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F


# ================================================================== #
#  Public API                                                          #
# ================================================================== #

def wiener_hopf_initialize(
    model,
    data_list: list[dict],
    num_types: int,
    device: torch.device,
    *,
    delta_t: float = 0.1,
    max_lag: float = 10.0,
    lambda_l1: float = 1e-3,
    lambda_hier: float = 1e-4,
    fista_lr: float = 1e-2,
    fista_steps: int = 500,
    fista_inner_steps: int = 20,
    refit_steps: int = 200,
    prune: bool = True,
    val_ratio: float = 0.15,
    lift_threshold: float = 1.0,
    pair_min_support: int = 2,
    pair_topk_per_target: int = 20,
    max_source_order: int = 3,
    synergy_window: float | None = None,
    phase3_support_tau: float = 20.0,
    phase3_source_pool_topk: int = 24,
    phase3_beta2: float = 0.5,
    phase3_beta3: float = 1.0,
    int_grid_mult: int = 3,
    phase4_source_cap: float | None = None,
    exc_prefit_steps: int = 40,
    phase6_protect_best_inh_single: bool = True,
    phase6_family_conflict_penalty: float = 0.35,
    fixed_target: int | None = None,
) -> None:
    """Run the full 6-phase deterministic initialisation.

    Parameters
    ----------
    model : HNSTPP
        Model whose parameters are overwritten in-place.
    data_list : list[dict]
        ``[{'time': [...], 'event': [...]}, ...]`` (already time-scaled).
    num_types : int
        Number of event types *D*.
    device : torch.device
    delta_t, max_lag : float
        Discretisation resolution / maximum lag for Wiener-Hopf.
    lambda_l1 : float
        L₁ regularisation strength for FISTA (Phase 5).
    lambda_hier : float
        Hierarchical penalty weight (Phase 5).
    fista_lr : float
        Initial learning rate (step size) for FISTA proximal gradient.
    fista_steps : int
        Maximum BCD outer iterations.
    fista_inner_steps : int
        FISTA inner-loop iterations per block per outer step.
    refit_steps : int
        Gradient steps for debiased refit (Phase 6).
    val_ratio : float
        Fraction of data held out for validation (Phase 6).
    lift_threshold : float
        Kept for backwards compatibility (unused in the new Phase 3).
    pair_min_support : int
        Minimum rule-activation support x1 = n11 + n10 for higher-order
        candidate creation.
    pair_topk_per_target : int
        Max number of higher-order candidates retained per target *per order*
        in Phase 3. Set <=0 for no cap.
    max_source_order : int
        Maximum source-set size used in Phase 3 candidate generation.
        1 => single-only, 2 => single+pair, 3 => single+pair+triplet.
    phase3_support_tau : float
        Base temperature used in Phase 3 support shrinkage and multi-rule
        warm-start blending. Effective shrinkage temperature is estimated
        adaptively per source-order from candidate supports.
    phase3_source_pool_topk : int
        Kept for API compatibility. Not used by the current IE+NLL
        Phase-3 policy (full non-target source pool is used).
    phase3_beta2, phase3_beta3 : float
        Kept for API compatibility. Not used by the current IE+NLL
        Phase-3 policy.
    synergy_window : float | None
        Time window for synergy mining; defaults to ``model.max_cap``.
    int_grid_mult : int
        Grid density multiplier for integral caching (Phase 4).
    exc_prefit_steps : int
        Excitation-only proximal steps before residual-based inhibition seeding.
    fixed_target : int | None
        If set, build/optimise only rules targeting this event type and keep
        rule heads fixed to this target after initialisation.
    """
    D = num_types
    R = model.num_rules
    if fixed_target is None:
        raise ValueError(
            "This initializer now assumes fixed-target mode. "
            "Set `wh_fixed_target` in config."
        )
    fixed_target = int(fixed_target)
    if fixed_target < 0 or fixed_target >= D:
        raise ValueError(f"fixed_target must be in [0, {D-1}], got {fixed_target}")

    _banner("Wiener-Hopf Deterministic Initialisation")

    # ── Train / Validation split ─────────────────────────────────
    if len(data_list) <= 1:
        # No meaningful hold-out split possible.
        n_val = len(data_list)
        val_data = data_list
        train_data = data_list
        print("  [Warning] Dataset too small for disjoint hold-out; "
              "train/val will share data.")
    else:
        n_val = min(max(1, int(len(data_list) * val_ratio)), len(data_list) - 1)
        val_data = data_list[:n_val]
        train_data = data_list[n_val:]

    # Phase 1-2 ──────────────────────────────────────────────────────
    R_eff, Lambda = _phase1_wiener_hopf(train_data, D, delta_t, max_lag)
    R_exc, R_inh, b_vec = _phase2_mean_field(R_eff, Lambda)

    # Phase 3 ── Overcomplete Dictionary ─────────────────────────────
    window = synergy_window or float(model.max_cap or max_lag)
    candidates = _phase3_overcomplete_dictionary(
        R_exc, R_inh, train_data, D, window, lift_threshold,
        pair_min_support=pair_min_support,
        pair_topk_per_target=pair_topk_per_target,
        max_source_order=max_source_order,
        support_tau=phase3_support_tau,
        source_pool_topk=phase3_source_pool_topk,
        beta2=phase3_beta2,
        beta3=phase3_beta3,
        fixed_target=fixed_target,
    )

    # Phase 4 ── Feature Caching ─────────────────────────────────────
    P_event, P_int, meta = _phase4_cache_features(
        candidates, train_data, D, model, device, int_grid_mult,
        source_cap=phase4_source_cap,
    )
    val_cache = None
    if len(val_data) > 0:
        Pv_event, Pv_int, meta_v = _phase4_cache_features(
            candidates, val_data, D, model, device, int_grid_mult=2,
            source_cap=phase4_source_cap,
        )
        val_cache = {
            'P_event': Pv_event,
            'P_int': Pv_int,
            'meta': meta_v,
        }

    # Phase 5 ── BCD + FISTA ─────────────────────────────────────────
    w_exc, w_inh = _phase5_bcd_fista(
        P_event, P_int, meta, b_vec, candidates, D, device,
        lambda_l1=lambda_l1, lambda_hier=lambda_hier,
        lr=fista_lr, max_steps=fista_steps,
        inner_steps=fista_inner_steps,
        val_cache=val_cache,
        exc_prefit_steps=exc_prefit_steps,
        fixed_target=fixed_target,
    )

    # Phase 6 ── Debiased Refit + Top-R Truncation ───────────────────
    _phase6_refit_and_inject(
        model, candidates, w_exc, w_inh, b_vec,
        val_data, D, R, device,
        P_event, P_int, meta,
        refit_steps=refit_steps, refit_lr=fista_lr,
        prune=prune,
        val_cache=val_cache,
        phase5_family_state=None,
        protect_best_inh_single=phase6_protect_best_inh_single,
        family_conflict_penalty=phase6_family_conflict_penalty,
        fixed_target=fixed_target,
    )

    _banner("Initialisation complete")


# ================================================================== #
#  Phase 1 — Wiener-Hopf cross-covariance → R_effective               #
# ================================================================== #

def _phase1_wiener_hopf(data_list, D, delta_t, max_lag):
    """FFT-based cross-covariance + Block-Toeplitz solve.

    Returns R_effective (D, D) and Lambda (D,).
    """
    print("\n[Phase 1] Wiener-Hopf cross-covariance (FFT)")
    if delta_t <= 0:
        raise ValueError(f"delta_t must be > 0, got {delta_t}")
    L = int(max_lag / delta_t)
    if L < 1:
        print(f"  [Warning] max_lag/delta_t < 1 ({max_lag}/{delta_t}); using L=1")
        L = 1

    # ---- discretise sequences & accumulate FFT cross-spectrum ------
    total_time = 0.0
    counts = np.zeros(D)
    seq_lens: list[int] = []

    # First pass: discretise
    count_arrays: list[np.ndarray] = []
    for item in data_list:
        times = np.asarray(item['time'], dtype=np.float64)
        events = np.asarray(item['event'], dtype=np.int64)
        if len(times) == 0:
            continue
        T = times[-1] + delta_t
        total_time += T
        n_bins = int(np.ceil(T / delta_t))
        cs = np.zeros((D, n_bins))
        valid = (events >= 0) & (events < D)
        b_idx = np.minimum((times[valid] / delta_t).astype(int), n_bins - 1)
        np.add.at(cs, (events[valid], b_idx), 1)
        counts += cs.sum(axis=1)
        count_arrays.append(cs)
        seq_lens.append(n_bins)

    Lambda = counts / max(total_time, 1e-9)
    print(f"  Mean rates Λ: {np.round(Lambda, 4)}")
    if not count_arrays:
        print("  [Warning] No events found; returning zero R_effective.")
        return torch.zeros(D, D, dtype=torch.float32), Lambda

    # ---- FFT cross-correlation C_raw(τ) for τ ∈ [0, L) ─────────
    max_n = max(seq_lens) if seq_lens else 1
    n_fft = int(2 ** np.ceil(np.log2(max_n + L)))
    cross_spec = np.zeros((D, D, n_fft // 2 + 1), dtype=np.complex128)

    for cs in count_arrays:
        CS = np.fft.rfft(cs, n=n_fft, axis=1)          # (D, n_fft//2+1)
        cross_spec += CS[:, None, :] * CS[None, :, :].conj()

    C_raw = np.fft.irfft(cross_spec, n=n_fft, axis=2)[:, :, :L]  # (D,D,L)

    # ---- normalise: C(τ) = raw / (n_overlap · Δt²) − Λ⊗Λ ──────
    sl = np.asarray(seq_lens)
    lags = np.arange(L)
    n_overlap = np.maximum(sl[:, None] - lags[None, :], 0).sum(axis=0)
    n_overlap = np.maximum(n_overlap, 1)                # avoid /0

    C = C_raw / (n_overlap[None, None, :] * delta_t ** 2) \
        - np.outer(Lambda, Lambda)[:, :, None]          # (D, D, L)

    # ---- Block-Toeplitz system  Y = X A Δt ──────────────────────
    # Y[i, (a·D+j)] = C_ij(τ_a)
    Y = C.transpose(0, 2, 1).reshape(D, L * D)

    # A[(a·D+k, b·D+j)] = C_kj(|a−b|) (transpose for negative lag)
    # Build block-row for lag 0..L-1, then use Toeplitz symmetry
    blocks = np.empty((2 * L - 1, D, D))
    for lag in range(L):
        blocks[L - 1 + lag] = C[:, :, lag]              # C(+τ)
        blocks[L - 1 - lag] = C[:, :, lag].T             # C(−τ)

    idx = np.arange(L)
    row_idx = idx[:, None] - idx[None, :]                # (L, L)  diff matrix
    A = blocks[L - 1 + row_idx].transpose(0, 2, 1, 3).reshape(D * L, D * L)

    # diagonal noise correction: Λ/Δt on all diag blocks, + C(0) on block-0
    diag_noise = np.diag(Lambda / delta_t)
    for a in range(L):
        blk = diag_noise.copy()
        if a == 0:
            blk = blk + C[:, :, 0]
        A[a * D:(a + 1) * D, a * D:(a + 1) * D] += blk

    # ---- solve & integrate ─────────────────────────────────────
    At = torch.as_tensor(A, dtype=torch.float64)
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    try:
        Xt = torch.linalg.solve(At.T, Yt.T).T / delta_t   # (D, D·L)
    except RuntimeError:
        # Fallback 1: tiny ridge for near-singular systems
        eye = torch.eye(At.shape[0], dtype=torch.float64, device=At.device)
        ridge = 1e-6
        try:
            Xt = torch.linalg.solve((At + ridge * eye).T, Yt.T).T / delta_t
            print("  [Warning] Singular WH system; solved with ridge regularisation.")
        except RuntimeError:
            # Fallback 2: least-squares pseudo-solution
            Xt = torch.linalg.lstsq(At.T, Yt.T).solution.T / delta_t
            print("  [Warning] Singular WH system; solved with least-squares fallback.")

    R_eff = Xt.reshape(D, L, D).sum(dim=1).float() * delta_t  # (D, D)
    if not torch.isfinite(R_eff).all():
        print("  [Warning] Non-finite R_effective detected; applying nan_to_num.")
        R_eff = torch.nan_to_num(R_eff, nan=0.0, posinf=1e6, neginf=-1e6)
    max_abs = float(R_eff.abs().max().item())
    if max_abs > 1e6:
        print(f"  [Warning] R_effective too large (max={max_abs:.2e}); clipping to ±1e6.")
        R_eff = R_eff.clamp(min=-1e6, max=1e6)
    print(f"  R_effective range: [{R_eff.min():.4f}, {R_eff.max():.4f}]")

    return R_eff, Lambda


# ================================================================== #
#  Phase 2 — Reverse mean-field mapping                                #
# ================================================================== #

def _phase2_mean_field(R_eff, Lambda):
    """Algebraic inversion from linear R̃ to nonlinear (R_exc, R_inh, b).

    Math
    ----
    R_inh[i,j] = R̃_inh[i,j] / Λ_i
    R_exc[i,j] = R̃_exc[i,j] · exp( Σ_j' R_inh[i,j'] Λ_j' )
    b[i]       = Λ_i · exp( Σ_j R_inh[i,j] Λ_j ) − Σ_j R_exc[i,j] Λ_j
    """
    print("\n[Phase 2] Reverse mean-field mapping")
    Lam = torch.as_tensor(Lambda, dtype=torch.float32)    # (D,)

    R_tilde_exc = torch.relu(R_eff)
    R_tilde_inh = torch.relu(-R_eff)                      # positive magnitude

    # R_inh = R̃_inh / Λ_i  (target rate in denominator)
    R_inh = R_tilde_inh / Lam.unsqueeze(1).clamp(min=1e-9)

    # Ī_i = Σ_j R_inh[i,j] Λ_j
    I_bar = (R_inh * Lam.unsqueeze(0)).sum(dim=1)         # (D,)
    exp_i = torch.exp(I_bar.clamp(min=0.0, max=20.0))
    R_exc = R_tilde_exc * exp_i.unsqueeze(1)
    R_exc = torch.nan_to_num(R_exc, nan=0.0, posinf=1e6, neginf=0.0)

    # Check spectral radius of R_exc for stability; scale down if ≥1.0
    try:
        eigenvalues = torch.linalg.eigvals(R_exc.to(torch.float64))
        spectral_radius = float(eigenvalues.abs().max().item())
    except RuntimeError:
        spectral_radius = float(torch.linalg.norm(R_exc.to(torch.float64), ord=2).item())
        print("  [Warning] eigvals failed; using spectral-norm upper bound for stability check.")
    if spectral_radius >= 1.0:
        print(f"  [Warning] Spectral radius {spectral_radius:.4f} >= 1.0. Scaling down R_exc.")
        R_exc = R_exc * (0.95 / spectral_radius)

    b_vec = (Lam * exp_i
             - (R_exc * Lam.unsqueeze(0)).sum(dim=1)).clamp(min=1e-4)

    print(f"  R_exc range: [{R_exc.min():.4f}, {R_exc.max():.4f}],"
          f"  R_inh range: [{R_inh.min():.4f}, {R_inh.max():.4f}]")
    print(f"  b_vec: {b_vec.numpy().round(4)}")
    return R_exc, R_inh, b_vec


# ================================================================== #
#  Phase 3 — Overcomplete Dictionary Generation                        #
# ================================================================== #

def _phase3_overcomplete_dictionary(
    R_exc, R_inh, data_list, D, window, lift_threshold, eps=1e-4,
    *,
    pair_min_support: int = 2,
    pair_topk_per_target: int = 20,
    max_source_order: int = 3,
    support_tau: float = 20.0,
    source_pool_topk: int = 24,
    beta2: float = 0.5,
    beta3: float = 1.0,
    fixed_target: int | None = None,
):
    """Build fixed-target candidate dictionary (single + higher-order).

    Strategy used here:
    - `single`: Phase-2 priors directly define warm-starts.
    - `pair/triplet`: IE-only warm-starts (no Phase-3 backoff blending).
    - Final ranking across higher-order candidates uses a unified
      projected-NLL gain proxy with order-wise support shrinkage.

    Notes
    -----
    `source_pool_topk`, `beta2`, and `beta3` are kept in the signature for
    backwards-compatibility, but are not used by this IE+NLL Phase-3 policy.
    """
    print(f"\n[Phase 3] Overcomplete dictionary generation")

    if fixed_target is None:
        raise ValueError("Phase 3 now requires fixed_target mode.")
    max_source_order = max(1, int(max_source_order))
    tgt = int(fixed_target)
    if tgt < 0 or tgt >= D:
        raise ValueError(f"fixed_target must be in [0, {D-1}], got {tgt}")
    _ = source_pool_topk, beta2, beta3

    R_exc_np = R_exc.detach().cpu().numpy() if torch.is_tensor(R_exc) else np.asarray(R_exc)
    R_inh_np = R_inh.detach().cpu().numpy() if torch.is_tensor(R_inh) else np.asarray(R_inh)

    windows = sorted(set([
        max(0.5, 0.35 * float(window)),
        max(0.8, 0.70 * float(window)),
        float(window),
    ]))
    window_cache: list[tuple[float, np.ndarray, np.ndarray, int, int, float]] = []
    for win in windows:
        X, y = _build_history_presence(data_list, D, win)
        if X.shape[0] == 0:
            continue
        y_mask = (y == tgt)
        y1 = int(y_mask.sum())
        n_total = int(X.shape[0])
        if y1 <= 0 or n_total <= 0:
            continue
        lam0 = float(np.clip(y1 / max(n_total, 1), 1e-6, 1.0 - 1e-6))
        window_cache.append((float(win), X, y_mask, y1, n_total, lam0))

    # ── 3.1 Single-source pool (source != target) ───────
    candidates: list[dict] = []
    seen_keys: set[tuple[int, tuple[int, ...]]] = set()
    for src in range(D):
        if src == tgt:
            continue
        key = (tgt, (src,))
        seen_keys.add(key)

        exc_val = float(R_exc[tgt, src])
        inh_val = float(R_inh[tgt, src])
        w_exc_init = exc_val if exc_val > eps else 0.0
        w_inh_init = inh_val if inh_val > eps else 0.0
        sign_hint = 1 if exc_val > inh_val else (-1 if inh_val > exc_val else 0)

        best_score = -1.0
        best_lor = 0.0
        for _, X, y_mask, y1, n_total, _ in window_cache:
            emp_exc, emp_inh, emp_score, emp_lor = _single_init_from_presence(
                X, y_mask, src, y1, n_total,
            )
            if emp_score <= best_score:
                continue
            best_score = emp_score
            best_lor = emp_lor
            w_exc_init = max(w_exc_init, emp_exc)
            w_inh_init = max(w_inh_init, emp_inh)

        if w_exc_init > w_inh_init + 1e-12:
            sign_hint = 1
        elif w_inh_init > w_exc_init + 1e-12:
            sign_hint = -1
        elif best_score > 0.0:
            sign_hint = 1 if best_lor >= 0.0 else -1

        candidates.append({
            'sources': [src],
            'target': tgt,
            'w_exc_init': float(w_exc_init),
            'w_inh_init': float(w_inh_init),
            'lor_sign': int(sign_hint),
        })
    Ks = len(candidates)
    print(f"  Single-source candidates: {Ks}")

    if max_source_order == 1:
        print(f"  Total overcomplete dictionary K = {len(candidates)}")
        return candidates

    multi_counts = {order: 0 for order in range(2, max_source_order + 1)}
    print("  Higher-order mode: aligned score-mine "
          f"(windows={np.round(windows, 3).tolist()}, min_x1_support={pair_min_support}, "
          f"topk/target/order={pair_topk_per_target})")

    # Keep best projected-NLL row per (order, target, canonical-sources).
    best_rows: dict[tuple[int, int, tuple[int, ...]], dict] = {}
    for _, X, y_mask, _, _, lam0 in window_cache:

        # Use full non-target pool in IE+NLL mode for fair single/pair/triplet competition.
        src_pool = [s for s in range(D) if s != tgt]

        for order in range(2, max_source_order + 1):
            for sources in combinations(src_pool, order):
                if order == 2:
                    a, b = sources
                    m = X[:, a] & X[:, b]
                    x1 = int(m.sum())
                    if x1 < max(1, int(pair_min_support)):
                        continue
                    n11 = int((m & y_mask).sum())
                    w_exc_ie, w_inh_ie, _ = _pair_ie_init_from_presence(X, y_mask, a, b)
                elif order == 3:
                    a, b, c = sources
                    w_exc_ie, w_inh_ie, x1, n11 = _triplet_ie_exact_from_presence(X, y_mask, a, b, c)
                    if x1 < max(1, int(pair_min_support)):
                        continue
                else:
                    continue

                # Higher-order candidates are IE-only in this Phase-3 policy.
                if (w_exc_ie + w_inh_ie) <= 0.0:
                    continue

                # Exact Binomial deviance with directional split.
                # Each active row contributes a binary target event indicator,
                # so a Binomial null is more appropriate than the previous
                # asymmetric proxy.
                mu = max(lam0 * x1, 1e-9)
                p0 = float(np.clip(lam0, 1e-9, 1.0 - 1e-9))
                phat = float(np.clip(n11 / max(x1, 1), 1e-9, 1.0 - 1e-9))
                dev = 2.0 * (
                    n11 * np.log(phat / p0)
                    + (x1 - n11) * np.log((1.0 - phat) / (1.0 - p0))
                )
                if n11 > mu:
                    d_exc = float(dev)
                    d_inh = 0.0
                elif n11 < mu:
                    d_exc = 0.0
                    d_inh = float(dev)
                else:
                    d_exc = 0.0
                    d_inh = 0.0
                nll_gain = float(max(d_exc, d_inh))
                if nll_gain <= 0.0:
                    continue

                w_exc_init = float(np.clip(w_exc_ie, 0.0, 2.0))
                w_inh_init = float(np.clip(w_inh_ie, 0.0, 2.0))
                # Keep the deviance-derived direction instead of overwriting it
                # with IE warm-start magnitudes.
                if d_exc > d_inh:
                    sign_hint = 1
                elif d_inh > d_exc:
                    sign_hint = -1
                else:
                    sign_hint = int(sign_hint)

                sources_key = tuple(int(s) for s in sources)
                key = (order, tgt, sources_key)
                cur = {
                    'order': int(order),
                    'target': int(tgt),
                    'sources': sources_key,
                    'support_x1': int(x1),
                    'nll_gain': nll_gain,
                    'w_exc_init': w_exc_init,
                    'w_inh_init': w_inh_init,
                    'sign_hint': int(sign_hint),
                }
                prev = best_rows.get(key)
                if prev is None or cur['nll_gain'] > prev['nll_gain']:
                    best_rows[key] = cur

    if not best_rows:
        for order in range(2, max_source_order + 1):
            print(f"  Order-{order} candidates: 0")
        print(f"  Total overcomplete dictionary K = {len(candidates)}")
        return candidates

    # Adaptive tau per order from support medians (stabilised by global median).
    all_supports = np.asarray(
        [int(v.get('support_x1', 0)) for v in best_rows.values() if int(v.get('support_x1', 0)) > 0],
        dtype=np.float64,
    )
    global_tau = float(max(np.median(all_supports), max(1, int(pair_min_support)))) \
        if len(all_supports) else float(max(1, int(pair_min_support)))

    tau_by_order: dict[int, float] = {}
    for order in range(2, max_source_order + 1):
        x_ord = [int(v['support_x1']) for (o, _, _), v in best_rows.items() if int(o) == order]
        tau_by_order[order] = _adaptive_tau_from_supports(
            x_ord,
            min_support=max(1, int(pair_min_support)),
            global_tau=global_tau,
        )
    tau_log = ", ".join([f"order{ord_k}={tau_by_order[ord_k]:.3f}" for ord_k in sorted(tau_by_order)])
    print(f"  Adaptive tau: {tau_log}")

    # Build order-wise scored rows.
    for order in range(2, max_source_order + 1):
        rows_t = [dict(v) for (o, tt, _), v in best_rows.items() if int(o) == order and int(tt) == tgt]
        if not rows_t:
            print(f"  Order-{order} candidates: 0")
            continue

        tau_ord = float(tau_by_order.get(order, max(1, int(pair_min_support))))
        for row in rows_t:
            x1 = int(row.get('support_x1', 0))
            q = np.sqrt(float(x1) / float(x1 + tau_ord)) if x1 > 0 else 0.0
            score_raw = float(float(row['nll_gain']) * float(q))
            row['score_raw'] = score_raw
            row['sign_hint'] = int(row.get('sign_hint', 0))

        raw = np.asarray([float(r['score_raw']) for r in rows_t], dtype=np.float64)
        med = float(np.median(raw))
        mad = float(np.median(np.abs(raw - med)))
        scale = max(1.4826 * mad, 1e-8)
        for r in rows_t:
            r['score_norm'] = float((float(r['score_raw']) - med) / scale)

        rows_t.sort(key=lambda r: float(r['score_norm']), reverse=True)
        rows_pos = [r for r in rows_t if int(r.get('sign_hint', 0)) > 0]
        rows_neg = [r for r in rows_t if int(r.get('sign_hint', 0)) < 0]

        selected: list[dict] = []
        if pair_topk_per_target > 0:
            k_main = int(pair_topk_per_target)
            k_side = max(1, k_main // 2)
            selected.extend(rows_t[:k_main])
            selected.extend(rows_pos[:k_side])
            selected.extend(rows_neg[:k_side])
        else:
            selected.extend(rows_t)

        seen_local: set[tuple[int, ...]] = set()
        for row in selected:
            sources = tuple(sorted(int(x) for x in row['sources']))
            if sources in seen_local:
                continue
            seen_local.add(sources)
            key = (tgt, sources)
            if key in seen_keys:
                continue
            if float(row.get('score_raw', 0.0)) <= 0.0:
                continue
            seen_keys.add(key)
            candidates.append({
                'sources': list(sources),
                'target': tgt,
                'w_exc_init': float(max(0.0, row.get('w_exc_init', 0.0))),
                'w_inh_init': float(max(0.0, row.get('w_inh_init', 0.0))),
                'lor_sign': int(row.get('sign_hint', 0)),
            })
            multi_counts[order] += 1

        print(f"  Order-{order} candidates: {multi_counts[order]}")

    print(f"  Total overcomplete dictionary K = {len(candidates)}")
    return candidates


def _build_history_presence(data_list, D: int, window: float):
    """Build causal binary history features X for one time window.

    Returns
    -------
    X : (N, D) bool
        X[n, s]=True iff type-s appeared in (t_n - window, t_n).
    y : (N,) int64
        Event type at each evaluated event.
    """
    from collections import deque

    window = float(max(window, 1e-6))
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []

    for item in data_list:
        times = np.asarray(item['time'], dtype=np.float64)
        events = np.asarray(item['event'], dtype=np.int64)
        n = len(times)
        if n <= 1:
            continue

        counts = np.zeros(D, dtype=np.int32)
        q: deque[tuple[float, int]] = deque()
        x_seq = np.zeros((n - 1, D), dtype=np.bool_)
        y_seq = np.empty((n - 1,), dtype=np.int64)
        out_i = 0

        for i in range(n):
            t_i = float(times[i])
            while q and (t_i - q[0][0] > window):
                _, old_type = q.popleft()
                if 0 <= old_type < D:
                    counts[old_type] -= 1

            e_i = int(events[i])
            if i > 0 and 0 <= e_i < D:
                x_seq[out_i] = counts > 0
                y_seq[out_i] = e_i
                out_i += 1

            if 0 <= e_i < D:
                q.append((t_i, e_i))
                counts[e_i] += 1

        if out_i > 0:
            rows_x.append(x_seq[:out_i])
            rows_y.append(y_seq[:out_i])

    if not rows_x:
        return np.zeros((0, D), dtype=np.bool_), np.zeros((0,), dtype=np.int64)
    return np.concatenate(rows_x, axis=0), np.concatenate(rows_y, axis=0)


def _lor_stats_from_counts(n11: int, x1: int, y1: int, n_total: int):
    """Return (log-odds-ratio, z-score) from 2x2 table with Haldane correction."""
    n10 = max(x1 - n11, 0)
    n01 = max(y1 - n11, 0)
    n00 = max(n_total - n11 - n10 - n01, 0)

    a = n11 + 0.5
    b = n10 + 0.5
    c = n01 + 0.5
    d = n00 + 0.5
    lor = float(np.log((a * d) / (b * c)))
    se = float(np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d))
    z = lor / max(se, 1e-9)
    return lor, z


def _safe_conditional_rate(
    mask: np.ndarray,
    y_mask: np.ndarray,
    *,
    eps: float = 1e-4,
    r_min: float = 1e-6,
    r_max: float = 1.0 - 1e-6,
):
    """Smoothed conditional target rate P(y=1|mask) with clipping."""
    den = int(mask.sum())
    num = int((mask & y_mask).sum())
    rate = (num + float(eps)) / (den + 2.0 * float(eps))
    rate = float(np.clip(rate, r_min, r_max))
    return rate, den


def _single_init_from_presence(
    X: np.ndarray,
    y_mask: np.ndarray,
    src: int,
    y1: int,
    n_total: int,
):
    """Single-source empirical warm-start and sign score on presence features."""
    A = X[:, int(src)]
    x1 = int(A.sum())
    if x1 <= 0:
        return 0.0, 0.0, 0.0, 0.0

    n11 = int((A & y_mask).sum())
    r0, _ = _safe_conditional_rate(~A, y_mask)
    r1, _ = _safe_conditional_rate(A, y_mask)
    lor, z = _lor_stats_from_counts(n11, x1, y1, n_total)

    w_exc = max(0.0, r1 - r0)
    w_inh = max(0.0, np.log(r0) - np.log(r1))
    score = float(abs(z) * max(w_exc, w_inh))
    return float(w_exc), float(w_inh), score, float(lor)


def _pair_ie_init_from_presence(X: np.ndarray, y_mask: np.ndarray, a: int, b: int):
    """Pair warm-start via disjoint-set inclusion-exclusion on presence features."""
    A = X[:, a]
    B = X[:, b]
    m_ab = A & B
    m_a_only = A & (~B)
    m_b_only = (~A) & B
    m_none = (~A) & (~B)

    r0, _ = _safe_conditional_rate(m_none, y_mask)
    r_a, _ = _safe_conditional_rate(m_a_only, y_mask)
    r_b, _ = _safe_conditional_rate(m_b_only, y_mask)
    r_ab, x_ab = _safe_conditional_rate(m_ab, y_mask)

    w_exc_ie = max(0.0, r_ab - (r_a + r_b - r0))
    w_inh_ie = max(0.0, np.log(r_a) + np.log(r_b) - np.log(r0) - np.log(r_ab))
    return float(w_exc_ie), float(w_inh_ie), int(x_ab)


def _triplet_ie_exact_from_presence(
    X: np.ndarray,
    y_mask: np.ndarray,
    a: int,
    b: int,
    c: int,
):
    """Triplet warm-start via exact 8-cell three-way log interaction.

    Returns
    -------
    w_exc_ie : float
        Positive interaction magnitude (excitation channel).
    w_inh_ie : float
        Negative interaction magnitude mapped to positive inhibition magnitude.
    x111 : int
        Support count for A&B&C cell.
    n11 : int
        Positive-event count inside A&B&C cell.
    """
    A = X[:, a]
    B = X[:, b]
    C = X[:, c]
    cells = {
        '000': (~A) & (~B) & (~C),
        '100': A & (~B) & (~C),
        '010': (~A) & B & (~C),
        '001': (~A) & (~B) & C,
        '110': A & B & (~C),
        '101': A & (~B) & C,
        '011': (~A) & B & C,
        '111': A & B & C,
    }

    rates: dict[str, float] = {}
    dens: dict[str, int] = {}
    for key, m in cells.items():
        r, d = _safe_conditional_rate(m, y_mask)
        rates[key] = float(r)
        dens[key] = int(d)

    x111 = int(dens['111'])
    if x111 <= 0:
        return 0.0, 0.0, 0, 0

    xi = (
        np.log(rates['111'])
        - np.log(rates['110']) - np.log(rates['101']) - np.log(rates['011'])
        + np.log(rates['100']) + np.log(rates['010']) + np.log(rates['001'])
        - np.log(rates['000'])
    )
    xi = float(np.clip(xi, -8.0, 8.0))

    n11 = int((cells['111'] & y_mask).sum())
    w_exc_ie = max(0.0, xi)
    w_inh_ie = max(0.0, -xi)
    return float(w_exc_ie), float(w_inh_ie), int(x111), int(n11)


def _triplet_ie_surrogate_from_presence(
    X: np.ndarray,
    y_mask: np.ndarray,
    a: int,
    b: int,
    c: int,
):
    """Backward-compatible wrapper around exact triplet IE."""
    w_exc, w_inh, x111, _ = _triplet_ie_exact_from_presence(X, y_mask, a, b, c)
    return float(w_exc), float(w_inh), int(x111)


def _feature_backoff_init_from_presence(
    X: np.ndarray,
    y_mask: np.ndarray,
    sources: tuple[int, ...],
    R_exc_row: np.ndarray,
    R_inh_row: np.ndarray,
    *,
    tau_mix: float = 20.0,
):
    """Backoff warm-start based on conditional single-feature contribution."""
    cond = np.ones(X.shape[0], dtype=bool)
    for s in sources:
        cond &= X[:, int(s)]
    x_multi = int(cond.sum())
    if x_multi <= 0:
        return 0.0, 0.0, 0.0

    lam_hat, _ = _safe_conditional_rate(cond, y_mask)
    mu = X[cond].mean(axis=0).astype(np.float64) if x_multi > 0 else np.zeros(X.shape[1], dtype=np.float64)
    b0 = float(np.clip(y_mask.mean(), 1e-6, 1.0 - 1e-6))

    E_base = b0 + float(np.dot(R_exc_row, mu))
    I_base = float(np.dot(R_inh_row, mu))
    E_base = float(np.clip(E_base, 1e-6, 1e6))
    I_base = float(np.clip(I_base, 0.0, 20.0))
    lam_exp = float(np.clip(E_base * np.exp(-I_base), 1e-6, 1.0 - 1e-6))

    w_exc_b = max(0.0, lam_hat * np.exp(I_base) - E_base)
    w_inh_b = max(0.0, np.log(lam_exp) - np.log(lam_hat))
    rho = float(x_multi / (x_multi + max(float(tau_mix), 1e-6)))
    return float(w_exc_b), float(w_inh_b), float(rho)


def _adaptive_tau_from_supports(
    supports: list[int],
    *,
    min_support: int,
    global_tau: float,
    stabilizer: float = 20.0,
):
    """Order-wise adaptive tau from median support with small-sample stabilisation."""
    if len(supports) == 0:
        return float(max(min_support, int(np.ceil(global_tau))))
    med = float(np.median(np.asarray(supports, dtype=np.float64)))
    tau = max(med, float(min_support))
    n = float(len(supports))
    rho = n / (n + max(float(stabilizer), 1e-6))
    tau = rho * tau + (1.0 - rho) * float(global_tau)
    return float(max(tau, float(min_support)))


def _mine_higher_order_rows(
    X: np.ndarray,
    y_mask: np.ndarray,
    target: int,
    D: int,
    *,
    max_source_order: int,
    min_support: int,
    support_tau: float,
    source_pool_topk: int,
    beta2: float,
    beta3: float,
):
    """Mine pair/triplet rows for one target and one window.

    Notes
    -----
    - Support gate uses x1 = n11 + n10 (rule activation count).
    - Sign-aware scoring separates excitation/inhibition channels:
      z+ = max(z, 0), z- = max(-z, 0).
    """
    n_total = int(X.shape[0])
    y1 = int(y_mask.sum())
    if n_total <= 0 or y1 <= 0:
        return []

    non_target = [s for s in range(D) if s != int(target)]
    if not non_target:
        return []

    X_nt = X[:, non_target]                           # (N, S)
    x1_single = X_nt.sum(axis=0).astype(np.int64)
    n11_single = (X_nt & y_mask[:, None]).sum(axis=0).astype(np.int64)

    single_z_pos: dict[int, float] = {}
    single_z_neg: dict[int, float] = {}
    single_score: dict[int, float] = {}
    tau = max(float(support_tau), 1e-6)
    for local_i, src in enumerate(non_target):
        n11 = int(n11_single[local_i])
        x1 = int(x1_single[local_i])
        n10 = max(x1 - n11, 0)
        _, z = _lor_stats_from_counts(n11, x1, y1, n_total)
        z_pos = max(float(z), 0.0)
        z_neg = max(float(-z), 0.0)
        q_exc = np.sqrt(max(n11, 0) / (max(n11, 0) + tau))
        q_inh = np.sqrt(max(n10, 0) / (max(n10, 0) + tau))
        single_z_pos[src] = z_pos
        single_z_neg[src] = z_neg
        single_score[src] = max(z_pos * float(q_exc), z_neg * float(q_inh))

    if source_pool_topk > 0:
        ranked_src = sorted(non_target, key=lambda s: single_score[s], reverse=True)
        src_pool = ranked_src[:min(int(source_pool_topk), len(ranked_src))]
    else:
        src_pool = non_target

    rows: list[dict] = []
    pair_z_pos: dict[tuple[int, int], float] = {}
    pair_z_neg: dict[tuple[int, int], float] = {}

    if max_source_order >= 2 and len(src_pool) >= 2:
        for a, b in combinations(src_pool, 2):
            x = X[:, a] & X[:, b]
            x1 = int(x.sum())
            if x1 < min_support:
                continue
            n11 = int((x & y_mask).sum())
            n10 = max(x1 - n11, 0)
            lor, z = _lor_stats_from_counts(n11, x1, y1, n_total)
            z_pos = max(float(z), 0.0)
            z_neg = max(float(-z), 0.0)
            delta_exc = max(0.0, z_pos - max(single_z_pos[a], single_z_pos[b]))
            delta_inh = max(0.0, z_neg - max(single_z_neg[a], single_z_neg[b]))
            q_exc = np.sqrt(max(n11, 0) / (max(n11, 0) + tau))
            q_inh = np.sqrt(max(n10, 0) / (max(n10, 0) + tau))
            score_exc = (z_pos + beta2 * delta_exc) * float(q_exc)
            score_inh = (z_neg + beta2 * delta_inh) * float(q_inh)
            if score_exc > score_inh + 1e-12:
                sign_hint = 1
                score_raw = score_exc
            elif score_inh > score_exc + 1e-12:
                sign_hint = -1
                score_raw = score_inh
            else:
                sign_hint = 1 if lor > 0 else (-1 if lor < 0 else 0)
                score_raw = max(score_exc, score_inh)
            pair_z_pos[(a, b)] = z_pos
            pair_z_neg[(a, b)] = z_neg
            rows.append({
                'order': 2,
                'target': int(target),
                'sources': (int(a), int(b)),
                'support_x1': x1,
                'support_n11': n11,
                'support_n10': n10,
                'lor': float(lor),
                'sign_hint': int(sign_hint),
                'score_exc': float(score_exc),
                'score_inh': float(score_inh),
                'score_raw': float(score_raw),
            })

    if max_source_order >= 3 and len(src_pool) >= 3:
        for a, b, c in combinations(src_pool, 3):
            x = X[:, a] & X[:, b] & X[:, c]
            x1 = int(x.sum())
            if x1 < min_support:
                continue
            n11 = int((x & y_mask).sum())
            n10 = max(x1 - n11, 0)
            lor, z = _lor_stats_from_counts(n11, x1, y1, n_total)
            z_pos = max(float(z), 0.0)
            z_neg = max(float(-z), 0.0)
            z_ab_pos = pair_z_pos.get((min(a, b), max(a, b)), 0.0)
            z_ac_pos = pair_z_pos.get((min(a, c), max(a, c)), 0.0)
            z_bc_pos = pair_z_pos.get((min(b, c), max(b, c)), 0.0)
            z_ab_neg = pair_z_neg.get((min(a, b), max(a, b)), 0.0)
            z_ac_neg = pair_z_neg.get((min(a, c), max(a, c)), 0.0)
            z_bc_neg = pair_z_neg.get((min(b, c), max(b, c)), 0.0)
            delta_exc = max(0.0, z_pos - max(z_ab_pos, z_ac_pos, z_bc_pos))
            delta_inh = max(0.0, z_neg - max(z_ab_neg, z_ac_neg, z_bc_neg))
            q_exc = np.sqrt(max(n11, 0) / (max(n11, 0) + tau))
            q_inh = np.sqrt(max(n10, 0) / (max(n10, 0) + tau))
            score_exc = (z_pos + beta3 * delta_exc) * float(q_exc)
            score_inh = (z_neg + beta3 * delta_inh) * float(q_inh)
            if score_exc > score_inh + 1e-12:
                sign_hint = 1
                score_raw = score_exc
            elif score_inh > score_exc + 1e-12:
                sign_hint = -1
                score_raw = score_inh
            else:
                sign_hint = 1 if lor > 0 else (-1 if lor < 0 else 0)
                score_raw = max(score_exc, score_inh)
            rows.append({
                'order': 3,
                'target': int(target),
                'sources': (int(a), int(b), int(c)),
                'support_x1': x1,
                'support_n11': n11,
                'support_n10': n10,
                'lor': float(lor),
                'sign_hint': int(sign_hint),
                'score_exc': float(score_exc),
                'score_inh': float(score_inh),
                'score_raw': float(score_raw),
            })
    return rows


# ================================================================== #
#  Phase 4 — Dual-Track Feature Caching                                #
# ================================================================== #

def _phase4_cache_features(
    candidates, data_list, D, model, device, int_grid_mult=3,
    source_cap: float | None = None,
):
    """Pre-compute P_event [N, K] and P_int [M_grid, K].

    For each candidate rule *c*, p_c(t) = ReLU(S_c(t) − bias_c),
    where  S_c(t) = Σ_{j: t_j<t} H_{k_j,c} · g_c(t − t_j)
    and H is the candidate's binary source mask.

    The kernel g and bias are taken from the *model* (flat-initialised).
    We evaluate once per event-time (P_event) and once per integration
    grid-point (P_int).

    Returns  P_event, P_int, meta  (all on *device*).
    """
    print("\n[Phase 4] Dual-track feature caching")
    K_cand = len(candidates)
    if source_cap is not None:
        source_cap = float(max(source_cap, 0.0))

    # ── Collect all events into flat arrays ─────────────────────
    all_times: list[np.ndarray] = []
    all_types: list[np.ndarray] = []
    all_seq_id: list[np.ndarray] = []
    all_T: list[float] = []
    seq_local = 0
    for item in data_list:
        t = np.asarray(item['time'], dtype=np.float64)
        e = np.asarray(item['event'], dtype=np.int64)
        if len(t) == 0:
            continue
        all_times.append(t)
        all_types.append(e)
        # IMPORTANT: use contiguous local sequence ids (0..n_seqs-1)
        # so indexing remains consistent when empty sequences exist.
        all_seq_id.append(np.full(len(t), seq_local, dtype=np.int64))
        all_T.append(float(t[-1]) + 1e-6)
        seq_local += 1

    if not all_times:
        # Degenerate case: no events in train_data.
        print("  [Warning] Empty training events; returning zero feature caches.")
        P_event = torch.zeros(1, K_cand, dtype=torch.float32, device=device)
        P_int = torch.zeros(1, K_cand, dtype=torch.float32, device=device)
        event_target_oh = torch.zeros(1, D, dtype=torch.float32, device=device)
        cand_target_oh = torch.zeros(K_cand, D, dtype=torch.float32, device=device)
        for c_idx, cand in enumerate(candidates):
            cand_target_oh[c_idx, cand['target']] = 1.0
        meta = {
            'event_target_oh': event_target_oh,
            'cand_target_oh': cand_target_oh,
            'grid_weights': torch.ones(1, dtype=torch.float32, device=device),
            'N': 1,
            'D': D,
            'K': K_cand,
        }
        return P_event, P_int, meta

    flat_times = np.concatenate(all_times)    # (N_total,)
    flat_types = np.concatenate(all_types)    # (N_total,)
    flat_seq   = np.concatenate(all_seq_id)   # (N_total,)
    N = len(flat_times)
    print(f"  Total events N = {N}")

    # ── Build integration grid ──────────────────────────────────
    M_grid = N * int_grid_mult
    grid_times_list: list[np.ndarray] = []
    grid_seq_list: list[np.ndarray] = []
    grid_weights_list: list[np.ndarray] = []
    for s_idx, T_s in enumerate(all_T):
        n_pts = max(2, int(M_grid * T_s / sum(all_T)))
        g_t = np.linspace(0, T_s, n_pts + 1)
        # Trapezoidal weights: dt * (0.5 at endpoints, 1.0 interior)
        dt = T_s / n_pts
        w = np.full(n_pts + 1, dt)
        w[0] *= 0.5
        w[-1] *= 0.5
        grid_times_list.append(g_t)
        grid_seq_list.append(np.full(n_pts + 1, s_idx, dtype=np.int64))
        grid_weights_list.append(w)

    grid_times = np.concatenate(grid_times_list)
    grid_seq   = np.concatenate(grid_seq_list)
    grid_wts   = np.concatenate(grid_weights_list)
    M_actual = len(grid_times)
    print(f"  Integration grid M = {M_actual}")

    # ── Kernel setup from model ─────────────────────────────────
    max_cap = float(model.max_cap or 10.0)
    M_bins = model.num_bins
    # Use a flat kernel height for overcomplete candidates.
    # We keep this scalar synchronized with bias design (A-option):
    #   bias_r = (|S_r| - 0.5) * h_unit
    # so that a perfect pair activation (two sources once each) stays active.
    h_unit = 0.5
    h_raw_flat = float(_sp_inv(h_unit))
    h_full = np.zeros(M_bins + 1)  # endpoints 0
    h_full[1:-1] = np.log1p(np.exp(h_raw_flat))  # softplus(h_raw)
    bin_w = max_cap / M_bins

    def _eval_kernel_np(dt_arr):
        """Evaluate piecewise-linear kernel on numpy array.  Returns (len,)."""
        valid = (dt_arr >= 0) & (dt_arr < max_cap)
        dt_c = np.clip(dt_arr, 0, max_cap * (1 - 1e-7))
        dt_norm = dt_c / bin_w
        idx = np.clip(dt_norm.astype(np.int64), 0, M_bins - 1)
        frac = dt_norm - idx
        vals = h_full[idx] + (h_full[idx + 1] - h_full[idx]) * frac
        return vals * valid

    # ── Candidate source mask and bias (A-option synchronized with h_unit) ──
    biases = np.array([
        max((len(c['sources']) - 0.5) * h_unit, 0.0) for c in candidates
    ], dtype=np.float32)  # (K_cand,)
    src_mask_mat = np.zeros((D, K_cand), dtype=np.float32)  # (D,K)
    for c_idx, cand in enumerate(candidates):
        for src in cand['sources']:
            if 0 <= src < D:
                src_mask_mat[src, c_idx] = 1.0

    # ── Compute S_src(t) once, then project to candidates ───────
    # Group events by sequence for efficient per-sequence processing
    n_seqs = len(all_T)
    seq_event_idx: list[np.ndarray] = []   # indices into flat_times per seq
    for s in range(n_seqs):
        seq_event_idx.append(np.where(flat_seq == s)[0])

    seq_grid_idx: list[np.ndarray] = []
    for s in range(n_seqs):
        seq_grid_idx.append(np.where(grid_seq == s)[0])

    # Output arrays
    P_event_np = np.zeros((N, K_cand), dtype=np.float32)
    P_int_np   = np.zeros((M_actual, K_cand), dtype=np.float32)

    for s in range(n_seqs):
        ev_idx = seq_event_idx[s]        # indices of events in this seq
        if len(ev_idx) == 0:
            continue
        t_ev = flat_times[ev_idx]        # times of events in seq s
        k_ev = flat_types[ev_idx]        # types of events in seq s

        n_ev = len(t_ev)
        S_src_ev = np.zeros((n_ev, D), dtype=np.float32)  # (n_ev, D)

        # Event features: only history inside [t-max_cap, t)
        if n_ev > 1:
            ev_starts = np.searchsorted(t_ev, t_ev - max_cap, side='left')
            for i in range(1, n_ev):
                st = int(ev_starts[i])
                en = i
                if st >= en:
                    continue
                dt = t_ev[i] - t_ev[st:en]
                kvals = _eval_kernel_np(dt)
                kk = k_ev[st:en]
                valid = (kk >= 0) & (kk < D)
                if valid.any():
                    S_src_ev[i] = np.bincount(
                        kk[valid], weights=kvals[valid], minlength=D
                    )[:D]

        if source_cap is not None:
            np.minimum(S_src_ev, source_cap, out=S_src_ev)

        # Project source features to candidate features
        S_rule_ev = S_src_ev @ src_mask_mat                           # (n_ev, K)
        P_event_np[ev_idx] = np.maximum(S_rule_ev - biases[None, :], 0.0)

        # Grid features: only history inside [t-max_cap, t)
        gr_idx = seq_grid_idx[s]
        if len(gr_idx) == 0:
            continue
        t_gr = grid_times[gr_idx]
        n_gr = len(t_gr)
        S_src_gr = np.zeros((n_gr, D), dtype=np.float32)            # (n_gr, D)
        gr_ends = np.searchsorted(t_ev, t_gr, side='left')          # causal: t_j < t
        gr_starts = np.searchsorted(t_ev, t_gr - max_cap, side='left')
        for g in range(n_gr):
            st = int(gr_starts[g])
            en = int(gr_ends[g])
            if st >= en:
                continue
            dt = t_gr[g] - t_ev[st:en]
            kvals = _eval_kernel_np(dt)
            kk = k_ev[st:en]
            valid = (kk >= 0) & (kk < D)
            if valid.any():
                S_src_gr[g] = np.bincount(
                    kk[valid], weights=kvals[valid], minlength=D
                )[:D]

        if source_cap is not None:
            np.minimum(S_src_gr, source_cap, out=S_src_gr)

        S_rule_gr = S_src_gr @ src_mask_mat                           # (n_gr, K)
        P_int_np[gr_idx] = np.maximum(S_rule_gr - biases[None, :], 0.0)

    # ── Convert to tensors ─────────────────────────────────────
    P_event = torch.as_tensor(P_event_np, dtype=torch.float32, device=device)
    P_int   = torch.as_tensor(P_int_np,   dtype=torch.float32, device=device)

    # ── Build event-target one-hot (N, D) ──────────────────────
    event_target_oh = torch.zeros(N, D, dtype=torch.float32, device=device)
    safe_types = np.clip(flat_types, 0, D - 1)
    event_rows = torch.arange(N, dtype=torch.long, device=device)
    event_cols = torch.as_tensor(safe_types, dtype=torch.long, device=device)
    event_target_oh[event_rows, event_cols] = 1.0

    # ── Build candidate → target one-hot (K_cand, D) ──────────
    cand_target_oh = torch.zeros(K_cand, D, dtype=torch.float32, device=device)
    for c_idx, cand in enumerate(candidates):
        cand_target_oh[c_idx, cand['target']] = 1.0

    grid_wts_t = torch.as_tensor(grid_wts, dtype=torch.float32, device=device)

    meta = {
        'event_target_oh': event_target_oh,   # (N, D)
        'cand_target_oh':  cand_target_oh,     # (K_cand, D)
        'grid_weights':    grid_wts_t,          # (M_actual,)
        'N': N,
        'D': D,
        'K': K_cand,
    }
    print(f"  P_event: {tuple(P_event.shape)},  P_int: {tuple(P_int.shape)}")
    return P_event, P_int, meta


# ================================================================== #
#  Cached NLL helper                                                   #
# ================================================================== #

def _cached_nll(
    w_exc, w_inh, P_event, P_int, event_oh, cand_oh, b_vec, grid_wts,
    fixed_target: int | None = None,
):
    """NLL computed from cached features.

    All tensors are torch tensors on the same device.
    Shapes:
      P_event: (N, K), P_int: (M, K), event_oh: (N, D), cand_oh: (K, D), b_vec: (D,)
    """
    if fixed_target is None:
        raise ValueError("_cached_nll expects fixed_target in fixed-target mode.")
    tgt = int(fixed_target)
    if tgt < 0 or tgt >= int(cand_oh.shape[1]):
        raise ValueError(
            f"fixed_target must be in [0, {int(cand_oh.shape[1]) - 1}], got {tgt}"
        )

    # Candidate contribution to the single target only.
    # cand_mask: (K,) ∈ {0,1}
    cand_mask = cand_oh[:, tgt]
    w_exc_t = w_exc * cand_mask
    w_inh_t = w_inh * cand_mask

    # Event term at target dimension only: (N,)
    E_ev_t = torch.matmul(P_event, w_exc_t)
    I_ev_t = torch.matmul(P_event, w_inh_t).clamp(max=20.0)
    lam_ev_t = (b_vec[tgt] + E_ev_t).clamp(min=1e-8) * torch.exp(-I_ev_t)
    event_mask_t = event_oh[:, tgt]
    log_ll = (torch.log(lam_ev_t.clamp(min=1e-8)) * event_mask_t).sum()

    # Integral term for target only: (M,)
    E_in_t = torch.matmul(P_int, w_exc_t)
    I_in_t = torch.matmul(P_int, w_inh_t).clamp(max=20.0)
    lam_in_t = (b_vec[tgt] + E_in_t).clamp(min=0.0) * torch.exp(-I_in_t)
    integral = (lam_in_t * grid_wts).sum()

    n_events_t = max(float(event_mask_t.sum().item()), 1.0)
    return (-log_ll + integral) / n_events_t


def _solve_nonneg_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    """Small non-negative ridge solve used for family residualisation."""
    if X.size == 0:
        return np.zeros((0,), dtype=np.float32)

    gram = X.T @ X
    gram.flat[:: gram.shape[0] + 1] += ridge
    rhs = X.T @ y
    try:
        alpha = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return np.maximum(alpha, 0.0).astype(np.float32)


def _build_family_residual_cache(
    P_event: torch.Tensor,
    P_int: torch.Tensor,
    candidates: list[dict],
    *,
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residualise pair/triplet features against their lower-order parents.

    This is used only as a Phase-5 preconditioner. Final selection/refit still
    happens on the raw cached features.
    """
    Pe_raw = P_event.detach().cpu().numpy().astype(np.float32)
    Pi_raw = P_int.detach().cpu().numpy().astype(np.float32)
    Pe_res = Pe_raw.copy()
    Pi_res = Pi_raw.copy()

    idx_by = {
        (int(c['target']), tuple(int(s) for s in c['sources'])): i
        for i, c in enumerate(candidates)
    }
    order_list = sorted(range(len(candidates)), key=lambda i: (len(candidates[i]['sources']), i))

    for idx in order_list:
        cand = candidates[idx]
        tgt = int(cand['target'])
        srcs = tuple(int(s) for s in cand['sources'])
        order = len(srcs)
        if order <= 1:
            continue

        if order == 2:
            parent_keys = [(tgt, (srcs[0],)), (tgt, (srcs[1],))]
        elif order == 3:
            a, b, c = srcs
            parent_keys = [
                (tgt, tuple(sorted(pair)))
                for pair in [(a, b), (a, c), (b, c)]
            ]
        else:
            continue

        parent_idx = [idx_by[k] for k in parent_keys if k in idx_by]
        if not parent_idx:
            continue

        X_ev = Pe_res[:, parent_idx]
        y_ev = Pe_raw[:, idx]
        if float(np.linalg.norm(y_ev)) <= 1e-12 or float(np.linalg.norm(X_ev)) <= 1e-12:
            continue

        alpha = _solve_nonneg_ridge(X_ev, y_ev, ridge=ridge)
        Pe_res[:, idx] = np.maximum(y_ev - X_ev @ alpha, 0.0)

        X_in = Pi_res[:, parent_idx]
        y_in = Pi_raw[:, idx]
        Pi_res[:, idx] = np.maximum(y_in - X_in @ alpha, 0.0)

    return (
        torch.as_tensor(Pe_res, dtype=P_event.dtype, device=P_event.device),
        torch.as_tensor(Pi_res, dtype=P_int.dtype, device=P_int.device),
    )


def _subset_cached_nll(
    subset_idx: list[int],
    w_exc_sub: torch.Tensor,
    w_inh_sub: torch.Tensor,
    P_event_full: torch.Tensor,
    P_int_full: torch.Tensor,
    cand_oh_full: torch.Tensor,
    event_oh: torch.Tensor,
    b_vec: torch.Tensor,
    grid_wts: torch.Tensor,
    *,
    fixed_target: int | None = None,
) -> torch.Tensor:
    """Evaluate cached NLL on a subset of candidate columns."""
    if len(subset_idx) == 0:
        empty_pe = P_event_full[:, :0]
        empty_pi = P_int_full[:, :0]
        empty_coh = cand_oh_full[:0]
        empty_w = torch.zeros((0,), dtype=torch.float32, device=P_event_full.device)
        return _cached_nll(
            empty_w, empty_w, empty_pe, empty_pi, event_oh, empty_coh, b_vec, grid_wts,
            fixed_target=fixed_target,
        )

    idx_t = torch.as_tensor(subset_idx, dtype=torch.long, device=P_event_full.device)
    return _cached_nll(
        w_exc_sub, w_inh_sub,
        P_event_full[:, idx_t], P_int_full[:, idx_t],
        event_oh, cand_oh_full[idx_t], b_vec, grid_wts,
        fixed_target=fixed_target,
    )


def _fit_cached_subset(
    subset_idx: list[int],
    init_we: torch.Tensor,
    init_wi: torch.Tensor,
    P_event_full: torch.Tensor,
    P_int_full: torch.Tensor,
    cand_oh_full: torch.Tensor,
    event_oh: torch.Tensor,
    b_vec: torch.Tensor,
    grid_wts: torch.Tensor,
    *,
    fixed_target: int | None = None,
    steps: int = 40,
    lr: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Short debiased refit restricted to a selected subset."""
    if len(subset_idx) == 0:
        empty = torch.zeros((0,), dtype=torch.float32, device=P_event_full.device)
        return empty, empty

    we = init_we.clone().detach().requires_grad_(True)
    wi = init_wi.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([we, wi], lr=lr)

    for _ in range(max(steps, 0)):
        opt.zero_grad()
        nll = _subset_cached_nll(
            subset_idx, we, wi,
            P_event_full, P_int_full, cand_oh_full, event_oh, b_vec, grid_wts,
            fixed_target=fixed_target,
        )
        nll.backward()
        with torch.no_grad():
            we.grad[we <= 0] = we.grad[we <= 0].clamp(max=0)
            wi.grad[wi <= 0] = wi.grad[wi <= 0].clamp(max=0)
        opt.step()
        with torch.no_grad():
            we.clamp_(min=0.0)
            wi.clamp_(min=0.0)

    return we.detach(), wi.detach()


def _candidate_sign_pref(cand: dict) -> int:
    if 'lor_sign' in cand:
        return int(cand['lor_sign'])
    we = float(cand.get('w_exc_init', 0.0))
    wi = float(cand.get('w_inh_init', 0.0))
    if wi > we:
        return -1
    if we > wi:
        return 1
    return 0


def _build_inh_family_specs(candidates: list[dict], *, fixed_target: int | None = None) -> list[dict]:
    idx_by = {
        (int(c['target']), tuple(int(s) for s in c['sources'])): i
        for i, c in enumerate(candidates)
    }
    specs = []
    for fam_idx, cand in enumerate(candidates):
        srcs = tuple(int(s) for s in cand['sources'])
        order = len(srcs)
        if order not in (2, 3):
            continue
        if fixed_target is not None and int(cand['target']) != int(fixed_target):
            continue
        if _candidate_sign_pref(cand) >= 0:
            continue
        single_idx = []
        ok = True
        for s in srcs:
            key = (int(cand['target']), (int(s),))
            if key not in idx_by:
                ok = False
                break
            single_idx.append(idx_by[key])
        if not ok:
            continue
        spec = {
            'family_idx': fam_idx,
            'target': int(cand['target']),
            'sources': srcs,
            'order': order,
            'single_idx': tuple(single_idx),
            'cand_score_raw': float(cand.get('score_raw', 0.0)),
            'support_x1': float(cand.get('support_x1', 0.0)),
        }
        specs.append(spec)
    return specs


def _build_family_proxy_and_atoms(
    P_event: torch.Tensor,
    P_int: torch.Tensor,
    single_idx: tuple[int, ...],
):
    pev = [P_event[:, idx] for idx in single_idx]
    pin = [P_int[:, idx] for idx in single_idx]
    order = len(single_idx)
    if order == 2:
        pa_ev, pb_ev = pev
        pa_in, pb_in = pin
        proxy_ev = torch.stack([pa_ev, pb_ev], dim=1)
        proxy_in = torch.stack([pa_in, pb_in], dim=1)
        ab_ev = torch.minimum(pa_ev, pb_ev)
        ab_in = torch.minimum(pa_in, pb_in)
        a_only_ev = torch.relu(pa_ev - pb_ev)
        b_only_ev = torch.relu(pb_ev - pa_ev)
        a_only_in = torch.relu(pa_in - pb_in)
        b_only_in = torch.relu(pb_in - pa_in)
        excl_ev = torch.stack([a_only_ev, b_only_ev, ab_ev], dim=1)
        excl_in = torch.stack([a_only_in, b_only_in, ab_in], dim=1)
    else:
        pa_ev, pb_ev, pc_ev = pev
        pa_in, pb_in, pc_in = pin
        proxy_ev = torch.stack([pa_ev, pb_ev, pc_ev], dim=1)
        proxy_in = torch.stack([pa_in, pb_in, pc_in], dim=1)
        abc_ev = torch.minimum(torch.minimum(pa_ev, pb_ev), pc_ev)
        ab_ev = torch.relu(torch.minimum(pa_ev, pb_ev) - pc_ev)
        ac_ev = torch.relu(torch.minimum(pa_ev, pc_ev) - pb_ev)
        bc_ev = torch.relu(torch.minimum(pb_ev, pc_ev) - pa_ev)
        a_only_ev = torch.relu(pa_ev - torch.maximum(pb_ev, pc_ev))
        b_only_ev = torch.relu(pb_ev - torch.maximum(pa_ev, pc_ev))
        c_only_ev = torch.relu(pc_ev - torch.maximum(pa_ev, pb_ev))
        abc_in = torch.minimum(torch.minimum(pa_in, pb_in), pc_in)
        ab_in = torch.relu(torch.minimum(pa_in, pb_in) - pc_in)
        ac_in = torch.relu(torch.minimum(pa_in, pc_in) - pb_in)
        bc_in = torch.relu(torch.minimum(pb_in, pc_in) - pa_in)
        a_only_in = torch.relu(pa_in - torch.maximum(pb_in, pc_in))
        b_only_in = torch.relu(pb_in - torch.maximum(pa_in, pc_in))
        c_only_in = torch.relu(pc_in - torch.maximum(pa_in, pb_in))
        excl_ev = torch.stack([a_only_ev, b_only_ev, c_only_ev, ab_ev, ac_ev, bc_ev, abc_ev], dim=1)
        excl_in = torch.stack([a_only_in, b_only_in, c_only_in, ab_in, ac_in, bc_in, abc_in], dim=1)
    return proxy_ev, proxy_in, excl_ev, excl_in


def _target_nll_from_terms(E_ev_t, I_ev_t, E_in_t, I_in_t, event_mask_t, b_tgt, grid_wts):
    lam_ev_t = (b_tgt + E_ev_t).clamp(min=1e-8) * torch.exp(I_ev_t.neg().clamp(min=-20.0, max=20.0))
    log_ll = (torch.log(lam_ev_t.clamp(min=1e-8)) * event_mask_t).sum()
    lam_in_t = (b_tgt + E_in_t).clamp(min=0.0) * torch.exp(I_in_t.neg().clamp(min=-20.0, max=20.0))
    integral = (lam_in_t * grid_wts).sum()
    n_events_t = max(float(event_mask_t.sum().item()), 1.0)
    return (-log_ll + integral) / n_events_t


def _fit_nonneg_family_weights(P_ev_atoms, P_in_atoms, E_base_ev, I_base_ev, E_base_in, I_base_in, event_mask_t, b_tgt, grid_wts, *, steps=120, lr=5e-2):
    w = torch.zeros((P_ev_atoms.shape[1],), dtype=torch.float32, device=P_ev_atoms.device, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(max(steps, 0)):
        opt.zero_grad()
        nll = _target_nll_from_terms(
            E_base_ev,
            I_base_ev + torch.matmul(P_ev_atoms, w),
            E_base_in,
            I_base_in + torch.matmul(P_in_atoms, w),
            event_mask_t,
            b_tgt,
            grid_wts,
        )
        nll.backward()
        with torch.no_grad():
            w.grad[w <= 0] = w.grad[w <= 0].clamp(max=0)
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0.0)
    return w.detach()


def _balanced_family_metric(E_base_ev, I_base_ev, E_base_in, I_base_in, add_ev, add_in, region_ev, region_in, event_mask_t, b_tgt, grid_wts):
    E_ev = E_base_ev
    I_ev = I_base_ev + add_ev
    E_in = E_base_in
    I_in = I_base_in + add_in
    lam_ev = (b_tgt + E_ev).clamp(min=1e-8) * torch.exp(I_ev.neg().clamp(min=-20.0, max=20.0))
    lam_in = (b_tgt + E_in).clamp(min=0.0) * torch.exp(I_in.neg().clamp(min=-20.0, max=20.0))
    scores = []
    for j in range(region_ev.shape[1]):
        z_ev = region_ev[:, j] * event_mask_t
        z_in = region_in[:, j] * grid_wts
        mass = float(z_ev.sum().item() + z_in.sum().item())
        if mass <= 1e-6:
            continue
        local = (-(z_ev * torch.log(lam_ev.clamp(min=1e-8))).sum() + (z_in * lam_in).sum()) / max(float(z_ev.sum().item()), 1.0)
        scores.append(local)
    if not scores:
        return torch.tensor(float('inf'), device=E_base_ev.device)
    return torch.stack(scores).mean()


def _build_maximal_inh_cliques(family_specs: list[dict]) -> list[tuple[int, ...]]:
    src_sets = sorted({tuple(int(s) for s in spec['sources']) for spec in family_specs}, key=lambda x: (len(x), x))
    maximal = []
    for srcs in src_sets:
        s = set(srcs)
        if any(s < set(other) for other in src_sets):
            continue
        maximal.append(srcs)
    return maximal


# ================================================================== #
#  Phase 5 — BCD + FISTA sparse optimisation                           #
# ================================================================== #

def _phase5_bcd_fista(
    P_event, P_int, meta, b_vec, candidates, D, device,
    *,
    lambda_l1: float = 1e-3,
    lambda_hier: float = 1e-4,
    lr: float = 1e-2,
    max_steps: int = 500,
    inner_steps: int = 20,
    tol: float = 1e-6,
    ls_beta: float = 0.5,
    ls_max_iter: int = 20,
    val_cache: dict | None = None,
    exc_prefit_steps: int = 40,
    inh_seed_scale: float = 0.1,
    fixed_target: int | None = None,
):
    """BCD outer loop with per-block FISTA inner loop and backtracking line search.

    Outer loop  (BCD):
        Alternate between the w_exc block and the w_inh block until the
        outer-loop loss change is below *tol* or *max_steps* is reached.

    Inner loop  (FISTA per block):
        Each block runs *inner_steps* FISTA steps with its own momentum
        counter reset to t=1.  This keeps the momentum from "leaking"
        between the two distinct convex sub-problems.

    Backtracking line search  (Armijo on the smooth NLL part):
        Before applying the proximal operator we shrink the trial step
        size by factor *ls_beta* until the sufficient-decrease condition
        (quadratic majorisation bound) is satisfied or *ls_max_iter*
        halvings are done.  This guarantees descent even when the NLL
        curvature is large.

    Optimise:  min_{w_exc, w_inh >= 0}  NLL(w) + λ₁‖w_exc‖₁ + λ₁‖w_inh‖₁
                                          + λ_h Ω_hier(w)

    Stage A (new):
        Excitation-only proximal prefit with w_inh fixed to zero.

    Stage B (new):
        Residual-based projected scores on train/val caches seed
        inhibition candidates (sign selection deferred to scores).

    Returns
    -------
    w_exc, w_inh : (K,), non-negative tensors.
    """
    print(f"\n[Phase 5] BCD + FISTA (λ₁={lambda_l1}, lr_init={lr}, "
          f"outer={max_steps}, inner={inner_steps})")

    # Residualised cache is used only for sparse optimisation so that
    # higher-order rules compete on their unique signal rather than on
    # lower-order shared mass.
    P_event_res, P_int_res = _build_family_residual_cache(P_event, P_int, candidates)
    raw_norm = torch.linalg.norm(P_event, dim=0).mean().item()
    res_norm = torch.linalg.norm(P_event_res, dim=0).mean().item()
    print(f"  Residualised cache: mean ||P_event|| raw={raw_norm:.4f}  resid={res_norm:.4f}")
    P_event = P_event_res
    P_int = P_int_res

    event_oh = meta['event_target_oh']   # (N, D)
    cand_oh  = meta['cand_target_oh']    # (K, D)
    grid_wts = meta['grid_weights']      # (M,)

    b = b_vec.to(device).clamp(min=1e-6)  # (D,)

    # Warm-start from Phase 3 initial weights
    w_exc = torch.tensor(
        [c['w_exc_init'] for c in candidates],
        dtype=torch.float32, device=device,
    ).clamp(min=0.0)   # (K,)
    # Inhibition starts from zero; seeded from residual projected scores.
    w_inh = torch.zeros_like(w_exc)

    # Hierarchical penalty + adaptive L1 by feature norm.
    is_multi = torch.tensor(
        [1.0 if len(c['sources']) > 1 else 0.0 for c in candidates],
        dtype=torch.float32, device=device,
    )  # (K,)
    feat_norm = torch.linalg.norm(P_event, dim=0).clamp(min=1e-6)  # (K,)
    feat_norm = feat_norm / feat_norm.mean().clamp(min=1e-6)
    feat_norm = feat_norm.clamp(min=0.25, max=4.0)
    penalty_rate = lambda_l1 * feat_norm + lambda_hier * is_multi  # (K,)
    is_high_order = (is_multi > 0.5)

    def _nll_smooth(we, wi):
        """Smooth part f(we, wi): cached NLL only."""
        return _cached_nll(
            we, wi, P_event, P_int, event_oh, cand_oh, b, grid_wts,
            fixed_target=fixed_target,
        )

    # ──────────────────────────────────────────────────────────────
    #  Full objective (NLL + L₁ regularisation)
    # ──────────────────────────────────────────────────────────────
    def _objective(we, wi):
        """f(we,wi)  +  g(we)  +  g(wi)  where g = λ‖·‖₁."""
        nll = _nll_smooth(we, wi)
        reg = (penalty_rate * (we + wi)).sum()
        return nll + reg

    def _projected_scores(we, wi, P_ev, P_in, event_oh_loc, cand_oh_loc, grid_wts_loc):
        """Projected first-order gains: positive score => expected NLL decrease."""
        we_var = we.detach().clone().requires_grad_(True)
        wi_var = wi.detach().clone().requires_grad_(True)
        nll = _cached_nll(
            we_var, wi_var, P_ev, P_in, event_oh_loc, cand_oh_loc, b, grid_wts_loc,
            fixed_target=fixed_target,
        )
        nll.backward()
        exc_score = torch.relu(-we_var.grad.detach())
        inh_score = torch.relu(-wi_var.grad.detach())
        return exc_score, inh_score

    # ──────────────────────────────────────────────────────────────
    #  Proximal gradient step with backtracking line search
    #  applied to one block  (the other block is held constant).
    #
    #  Returns the new iterate w_new and the step-size used.
    # ──────────────────────────────────────────────────────────────
    def _prox_grad_step(w_y, grad_y, f_y, wi_fixed, we_fixed, is_exc_block,
                        step_size):
        """One proximal-gradient step with Armijo backtracking.

        Quadratic majorisation sufficient-decrease condition:
            f(prox(y - α g))  ≤  f(y) - α‖g‖²/2

        Args:
            w_y        : (K,) momentum extrapolation point y
            grad_y     : (K,) gradient of smooth NLL at y
            f_y        : scalar  smooth NLL at y (for line-search check)
            wi_fixed   : (K,) the *other* block (frozen)
            we_fixed   : (K,) the *other* block (frozen, for is_exc_block=False)
            is_exc_block : True → updating w_exc; False → updating w_inh
            step_size  : initial step size α

        Returns:
            w_new      : (K,) updated iterate
            step_size  : (possibly reduced) step size for next call
        """
        alpha = step_size
        grad_sq = (grad_y * grad_y).sum().item()  # ‖g‖²

        for _ in range(ls_max_iter):
            # Gradient step
            v = w_y - alpha * grad_y
            # Proximal operator: soft-threshold + non-negativity
            w_new = (v - alpha * penalty_rate).clamp(min=0.0)  # (K,)

            # Evaluate smooth NLL at candidate
            with torch.no_grad():
                if is_exc_block:
                    f_new = _nll_smooth(w_new, wi_fixed)
                else:
                    f_new = _nll_smooth(we_fixed, w_new)

            # Sufficient-decrease (Armijo) check
            if f_new.item() <= f_y - 0.5 * alpha * grad_sq + 1e-10:
                break
            alpha *= ls_beta

        return w_new, alpha

    # ──────────────────────────────────────────────────────────────
    #  Inner FISTA loop for one block
    #
    #  Each call runs *inner_steps* FISTA iterations with momentum
    #  reset to t=1, solving the sub-problem for one block while the
    #  other block is frozen.
    # ──────────────────────────────────────────────────────────────
    def _fista_block(w_init, wi_frozen, we_frozen, is_exc_block,
                     step_size, n_inner):
        """Run FISTA for one BCD block.

        Args:
            w_init      : (K,) current iterate for this block
            wi_frozen   : (K,) frozen *other* block
            we_frozen   : (K,) same (for is_exc_block=False case)
            is_exc_block: True  → this block = w_exc, other = wi_frozen
                          False → this block = w_inh, other = we_frozen
            step_size   : initial α (adapted by line search inside)
            n_inner     : number of FISTA steps

        Returns:
            w_out       : (K,) updated block iterate
            step_size   : step size after last line search
        """
        w_prev = w_init.clone()   # w_{k-1}
        w_cur  = w_init.clone()   # w_k
        t_cur  = 1.0              # FISTA momentum counter (reset to 1)

        for _ in range(n_inner):
            # Momentum extrapolation: y = w_cur + (t_cur-1)/t_new * (w_cur - w_prev)
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_cur ** 2)) / 2.0
            momentum = (t_cur - 1.0) / t_new
            y = (w_cur + momentum * (w_cur - w_prev)).clamp(min=0.0)  # (K,)

            # Gradient of smooth NLL at extrapolation point y
            y_var = y.detach().requires_grad_(True)
            if is_exc_block:
                f_y = _nll_smooth(y_var, wi_frozen.detach())
            else:
                f_y = _nll_smooth(we_frozen.detach(), y_var)
            f_y.backward()
            grad_y = y_var.grad.detach()   # (K,)

            # Proximal step with backtracking line search
            w_new, step_size = _prox_grad_step(
                y, grad_y, f_y.item(),
                wi_frozen, we_frozen, is_exc_block,
                step_size,
            )

            w_prev = w_cur
            w_cur  = w_new
            t_cur  = t_new

        return w_cur.detach(), step_size

    # ──────────────────────────────────────────────────────────────
    #  Stage A: excitation-only prefit (w_inh = 0)
    # ──────────────────────────────────────────────────────────────
    print(f"  Stage A: excitation-only prefit ({exc_prefit_steps} steps)")
    lr_exc = lr
    w_inh.zero_()
    for _ in range(max(exc_prefit_steps, 0)):
        w_exc, lr_exc = _fista_block(
            w_exc, w_inh, w_exc,
            is_exc_block=True,
            step_size=lr_exc,
            n_inner=1,
        )

    # ──────────────────────────────────────────────────────────────
    #  Stage B: residual projected-score seeding for inhibition
    # ──────────────────────────────────────────────────────────────
    print("  Stage B: train/val projected-score seeding")
    exc_tr, inh_tr = _projected_scores(
        w_exc, torch.zeros_like(w_inh),
        P_event, P_int, event_oh, cand_oh, grid_wts,
    )
    if val_cache is not None:
        meta_v = val_cache['meta']
        exc_v, inh_v = _projected_scores(
            w_exc, torch.zeros_like(w_inh),
            val_cache['P_event'], val_cache['P_int'],
            meta_v['event_target_oh'], meta_v['cand_target_oh'], meta_v['grid_weights'],
        )
        exc_score = torch.minimum(exc_tr, exc_v)
        inh_score = torch.minimum(inh_tr, inh_v)
    else:
        exc_score, inh_score = exc_tr, inh_tr

    # Projected scores provide only a soft inhibition seed. Higher-order
    # excitation warm-start from Phase 3 is preserved; no hard sign decision
    # is made here.
    prefer_inh = is_high_order & (inh_score > exc_score) & (inh_score > 0)
    prefer_exc = is_high_order & (exc_score >= inh_score) & (exc_score > 0)
    with torch.no_grad():
        w_inh.zero_()
        if is_high_order.any():
            s = inh_score[is_high_order].clamp(min=0)
            if s.numel() > 0 and float(s.max()) > 0.0:
                s = s / s.max().clamp(min=1e-9)
                base = (
                    torch.quantile(w_exc[w_exc > 0], 0.5)
                    if (w_exc > 0).any()
                    else torch.tensor(0.05, device=device)
                )
                w_inh[is_high_order] = inh_seed_scale * base * s
    print(f"    high-order sign by score (soft seed only): inh={int(prefer_inh.sum().item())}, "
          f"exc={int(prefer_exc.sum().item())}")

    # ──────────────────────────────────────────────────────────────
    #  Outer BCD loop
    # ──────────────────────────────────────────────────────────────
    lr_exc = lr_exc   # keep adapted excitation step size from Stage A
    lr_inh = lr

    prev_obj = float('inf')

    for outer in range(max_steps):
        # ── Block 1: update w_exc (w_inh frozen) ───────────────
        w_exc, lr_exc = _fista_block(
            w_exc, w_inh, w_exc,
            is_exc_block=True,
            step_size=lr_exc,
            n_inner=inner_steps,
        )

        # ── Block 2: update w_inh (w_exc frozen) ───────────────
        w_inh, lr_inh = _fista_block(
            w_inh, w_inh, w_exc,
            is_exc_block=False,
            step_size=lr_inh,
            n_inner=inner_steps,
        )

        cur_obj = float(_objective(w_exc, w_inh).item())

        if outer % 10 == 0 or outer == max_steps - 1:
            n_alive_exc = int((w_exc > 1e-6).sum().item())
            n_alive_inh = int((w_inh > 1e-6).sum().item())
            print(f"  outer {outer:4d}  obj={cur_obj:.6f}  "
                  f"lr_exc={lr_exc:.2e}  lr_inh={lr_inh:.2e}  "
                  f"alive_exc={n_alive_exc}  alive_inh={n_alive_inh}")

        if abs(prev_obj - cur_obj) < tol and outer > 2:
            print(f"  Converged at outer step {outer}")
            break
        prev_obj = cur_obj

    n_exc = int((w_exc > 1e-6).sum().item())
    n_inh = int((w_inh > 1e-6).sum().item())
    print(f"  Final survivors: {n_exc} exc + {n_inh} inh")
    return w_exc.detach(), w_inh.detach()


# ================================================================== #
#  Phase 6 — Debiased Refit + Drop-one ΔNLL + Top-R Truncation         #
# ================================================================== #

def _phase6_refit_and_inject(
    model, candidates, w_exc, w_inh, b_vec,
    val_data, D, R, device,
    P_event, P_int, meta,
    *,
    refit_steps: int = 200,
    refit_lr: float = 1e-2,
    prune: bool = True,
    val_cache: dict | None = None,
    phase5_family_state: dict | None = None,
    protect_best_inh_single: bool = True,
    family_conflict_penalty: float = 0.35,
    fixed_target: int | None = None,
):
    """Step 6.1 debiased refit, 6.2 global incremental selection, 6.3 Top-R injection."""
    print(f"\n[Phase 6] Debiased refit + Global incremental Top-R truncation")

    # ── 6.1  Keep only surviving rules ─────────────────────────
    alive = (w_exc > 1e-6) | (w_inh > 1e-6)  # (K,)
    alive_idx = torch.where(alive)[0]         # indices into candidates
    n_alive = len(alive_idx)
    print(f"  6.1  Survivors after FISTA: {n_alive}")

    if n_alive == 0:
        print("  No rules survived — falling back to random init")
        _inject_random(model, D, R, b_vec, device, fixed_target=fixed_target)
        return

    # Sub-select cached features for alive rules only
    P_ev_sub = P_event[:, alive_idx]           # (N, n_alive)
    P_in_sub = P_int[:, alive_idx]             # (M, n_alive)
    cand_oh  = meta['cand_target_oh'][alive_idx]  # (n_alive, D)
    grid_wts = meta['grid_weights']            # (M,)
    event_oh = meta['event_target_oh']         # (N, D)
    b = b_vec.to(device).clamp(min=1e-6)       # (D,)

    # Alive weights
    we = w_exc[alive_idx].clone().requires_grad_(True)
    wi = w_inh[alive_idx].clone().requires_grad_(True)

    # ── Debiased refit (pure NLL, no L₁) ──────────────────────
    opt = torch.optim.Adam([we, wi], lr=refit_lr)
    for step in range(refit_steps):
        opt.zero_grad()
        nll = _cached_nll(
            we, wi, P_ev_sub, P_in_sub, event_oh, cand_oh, b, grid_wts,
            fixed_target=fixed_target,
        )
        nll.backward()

        # Project gradients to keep w >= 0
        with torch.no_grad():
            we.grad[we <= 0] = we.grad[we <= 0].clamp(max=0)
            wi.grad[wi <= 0] = wi.grad[wi <= 0].clamp(max=0)

        opt.step()
        with torch.no_grad():
            we.clamp_(min=0.0)
            wi.clamp_(min=0.0)

        if step % 50 == 0 or step == refit_steps - 1:
            print(f"  Refit step {step:3d}  NLL={nll.item():.6f}")

    we_final = we.detach()
    wi_final = wi.detach()

    # ── 6.2  Ranking on validation data ───────────────────────
    if prune:
        print(f"  6.2  Global incremental selection on {len(val_data)} val sequences")
    else:
        print("  6.2  Global incremental selection disabled (prune=False); using weight score.")

    # Validation cache: reuse precomputed full-candidate cache when available.
    if (val_cache is not None
            and 'P_event' in val_cache
            and val_cache['P_event'].shape[1] == len(candidates)):
        P_ev_v = val_cache['P_event'][:, alive_idx]   # (N_val, n_alive)
        P_in_v = val_cache['P_int'][:, alive_idx]     # (M_val, n_alive)
        event_oh_v = val_cache['meta']['event_target_oh']  # (N_val, D)
        grid_wts_v = val_cache['meta']['grid_weights']     # (M_val,)
    else:
        quick_cache = _build_quick_cache(
            [candidates[i] for i in alive_idx.cpu().tolist()],
            val_data, D, model, device,
        )
        P_ev_v = quick_cache['P_event']   # (N_val, n_alive)
        P_in_v = quick_cache['P_int']     # (M_val, n_alive)
        event_oh_v = quick_cache['event_target_oh']  # (N_val, D)
        grid_wts_v = quick_cache['grid_weights']     # (M_val,)

    if prune:
        with torch.no_grad():
            baseline_nll = _cached_nll(
                we_final, wi_final, P_ev_v, P_in_v, event_oh_v, cand_oh, b, grid_wts_v,
                fixed_target=fixed_target,
            )
        delta_nlls = torch.zeros(n_alive, device=device)
        with torch.no_grad():
            for r_local in range(n_alive):
                we_tmp = we_final.clone()
                wi_tmp = wi_final.clone()
                we_tmp[r_local] = 0.0
                wi_tmp[r_local] = 0.0
                nll_r = _cached_nll(
                    we_tmp, wi_tmp, P_ev_v, P_in_v, event_oh_v, cand_oh, b, grid_wts_v,
                    fixed_target=fixed_target,
                )
                delta_nlls[r_local] = nll_r - baseline_nll  # diagnostic only

        selected_local: list[int] = []
        remaining = set(range(n_alive))
        cur_we = torch.zeros((0,), dtype=torch.float32, device=device)
        cur_wi = torch.zeros((0,), dtype=torch.float32, device=device)
        cur_val_nll = _subset_cached_nll(
            [], cur_we, cur_wi,
            P_ev_v, P_in_v, cand_oh, event_oh_v, b, grid_wts_v,
            fixed_target=fixed_target,
        )
        incremental_gain: list[float] = []

        while len(selected_local) < min(R, n_alive) and remaining:
            best_loc = None
            best_gain = None
            for r_local in list(remaining):
                trial_idx = selected_local + [r_local]
                if selected_local:
                    trial_we = torch.cat([cur_we, we_final[torch.tensor([r_local], device=device)]], dim=0)
                    trial_wi = torch.cat([cur_wi, wi_final[torch.tensor([r_local], device=device)]], dim=0)
                else:
                    trial_we = we_final[torch.tensor([r_local], device=device)]
                    trial_wi = wi_final[torch.tensor([r_local], device=device)]
                trial_nll = _subset_cached_nll(
                    trial_idx, trial_we, trial_wi,
                    P_ev_v, P_in_v, cand_oh, event_oh_v, b, grid_wts_v,
                    fixed_target=fixed_target,
                )
                gain = float((cur_val_nll - trial_nll).item())
                if best_loc is None or gain > best_gain:
                    best_loc = r_local
                    best_gain = gain

            if best_loc is None:
                break

            selected_local.append(int(best_loc))
            remaining.remove(int(best_loc))
            init_we = we_final[torch.as_tensor(selected_local, dtype=torch.long, device=device)]
            init_wi = wi_final[torch.as_tensor(selected_local, dtype=torch.long, device=device)]
            cur_we, cur_wi = _fit_cached_subset(
                selected_local, init_we, init_wi,
                P_ev_sub, P_in_sub, cand_oh, event_oh, b, grid_wts,
                fixed_target=fixed_target, steps=40, lr=refit_lr,
            )
            cur_val_nll = _subset_cached_nll(
                selected_local, cur_we, cur_wi,
                P_ev_v, P_in_v, cand_oh, event_oh_v, b, grid_wts_v,
                fixed_target=fixed_target,
            )
            incremental_gain.append(float(best_gain))
    else:
        delta_nlls = we_final + wi_final
        selected_local = torch.argsort(-delta_nlls).cpu().tolist()[: min(R, n_alive)]
        cur_we = we_final[torch.as_tensor(selected_local, dtype=torch.long, device=device)]
        cur_wi = wi_final[torch.as_tensor(selected_local, dtype=torch.long, device=device)]
        incremental_gain = [float(delta_nlls[i].item()) for i in selected_local]

    selected_items = []
    for rank, loc in enumerate(selected_local):
        glob = int(alive_idx[loc].item())
        selected_items.append({
            'kind': 'raw',
            'local': int(loc),
            'global': glob,
            'cand': candidates[glob],
            'w_exc': float(cur_we[rank].item()),
            'w_inh': float(cur_wi[rank].item()),
            'gain': float(incremental_gain[rank]) if rank < len(incremental_gain) else 0.0,
            'delta_drop': float(delta_nlls[loc].item()),
        })

    protected_pos = set(range(min(3, len(selected_items))))
    best_inh_pos = None
    best_inh_strength = None
    best_inh_fallback = None
    selected_global_to_pos = {int(item['global']): pos for pos, item in enumerate(selected_items)}
    for loc in range(n_alive):
        glob = int(alive_idx[loc].item())
        cand = candidates[glob]
        if len(cand['sources']) != 1:
            continue
        we_loc = float(we_final[loc].item())
        wi_loc = float(wi_final[loc].item())
        if wi_loc <= we_loc:
            continue
        strength = wi_loc - we_loc
        if best_inh_strength is None or strength > best_inh_strength:
            best_inh_strength = strength
            if glob in selected_global_to_pos:
                best_inh_pos = selected_global_to_pos[glob]
                best_inh_fallback = None
            else:
                best_inh_pos = None
                best_inh_fallback = {
                    'kind': 'raw',
                    'local': int(loc),
                    'global': glob,
                    'cand': cand,
                    'w_exc': we_loc,
                    'w_inh': wi_loc,
                    'gain': 0.0,
                    'delta_drop': float(delta_nlls[loc].item()),
                }
    if protect_best_inh_single and best_inh_pos is not None:
        protected_pos.add(best_inh_pos)

    protected_items = [selected_items[p] for p in sorted(protected_pos)]
    if protect_best_inh_single and best_inh_fallback is not None:
        protected_items.append(best_inh_fallback)
    if phase5_family_state is not None:
        zeroed_global = set(int(i) for i in phase5_family_state.get('zeroed_global_idx', ()))
        if zeroed_global:
            protected_items = [
                item for item in protected_items
                if not (
                    item['kind'] == 'raw'
                    and item['global'] in zeroed_global
                    and item['w_inh'] > item['w_exc']
                )
            ]
    num_family_slots = max(0, R - len(protected_items))
    chosen_family_items = []
    feasible_families = 0
    family_items = []

    if phase5_family_state is not None:
        preselected = list(phase5_family_state.get('family_items', []))
        if preselected:
            used = set()
            for item in preselected:
                srcs = set(int(s) for s in item['sources'])
                if used & srcs:
                    continue
                used |= srcs
                chosen_family_items.append(item)
            num_family_slots = max(0, R - len(protected_items) - len(chosen_family_items))

    if fixed_target is not None and num_family_slots > 0:
        family_specs = _build_inh_family_specs(candidates, fixed_target=fixed_target)
        if family_specs:
            score_by_order = {}
            support_by_order = {}
            for spec in family_specs:
                score_by_order.setdefault(int(spec['order']), []).append(float(spec.get('cand_score_raw', 0.0)))
                support_by_order.setdefault(int(spec['order']), []).append(float(np.log1p(spec.get('support_x1', 0.0))))
            order_score_stats = {}
            order_support_stats = {}
            for order, vals in score_by_order.items():
                arr = np.asarray(vals, dtype=np.float64)
                med = float(np.median(arr)) if len(arr) else 0.0
                mad = float(np.median(np.abs(arr - med))) if len(arr) else 0.0
                order_score_stats[int(order)] = (med, max(1.4826 * mad, 1e-6))
            for order, vals in support_by_order.items():
                arr = np.asarray(vals, dtype=np.float64)
                med = float(np.median(arr)) if len(arr) else 0.0
                mad = float(np.median(np.abs(arr - med))) if len(arr) else 0.0
                order_support_stats[int(order)] = (med, max(1.4826 * mad, 1e-6))
            protected_inh_strength = {}
            for item in protected_items:
                if item['kind'] == 'raw' and len(item['cand']['sources']) == 1 and item['w_inh'] > item['w_exc']:
                    s = int(item['cand']['sources'][0])
                    protected_inh_strength[s] = float(item['w_inh'] - item['w_exc'])
            event_mask_tr = event_oh[:, int(fixed_target)]
            b_tgt = b[int(fixed_target)]

            if (val_cache is not None
                    and 'P_event' in val_cache
                    and val_cache['P_event'].shape[1] == len(candidates)):
                P_ev_v_full = val_cache['P_event']
                P_in_v_full = val_cache['P_int']
                event_mask_v = val_cache['meta']['event_target_oh'][:, int(fixed_target)]
                grid_wts_v = val_cache['meta']['grid_weights']
            else:
                quick_full = _build_quick_cache(candidates, val_data, D, model, device)
                P_ev_v_full = quick_full['P_event']
                P_in_v_full = quick_full['P_int']
                event_mask_v = quick_full['event_target_oh'][:, int(fixed_target)]
                grid_wts_v = quick_full['grid_weights']

            def _add_family_base(item, E_base_ev_tr, I_base_ev_tr, E_base_in_tr, I_base_in_tr,
                                 E_base_ev_v, I_base_ev_v, E_base_in_v, I_base_in_v):
                srcs = tuple(int(s) for s in item['sources'])
                single_idx_local = tuple(
                    idx for idx in (
                        next(
                            i for i, c in enumerate(candidates)
                            if int(c['target']) == int(item['target'])
                            and tuple(int(x) for x in c['sources']) == (int(s),)
                        )
                        for s in srcs
                    )
                )
                proxy_ev_tr, proxy_in_tr, excl_ev_tr, excl_in_tr = _build_family_proxy_and_atoms(
                    P_event, P_int, single_idx_local
                )
                proxy_ev_v, proxy_in_v, excl_ev_v, excl_in_v = _build_family_proxy_and_atoms(
                    P_ev_v_full, P_in_v_full, single_idx_local
                )
                if int(item['mode']) == 0:
                    add_ev_tr = torch.matmul(proxy_ev_tr, item['weights'])
                    add_in_tr = torch.matmul(proxy_in_tr, item['weights'])
                    add_ev_v = torch.matmul(proxy_ev_v, item['weights'])
                    add_in_v = torch.matmul(proxy_in_v, item['weights'])
                else:
                    add_ev_tr = torch.matmul(excl_ev_tr, item['weights'])
                    add_in_tr = torch.matmul(excl_in_tr, item['weights'])
                    add_ev_v = torch.matmul(excl_ev_v, item['weights'])
                    add_in_v = torch.matmul(excl_in_v, item['weights'])
                return (
                    E_base_ev_tr,
                    I_base_ev_tr + add_ev_tr,
                    E_base_in_tr,
                    I_base_in_tr + add_in_tr,
                    E_base_ev_v,
                    I_base_ev_v + add_ev_v,
                    E_base_in_v,
                    I_base_in_v + add_in_v,
                )

            used_src = set()
            for item in chosen_family_items:
                used_src |= set(int(s) for s in item['sources'])

            while len(chosen_family_items) < num_family_slots:
                family_items = []
                E_base_ev_tr = torch.zeros_like(event_mask_tr)
                I_base_ev_tr = torch.zeros_like(event_mask_tr)
                E_base_in_tr = torch.zeros_like(grid_wts)
                I_base_in_tr = torch.zeros_like(grid_wts)
                E_base_ev_v = torch.zeros_like(event_mask_v)
                I_base_ev_v = torch.zeros_like(event_mask_v)
                E_base_in_v = torch.zeros_like(grid_wts_v)
                I_base_in_v = torch.zeros_like(grid_wts_v)

                for item in protected_items:
                    loc = item['local']
                    E_base_ev_tr = E_base_ev_tr + P_ev_sub[:, loc] * item['w_exc']
                    I_base_ev_tr = I_base_ev_tr + P_ev_sub[:, loc] * item['w_inh']
                    E_base_in_tr = E_base_in_tr + P_in_sub[:, loc] * item['w_exc']
                    I_base_in_tr = I_base_in_tr + P_in_sub[:, loc] * item['w_inh']
                    E_base_ev_v = E_base_ev_v + P_ev_v[:, loc] * item['w_exc']
                    I_base_ev_v = I_base_ev_v + P_ev_v[:, loc] * item['w_inh']
                    E_base_in_v = E_base_in_v + P_in_v[:, loc] * item['w_exc']
                    I_base_in_v = I_base_in_v + P_in_v[:, loc] * item['w_inh']

                for item in chosen_family_items:
                    (
                        E_base_ev_tr, I_base_ev_tr, E_base_in_tr, I_base_in_tr,
                        E_base_ev_v, I_base_ev_v, E_base_in_v, I_base_in_v,
                    ) = _add_family_base(
                        item,
                        E_base_ev_tr, I_base_ev_tr, E_base_in_tr, I_base_in_tr,
                        E_base_ev_v, I_base_ev_v, E_base_in_v, I_base_in_v,
                    )

                for spec in family_specs:
                    srcs_set = set(int(s) for s in spec['sources'])
                    if used_src & srcs_set:
                        continue
                    proxy_ev_tr, proxy_in_tr, excl_ev_tr, excl_in_tr = _build_family_proxy_and_atoms(
                        P_event, P_int, spec['single_idx']
                    )
                    proxy_ev_v, proxy_in_v, excl_ev_v, excl_in_v = _build_family_proxy_and_atoms(
                        P_ev_v_full, P_in_v_full, spec['single_idx']
                    )
                    w_proxy = _fit_nonneg_family_weights(
                        proxy_ev_tr, proxy_in_tr,
                        E_base_ev_tr, I_base_ev_tr, E_base_in_tr, I_base_in_tr,
                        event_mask_tr, b_tgt, grid_wts,
                    )
                    w_excl = _fit_nonneg_family_weights(
                        excl_ev_tr, excl_in_tr,
                        E_base_ev_tr, I_base_ev_tr, E_base_in_tr, I_base_in_tr,
                        event_mask_tr, b_tgt, grid_wts,
                    )
                    base_metric_tr = _balanced_family_metric(
                        E_base_ev_tr, I_base_ev_tr, E_base_in_tr, I_base_in_tr,
                        torch.zeros_like(event_mask_tr), torch.zeros_like(grid_wts),
                        excl_ev_tr, excl_in_tr, event_mask_tr, b_tgt, grid_wts,
                    )
                    proxy_metric_tr = _balanced_family_metric(
                        E_base_ev_tr, I_base_ev_tr, E_base_in_tr, I_base_in_tr,
                        torch.matmul(proxy_ev_tr, w_proxy), torch.matmul(proxy_in_tr, w_proxy),
                        excl_ev_tr, excl_in_tr, event_mask_tr, b_tgt, grid_wts,
                    )
                    excl_metric_tr = _balanced_family_metric(
                        E_base_ev_tr, I_base_ev_tr, E_base_in_tr, I_base_in_tr,
                        torch.matmul(excl_ev_tr, w_excl), torch.matmul(excl_in_tr, w_excl),
                        excl_ev_tr, excl_in_tr, event_mask_tr, b_tgt, grid_wts,
                    )
                    base_metric = _balanced_family_metric(
                        E_base_ev_v, I_base_ev_v, E_base_in_v, I_base_in_v,
                        torch.zeros_like(event_mask_v), torch.zeros_like(grid_wts_v),
                        excl_ev_v, excl_in_v, event_mask_v, b_tgt, grid_wts_v,
                    )
                    proxy_metric = _balanced_family_metric(
                        E_base_ev_v, I_base_ev_v, E_base_in_v, I_base_in_v,
                        torch.matmul(proxy_ev_v, w_proxy), torch.matmul(proxy_in_v, w_proxy),
                        excl_ev_v, excl_in_v, event_mask_v, b_tgt, grid_wts_v,
                    )
                    excl_metric = _balanced_family_metric(
                        E_base_ev_v, I_base_ev_v, E_base_in_v, I_base_in_v,
                        torch.matmul(excl_ev_v, w_excl), torch.matmul(excl_in_v, w_excl),
                        excl_ev_v, excl_in_v, event_mask_v, b_tgt, grid_wts_v,
                    )
                    proxy_impr = min(
                        float((base_metric_tr - proxy_metric_tr).item()),
                        float((base_metric - proxy_metric).item()),
                    )
                    excl_impr = min(
                        float((base_metric_tr - excl_metric_tr).item()),
                        float((base_metric - excl_metric).item()),
                    )

                    best_mode = None
                    best_weights = None
                    best_improvement = 0.0
                    if proxy_impr > best_improvement + 1e-6:
                        best_mode = 0
                        best_weights = w_proxy
                        best_improvement = proxy_impr
                    if excl_impr > best_improvement + 1e-6:
                        best_mode = 1
                        best_weights = w_excl
                        best_improvement = excl_impr
                    if best_mode is None or best_improvement <= 1e-6:
                        continue
                    feasible_families += 1
                    score_med, score_scale = order_score_stats.get(int(spec['order']), (0.0, 1.0))
                    support_med, support_scale = order_support_stats.get(int(spec['order']), (0.0, 1.0))
                    z_score = (float(spec.get('cand_score_raw', 0.0)) - score_med) / score_scale
                    z_support = (float(np.log1p(spec.get('support_x1', 0.0))) - support_med) / support_scale
                    conflict = sum(float(protected_inh_strength.get(int(s), 0.0)) for s in spec['sources'])
                    mode_penalty = 0.03 if best_mode == 0 else 0.0
                    family_items.append({
                        'kind': 'family',
                        'sources': spec['sources'],
                        'target': spec['target'],
                        'order': spec['order'],
                        'mode': best_mode,
                        'weights': best_weights,
                        'improvement': float(best_improvement),
                        'score': float(best_improvement) + 0.02 * z_score + 0.01 * z_support - float(family_conflict_penalty) * conflict - mode_penalty,
                    })

                if not family_items:
                    break

                for i in range(len(family_items)):
                    src_i = set(int(s) for s in family_items[i]['sources'])
                    pen = 0.0
                    for j in range(len(family_items)):
                        if i == j:
                            continue
                        src_j = set(int(s) for s in family_items[j]['sources'])
                        if src_i < src_j and family_items[j]['order'] > family_items[i]['order']:
                            sup_score = float(family_items[j]['score'])
                            cur_score = float(family_items[i]['score'])
                            if sup_score > 0.0 and sup_score >= 0.8 * cur_score:
                                pen = max(pen, 0.75 * sup_score)
                    family_items[i]['score'] -= pen

                best_item = max(family_items, key=lambda x: float(x['score']))
                if float(best_item['score']) <= 0.0:
                    break
                chosen_family_items.append(best_item)
                used_src |= set(int(s) for s in best_item['sources'])

    used_family_sources = set()
    for item in chosen_family_items:
        used_family_sources |= set(int(s) for s in item['sources'])
    backfill_items = []
    if len(protected_items) + len(chosen_family_items) < R:
        for item in selected_items:
            if item in protected_items:
                continue
            if len(set(int(s) for s in item['cand']['sources']) & used_family_sources) > 0:
                continue
            backfill_items.append(item)
            if len(protected_items) + len(chosen_family_items) + len(backfill_items) >= R:
                break

    final_items = protected_items + chosen_family_items + backfill_items
    top_R = min(R, len(final_items))

    print(f"  6.3  Top-{R} truncation (from {n_alive} alive)")
    print(f"  Family hypotheses feasible={feasible_families} active={len(chosen_family_items)}")
    print(f"  Incremental ranking (top {top_R}):")
    for rank, item in enumerate(final_items[:top_R]):
        if item['kind'] == 'raw':
            src_str = ','.join(str(s) for s in item['cand']['sources'])
            print(f"    #{rank}: {{{src_str}}}→{item['cand']['target']}  "
                  f"gain={item['gain']:+.4f}  "
                  f"Δdrop={item['delta_drop']:+.4f}  "
                  f"w_exc={item['w_exc']:.4f}  w_inh={item['w_inh']:.4f}")
        else:
            src_str = ','.join(str(s) for s in item['sources'])
            print(f"    #{rank}: FAMILY-{('EXCL' if item['mode'] == 1 else 'PROXY')}{{{src_str}}}→{item['target']}  "
                  f"impr={item['improvement']:+.4f}  "
                  f"w={tuple(round(float(x),4) for x in item['weights'].tolist())}")

    # Inject final units into model
    with torch.no_grad():
        model.b0.data = _sp_inv_t(b_vec.to(device))
        if hasattr(model, 'clear_family_hypotheses'):
            model.clear_family_hypotheses()

        for slot in range(R):
            if slot < top_R:
                item = final_items[slot]
                if item['kind'] == 'raw':
                    _inject_single_rule(
                        model, slot, item['cand'],
                        float(item['w_exc']), float(item['w_inh']),
                        D, device,
                    )
                else:
                    _inject_family_hypothesis_rule(model, slot, item, D, device)
            else:
                _inject_noise_rule(model, slot, D, device, fixed_target=fixed_target)

        # Freeze bias
        model.rule_bias_raw.requires_grad = False
        if fixed_target is not None:
            tgt = int(fixed_target)
            model.rule_target_logits.data.fill_(-3.0)
            model.rule_target_logits.data[:, tgt] = 3.0
            model.rule_target_logits.requires_grad = False
        else:
            model.rule_target_logits.requires_grad = True

        # Kernel: flat init at 0.5 for all rules
        model.kernel_height_raw.data.fill_(_sp_inv(0.5))

    bk = F.softplus(model.b0).data.cpu().numpy()
    bias = F.softplus(model.rule_bias_raw).data.cpu().numpy()
    print(f"  b_k  (softplus):  {np.round(bk, 4)}")
    print(f"  bias (frozen):    {np.round(bias, 3)}")


# ================================================================== #
#  Parameter injection helpers                                         #
# ================================================================== #

def _inject_single_rule(model, slot, cand, w_exc_val, w_inh_val, D, device):
    """Write one candidate rule into model slot."""
    sources = cand['sources']
    tgt = cand['target']

    # H: source mask
    model.theta.data[:, slot] = -3.0
    for src in sources:
        model.theta.data[src, slot] = 3.0

    # Head: target
    model.rule_target_logits.data[slot, :] = -3.0
    model.rule_target_logits.data[slot, tgt] = 3.0

    # Weights & sign
    is_exc = w_exc_val > w_inh_val
    if is_exc:
        model.sign_logits.data[slot] = 3.0
        model.w_exc_raw.data[slot] = _sp_inv(max(w_exc_val, 1e-4))
        model.w_inh_raw.data[slot] = _sp_inv(max(w_inh_val, 1e-6)) if w_inh_val > 1e-6 else -3.0
    else:
        model.sign_logits.data[slot] = -3.0
        model.w_inh_raw.data[slot] = _sp_inv(max(w_inh_val, 1e-4))
        model.w_exc_raw.data[slot] = _sp_inv(max(w_exc_val, 1e-6)) if w_exc_val > 1e-6 else -3.0

    # Bias (A-option): synchronized with Phase-4 candidate feature cache.
    # With flat kernel height h_unit=0.5:
    #   bias_r = (N_sources - 0.5) * h_unit
    N_r = len(sources)
    h_unit = 0.5
    bv = max((N_r - 0.5) * h_unit, 0.0)
    model.rule_bias_raw.data[slot] = _sp_inv(bv) if bv > 0 else -5.0


def _inject_family_hypothesis_rule(model, slot, item, D, device):
    model.theta.data[:, slot] = -3.0
    model.rule_target_logits.data[slot, :] = -3.0
    model.rule_target_logits.data[slot, int(item['target'])] = 3.0
    model.sign_logits.data[slot] = -3.0
    model.w_exc_raw.data[slot] = -3.0
    model.w_inh_raw.data[slot] = -3.0
    model.rule_bias_raw.data[slot] = -5.0
    if hasattr(model, 'register_family_hypothesis'):
        model.register_family_hypothesis(
            slot=slot,
            target=int(item['target']),
            sources=tuple(int(s) for s in item['sources']),
            mode=int(item['mode']),
            weights=item['weights'],
        )


def _inject_noise_rule(model, slot, D, device, fixed_target: int | None = None):
    """Fill empty slot with weak random noise."""
    model.theta.data[:, slot] = torch.randn(D, device=device) * 0.3 - 1.0
    if fixed_target is None:
        model.rule_target_logits.data[slot] = torch.randn(D, device=device) * 0.3
    else:
        tgt = int(fixed_target)
        model.rule_target_logits.data[slot] = -3.0
        model.rule_target_logits.data[slot, tgt] = 3.0
    model.sign_logits.data[slot] = 0.0
    model.w_exc_raw.data[slot] = -3.0
    model.w_inh_raw.data[slot] = -3.0
    # Bias for noise rule
    model.rule_bias_raw.data[slot] = -5.0


def _inject_random(model, D, R, b_vec, device, fixed_target: int | None = None):
    """Fallback: random initialisation if no rules survived."""
    with torch.no_grad():
        if hasattr(model, 'clear_family_hypotheses'):
            model.clear_family_hypotheses()
        model.b0.data = _sp_inv_t(b_vec.to(device))
        for r in range(R):
            _inject_noise_rule(model, r, D, device, fixed_target=fixed_target)
        model.rule_bias_raw.requires_grad = False
        if fixed_target is not None:
            tgt = int(fixed_target)
            model.rule_target_logits.data.fill_(-3.0)
            model.rule_target_logits.data[:, tgt] = 3.0
            model.rule_target_logits.requires_grad = False
        else:
            model.rule_target_logits.requires_grad = True
        model.kernel_height_raw.data.fill_(_sp_inv(0.5))


# ================================================================== #
#  Quick validation-set cache builder                                  #
# ================================================================== #

def _build_quick_cache(alive_candidates, data_list, D, model, device):
    """Validation cache wrapper reusing the same logic as Phase 4."""
    P_event, P_int, meta = _phase4_cache_features(
        alive_candidates, data_list, D, model, device, int_grid_mult=2,
    )
    return {
        'P_event': P_event,
        'P_int': P_int,
        'event_target_oh': meta['event_target_oh'],
        'grid_weights': meta['grid_weights'],
    }


# ================================================================== #
#  NLL evaluation helpers  (kept for backwards compat / diagnostics)    #
# ================================================================== #

def _make_eval_batches(data_list, D, device, max_seqs=200, batch_size=32):
    """Create evaluation batches from raw data_list (no DataLoader)."""
    pad_id = D
    subset = data_list[:max_seqs]
    batches = []

    for start in range(0, len(subset), batch_size):
        chunk = subset[start:start + batch_size]
        max_len = max(len(x['time']) for x in chunk)
        B = len(chunk)
        ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        tds = torch.zeros(B, max_len, dtype=torch.float32)
        for i, item in enumerate(chunk):
            t = torch.as_tensor(item['time'], dtype=torch.float32)
            e = torch.as_tensor(item['event'], dtype=torch.long)
            n = len(t)
            ids[i, :n] = e
            if n > 0:
                tds[i, :n] = torch.cat([t[:1], t[1:] - t[:-1]])
        batches.append({
            'input_ids': ids.to(device),
            'time_diffs': tds.to(device),
            'attention_mask': (ids != pad_id).float().to(device),
        })
    return batches


def _eval_nll(model, batches, device):
    """Mean NLL over pre-built batches."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            mo = model(tau=0.5)
            ld = model.compute_loss(batch, mo)
            total += ld['nll_loss'].item()
            n += 1
    return total / max(n, 1)


# ================================================================== #
#  Numeric helpers                                                     #
# ================================================================== #

def _sp_inv(x: float) -> float:
    """softplus⁻¹(x) — numerically stable."""
    if x > 20:
        return x
    return float(np.log(np.expm1(x) + 1e-9))


def _sp_inv_t(x: torch.Tensor) -> torch.Tensor:
    """Batched softplus⁻¹ for a tensor."""
    x = x.clamp(min=1e-6)
    return torch.where(
        x > 20.0,
        x,
        torch.log(torch.expm1(x) + 1e-9),
    )


def _banner(msg: str):
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60)
