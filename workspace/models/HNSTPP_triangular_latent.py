"""
HNSTPP with Triangular Kernel and Closed-Form Integration.

This implementation uses ordered triangular kernels (3-piece linear basis with positions a, b, c)
instead of Gaussian kernels. The triangular kernel enables exact closed-form integration,
eliminating Monte Carlo approximation errors.

Triangular Kernel Definition:
    K(dt; a, b, c) = {
        (dt - a) / (b - a)    if a <= dt < b   (rising slope)
        (c - dt) / (c - b)    if b <= dt <= c  (falling slope)
        0                      otherwise
    }
    where: a < b < c (enforced via cumulative positive deltas)
"""

from .base import BaseTPP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class PolynomialIntegrator:
    """Helper to compute exact integral of product of N linear functions (m*t + c)."""
    
    @staticmethod
    def get_poly_coeffs_variable(slopes, intercepts):
        """
        Computes coeffs for product of N linear terms: Prod(m_i * t + c_i)
        Works with variable number of terms.
        
        Input: 
            slopes: (..., N) slopes
            intercepts: (..., N) intercepts
        Output: 
            coeffs: (..., N+1) -> [a_N, a_{N-1}, ..., a_1, a_0]
        """
        shape = slopes.shape[:-1]
        N = slopes.shape[-1]
        device = slopes.device
        
        # Start with constant 1
        current_coeffs = torch.ones(*shape, 1, device=device)
        
        for i in range(N):
            m = slopes[..., i:i+1]  # (..., 1)
            c = intercepts[..., i:i+1]  # (..., 1)
            
            prev_len = current_coeffs.shape[-1]
            new_coeffs = torch.zeros(*shape, prev_len + 1, device=device)
            
            # (a_n t^n + ... + a_0) * (mt + c) = m*(...) shifted + c*(...)
            new_coeffs[..., :-1] = new_coeffs[..., :-1] + current_coeffs * m
            new_coeffs[..., 1:] = new_coeffs[..., 1:] + current_coeffs * c
            
            current_coeffs = new_coeffs
            
        return current_coeffs

    @staticmethod
    def integrate_poly(coeffs, t_start, t_end):
        """
        Exact polynomial integration using Power Rule.
        
        Input: 
            coeffs: (..., D+1) coefficients [a_D, a_{D-1}, ..., a_1, a_0]
            t_start: (...) start times
            t_end: (...) end times
        Output: 
            integral: (...) integral values
        """
        deg_plus_1 = coeffs.shape[-1]
        degree = deg_plus_1 - 1
        device = coeffs.device
        
        if degree == 0:
            # Constant function
            return coeffs[..., 0] * (t_end - t_start)
        
        # Powers: [D+1, D, ..., 2, 1] for integration
        powers = torch.arange(degree + 1, 0, -1, device=device, dtype=coeffs.dtype)
        
        # Integral coefficients: a_i / (power)
        int_coeffs = coeffs / powers  # (..., D+1)
        
        def eval_primitive(t):
            # t: (...)
            t_exp = t.unsqueeze(-1)  # (..., 1)
            # Compute t^{D+1}, t^D, ..., t^1 - vectorized
            t_powers = t_exp ** powers.unsqueeze(0)  # (..., D+1)
            return (int_coeffs * t_powers).sum(dim=-1)
            
        result = eval_primitive(t_end) - eval_primitive(t_start)
        return result

    @staticmethod
    def integrate_exact(coeffs, t_start, t_end):
        """Legacy interface."""
        return F.relu(PolynomialIntegrator.integrate_poly(coeffs, t_start, t_end))


class HNSTPPTriangularLatent(BaseTPP):
    """Hypergraph Neuro-Symbolic TPP with Triangular Kernels and Latent Nodes.
    
    Key features:
    1. Triangular kernel (a, b, c) instead of Gaussian (mu, sigma)
    2. Closed-form integral computation
    3. Rule weight and interaction weight learned directly
    4. Latent nodes for interaction space
    5. Rule length correction factor C^{num_events - 1} where C = 1/num_components
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_types = config['num_types']
        self.num_rules = config['num_rules']
        self.emb_dim = config['embed_dim']
        self.tau = config['start_tau']
        self.pad_token_id = config['pad_token_id']
        self.num_latent_nodes = config.get('num_latent_nodes', 0)
        
        self.total_nodes = self.num_types + self.num_latent_nodes

        # Rule Structure Learning
        init_std = 1.0 / np.sqrt(self.total_nodes)
        self.theta = nn.Parameter(torch.randn(self.total_nodes, self.num_rules) * init_std)
        with torch.no_grad():
            self.theta[self.num_types:, :] -= 2.0
        
        self.rule_target_logits = nn.Parameter(torch.randn(self.num_rules, self.num_types) * 0.5)
        self.rule_weights = nn.Parameter(torch.ones(self.num_rules) * 1.0e-2)

        # Triangular Kernel Parameters
        # Mixture of 3 symmetric triangular kernels per (source_type, rule) pair
        self.num_components = 3
        
        self.raw_deltas = nn.Parameter(torch.ones(self.num_types, self.num_rules, self.num_components) * 1.5)
        self.raw_widths = nn.Parameter(torch.ones(self.num_types, self.num_rules, self.num_components) * 3.0)
        self.raw_logits = nn.Parameter(torch.ones(self.num_types, self.num_rules, self.num_components) * 0.5)
        self.min_width = 0.01
        self.eps = 1e-8

        # Interaction Learning
        self.interaction_weights = nn.Parameter(torch.randn(self.num_rules, self.num_rules) * 0.01)
        with torch.no_grad():
            self.interaction_weights.fill_diagonal_(0.0)
        
        self.raw_deltas_inter = nn.Parameter(torch.ones(self.num_rules, self.num_rules, self.num_components) * 1.5)
        self.raw_widths_inter = nn.Parameter(torch.ones(self.num_rules, self.num_rules, self.num_components) * 3.0)
        self.raw_logits_inter = nn.Parameter(torch.ones(self.num_rules, self.num_rules, self.num_components) * 0.5)
        
        # Base Intensity
        self.b0 = nn.Parameter(torch.full((self.num_types,), -6.0))

        # Context Encoder
        self.history_encoder = nn.GRU(self.emb_dim, self.emb_dim, batch_first=True)
        self.context_gate = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.num_rules),
            nn.Sigmoid()
        )

    # Parameter Accessors
    
    def get_triangular_params(self):
        """Get peaks, widths, and weights for symmetric triangular mixture.
        
        Returns:
            peaks: (num_types, num_rules, 3) peak positions (cumulative from deltas)
            widths: (num_types, num_rules, 3) half-widths (symmetric)
            weights: (num_types, num_rules, 3) mixture weights (sum to 1 via softmax)
        """
        deltas = F.softplus(self.raw_deltas) + self.eps
        widths = F.softplus(self.raw_widths) + self.min_width
        peaks = torch.cumsum(deltas, dim=-1)
        weights = F.softmax(self.raw_logits, dim=-1)
        
        return peaks, widths, weights
    
    def get_interaction_triangular_params(self):
        """Get peaks, widths, and weights for interaction kernels.
        
        Returns:
            peaks: (num_rules, num_rules, 3)
            widths: (num_rules, num_rules, 3)
            weights: (num_rules, num_rules, 3)
        """
        deltas = F.softplus(self.raw_deltas_inter) + self.eps
        widths = F.softplus(self.raw_widths_inter) + self.min_width
        peaks = torch.cumsum(deltas, dim=-1)
        weights = F.softmax(self.raw_logits_inter, dim=-1)
        
        return peaks, widths, weights
    
    def get_interaction_mask_from_dual(self, H_hard, H_soft, Head_hard=None, Head_soft=None):
        """Compute interaction adjacency mask using Target-Source connection.
        
        Two rules can interact if:
        - Rule A's target type is one of Rule B's source types, OR
        - Both rules share source nodes (including latent nodes)
        
        Args:
            H_soft: (total_nodes, M) soft incidence matrix (source nodes per rule, includes latent)
            H_hard: (total_nodes, M) hard incidence matrix
            Head_hard: (M, K) rule target mapping (event types only)
            Head_soft: (M, K) soft rule target mapping
            
        Returns:
            mask: (M, M) interaction adjacency
        """
        M = H_hard.shape[1]
        device = H_hard.device
        
        # Method 1: Source node sharing (including latent nodes)
        # A_dual[i,j] > 0 if rules i and j share any source node
        A_source_soft = torch.matmul(H_soft.t(), H_soft)  # (M, M)
        A_source_hard = torch.matmul(H_hard.t(), H_hard)
        
        # Method 2: Target-Source connection (event types only)
        # A_target[i,j] > 0 if rule i's target is in rule j's sources
        if Head_hard is not None and Head_soft is not None:
            # Head: (M, K), H_event: (K, M) - event types만
            K = Head_hard.shape[1]
            H_soft_event = H_soft[:K, :]  # (K, M)
            H_hard_event = H_hard[:K, :]  # (K, M)
            
            # Target-Source: Head @ H_event gives (M, M)
            A_target_soft = torch.matmul(Head_soft, H_soft_event)  # (M, M)
            A_target_hard = torch.matmul(Head_hard, H_hard_event)
        else:
            A_target_soft = torch.zeros(M, M, device=device)
            A_target_hard = torch.zeros(M, M, device=device)
        
        # Combined adjacency: union of both connection types
        A_combined_soft = A_source_soft + A_target_soft
        A_combined_hard = A_source_hard + A_target_hard
        
        # Threshold for hard mask
        mask_hard = (A_combined_hard > 0.5).float()
        mask_hard.fill_diagonal_(0.0)
        
        # STE: forward uses hard, backward uses soft
        mask = (mask_hard - A_combined_soft).detach() + A_combined_soft
        
        return mask
    
    def symmetric_triangular_kernel(self, dt, peak, width, weight, num_events_in_rule=1):
        """
        Compute symmetric triangular kernel value with area=1 normalization.
        Includes correction factor C^{num_events - 1} where C = 1/num_components.
        
        Args:
            dt: time difference
            peak: center position
            width: half-width (symmetric)
            weight: mixture weight (alpha_k from softmax)
            num_events_in_rule: number of events in the rule (for correction factor)
            
        Returns:
            kernel values with correction factor applied
        """
        t_start = peak - width
        t_end = peak + width
        w_height = weight / (width + self.eps)
        dist = torch.abs(dt - peak)
        kernel = w_height * torch.clamp(1.0 - dist / (width + self.eps), 0.0, 1.0)
        kernel = torch.where((dt >= t_start) & (dt <= t_end), kernel, torch.zeros_like(kernel))
        
        # Apply correction factor: C^{num_events - 1} where C = 1/num_components
        C = 1.0 / self.num_components
        correction = C ** (num_events_in_rule - 1)
        kernel = kernel * correction
        
        return kernel
    
    def triangular_kernel_mixture(self, dt, peaks, widths, weights, num_events_in_rule=1):
        """
        Compute mixture of symmetric triangular kernels with correction factor.
        
        Args:
            dt: time difference, shape (...)
            peaks: component peaks, shape (..., num_components)
            widths: component half-widths, shape (..., num_components)
            weights: mixture weights (sum to 1), shape (..., num_components)
            num_events_in_rule: number of events in the rule (for correction factor)
            
        Returns:
            kernel mixture value with correction factor
        """
        dt_exp = dt.unsqueeze(-1)
        components = self.symmetric_triangular_kernel(dt_exp, peaks, widths, weights, num_events_in_rule)
        kernel = components.sum(dim=-1)
        return kernel
    
    def symmetric_triangular_integral(self, t_start, t_end, t_event, peak, width, weight, num_events_in_rule=1):
        """
        Compute exact integral of a symmetric triangular kernel with correction factor.
        
        Args:
            t_start: integration start time
            t_end: integration end time
            t_event: time of the triggering event
            peak: relative peak position from t_event
            width: half-width of triangle
            weight: mixture weight (alpha_k)
            num_events_in_rule: number of events in the rule (for correction factor)
            
        Returns:
            Exact integral value with correction factor applied
        """
        abs_peak = t_event + peak
        abs_start = abs_peak - width
        abs_end = abs_peak + width
        height = weight / (width + self.eps)
        int_start = torch.clamp(t_start, min=abs_start, max=abs_end)
        int_end = torch.clamp(t_end, min=abs_start, max=abs_end)
        valid = (int_end > int_start) & (t_end > abs_start) & (t_start < abs_end)
        
        def triangle_segment_integral(s, e, p, w, h):
            left_valid = (s < p) & (e > s)
            right_valid = (e > p) & (e > s)
            left_end = torch.where(e < p, e, p)
            left_int = h * ((left_end - s) - (p/w)*(left_end - s) + (left_end**2 - s**2)/(2*w))
            left_int = torch.where(left_valid & (left_end > s), left_int, torch.zeros_like(left_int))
            right_start = torch.where(s > p, s, p)
            right_int = h * ((e - right_start) - (e**2 - right_start**2)/(2*w) + (p/w)*(e - right_start))
            right_int = torch.where(right_valid & (e > right_start), right_int, torch.zeros_like(right_int))
            return left_int + right_int
        
        integral = torch.where(
            valid,
            triangle_segment_integral(int_start, int_end, abs_peak, width, height),
            torch.zeros_like(int_start)
        )
        
        # Apply correction factor
        C = 1.0 / self.num_components
        correction = C ** (num_events_in_rule - 1)
        integral = integral * correction
        
        return integral
    
    def triangular_mixture_integral(self, t_start, t_end, t_event, peaks, widths, weights, num_events_in_rule=1):
        """
        Compute exact integral of triangular mixture with correction factor.
        
        Args:
            t_start, t_end: integration bounds
            t_event: triggering event time
            peaks: (num_components,) or broadcastable
            widths: (num_components,) or broadcastable
            weights: (num_components,) or broadcastable
            num_events_in_rule: number of events in the rule (for correction factor)
            
        Returns:
            Total integral with correction factor
        """
        num_comp = peaks.shape[-1]
        total = 0.0
        for i in range(num_comp):
            comp_integral = self.symmetric_triangular_integral(
                t_start, t_end, t_event,
                peak=peaks[..., i],
                width=widths[..., i],
                weight=weights[..., i],
                num_events_in_rule=num_events_in_rule
            )
            total = total + comp_integral
        return total

    def straight_through_estimator(self, logits, tau=None):
        """Stable Gumbel-Sigmoid for differentiable discrete selection."""
        if tau is None:
            tau = self.tau
        if not self.training:
            y_soft = torch.sigmoid(logits)
            y_hard = (y_soft > 0.5).float()
            return y_hard, y_soft
        noise_scale = min(1.0, tau / 2.0)
        u1 = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
        u2 = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
        gumbel_noise = (-torch.log(-torch.log(u1)) + torch.log(-torch.log(u2))) * noise_scale
        y_soft = torch.sigmoid((logits + gumbel_noise) / tau)
        y_hard = (y_soft > 0.5).float()
        return (y_hard - y_soft).detach() + y_soft, y_soft
    
    def forward(self, x, tau=None):
        if tau is None:
            tau = self.tau
            
        # Rule source node selection (includes latent nodes)
        H_hard, H_soft = self.straight_through_estimator(self.theta, tau)
        
        # Rule target type selection
        Head_soft = F.softmax(self.rule_target_logits, dim=-1)
        Head_hard = F.one_hot(Head_soft.argmax(dim=-1), num_classes=self.num_types).float()
        
        # Rule intensity matrices for event nodes only
        H_soft_event = H_soft[:self.num_types, :]
        H_hard_event = H_hard[:self.num_types, :]
        
        # Interaction matrices (full with latent nodes)
        H_soft_full = H_soft
        H_hard_full = H_hard
        
        # Rule weights (allow negative)
        rule_weights = self.rule_weights
        
        # Triangular kernel params (peaks, widths, weights for 3 symmetric triangles)
        tri_peaks, tri_widths, tri_weights = self.get_triangular_params()
        
        # Compute interaction mask using dual hypergraph with Target-Source connection
        # Use FULL hypergraph (latent nodes 포함) for interaction detection
        inter_mask = self.get_interaction_mask_from_dual(H_hard_full, H_soft_full, Head_hard, Head_soft)
        
        # Effective interaction weights (masked by adjacency)
        effective_inter_weights = self.interaction_weights * inter_mask
        
        # Interaction triangular params (peaks, widths, weights for 3 symmetric triangles)
        inter_peaks, inter_widths, inter_tri_weights = self.get_interaction_triangular_params()
        
        # Context encoding
        event_embs = self.event_embedding(x['input_ids'])
        context_h, _ = self.history_encoder(event_embs)
        
        return {
            'H_soft': H_soft_event,
            'H_hard': H_hard_event,
            'H_soft_full': H_soft_full,
            'H_hard_full': H_hard_full,
            'Head_soft': Head_soft,
            'Head_hard': Head_hard if not self.training else Head_soft,
            'rule_weights': rule_weights,
            'tri_peaks': tri_peaks,
            'tri_widths': tri_widths,
            'tri_weights': tri_weights,
            'inter_mask': inter_mask,
            'effective_inter_weights': effective_inter_weights,
            'inter_peaks': inter_peaks,
            'inter_widths': inter_widths,
            'inter_tri_weights': inter_tri_weights,
            'context_h': context_h,
        }

    # ============== Intensity Computation ==============
    
    def _get_most_recent_event_per_type(self, eval_times, event_times, event_types):
        """
        For each (batch, eval_time, type), find the most recent event of that type.
        Optimized version with reduced memory usage.
        
        Args:
            eval_times: (B, Q) evaluation times
            event_times: (B, L) event times
            event_types: (B, L) event types
            
        Returns:
            last_event_times: (B, Q, K) time of most recent event of each type (0 if none)
            last_event_mask: (B, Q, K) 1 if there's a valid event, 0 otherwise
        """
        B, Q = eval_times.shape
        _, L = event_times.shape
        K = self.num_types
        device = eval_times.device
        
        # Build type one-hot: (B, L, K)
        type_onehot = F.one_hot(event_types.clamp(0, K-1), num_classes=K).float()  # (B, L, K)
        
        # dt[b, q, l] = eval_times[b, q] - event_times[b, l]
        dt = eval_times.unsqueeze(2) - event_times.unsqueeze(1)  # (B, Q, L)
        
        # Causal mask: only consider past events (dt > 0) and not padded
        pad_mask = (event_types == self.pad_token_id)  # (B, L)
        causal_mask = (dt > 0) & (~pad_mask.unsqueeze(1))  # (B, Q, L) - boolean for efficiency
        
        # Create masked times: (B, Q, L)
        # Use -inf for invalid entries so max() will ignore them
        masked_times = torch.where(
            causal_mask,
            event_times.unsqueeze(1).expand(B, Q, L),
            torch.full((B, Q, L), -1e9, device=device, dtype=event_times.dtype)
        )
        
        # Expand for types: (B, Q, L, K) - only compute where type matches
        masked_times_k = masked_times.unsqueeze(-1) * type_onehot.unsqueeze(1)
        masked_times_k = torch.where(
            type_onehot.unsqueeze(1) > 0.5,
            masked_times_k,
            torch.full_like(masked_times_k, -1e9)
        )

        # Find max over L for each type: (B, Q, K)
        last_event_times, _ = masked_times_k.max(dim=2)  # (B, Q, K)
        
        # Mask for valid events
        last_event_mask = (last_event_times > -1e8).float()
        last_event_times = last_event_times.clamp(min=0.0) * last_event_mask  # Set invalid to 0
        
        return last_event_times, last_event_mask

    def _compute_product_kernel_integral_exact(self, t_start, t_end, last_event_times, last_event_mask,
                                               tri_peaks, tri_widths, tri_weights, H, rule_idx=None):
        """
        Compute exact integral of product of kernels for rules using breakpoint-based piecewise integration.
        
        For each rule r:
            φ_r(t) = Π_{k: H[k,r]>0.5} K_k(t)
        where K_k(t) = Σ_{c=1}^{3} tri_c(t) is the sum of 3 triangular components.
        
        Since each tri_c is piecewise linear, K_k is piecewise linear, and their product is piecewise polynomial.
        We compute the exact integral by:
        1. Collecting all breakpoints (triangle starts, peaks, ends)
        2. Sorting and deduplicating
        3. Integrating polynomial over each segment
        
        Args:
            t_start: (B, Q) integration start
            t_end: (B, Q) integration end
            last_event_times: (B, Q, K) most recent event time per type
            last_event_mask: (B, Q, K) which types have valid events
            tri_peaks: (K, M, C) relative peak offsets
            tri_widths: (K, M, C) half-widths
            tri_weights: (K, M, C) component weights
            H: (K, M) rule membership matrix
            rule_idx: if specified, compute for single rule; else all rules
            
        Returns:
            integrals: (B, Q, M) or (B, Q) integral of product kernel
        """
        B, Q, K = last_event_times.shape
        _, M, C = tri_peaks.shape
        device = t_start.device
        
        if rule_idx is not None:
            M_out = 1
            rule_range = [rule_idx]
        else:
            M_out = M
            rule_range = range(M)
        
        # Maximum number of breakpoints per rule:
        # Each required type has 3 components, each with 3 breakpoints (start, peak, end)
        # Max 4 required types * 3 components * 3 breakpoints = 36, plus 2 for integration bounds
        MAX_BP = 40
        
        # Output tensor
        integrals = torch.zeros(B, Q, M_out, device=device)
        
        # Process each rule
        for r_idx, r in enumerate(rule_range):
            # Find required types for this rule
            required_mask = H[:, r] > 0.5  # (K,)
            required_types = required_mask.nonzero(as_tuple=True)[0]  # indices of required types
            n_req = len(required_types)
            
            if n_req == 0:
                continue
            
            # Collect all breakpoints for this rule: (B, Q, n_bp)
            # Absolute peak positions for required types: (B, Q, n_req, C)
            req_last_times = last_event_times[:, :, required_types]  # (B, Q, n_req)
            req_peaks = tri_peaks[required_types, r, :]  # (n_req, C)
            req_widths = tri_widths[required_types, r, :]  # (n_req, C)
            req_weights = tri_weights[required_types, r, :]  # (n_req, C)
            req_mask = last_event_mask[:, :, required_types]  # (B, Q, n_req)
            
            # Absolute peaks: (B, Q, n_req, C)
            abs_peaks = req_last_times.unsqueeze(-1) + req_peaks.unsqueeze(0).unsqueeze(0)
            
            # Triangle starts and ends: (B, Q, n_req, C)
            tri_starts = abs_peaks - req_widths.unsqueeze(0).unsqueeze(0)
            tri_ends = abs_peaks + req_widths.unsqueeze(0).unsqueeze(0)
            
            # Collect all breakpoints: starts, peaks, ends
            # Shape: (B, Q, n_req * C * 3)
            all_bp = torch.cat([
                tri_starts.reshape(B, Q, -1),
                abs_peaks.reshape(B, Q, -1),
                tri_ends.reshape(B, Q, -1)
            ], dim=-1)  # (B, Q, n_req * C * 3)
            
            # Add integration bounds
            all_bp = torch.cat([
                all_bp,
                t_start.unsqueeze(-1),
                t_end.unsqueeze(-1)
            ], dim=-1)  # (B, Q, n_bp)
            
            # Clamp to integration interval
            all_bp = torch.clamp(all_bp, 
                                t_start.unsqueeze(-1).expand_as(all_bp),
                                t_end.unsqueeze(-1).expand_as(all_bp))
            
            # Sort breakpoints
            all_bp_sorted, _ = torch.sort(all_bp, dim=-1)  # (B, Q, n_bp)
            
            # Create segments: (start, end) pairs
            seg_starts = all_bp_sorted[..., :-1]  # (B, Q, n_seg)
            seg_ends = all_bp_sorted[..., 1:]  # (B, Q, n_seg)
            seg_valid = seg_ends > seg_starts + 1e-10  # Valid non-zero-length segments
            
            n_seg = seg_starts.shape[-1]
            
            # For each segment, compute the product of kernel values
            # Take midpoint of segment to determine which piece of each triangle we're in
            seg_mids = (seg_starts + seg_ends) * 0.5  # (B, Q, n_seg)
            
            # For each required type and component, get linear params at segment midpoint
            # Use broadcasting instead of expand for memory efficiency
            seg_mids_exp = seg_mids.unsqueeze(-1).unsqueeze(-1)  # (B, Q, n_seg, 1, 1)
            abs_peaks_exp = abs_peaks.unsqueeze(2)  # (B, Q, 1, n_req, C)
            widths_exp = req_widths.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1, n_req, C)
            weights_exp = req_weights.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1, 1, 1, n_req, C)
            
            # Heights (use in-place for efficiency)
            heights_exp = weights_exp / (widths_exp + self.eps)
            
            # Apply correction factor: C^{num_events_in_rule - 1} where C = 1/num_components
            C = 1.0 / self.num_components
            correction = C ** (n_req - 1)
            
            # Triangle bounds (compute only once)
            tri_starts_exp = abs_peaks_exp - widths_exp
            tri_ends_exp = abs_peaks_exp + widths_exp
            
            # Determine region for each segment midpoint (boolean masks)
            in_left = (seg_mids_exp >= tri_starts_exp) & (seg_mids_exp < abs_peaks_exp)
            in_right = (seg_mids_exp >= abs_peaks_exp) & (seg_mids_exp <= tri_ends_exp)
            
            # Compute slope/intercept only for active regions (more efficient)
            inv_width = 1.0 / (widths_exp + self.eps)
            slope_mag = heights_exp * inv_width
            
            # Linear params - reuse computations
            slope_left = slope_mag
            intercept_left = heights_exp * (1.0 - abs_peaks_exp * inv_width)
            slope_right = -slope_mag
            intercept_right = heights_exp * (1.0 + abs_peaks_exp * inv_width)
            
            # Select based on region: (B, Q, n_seg, n_req, C)
            comp_slopes = torch.where(in_left, slope_left,
                                      torch.where(in_right, slope_right, torch.zeros_like(slope_left)))
            comp_intercepts = torch.where(in_left, intercept_left,
                                         torch.where(in_right, intercept_right, torch.zeros_like(intercept_left)))
            
            # Sum over components to get K_k(t) = Σ_c tri_c(t) for each type
            type_slopes = comp_slopes.sum(dim=-1)  # (B, Q, n_seg, n_req)
            type_intercepts = comp_intercepts.sum(dim=-1)  # (B, Q, n_seg, n_req)
            
            # Mask invalid types (use broadcasting instead of expand)
            req_mask_exp = req_mask.unsqueeze(2)  # (B, Q, 1, n_req)
            type_slopes = torch.where(req_mask_exp > 0.5, type_slopes, torch.zeros_like(type_slopes))
            type_intercepts = torch.where(req_mask_exp > 0.5, type_intercepts, torch.ones_like(type_intercepts))
            
            # Now compute product of n_req linear functions per segment
            # Using PolynomialIntegrator.get_poly_coeffs_variable
            # Reshape to (B*Q*n_seg, n_req)
            slopes_flat = type_slopes.reshape(-1, n_req)
            intercepts_flat = type_intercepts.reshape(-1, n_req)
            
            # Get polynomial coefficients: (B*Q*n_seg, n_req+1)
            poly_coeffs = PolynomialIntegrator.get_poly_coeffs_variable(slopes_flat, intercepts_flat)
            
            # Reshape back: (B, Q, n_seg, n_req+1)
            poly_coeffs = poly_coeffs.reshape(B, Q, n_seg, -1)
            
            # Integrate polynomial over each segment
            # (B, Q, n_seg)
            seg_integrals = PolynomialIntegrator.integrate_poly(
                poly_coeffs, seg_starts, seg_ends
            )  # (B, Q, n_seg)
            
            # Mask invalid segments
            seg_integrals = seg_integrals * seg_valid.float()
            
            # Sum over segments
            rule_integral = seg_integrals.sum(dim=-1)  # (B, Q)
            
            # Clamp negative values (numerical errors)
            rule_integral = F.relu(correction*rule_integral)
            
            integrals[:, :, r_idx] = rule_integral
        
        return integrals
    
    def compute_intensity_at_times(self, eval_times, event_times, event_types, model_output, return_per_rule=False, return_event_info=False):
        """
        Compute intensity at specific evaluation times.
        
        Uses Top-1 event selection per type and proper rule activation.
        
        Args:
            eval_times: (B, Q) times to evaluate intensity
            event_times: (B, L) times of past events
            event_types: (B, L) types of past events
            model_output: output from forward()
            return_per_rule: if True, also return per-rule contributions
            return_event_info: if True, return the 4 events used per rule for explanation
            
        Returns:
            lambda_per_type: (B, Q, num_types) intensity per event type
            (optional) rule_intensity: (B, Q, M) per-rule contributions
            (optional) event_info: dict with 'last_event_times', 'last_event_types', 'used_types_per_rule'
        """
        B, Q = eval_times.shape
        _, L = event_times.shape
        M = self.num_rules
        K = self.num_types
        device = eval_times.device
        
        H_soft = model_output['H_soft']  # (K, M)
        H_hard = model_output['H_hard']
        Head = model_output['Head_soft'] if self.training else model_output['Head_hard']  # (M, K)
        rule_weights = model_output['rule_weights']  # (M,) - allow negative
        tri_peaks = model_output['tri_peaks']  # (K, M, 3)
        tri_widths = model_output['tri_widths']  # (K, M, 3)
        tri_weights = model_output['tri_weights']  # (K, M, 3)
        effective_inter_weights = model_output['effective_inter_weights']  # (M, M)
        
        H = H_soft if self.training else H_hard
        
        # ============== Step 1: Top-1 Event Selection ==============
        # Get most recent event of each type for each eval time
        last_event_times, last_event_mask = self._get_most_recent_event_per_type(
            eval_times, event_times, event_types
        )  # (B, Q, K), (B, Q, K)
        
        # Compute dt from most recent event of each type: (B, Q, K)
        dt_per_type = eval_times.unsqueeze(-1) - last_event_times  # (B, Q, K)
        
        # ============== Step 2: Compute Kernel Values (Top-1 per type) ==============
        # Compute kernel values for each (type, rule) pair
        # dt_per_type: (B, Q, K) -> (B, Q, K, 1) for broadcasting with rule dimension
        dt_expanded = dt_per_type.unsqueeze(-1)  # (B, Q, K, 1)
        
        # Kernel params: (K, M, 3) - for each type k and rule m
        # Expand to (1, 1, K, M, 3) for broadcasting
        peaks_exp = tri_peaks.view(1, 1, K, M, self.num_components)  # (1, 1, K, M, 3)
        widths_exp = tri_widths.view(1, 1, K, M, self.num_components)
        weights_exp = tri_weights.view(1, 1, K, M, self.num_components)
        
        # For triangular kernel with mixture: compute directly without calling mixture method
        # kernel(dt) = sum_c weight_c * max(0, 1 - |dt - peak_c| / width_c)
        dt_for_kernel = dt_expanded.unsqueeze(-1)  # (B, Q, K, 1, 1) -> broadcast with (1,1,K,M,3)
        
        # Symmetric triangular: f(t) = max(0, weight/width * (1 - |t - peak| / width))
        distance = torch.abs(dt_for_kernel - peaks_exp)  # (B, Q, K, M, 3)
        normalized_dist = distance / (widths_exp + self.eps)
        height = weights_exp / (widths_exp + self.eps)
        
        # Apply correction factor per rule: C^{num_events_in_rule - 1}
        # Count number of required types per rule: H.sum(dim=0) -> (M,)
        num_events_per_rule = H.sum(dim=0)  # (M,)
        C = 1.0 / self.num_components
        correction_per_rule = C ** (num_events_per_rule - 1)  # (M,)
        correction_exp = correction_per_rule.view(1, 1, 1, M, 1)  # (1, 1, 1, M, 1)
        
        component_vals = correction_exp * height * F.relu(1.0 - normalized_dist)  # (B, Q, K, M, 3)
        
        # Sum over components to get kernel mixture
        kernel_vals = component_vals.sum(dim=-1)  # (B, Q, K, M)
        
        # Mask by whether there's a valid event for this type
        kernel_vals = kernel_vals * last_event_mask.unsqueeze(-1)  # (B, Q, K, M)
        
        # ============== Step 3: Rule Activation (AND logic) ==============
        # type_satisfaction is now just the kernel value (Top-1 selection)
        type_satisfaction = kernel_vals  # (B, Q, K, M)
        
        # Rule pattern: H[k, m] indicates if type k is required for rule m
        H_exp = H.view(1, 1, K, M)
        
        # ============== AND-logic ==============
        # AND logic: rule activation = product of kernel values for required types
        # For non-required types (H≈0), we use kernel^H ≈ 1 (no contribution)
        # For required types (H≈1), we use kernel^H ≈ kernel (full contribution)
        
        eps = 1e-6
        
        # Use H_soft during training for gradient flow
        if self.training:
            H_weight = H_soft.view(1, 1, K, M)
        else:
            H_weight = H_hard.view(1, 1, K, M)
        
        # Product AND: prod_k (kernel_k ^ H_k) = exp(sum_k H_k * log(kernel_k))
        # Clamp kernel values to prevent log(0)
        safe_kernel = torch.clamp(kernel_vals, min=eps)
        
        # Weighted log: H * log(kernel)
        # When H=0, contribution is 0 (kernel^0 = 1, log contribution = 0)
        # When H=1, contribution is log(kernel) (full AND logic)
        log_kernel = torch.log(safe_kernel)
        weighted_log_kernel = log_kernel * H_weight  # (B, Q, K, M)
        
        # Sum over types: sum_k H_k * log(kernel_k)
        log_product = weighted_log_kernel.sum(dim=2)  # (B, Q, M)
        
        # Exp to get product, clamp for numerical stability
        rule_activation = torch.exp(log_product.clamp(min=-30, max=10))  # (B, Q, M)
        
        # ============== Step 4: Compute Interaction via Satisfaction Factor ==============
        # S_r = 1 - exp(-I_r) where I_r is EXACT integral of rule intensity
        # from second-latest event time to current eval time
        
        # Find the second-latest event time before each eval_time
        dt_all = eval_times.unsqueeze(2) - event_times.unsqueeze(1)  # (B, Q, L)
        pad_mask = (event_types == self.pad_token_id).unsqueeze(1)  # (B, 1, L)
        causal_mask = (dt_all > 0).float() * (~pad_mask).float()
        masked_times = event_times.unsqueeze(1) * causal_mask + (-1e9) * (1 - causal_mask)
        prev_event_time, _ = masked_times.max(dim=2)  # (B, Q)
        prev_event_time = torch.clamp(prev_event_time, min=0)  # Set -inf to 0
        
        # Compute EXACT satisfaction factor using polynomial integration
        # S_r = 1 - exp(-∫φ_r(t)dt) from prev_event_time to eval_times
        satisfaction_factor = self.compute_rule_satisfaction(
            t_start=prev_event_time,
            t_end=eval_times,
            event_times=event_times,
            event_types=event_types,
            model_output=model_output
        )  # (B, Q, M)
        
        # Interaction contribution: weighted sum of source satisfactions
        inter_contrib = torch.matmul(satisfaction_factor, effective_inter_weights.t())  # (B, Q, M)
        
        # Effective rule weight = base_weight + interaction_contribution
        # Note: effective_weights can be negative (for inhibition)
        effective_weights = rule_weights.view(1, 1, M) + inter_contrib  # (B, Q, M)
        
        # Rule contribution to intensity (can be negative before ReLU)
        rule_intensity = rule_activation * effective_weights  # (B, Q, M)
        
        # Project to event types via Head
        lambda_per_type = torch.matmul(rule_intensity, Head)  # (B, Q, K)
        
        # Add base intensity
        b0 = F.softplus(self.b0).view(1, 1, K)
        lambda_per_type = lambda_per_type + b0
        
        # Ensure non-negative intensity (ReLU handles negative weights)
        lambda_per_type = F.relu(lambda_per_type)
        
        # Prepare event info for explanations (Fix #6)
        if return_event_info:
            # For each rule, identify the top 4 types used
            # H: (K, M), get indices of required types
            _, type_order = H.t().sort(dim=1, descending=True)  # (M, K)
            used_types_per_rule = type_order[:, :4]  # (M, 4) - top 4 types per rule
            
            # Get the actual event times and types for these
            event_info = {
                'last_event_times': last_event_times,  # (B, Q, K) - most recent event time per type
                'last_event_mask': last_event_mask,  # (B, Q, K) - which types have valid events
                'used_types_per_rule': used_types_per_rule,  # (M, 4) - which 4 types each rule uses
                'H_matrix': H,  # (K, M) - which types are actually required
            }
            if return_per_rule:
                return lambda_per_type, rule_intensity, event_info
            return lambda_per_type, event_info
        
        if return_per_rule:
            return lambda_per_type, rule_intensity
        return lambda_per_type

    # ============== Closed-Form Integral ==============
    
    def compute_integral_closed_form(self, t_start, t_end, event_times, event_types, model_output):
        """
        Compute exact integral of intensity over [t_start, t_end] using breakpoint-based piecewise integration.
        
        For each rule, intensity = product of K_k(t) for required types (AND logic).
        Each K_k(t) = sum of 3 triangular components (piecewise linear).
        Their product is piecewise polynomial, integrated exactly.
        
        Args:
            t_start: (B,) or (B, Q) start times
            t_end: (B,) or (B, Q) end times
            event_times: (B, L) times of past events
            event_types: (B, L) types of past events
            model_output: output from forward()
            
        Returns:
            integral: (B,) or (B, Q) total integral values
        """
        B, L = event_times.shape
        K = self.num_types
        M = self.num_rules
        device = event_times.device
        
        # Ensure t_start/t_end are 2D
        if t_start.dim() == 1:
            t_start = t_start.unsqueeze(1)
            t_end = t_end.unsqueeze(1)
        Q = t_start.shape[1]
        
        H_soft = model_output['H_soft']  # (K, M)
        H_hard = model_output['H_hard']
        Head = model_output['Head_soft'] if self.training else model_output['Head_hard']  # (M, K)
        rule_weights = model_output['rule_weights']  # (M,)
        tri_peaks = model_output['tri_peaks']  # (K, M, 3)
        tri_widths = model_output['tri_widths']  # (K, M, 3)
        tri_weights = model_output['tri_weights']  # (K, M, 3)
        effective_inter_weights = model_output['effective_inter_weights']  # (M, M)
        
        H = H_soft if self.training else H_hard
        
        # Get most recent event per type
        last_event_times, last_event_mask = self._get_most_recent_event_per_type(
            t_end, event_times, event_types
        )  # (B, Q, K), (B, Q, K)
        
        # Compute exact integral using product of kernels (AND logic) with breakpoint-based integration
        rule_integrals = self._compute_product_kernel_integral_exact(
            t_start, t_end, last_event_times, last_event_mask,
            tri_peaks, tri_widths, tri_weights, H
        )  # (B, Q, M)
        

        source_satisfaction = 1.0 - torch.exp(-rule_integrals.clamp(min=0, max=20))
        delta_weights = torch.matmul(source_satisfaction, effective_inter_weights.t())
        effective_rule_weights = rule_weights.view(1, 1, M) + delta_weights
        weighted_rule_integrals = rule_integrals * effective_rule_weights
        
        # Project to types: integral_per_type = weighted_rule_integrals @ Head
        # This can be negative if weights are negative
        integral_per_type = torch.matmul(weighted_rule_integrals, Head)  # (B, Q, K)
        
        # Add base intensity integral: b0 * (t_end - t_start)
        b0 = F.softplus(self.b0).view(1, 1, K)
        interval_length = (t_end - t_start).unsqueeze(-1)  # (B, Q, 1)
        integral_per_type = integral_per_type + b0 * interval_length
        
        # Apply ReLU to ensure non-negative integral (matching intensity ReLU)
        # This is mathematically correct: ∫ReLU(λ(t))dt ≥ 0
        integral_per_type = F.relu(integral_per_type)
        
        # Total integral across all types
        total_integral = integral_per_type.sum(dim=-1)  # (B, Q)
        
        return total_integral.squeeze(1) if Q == 1 else total_integral
    
    def compute_rule_satisfaction(self, t_start, t_end, event_times, event_types, model_output):
        """
        Compute rule satisfaction S_r = 1 - exp(-∫φ_r(t)dt) using EXACT integration with AND logic.
        
        Rule φ_r = PRODUCT of K_k(t) for required types (AND logic).
        Each K_k(t) = sum of 3 triangular components (piecewise linear).
        Their product is piecewise polynomial, integrated exactly using breakpoint-based method.
        
        This implementation is fully VECTORIZED - no for loops over batches.
        
        Args:
            t_start: (B,) or (B, Q) start times
            t_end: (B,) or (B, Q) end times
            event_times: (B, L) event times
            event_types: (B, L) event types
            model_output: forward output
            
        Returns:
            satisfaction: (B, Q, M) satisfaction factor per rule, S_r = 1 - exp(-∫φ_r(t)dt)
        """
        # Ensure 2D
        if t_start.dim() == 1:
            t_start = t_start.unsqueeze(1)
            t_end = t_end.unsqueeze(1)
        
        B, Q = t_start.shape
        M = self.num_rules
        K = self.num_types
        device = t_start.device
        
        H_soft = model_output['H_soft']  # (K, M)
        H_hard = model_output['H_hard']
        tri_peaks = model_output['tri_peaks']  # (K, M, 3)
        tri_widths = model_output['tri_widths']
        tri_weights = model_output['tri_weights']
        
        H = H_soft if self.training else H_hard
        
        # Get most recent event per type before t_end
        last_event_times, last_event_mask = self._get_most_recent_event_per_type(
            t_end, event_times, event_types
        )  # (B, Q, K), (B, Q, K)
        
        # Compute exact integral using breakpoint-based piecewise integration
        rule_integrals = self._compute_product_kernel_integral_exact(
            t_start, t_end, last_event_times, last_event_mask,
            tri_peaks, tri_widths, tri_weights, H
        )  # (B, Q, M)
        
        # S_r = 1 - exp(-I_r)
        # Clamp integral to prevent overflow in exp, but allow larger values for better gradients
        satisfaction = 1.0 - torch.exp(-rule_integrals.clamp(min=0, max=50))
        
        return satisfaction

    # ============== NLL Calculation ==============
    
    def _calculate_nll(self, batch, model_output):
        """Compute NLL using closed-form integration."""
        time_diffs = batch['time_diffs']
        input_ids = batch['input_ids']
        mask = batch.get('attention_mask', torch.ones_like(time_diffs))
        
        B, L = time_diffs.shape
        device = time_diffs.device
        
        # Compute absolute event times
        times = torch.cumsum(time_diffs, dim=1)  # (B, L)
        
        # ============== Event Log-Likelihood ==============
        # Compute intensity at each event time
        lambda_at_events = self.compute_intensity_at_times(
            eval_times=times,
            event_times=times,
            event_types=input_ids,
            model_output=model_output
        )  # (B, L, K)
        
        # Get intensity for the actual event type
        safe_input_ids = input_ids.clone()
        safe_input_ids[safe_input_ids == self.pad_token_id] = 0
        target_lambda = lambda_at_events.gather(2, safe_input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L)
        
        # Log-likelihood of events with better numerical stability
        # Clamp to reasonable range [1e-5, 1e3] to avoid log(0) and log(huge)
        safe_lambda = torch.where(mask > 0, target_lambda, torch.ones_like(target_lambda))
        safe_lambda = safe_lambda.clamp(min=1e-6, max=1e4)  # Wider range for stability
        event_ll = torch.sum(torch.log(safe_lambda) * mask)
        
        # ============== Integral (Non-Event) Loss ==============
        # Vectorized computation of integrals for all intervals
        t_starts = torch.cat([torch.zeros(B, 1, device=device), times[:, :-1]], dim=1)  # (B, L)
        t_ends = times  # (B, L)
        
        # Process in chunks to manage memory (chunk_size adjustable based on VRAM)
        chunk_size = min(L, 32)  # Process 32 intervals at once
        integral_loss = torch.zeros(B, device=device)
        
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            chunk_len = chunk_end - chunk_start
            
            # Get chunk of intervals
            chunk_t_start = t_starts[:, chunk_start:chunk_end]  # (B, chunk_len)
            chunk_t_end = t_ends[:, chunk_start:chunk_end]  # (B, chunk_len)
            chunk_mask = mask[:, chunk_start:chunk_end]  # (B, chunk_len)
            
            # Create past event mask for each interval in chunk: (B, chunk_len, L)
            # past_mask[b, i, j] = 1 if event j happened before interval i's end time
            chunk_t_end_exp = chunk_t_end.unsqueeze(-1)  # (B, chunk_len, 1)
            times_exp = times.unsqueeze(1)  # (B, 1, L)
            past_mask = (times_exp < chunk_t_end_exp).float()  # (B, chunk_len, L)
            
            # Batch process all intervals in chunk
            # Reshape to process all (B*chunk_len) intervals together
            flat_t_start = chunk_t_start.reshape(-1)  # (B*chunk_len,)
            flat_t_end = chunk_t_end.reshape(-1)  # (B*chunk_len,)
            
            # Expand event data for all intervals
            flat_times = times.unsqueeze(1).expand(B, chunk_len, L).reshape(B * chunk_len, L)
            flat_types = input_ids.unsqueeze(1).expand(B, chunk_len, L).reshape(B * chunk_len, L)
            flat_past_mask = past_mask.reshape(B * chunk_len, L)
            
            # Apply past mask
            masked_times = flat_times * flat_past_mask
            
            # Compute integrals for all intervals in chunk at once
            chunk_integrals = self.compute_integral_closed_form(
                t_start=flat_t_start,
                t_end=flat_t_end,
                event_times=masked_times,
                event_types=flat_types,
                model_output=model_output
            )  # (B*chunk_len,)
            
            # Reshape back and apply mask
            chunk_integrals = chunk_integrals.reshape(B, chunk_len)
            # Clamp integrals to prevent explosion
            chunk_integrals = chunk_integrals.clamp(max=100.0)
            integral_loss = integral_loss + (chunk_integrals * chunk_mask).sum(dim=1)
        
        total_integral_loss = integral_loss.sum()
        
        num_events = mask.sum() + 1e-9
        nll = -(event_ll - total_integral_loss) / num_events
        
        return nll

    # ============== Regularization ==============
    
    def _calculate_regularization(self, model_output, sparsity_weight=None, base_weight=None, **kwargs):
        """Regularization for interpretable rules."""
        H_soft = model_output['H_soft']
        Head_soft = model_output['Head_soft']
        H_soft_full = model_output['H_soft_full']
        inter_mask = model_output['inter_mask']
        effective_inter_weights = model_output['effective_inter_weights']
        
        lambda_ortho = float(self.config.get('lambda_ortho', 1e-3))
        lambda_sparse = sparsity_weight if sparsity_weight is not None else float(self.config.get('lambda_sparse', 1e-3))
        lambda_struct = float(self.config.get('lambda_struct', 1e-3))
        lambda_interaction = float(self.config.get('lambda_interaction', 1e-3))
        reg_rule_num = float(self.config.get('reg_rule_num', 3))
        min_rule_num = float(self.config.get('min_rule_num', 1))
        lambda_base = base_weight if base_weight is not None else float(self.config.get('lambda_base', 1e-4))
        lambda_latent_sparse = float(self.config.get('lambda_latent_sparse', 1e-4))
        
        # Rule orthogonality: encourage different rules to use different types
        H_norm = H_soft_full / (H_soft_full.norm(dim=0, keepdim=True) + 1e-9)
        gram = torch.matmul(H_norm.t(), H_norm)
        eye = torch.eye(self.num_rules, device=H_soft.device)
        reg_ortho = -torch.logdet(gram + 1e-5 * eye)
        
        # Rule sparsity: penalize having too many types per rule
        types_per_rule = H_soft.sum(dim=0)
        excess = torch.relu(types_per_rule - reg_rule_num)
        reg_sparse = torch.sum(excess ** 2)
        H_soft_latent = H_soft_full[self.num_types:, :]
        latents_per_rule = H_soft_latent.sum(dim=0)
        excess_latent = torch.relu(latents_per_rule - 1.0) 
        reg_sparse_latent = torch.sum(excess_latent ** 2)

        
        # Minimum rule activation: encourage each rule to have at least min_rule_num types
        deficit = torch.relu(min_rule_num - types_per_rule)
        reg_min_rule = torch.sum(deficit ** 2) * 0.5
        
        # Target-Source separation: penalize overlap between target and source types
        # Head_soft: (M, K), H_soft: (K, M)
        # For each rule m, penalize H_soft[target_type, m] where target_type = argmax(Head_soft[m])
        # This discourages a rule's target type from being in its own sources
        # Strong penalty: use squared overlap for stronger gradient
        overlap = (Head_soft.t() * H_soft).sum()  # Penalize when target type is in sources
        reg_target_source = overlap ** 2  # Much stronger penalty (was 0.1 * overlap)
        
        # Interaction structure sparsity
        reg_struct = torch.mean(inter_mask)
        
        # Interaction weight sparsity
        reg_interaction = torch.mean(torch.abs(effective_inter_weights))

        # Base intensity regularization
        reg_base = torch.abs(F.softplus(self.b0)).mean()
        
        total_reg = (lambda_ortho * reg_ortho +
                    lambda_sparse * reg_sparse +
                    lambda_sparse * reg_min_rule +
                    lambda_sparse * reg_target_source +
                    lambda_struct * reg_struct +
                    lambda_interaction * reg_interaction +
                    lambda_base * reg_base +
                    lambda_latent_sparse * reg_sparse_latent)
        
        return total_reg

    # ============== Structure Extraction ==============
    
    def get_structure(self):
        """Extract learned rule and interaction structures."""
        with torch.no_grad():
            tau = float(self.config.get('end_tau', self.tau))
            H_hard_full, H_soft_full = self.straight_through_estimator(self.theta, tau=tau)
            
            # Extract event types only (exclude latent nodes)
            H_hard = H_hard_full[:self.num_types, :]  # (K, M)
            H_soft = H_soft_full[:self.num_types, :]  # (K, M)
            
            Head_soft = F.softmax(self.rule_target_logits, dim=-1)
            Head_hard = F.one_hot(Head_soft.argmax(dim=-1), num_classes=self.num_types).float()
            
            # Apply target mask to exclude target types from sources (for interpretation only)
            target_mask = 1.0 - Head_hard.t()  # (K, M)
            H_hard_masked = H_hard * target_mask
            
            tri_peaks, tri_widths, tri_weights = self.get_triangular_params()
            inter_peaks, inter_widths, inter_tri_weights = self.get_interaction_triangular_params()
            
            # Compute interaction mask using full hypergraph (latent nodes 포함)
            inter_mask = self.get_interaction_mask_from_dual(H_hard_full, H_soft_full, Head_hard, Head_soft)
            
            return {
                'rule_definitions': H_hard_masked.t().detach(),  # (M, K) - event types only
                'rule_targets': Head_hard.detach(),
                'rule_weights': self.rule_weights.detach(),
                'tri_peaks': tri_peaks.detach(),  # (K, M, 3)
                'tri_widths': tri_widths.detach(),
                'tri_weights': tri_weights.detach(),
                'interaction_mask': inter_mask.detach(),
                'interaction_weights': self.interaction_weights.detach(),
                'inter_peaks': inter_peaks.detach(),
                'inter_widths': inter_widths.detach(),
                'inter_tri_weights': inter_tri_weights.detach(),
                'b0': F.softplus(self.b0).detach(),
            }
    
    def explain_model_parameters(self, event_names=None):
        """Generate human-readable explanation of learned parameters."""
        struct = self.get_structure()
        
        rule_defs = struct['rule_definitions'].cpu().numpy()
        rule_targets = struct['rule_targets'].cpu().numpy()
        rule_weights = struct['rule_weights'].cpu().numpy()
        tri_peaks = struct['tri_peaks'].cpu().numpy()  # (K, M, 3)
        tri_widths = struct['tri_widths'].cpu().numpy()
        tri_weights = struct['tri_weights'].cpu().numpy()  # (K, M, 3)
        inter_mask = struct['interaction_mask'].cpu().numpy()
        inter_weights = struct['interaction_weights'].cpu().numpy()
        inter_peaks = struct['inter_peaks'].cpu().numpy()  # (M, M, 3)
        inter_widths = struct['inter_widths'].cpu().numpy()
        inter_tri_weights = struct['inter_tri_weights'].cpu().numpy()  # (M, M, 3)
        b0 = struct['b0'].cpu().numpy()
        
        num_rules = self.num_rules
        num_types = self.num_types
        
        if event_names is None:
            event_names = [f"Type {i}" for i in range(num_types)]
        
        # Build rule descriptions
        rule_list = []
        for r_idx in range(num_rules):
            required_types = np.where(rule_defs[r_idx] > 0.5)[0]
            target_type = np.argmax(rule_targets[r_idx])
            
            if len(required_types) == 0:
                condition_str = "Empty rule"
            else:
                conditions = []
                for k in required_types:
                    # Show dominant component (highest weight)
                    weights_k = tri_weights[k, r_idx]  # (3,)
                    dominant_idx = np.argmax(weights_k)
                    peak_val = tri_peaks[k, r_idx, dominant_idx]
                    width_val = tri_widths[k, r_idx, dominant_idx]
                    w_val = weights_k[dominant_idx]
                    left = peak_val - width_val
                    right = peak_val + width_val
                    cond = f"[{event_names[k]}: peak={peak_val:.2f}, width={width_val:.2f}, weight={w_val:.2f}]"
                    conditions.append(cond)
                condition_str = " AND ".join(conditions)
            
            rule_list.append({
                "Rule ID": f"Rule {r_idx}",
                "Conditions": condition_str,
                "Target": event_names[target_type],
                "Weight": f"{rule_weights[r_idx]:.4f}",
                "Active Types": len(required_types),
            })
        
        df_rules = pd.DataFrame(rule_list)
        
        # Build interaction descriptions
        interaction_list = []
        for tgt in range(num_rules):
            for src in range(num_rules):
                if inter_mask[tgt, src] < 0.1:
                    continue
                weight = inter_weights[tgt, src]
                if abs(weight) < 1e-4:
                    continue
                
                # Show dominant component (highest weight)
                weights_inter = inter_tri_weights[tgt, src]  # (3,)
                dominant_idx = np.argmax(weights_inter)
                peak_val = inter_peaks[tgt, src, dominant_idx]
                width_val = inter_widths[tgt, src, dominant_idx]
                w_val = weights_inter[dominant_idx]
                left = peak_val - width_val
                right = peak_val + width_val
                
                interaction_list.append({
                    "Source Rule": f"Rule {src}",
                    "Target Rule": f"Rule {tgt}",
                    "Type": "Excitation" if weight > 0 else "Inhibition",
                    "Weight": f"{weight:.4f}",
                    "Timing": f"peak@{peak_val:.2f}, width={width_val:.2f}, range({left:.2f}-{right:.2f}), mix_w={w_val:.2f}",
                })
        
        df_interactions = pd.DataFrame(interaction_list)
        
        # Base intensities
        b0_df = pd.DataFrame({
            'Event Type': event_names,
            'Base Intensity': b0.tolist()
        })
        
        print('\n[Base Intensities per Type]')
        print(b0_df)
        
        return df_rules, df_interactions, b0_df
    
    def get_rule_diagnostics(self):
        """Get diagnostic information about learned rules for debugging."""
        with torch.no_grad():
            tau = float(self.config.get('end_tau', self.tau))
            H_hard, H_soft = self.straight_through_estimator(self.theta, tau=tau)
            
            theta_values = self.theta.detach().cpu().numpy()
            sigmoid_theta = torch.sigmoid(self.theta).detach().cpu().numpy()
            active_per_rule = (sigmoid_theta > 0.5).sum(axis=0)
            mass_per_rule = sigmoid_theta.sum(axis=0)
            
            # Get triangular kernel params
            tri_peaks, tri_widths, tri_weights = self.get_triangular_params()
            peak_vals = tri_peaks.detach().cpu().numpy()
            width_vals = tri_widths.detach().cpu().numpy()
            
            return {
                'theta_raw_mean': theta_values.mean(),
                'theta_raw_std': theta_values.std(),
                'sigmoid_theta_mean': sigmoid_theta.mean(),
                'active_types_per_rule': active_per_rule.tolist(),
                'selection_mass_per_rule': mass_per_rule.tolist(),
                'mu_mean': peak_vals.mean(),  # Using peak as analogous to mu
                'mu_std': peak_vals.std(),
                'sigma_mean': width_vals.mean(),  # Using width as analogous to sigma
                'sigma_std': width_vals.std(),
                'theta_event': 0.0,  # Placeholder for compatibility
                'theta_type': float(sigmoid_theta.mean()),  # Use mean theta as proxy
                'b0_values': F.softplus(self.b0).detach().cpu().numpy().tolist(),
            }
