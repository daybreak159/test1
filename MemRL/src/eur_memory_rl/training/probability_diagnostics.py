"""Behavior/recompute differences remain observable even without a PPO gate."""


def probability_diagnostics(new, old, mask=None, tolerance=0.03):
    import torch

    actual = new.detach().float()
    behavior = torch.as_tensor(old, dtype=actual.dtype, device=actual.device).detach()
    if actual.shape != behavior.shape:
        raise ValueError('Behavior/recompute probability shapes differ')
    if mask is not None:
        actual, behavior = actual[mask], behavior[mask]
    if not torch.isfinite(actual).all() or not torch.isfinite(behavior).all():
        raise RuntimeError('Nonfinite behavior/recompute logprob; no update published')
    delta = (actual - behavior).abs().reshape(-1)
    if not delta.numel():
        return {'count': 0, 'max_abs': 0.0, 'mean_abs': 0.0, 'p50_abs': 0.0,
                'p95_abs': 0.0, 'p99_abs': 0.0, 'above_tolerance_fraction': 0.0,
                'tolerance': tolerance, 'exceeds_tolerance': False}
    quantiles = torch.quantile(delta, delta.new_tensor([0.5, 0.95, 0.99])).cpu().tolist()
    maximum = float(delta.max())
    return {'count': delta.numel(), 'max_abs': maximum, 'mean_abs': float(delta.mean()),
            'p50_abs': quantiles[0], 'p95_abs': quantiles[1], 'p99_abs': quantiles[2],
            'above_tolerance_fraction': float((delta > tolerance).float().mean()),
            'tolerance': tolerance, 'exceeds_tolerance': maximum > tolerance}
