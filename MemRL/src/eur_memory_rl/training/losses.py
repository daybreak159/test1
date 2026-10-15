from __future__ import annotations


def clipped_surrogate(new_logprob, old_logprob, advantage, epsilon=0.2):
    import torch

    old = torch.as_tensor(old_logprob, dtype=new_logprob.dtype, device=new_logprob.device).detach()
    ratio = (new_logprob - old).exp()
    advantage = torch.as_tensor(advantage, dtype=new_logprob.dtype, device=new_logprob.device).detach()
    return torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)


def generation_loss(new_logprobs, old_logprobs, mask, advantage, row_weight, divisor=768, epsilon=0.2):
    if divisor <= 0:
        raise ValueError("Common divisor must be positive")
    per_token = clipped_surrogate(new_logprobs, old_logprobs, advantage, epsilon)
    return -(per_token * mask.to(per_token.dtype)).sum() * row_weight / divisor


def slate_loss(new_joint_logprob, old_joint_logprob, advantage, row_weight, divisor=768, epsilon=0.2):
    return -clipped_surrogate(new_joint_logprob, old_joint_logprob, advantage, epsilon) * row_weight / divisor


def ddp_local_weighted_sum(loss, world_size):
    """DDP averages gradients; recover the global weighted sum, not a rank/microbatch mean."""
    return loss * world_size
