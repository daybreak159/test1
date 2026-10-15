from __future__ import annotations

import copy
from pathlib import Path
import time
from collections import defaultdict

from ..contracts import GenerationRecord
from ..policies.direct_choice import replay_slate_logprob
from .checkpoint import adapter_state, restore_adapters
from .losses import generation_loss, slate_loss, policy_gradient_loss
from .probability_diagnostics import probability_diagnostics
from .rows import validate_rows


class Coordinator:
    def __init__(
        self,
        shared,
        generator,
        selector,
        domain,
        checkpoint_store,
        lr=None,
        microbatch_size=1,
        divisor=768,
        epsilon=0.2,
        max_grad_norm=1.0,
        train_roles=("E", "U", "R"),
        behavior_tolerance=0.03,
        rollout_correction=None,
        policy_loss="single_step_pg",
        probability_diagnostics_path=None,
    ):
        import torch

        if microbatch_size < 1:
            raise ValueError("microbatch_size >= 1")
        self.shared, self.generator, self.selector = shared, generator, selector
        self.domain, self.store = domain, checkpoint_store
        self.microbatch, self.divisor, self.epsilon = microbatch_size, divisor, epsilon
        self.max_grad_norm, self.train_roles = max_grad_norm, set(train_roles)
        self.behavior_tolerance = behavior_tolerance
        self.rollout_correction = rollout_correction
        if policy_loss not in {"single_step_pg", "ppo"}:
            raise ValueError("Unknown policy_loss")
        if rollout_correction and policy_loss != "ppo":
            raise ValueError("Historical rollout correction requires explicit policy_loss=ppo")
        self.policy_loss = policy_loss
        self.probability_diagnostics_path = probability_diagnostics_path
        self.probability_audit = []
        if bool(rollout_correction) != bool(getattr(generator, "mixed_backend", False)):
            raise ValueError("Mixed rollout requires explicit correction; HF-only retains its original gate")
        if rollout_correction and rollout_correction.get("mode") != "decoupled":
            raise ValueError("Only the validated decoupled correction is supported")
        self.correction_audit = []
        rates = lr or {r: 1e-5 for r in shared.role_parameters}
        if shared.adapter_layout == "shared":
            if len(set(rates.values())) != 1:
                raise ValueError("The shared-LoRA ablation requires one learning rate")
            rates = {"shared": next(iter(rates.values()))}
        self.optimizers = {
            r: torch.optim.AdamW([p for _, p in group], lr=rates[r], weight_decay=0, betas=(0.9, 0.999))
            for r, group in shared.role_parameters.items()
        }
        self.rl_learning_rates = dict(rates)

    def initialize_from_stage(self, stage):
        """SFT supplies weights, not momentum for the new RL objective."""
        if stage != "sft":
            return False
        for role, optimizer in self.optimizers.items():
            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
            for group in optimizer.param_groups:
                group["lr"] = self.rl_learning_rates[role]
        return True

    def _generation_backward(self, rows):
        import torch

        total = 0.0
        for offset in range(0, len(rows), self.microbatch):
            batch = rows[offset : offset + self.microbatch]
            self._synchronize()
            started = time.perf_counter()
            values, masks = self.generator.recompute([row.record for row in batch])
            self._synchronize()
            recompute_seconds = time.perf_counter() - started
            terms = []
            for row, new, mask in zip(batch, values, masks):
                old = torch.tensor(row.record.old_logprobs, device=new.device)
                stats = probability_diagnostics(new, old, mask, self.behavior_tolerance)
                self._record_probability({"event": "recompute", "row_id": row.id, "role": row.role,
                    "input_tokens": len(row.record.input_ids), "output_tokens": len(row.record.response_token_ids),
                    "seq_len": len(row.record.input_ids) + len(row.record.response_token_ids),
                    "recompute_seconds": recompute_seconds, **stats})
                if self.policy_loss == "single_step_pg":
                    terms.append(policy_gradient_loss(new, row.advantage, row.row_weight, self.divisor, mask))
                    continue
                if self.rollout_correction:
                    reference = self.references[row.id].to(new.device)
                    if mask.any() and float((new.detach()[mask] - reference[mask]).abs().max()) > self.behavior_tolerance:
                        raise RuntimeError("HF reference/recompute mismatch before mixed update")
                    terms.append(self._corrected_loss(new, reference, old, row, mask))
                    continue
                if (
                    stats["exceeds_tolerance"]
                ):
                    raise RuntimeError("E/U behavior/recompute logprob mismatch; no update published")
                terms.append(
                    generation_loss(new, old, mask, row.advantage, row.row_weight, self.divisor, self.epsilon)
                )
            loss = torch.stack(terms).sum()
            self._synchronize()
            started = time.perf_counter()
            loss.backward()
            self._synchronize()
            self._record_probability({"event": "backward", "role": batch[0].role,
                "row_ids": [row.id for row in batch], "backward_seconds": time.perf_counter() - started,
                "recompute_seconds": recompute_seconds,
                "peak_allocated_bytes_since_update": torch.cuda.max_memory_allocated(self.shared.device)
                    if self.shared.device.type == "cuda" else None,
                "batch_seq_len": max(len(r.record.input_ids) + len(r.record.response_token_ids) for r in batch)})
            total += float(loss.detach())
        return total

    def _selector_backward(self, rows):
        import torch

        groups = defaultdict(list)
        for row in rows:
            groups[row.record.forward.id].append(row)
        total = 0.0
        for group in groups.values():
            forward = group[0].record.forward
            if any(r.record.forward != forward for r in group):
                raise ValueError("A shared forward ID refers to different inputs")
            if not forward.forward_count:
                continue
            logits = self.selector.recompute_logits(forward)
            terms = []
            for row in group:
                new = replay_slate_logprob(logits, row.record, self.selector.tokenizer)
                stats = probability_diagnostics(new, row.record.old_slate_logprob, tolerance=self.behavior_tolerance)
                self._record_probability({"event": "recompute", "row_id": row.id, "role": "R",
                    "input_tokens": len(forward.input_ids), "output_tokens": 0,
                    "seq_len": len(forward.input_ids), "probability_level": "joint_slate", **stats})
                if self.policy_loss == "single_step_pg":
                    terms.append(policy_gradient_loss(new, row.advantage, row.row_weight, self.divisor))
                    continue
                if self.rollout_correction:
                    behavior = torch.tensor(row.record.old_slate_logprob, device=new.device)
                    if abs(float(new.detach()) - float(self.references[row.id])) > self.behavior_tolerance:
                        raise RuntimeError("HF R reference/recompute mismatch before mixed update")
                    terms.append(self._corrected_loss(new, self.references[row.id].to(new.device),
                                                     behavior, row))
                    continue
                if abs(float(new.detach()) - row.record.old_slate_logprob) > self.behavior_tolerance:
                    raise RuntimeError("R behavior/recompute joint probability mismatch")
                terms.append(
                    slate_loss(
                        new,
                        row.record.old_slate_logprob,
                        row.advantage,
                        row.row_weight,
                        self.divisor,
                        self.epsilon,
                    )
                )
            # L slates share this graph; each has its own identity/advantage/weight.
            loss = torch.stack(terms).sum()
            loss.backward()
            total += float(loss.detach())
        return total

    def _synchronize(self):
        import torch
        if self.shared.device.type == "cuda":
            torch.cuda.synchronize(self.shared.device)

    def _record_probability(self, stats):
        import json
        row = {"policy_version": self.shared.version, "policy_loss": self.policy_loss,
               "mismatch_action": "log" if self.policy_loss == "single_step_pg" else "error", **stats}
        self.probability_audit.append(row)
        if self.probability_diagnostics_path:
            path = Path(self.probability_diagnostics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def _corrected_loss(self, new, reference, behavior, row, mask=None):
        from .rollout_correction import corrected_surrogate
        terms, stats = corrected_surrogate(new, reference, behavior, row.advantage,
            epsilon=self.epsilon, correction_cap=self.rollout_correction.get("cap", 2.0), mask=mask)
        w = stats["correction"].reshape(-1)
        if mask is not None:
            w = w[mask]
        self.correction_audit.append({"row_id": row.id, "role": row.role, "count": w.numel(),
            "min": float(w.min()) if w.numel() else 0.0,
            "max": float(w.max()) if w.numel() else 0.0,
            "mean": float(w.mean()) if w.numel() else 0.0,
            "ess_fraction": float(w.sum().square() / (w.numel() * w.square().sum())) if w.numel() else 0.0,
            "log_correction_min": float(stats["log_correction"].min()),
            "log_correction_max": float(stats["log_correction"].max()),
            "learning_ratio_max_error_before_update": float((stats["learning_ratio"] - 1).abs().max()),
            "reference_logprobs": reference.detach().cpu().tolist(),
            "behavior_logprobs": behavior.detach().cpu().tolist()})
        return -terms.sum() * row.row_weight / self.divisor

    def _reference_pass(self, rows):
        """Score the actual vLLM actions once before backward; never generate again."""
        references = {}
        for role in ("E", "U", "R"):
            selected = [r for r in rows if r.role == role and r.row_weight and role in self.train_roles]
            # Some hybrid kernels select a different path with no_grad. Score
            # with the exact backward-time forward path, then discard graphs.
            with self.shared.activate(role, train=True):
                if role in ("E", "U"):
                    for row in selected:
                        values, _ = self.generator.recompute([row.record])
                        references[row.id] = values[0].detach().cpu()
                        del values
                else:
                    groups = defaultdict(list)
                    for row in selected:
                        if row.record.forward.forward_count:
                            groups[row.record.forward.id].append(row)
                    for group in groups.values():
                        logits = self.selector.recompute_logits(group[0].record.forward)
                        for row in group:
                            references[row.id] = replay_slate_logprob(
                                logits, row.record, self.selector.tokenizer).detach().cpu()
                        del logits
        return references

    def update(self, rows, iteration, data_state, extra=None):
        import torch

        validate_rows(rows, self.shared.version, self.domain)
        if iteration != int(self.shared.version.split("-")[-1]) + 1:
            raise ValueError("Exactly one on-policy update per jointly frozen batch")
        weights_before = adapter_state(self.shared)
        optimizer_before = {r: copy.deepcopy(o.state_dict()) for r, o in self.optimizers.items()}
        old_version = self.shared.version
        logs = {"losses": {}, "gradient_norms": {}, "parameter_update_norms": {}, "policy_loss": self.policy_loss}
        try:
            self.probability_audit = []
            self.correction_audit = []
            if self.rollout_correction:
                self.references = self._reference_pass(rows)
            for optimizer in self.optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            # No optimizer steps until all role backwards finish. Never switch mid-backward.
            for role in ("E", "U", "R"):
                selected = [row for row in rows if row.role == role and row.row_weight != 0]
                if role not in self.train_roles or not selected:
                    logs["losses"][role] = 0.0
                    continue
                with self.shared.activate(role, train=True):
                    if isinstance(selected[0].record, GenerationRecord):
                        logs["losses"][role] = self._generation_backward(selected)
                    else:
                        logs["losses"][role] = self._selector_backward(selected)
            self.shared.assert_base_unchanged()
            logs["probability_diagnostics"] = self.probability_audit
            if self.rollout_correction:
                logs["rollout_correction"] = {"config": self.rollout_correction,
                                             "rows": self.correction_audit}
            for role, group in self.shared.role_parameters.items():
                params = [p for _, p in group]
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                    raise RuntimeError(f"Nonfinite {role} gradient")
                logs["gradient_norms"][role] = float(
                    torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm, error_if_nonfinite=True)
                )
            for role, optimizer in self.optimizers.items():
                optimizer.step()
                logs["parameter_update_norms"][role] = (
                    sum(
                        float((p.detach().float().cpu() - weights_before[role][n].float()).square().sum())
                        for n, p in self.shared.role_parameters[role]
                    )
                    ** 0.5
                )
            self.shared.assert_base_unchanged()
            path = self.store.publish(
                self.shared, self.optimizers, iteration, data_state, {**(extra or {}), **logs}
            )
            self.shared.version = f"iteration-{iteration}"
            logs["checkpoint"] = str(path)
            return logs
        except Exception:
            restore_adapters(self.shared, weights_before)
            for role, optimizer in self.optimizers.items():
                optimizer.load_state_dict(optimizer_before[role])
                optimizer.zero_grad(set_to_none=True)
            self.shared.version = old_version
            raise
