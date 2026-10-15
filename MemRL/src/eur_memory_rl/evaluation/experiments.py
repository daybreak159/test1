from __future__ import annotations

import copy

from ..util import atomic_json


def ablation_configs(base):
    """Only this algorithm's internal experiments; no external baseline training."""
    variants = {
        "no_memory": {"memory_enabled": False, "train_roles": []},
        "fixed_initial_EUR": {"train_roles": []},
        "EU_RL_R0": {"train_roles": ["E", "U"], "selector_mode": "R0"},
        "R_RL_only": {"train_roles": ["R"]},
        "full_EUR": {"train_roles": ["E", "U", "R"]},
        "outcome_only": {"objective": "outcome_only"},
        "local_only": {"objective": "local_only"},
        "dual_reward": {"objective": "dual"},
        "direct_quality_LOO": {"local_score": "direct_quality"},
        "paired_process_LOO": {"local_score": "paired"},
        "no_old_reference": {"old_reference": False},
        "flat_credit": {"credit": "flat"},
        "conditional_credit": {"credit": "conditional"},
        "shared_lora": {"adapter_layout": "shared"},
        "independent_lora": {"adapter_layout": "independent"},
    }
    variants.update({f"load_{k}": {"max_selected": k} for k in (1, 2, 4, 8)})
    variants.update({f"recall_{n}": {"candidate_slots": n} for n in (8, 12, 16)})
    variants.update({f"stream_length_{n}": {"stream_length": n} for n in (2, 4, 8, 16)})
    results = {}
    for name, patch in variants.items():
        cfg = copy.deepcopy(base)
        cfg.setdefault("experiment", {}).update(patch)
        cfg["experiment"]["name"] = name
        cfg["experiment"]["budget_comparison"] = ["equal_training_tasks", "equal_compute"]
        if name in {"direct_quality_LOO", "paired_process_LOO", "no_old_reference"} and cfg["domain"] in {
            "swe",
            "web",
            "alf",
        }:
            cfg["local"]["budget"] = copy.deepcopy(cfg["consumer"]["profile"]["budget"])
        cfg["output_dir"] = str(base.get("output_dir", "runs")) + "/" + name
        results[name] = cfg
    return results


def write_experiment_plan(base, output):
    atomic_json(
        output,
        {
            "configs": ablation_configs(base),
            "seeds": [11, 23, 37, 51, 71],
            "audits": [
                "oracle_evidence_coverage_only",
                "content_selection_four_cells",
                "same_domain_consumer_transfer",
                "locked_candidates",
            ],
            "external_baseline_retraining": False,
        },
    )


def matched_budget_curves(records, x_key):
    if x_key not in {"training_tasks", "consumer_calls", "total_tokens", "monetary_cost", "wall_seconds"}:
        raise ValueError("Declare a measured common budget axis")
    methods = {}
    for record in records:
        methods.setdefault(record["method"], []).append(record)
    if not methods:
        return {}
    max_common = min(max(r[x_key] for r in runs) for runs in methods.values())
    return {
        method: sorted((r for r in runs if r[x_key] <= max_common), key=lambda r: r[x_key])
        for method, runs in methods.items()
    }


def execute_experiment_plan(plan_path, output_root, methods=None):
    """Explicit sequential launcher. Each process owns a separate domain/seed/checkpoint run."""
    import subprocess
    import sys
    from pathlib import Path
    from ..util import read_json

    plan = read_json(plan_path)
    root = Path(output_root).resolve()
    results = []
    for name, base in plan["configs"].items():
        if methods and name not in methods:
            continue
        for seed in plan["seeds"]:
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            # Preserve the original config-relative deployment/data paths in generated configs.
            config_dir = Path(cfg.pop("_config_dir", Path(plan_path).resolve().parent))
            for section, keys in {
                "data": ("public_manifest", "private_labels", "duplicate_manifest"),
                "runtime": ("dependency_lock", "adapter_gradient_audit", "old_new_logprob_audit"),
                "environment": ("image_manifest", "reset_audit"),
            }.items():
                for key in keys:
                    value = cfg[section].get(key)
                    if value and not Path(value).is_absolute():
                        cfg[section][key] = str(config_dir / value)
            cfg["output_dir"] = str(root / name / f"seed-{seed}" / "results")
            cfg["checkpoint_dir"] = str(root / "checkpoints")
            config_path = root / name / f"seed-{seed}" / "config.json"
            audit_path = root / name / f"seed-{seed}" / "model-audit.json"
            cfg["runtime"]["adapter_gradient_audit"] = str(audit_path)
            cfg["runtime"]["old_new_logprob_audit"] = str(audit_path)
            atomic_json(config_path, cfg)
            command = "train" if cfg["experiment"].get("train_roles") else "initialize"
            if command in {"train", "initialize"}:
                audit = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "eur_memory_rl",
                        "audit-model",
                        str(config_path),
                        "--output",
                        str(audit_path),
                    ],
                    check=False,
                )
                if audit.returncode:
                    raise RuntimeError(f"Model audit failed for {name}, seed {seed}")
            run = subprocess.run(
                [sys.executable, "-m", "eur_memory_rl", command, str(config_path)], check=False
            )
            results.append(
                {"method": name, "seed": seed, "exit_code": run.returncode, "config": str(config_path)}
            )
            atomic_json(root / "execution_manifest.json", results)
            if run.returncode:
                raise RuntimeError(
                    f"Experiment {name}, seed {seed} blocked/failed; see its concrete preflight"
                )
            best_path = Path(cfg["output_dir"]) / "best_checkpoint.json"
            if best_path.is_file():
                checkpoint = read_json(best_path)["checkpoint"]
            else:
                checkpoint_root = (
                    Path(cfg["checkpoint_dir"])
                    / name
                    / f"seed-{seed}"
                    / cfg["domain"]
                    / cfg["architecture"]["base_revision"]
                    / "E_U_R"
                )
                completed = sorted(
                    p for p in checkpoint_root.glob("iteration-*") if (p / "manifest.json").is_file()
                )
                if not completed:
                    raise RuntimeError("Experiment produced no complete checkpoint")
                checkpoint = str(completed[-1])
            evaluation = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eur_memory_rl",
                    "evaluate",
                    str(config_path),
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    str(Path(cfg["output_dir"]) / "test.json"),
                ],
                check=False,
            )
            results[-1]["evaluation_exit_code"] = evaluation.returncode
            results[-1]["evaluated_checkpoint"] = checkpoint
            atomic_json(root / "execution_manifest.json", results)
            if evaluation.returncode:
                raise RuntimeError(f"Locked evaluation failed for {name}, seed {seed}")
    return results
