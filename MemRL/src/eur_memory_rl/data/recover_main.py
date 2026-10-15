"""Rebuild an unupdated main group from immutable execution/generation logs.

Only the local evaluation budget may change. No previous local scores are reused.
"""
from pathlib import Path
import json

from ..contracts import Execution, GenerationRecord, Grade
from ..policies.prompts import encode_messages
from ..util import atomic_json, digest, file_hash, read_json, plain


def recover_main_group(rt, plan, replicas, seed, debug_zero_baseline, selector_mode, recovery):
    old = Path(recovery['root'])
    previous = read_json(old / 'config.json')
    for key in ('architecture', 'generation', 'selector', 'consumer', 'memory', 'recall',
                'experiment', 'global', 'seed', 'source_packing', 'memory_feedback',
                'retrieval', 'domain'):
        before, after = previous.get(key), rt.cfg.get(key)
        if key == 'experiment':
            before = {k: v for k, v in before.items() if k != 'name'}
            after = {k: v for k, v in after.items() if k != 'name'}
        if before != after:
            raise ValueError('Main recovery changed policy/task setting: ' + key)
    for name, expected in recovery['sha256'].items():
        if file_hash(old / name) != expected:
            raise ValueError('Main recovery log changed: ' + name)
    initial = read_json(old / 'initialization.json')['initial_checkpoint']
    expected_weights = file_hash(Path(initial) / 'state.pt')
    import torch
    from ..training.checkpoint import adapter_state
    saved_manifest = read_json(Path(initial) / 'manifest.json')
    if saved_manifest['state_sha256'] != expected_weights:
        raise ValueError('Saved initial checkpoint integrity failure')
    bundle = torch.load(Path(initial) / 'state.pt', map_location='cpu', weights_only=False)
    if bundle['iteration'] != 0 or bundle['base_hash'] != rt.shared.base_hash:
        raise ValueError('Recovery requires an unupdated identical base model')
    actual = adapter_state(rt.shared)
    if set(actual) != set(bundle['adapters']) or any(
        set(actual[r]) != set(bundle['adapters'][r]) or any(
            not torch.equal(t, bundle['adapters'][r][n]) for n, t in actual[r].items())
        for r in actual
    ):
        raise ValueError('Main recovery adapter weights differ from original behavior policy')
    if any((old / 'rl').glob('update-*.json')) or any(
        int(p.name.split('-')[-1]) > 0 for p in (old / 'checkpoints').rglob('iteration-*')
        if p.is_dir()
    ):
        raise ValueError('Recovery source already contains parameter updates')
    for key in ('public_manifest', 'private_labels', 'duplicate_manifest'):
        if file_hash(previous['data'][key]) != file_hash(rt.cfg['data'][key]):
            raise ValueError('Recovery dataset changed: ' + key)
    # Both runs must initialize from precisely the same trusted SFT file.
    if read_json(old / 'launch_provenance.json')['sft_checkpoint'] != recovery['sft_checkpoint']:
        raise ValueError('Main recovery SFT source mismatch')
    read_lines = lambda name: [json.loads(x) for x in (old / name).read_text().splitlines() if x.strip()]
    executions = {x['execution']['execution_id']: x['execution'] for x in read_lines('executions.jsonl')}
    grades = {x['execution_id']: x['grade'] for x in read_lines('grades.jsonl')}
    generations = read_lines('generations.jsonl')
    consumed_ids, generated_ids = [], []
    original_consumer, original_grade, original_generate = rt.adapter.run_consumer, rt.adapter.grade, rt.generator.generate

    def consumed(task, execution_id, seed, read, budget=None):
        if budget is not None:
            raise ValueError('Recovery attempted to reuse a local branch')
        raw = executions[execution_id]
        if raw['task_id'] != task.id or raw['seed'] != seed or raw['consumer_profile_hash'] != rt.adapter.profile.hash:
            raise ValueError('Saved main execution identity mismatch')
        slate = read(task.turns[0])
        first = raw['public_trace'][0]
        exposure = digest([task.turns[0], slate.memory_block, first['observation']])
        if digest([[exposure], rt.adapter.profile.budget, rt.adapter.profile.hash]) != raw['input_hash']:
            raise ValueError('Reconstructed R selection changed actual Solver input')
        consumed_ids.append(execution_id)
        return Execution(**{**raw, 'public_trace': tuple(raw['public_trace'])})

    def graded(task, execution):
        return Grade(**grades[execution.execution_id])

    def generated(role, messages, seed, deterministic=False):
        raw = generations[len(generated_ids)]['record']
        ids, input_hash, _ = encode_messages(rt.shared.tokenizer, rt.generator.prepare_messages(role, messages),
                                             rt.generator.input_limit(role))
        if raw['role'] != role or list(ids) != raw['input_ids'] or input_hash != raw['input_hash']:
            raise ValueError('Saved E/U prompt mismatch; refusing to sample a replacement')
        if raw['policy_version'] != rt.shared.version:
            raise ValueError('Saved E/U policy version mismatch')
        generated_ids.append({'role': role, 'input_hash': input_hash})
        return GenerationRecord(**{**raw, **{k: tuple(raw[k]) for k in
            ('input_ids', 'response_token_ids', 'response_mask', 'old_logprobs')}})

    rt.adapter.run_consumer, rt.adapter.grade, rt.generator.generate = consumed, graded, generated
    try:
        group = rt.stream.run_group(plan, replicas, seed, debug_zero_baseline, selector_mode)
        expected = {f'{plan.id}/replica-{i}/{e.task.id}' for i in range(replicas) for e in plan.events if e.task}
        if set(consumed_ids) != expected:
            raise ValueError('Main recovery did not restore every declared execution')
        # Verify rebuilt transactions, including invalid writes, against the old main records.
        validations = read_lines('memory_validation.jsonl')[:len(generated_ids)]
        rows = [row for episode in group for row in episode.rows if row.role in ('E', 'U')]
        if len(rows) != len(validations):
            raise ValueError('Incomplete recovered E/U rows')
        for row, saved in zip(rows, validations):
            if plain(row.record) != saved['record']:
                raise ValueError('Reconstructed E/U validation changed')
            if row.role == 'U' and row.bank_after_hash != digest(saved['transaction']['bank']):
                # Bank.hash includes its canonical dataclass representation.
                from .serde import bank
                if row.bank_after_hash != bank(saved['transaction']['bank']).hash:
                    raise ValueError('Reconstructed U memory bank changed')
        atomic_json(Path(rt.cfg['output_dir']) / 'main_recovery.json', {
            'source': str(old), 'log_sha256': recovery['sha256'], 'execution_ids': consumed_ids,
            'generations': generated_ids, 'new_solver_calls_for_main': 0,
            'main_costs': 'historical costs retained for original rewards; not newly incurred',
            'R': 'recomputed with unchanged weights/seeds; actual memory injection input hashes verified',
            'initial_state_sha256': expected_weights, 'old_local_results_reused': False,
        })
        return group
    finally:
        rt.adapter.run_consumer, rt.adapter.grade, rt.generator.generate = original_consumer, original_grade, original_generate
