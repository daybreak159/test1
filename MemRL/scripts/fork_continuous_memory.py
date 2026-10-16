"""Explicitly fork a completed RL boundary into per-replica continuous memory.

Preserves adapters, optimizers, RNG, data cursor and the exact main replica banks.
This is a recorded protocol transition, never a silent same-protocol resume.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'src'))
from eur_memory_rl.config import load_config, protocol_hash
from eur_memory_rl.data.serde import bank
from eur_memory_rl.util import atomic_json, file_hash, plain, read_json


def fork(source_root, iteration, root):
    spec = importlib.util.spec_from_file_location('memrl_launcher', SOURCE / 'start_memrl.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    source_root, root = source_root.resolve(), root.resolve()
    if iteration < 1 or root.exists():
        raise ValueError('Use a completed positive iteration and a new destination')
    cfg, _ = launcher.load_prepared(source_root)
    if cfg.get('memory_continuity', {}).get('enabled'):
        raise ValueError('Already continuous: use normal checkpoint resume instead')
    if cfg['schedule']['plans_per_batch'] != 1 or cfg.get('main_recovery'):
        raise ValueError('Fork requires one completed plan per update without main recovery')
    for n in range(1, iteration + 1):
        for prefix in ('rollout', 'update', 'main-before-local'):
            if not (source_root / f'rl/{prefix}-{n:06d}.json').is_file():
                raise ValueError('Missing completed source history')
    update = read_json(source_root / f'rl/update-{iteration:06d}.json')
    checkpoint = Path(update['checkpoint']).resolve()
    if not checkpoint.is_relative_to(source_root / 'checkpoints'):
        raise ValueError('Source checkpoint outside run')
    manifest = read_json(checkpoint / 'manifest.json')
    if file_hash(checkpoint / 'state.pt') != manifest['state_sha256']:
        raise ValueError('Source checksum mismatch')
    state = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
    if state['iteration'] != iteration or manifest['iteration'] != iteration:
        raise ValueError('Source iteration mismatch')
    if state['protocol_hash'] != protocol_hash(load_config(source_root / 'config.json')):
        raise ValueError('Source checkpoint protocol mismatch')
    data = state['data_state']
    if data['cursor'] != iteration or data['main_batches'] != iteration:
        raise ValueError('Fork supports declared one-plan-per-update cursors only')
    rollout = read_json(source_root / f'rl/rollout-{iteration:06d}.json')
    banks = [bank(episode['bank']) for episode in rollout['main']]
    if len(banks) != cfg['global']['replicas']:
        raise ValueError('Source replica bank count mismatch')
    for index, episode in enumerate(rollout['main']):
        if not episode['id'].endswith(f'/replica-{index}'):
            raise ValueError('Source replica order mismatch')
    public = read_json(Path(cfg['data']['public_manifest']))
    train = [p for p in public['episodes'] if p['split'] == 'train']
    if any(p['scope'] != b.scope for p in train for b in banks):
        raise ValueError('Continuous memory cannot cross scopes')
    prepared = launcher.prepare(source_root / 'config.json', checkpoint, root, continuous_memory=True)
    new_protocol = protocol_hash(load_config(root / 'config.json'))
    migrated = dict(state, protocol_hash=new_protocol,
                    data_state={**data, 'memory_banks': plain(banks)})
    transition = {
        'source_root': str(source_root), 'source_checkpoint': str(checkpoint),
        'source_state_sha256': manifest['state_sha256'], 'iteration': iteration,
        'source_protocol_hash': state['protocol_hash'], 'destination_protocol_hash': new_protocol,
        'preserved': ['adapters', 'optimizers', 'rng', 'cursor', 'main_batches', 'manifest_hash'],
        'replica_banks': [{'replica': i, 'hash': b.hash, 'entries': len(b.entries),
                           'active': len(b.active)} for i, b in enumerate(banks)],
        'credit_scope': 'current group only; no retroactive credit',
        'replay': 'disabled for subsequent groups; first task runs with inherited memory',
        'test_bank': 'fixed replica 0; no final rebuild; no merging',
        'historical_artifacts': 'prior rollout/update files copied verbatim; paths retain their original meaning',
        'source_core': launcher.read_json(source_root / 'launch_provenance.json')['source'],
        'destination_core': launcher.source_identity(SOURCE),
        'source_rollout_sha256': file_hash(source_root / f'rl/rollout-{iteration:06d}.json'),
    }
    migrated['extra'] = {**state.get('extra', {}), 'protocol_transition': transition}
    target = (root / 'checkpoints' / root.name / f"seed-{cfg['seed']}" / cfg['domain'] /
              cfg['architecture']['base_revision'] / 'E_U_R' / f'iteration-{iteration:06d}')
    stage = target.parent / ('.transition-' + target.name)
    stage.mkdir(parents=True)
    torch.save(migrated, stage / 'state.pt')
    destination_sha = file_hash(stage / 'state.pt')
    atomic_json(stage / 'manifest.json', {**manifest, 'protocol_hash': new_protocol,
                                        'state_sha256': destination_sha})
    stage.rename(target)
    atomic_json(target.parent / 'latest.json', {'checkpoint': target.name, 'iteration': iteration})
    (root / 'rl').mkdir(exist_ok=True)
    for n in range(1, iteration + 1):
        for prefix in ('rollout', 'update', 'main-before-local'):
            name = f'rl/{prefix}-{n:06d}.json'
            shutil.copy2(source_root / name, root / name)
    atomic_json(root / 'protocol_transition.json', {**transition, 'checkpoint': str(target),
                                                  'destination_state_sha256': destination_sha})
    print(json.dumps({'root': str(root), 'checkpoint': str(target),
                      'next_max_updates': iteration + 1}, ensure_ascii=False))
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    fork(args.source_root, args.iteration, args.root)
