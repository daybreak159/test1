"""Freeze disjoint SWE main/probe groups and independently verify clean resets.

This prepares data only. Model loading and training use scripts/memrl.sh.
Group selection must be declared from public descriptions before rollout.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eur_memory_rl.contracts import DataIdentity, Task, Turn
from eur_memory_rl.data.manifest import audit_manifests, load_manifest
from eur_memory_rl.domains.swe import SWEAdapter
from eur_memory_rl.services.backends import SWEEnvironment
from eur_memory_rl.util import atomic_json, digest, file_hash, plain, read_json


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def validate_groups(spec, rows, train_ids, test_ids, sft_ids, component_by_id):
    groups = spec['groups']
    if len(groups) != 5 or any(len(g['main']) != 4 or len(g['probes']) != 2 for g in groups):
        raise ValueError('Expected five groups of four main and two probes')
    main = [iid for g in groups for iid in g['main']]
    probes = [iid for g in groups for iid in g['probes']]
    all_ids = main + probes + test_ids
    if len(set(all_ids)) != 42 or len(test_ids) != 12 or set(main + probes) != set(train_ids):
        raise ValueError('Groups must use exactly the original 30 training and 12 test IDs without overlap')
    if len(set(g['id'] for g in groups)) != 5:
        raise ValueError('Group names must be unique')
    if set(sft_ids) & set(test_ids):
        raise ValueError('SFT/test ID overlap')
    components, roles = {}, {}
    for role, ids in [('train', main), ('train_probe', probes), ('test', test_ids)]:
        for iid in ids:
            normalized = ' '.join(rows[iid]['problem_statement'].split())
            keys = ('description:' + digest(normalized), component_by_id.get(iid, 'id:' + iid))
            for key in keys:
                if key in roles and roles[key] != role:
                    raise ValueError('Duplicate public task crosses main/probe/test roles: ' + iid)
                roles[key] = role
            components[iid] = component_by_id.get(iid, keys[0])
    test_components = {components[iid] for iid in test_ids}
    test_descriptions = {digest(' '.join(rows[iid]['problem_statement'].split())) for iid in test_ids}
    for iid in sft_ids:
        if iid not in rows:
            raise ValueError('Cannot audit SFT public description: ' + iid)
        if (component_by_id.get(iid, 'id:' + iid) in test_components
                or digest(' '.join(rows[iid]['problem_statement'].split())) in test_descriptions):
            raise ValueError('SFT/test duplicate component or public-description overlap: ' + iid)
    return main, probes, components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--public-tasks', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--public-pool', type=Path, action='append', default=[])
    parser.add_argument('--sft-manifest', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    spec, split = read_json(args.spec), read_json(args.split)
    pools = [r for path in args.public_pool for r in lines(path)]
    public = lines(args.public_tasks)
    rows = {r['instance_id']: r for r in pools + public}
    components = {r['instance_id']: r['component_id'] for r in pools if 'component_id' in r}
    sft_ids = [r['sample_id'] for r in read_json(args.sft_manifest)['records']]
    train_ids, test_ids = split['train_ids'], split['test_ids']
    main_ids, probe_ids, assigned = validate_groups(spec, rows, train_ids, test_ids, sft_ids, components)
    args.root.mkdir(parents=True, exist_ok=False)
    atomic_json(args.root / 'groups.json', spec)
    cfg, envdoc = read_json(args.config), read_json(args.environment)
    env = envdoc['swe']
    env['initial_observation_profile'] = 'sorted-names-v2'
    env['revision'] = 'eur-swe-clean-digest-stable-observation-v2'
    cfg['consumer']['profile']['harness_revision'] += '+stable-initial-observation-v2'
    env['allowed_images'] = {}
    env['output_artifact_root'] = str(args.root / 'reset_tool_outputs')
    records, tasks = [], {}
    roles = {iid: role for role, ids in [('train', main_ids), ('train_probe', probe_ids), ('test', test_ids)] for iid in ids}
    scope = 'verified30-five-groups-shared-scope'
    for iid in main_ids + probe_ids + test_ids:
        row = rows[iid]
        tag = 'swebench/sweb.eval.x86_64.' + iid.replace('__', '_1776_') + ':latest'
        inspected = subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True, text=True)
        if inspected.returncode:
            if shutil.disk_usage(args.root).free < 32 * 1024**3:
                raise RuntimeError('Less than 32GiB disk free before image pull')
            subprocess.run(['docker', 'pull', tag], check=True)
            inspected = subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True, text=True, check=True)
        pinned = next(v for v in json.loads(inspected.stdout)[0]['RepoDigests'] if v.startswith(tag.split(':')[0] + '@'))
        env['allowed_images'][iid] = pinned
        payload = {'image_key': iid, 'base_commit': row['base_commit']}
        pair = []
        for repeat in range(2):
            instance = SWEEnvironment(payload, 11, env)
            try:
                initial_hash = instance.state_hash
                initial_observation_hash = digest(instance.observation)
                observation, _ = instance.step({'command': 'git rev-parse HEAD && python -c "import sys; print(sys.version)"'}, 60)
                if observation['exit_code'] != 0 or instance.finalize('reset audit')['patch'].strip():
                    raise ValueError('Clean reset audit failed: ' + iid)
                pair.append({'task_id': iid, 'repeat': repeat, 'image': pinned,
                             'initial_state_hash': initial_hash, 'final_state_hash': instance.state_hash,
                             'initial_observation_hash': initial_observation_hash,
                             'observation': observation, 'clean_patch': True})
            finally:
                instance.close()
        if (pair[0]['initial_state_hash'] != pair[1]['initial_state_hash']
                or pair[0]['initial_observation_hash'] != pair[1]['initial_observation_hash']
                or pair[0]['observation'] != pair[1]['observation']):
            raise ValueError('Independent resets differ: ' + iid)
        records.extend(pair)
        atomic_json(args.root / 'reset_progress.json', {'completed_tasks': len(tasks) + 1, 'records': records})
        payload['expected_initial_state_hash'] = pair[0]['initial_state_hash']
        payload['expected_initial_observation_hash'] = pair[0]['initial_observation_hash']
        cert = {}
        if roles[iid] == 'train_probe':
            group = next(g for g in spec['groups'] if iid in g['probes'])
            cert = {'prefix_answerable': True, 'validated_before_sampling': True, 'cutoff': 0,
                    'validation_version': 'independent-public-issue-clean-reset-v2',
                    'probe_kind': 'related' if iid == group['probes'][0] else 'retention',
                    'source_task_ids': group['main'], 'reset_hash': pair[0]['initial_state_hash'],
                    'rationale': group['rationale'] + ' Independent public issue; not a solved certificate.'}
        identity = DataIdentity('swe_bench', row['version'], 'verified30-five-4plus2-v1',
                                roles[iid], iid, assigned[iid], scope, (), (), 0,
                                pair[0]['initial_state_hash'], iid)
        tasks[iid] = plain(Task(identity, (Turn(iid, iid + '/external-0', row['problem_statement'], public_time='2026-10-15'),),
                                payload, probe_certificate=cert))
        print('RESET', len(tasks), iid, flush=True)
    episodes = [{'id': g['id'], 'scope': scope, 'split': 'train', 'sources': [],
                 'tasks': [tasks[iid] for iid in g['main']]} for g in spec['groups']]
    episodes += [{'id': 'frozen-' + iid, 'scope': scope, 'split': 'test', 'sources': [],
                  'tasks': [tasks[iid]]} for iid in test_ids]
    manifest = {'format': 'eur-public-manifest-v1', 'domain': 'swe', 'dataset': 'swe_bench',
                'episodes': episodes, 'probes': [tasks[iid] for iid in probe_ids]}
    duplicates = {'reviewed': True, 'version': 'public-components-and-normalized-description-v2',
                  'assignments': {'swe_bench:' + iid: c for iid, c in assigned.items()}}
    for name, value in [('public_manifest', manifest), ('duplicate_manifest', duplicates),
                        ('private_labels', {'labels': {}}), ('environment', envdoc),
                        ('reset_audit', {'passed': True, 'complete_state_verified': True, 'records': records})]:
        atomic_json(args.root / (name + '.json'), value)
    cfg['schedule'].update(batches=5, plans_per_batch=1, dev_interval=100, require_declared_plans=True)
    cfg['experiment']['name'] = 'verified30-five-4plus2-hf-pg-v1'
    cfg['local'].update(pin_probe_backend=True, min_legal_candidates=2)
    cfg['local_solver_endpoints'] = spec['local_solver_endpoints']
    cfg['judge'].update(structured_output=True, repair_attempts=2, rubric_version='eur-pair-public-evidence-schema-repair-v2')
    cfg['environment'].update(image_manifest=str(args.root / 'environment.json'), reset_audit=str(args.root / 'reset_audit.json'))
    for key in ('public_manifest', 'private_labels', 'duplicate_manifest'):
        cfg['data'][key] = str(args.root / (key + '.json'))
        cfg['data'][key + '_sha256'] = file_hash(Path(cfg['data'][key]))
    # Resolve shared dependency lock before launcher migrates the config.
    lock = Path(cfg['runtime']['dependency_lock'])
    if not lock.is_absolute():
        cfg['runtime']['dependency_lock'] = str((args.config.parent / lock).resolve())
    cfg.pop('main_recovery', None)
    cfg.pop('first_task_replay', None)
    atomic_json(args.root / 'config.json', cfg)
    bundle = load_manifest(args.root / 'public_manifest.json', SWEAdapter(None, None, None))
    audit = audit_manifests([bundle], args.root / 'duplicate_manifest.json')
    audit.update(sft_test_id_overlap=[], sft_test_component_or_description_overlap=[],
                 sft_train_id_overlap=sorted(set(sft_ids) & set(train_ids)),
                 source_hashes={str(p): file_hash(p) for p in [args.spec, args.public_tasks, args.split, args.sft_manifest, *args.public_pool]},
                 main_order=main_ids, probe_order=probe_ids, test_order=test_ids)
    atomic_json(args.root / 'data_checks.json', audit)
    print('PREPARED', args.root, flush=True)


if __name__ == '__main__':
    main()
