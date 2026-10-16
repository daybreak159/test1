#!/usr/bin/env python3
"""Entry point for the existing coding deployment; no login or package installation."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

CODING = Path('/home/ubuntu/coding')
PROJECT = CODING / 'MemRL'
RESULTS = PROJECT / '03_实验结果'
DEFAULT_SFT = RESULTS / 'checkpoints/E文本SFT30_RL初始化权重'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def referenced(value, base):
    return (base / Path(value)).resolve()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_files(cfg, base, checkpoint):
    paths = [referenced(cfg['data'][k], base) for k in
             ('public_manifest', 'private_labels', 'duplicate_manifest')]
    env_path = referenced(cfg['environment']['image_manifest'], base)
    paths += [env_path, referenced(cfg['environment']['reset_audit'], base),
              referenced(cfg['runtime']['dependency_lock'], base),
              checkpoint / 'manifest.json', checkpoint / 'state.pt']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    for key in ('public_manifest', 'private_labels', 'duplicate_manifest'):
        if file_sha(referenced(cfg['data'][key], base)) != cfg['data'][key + '_sha256']:
            raise ValueError(f'任务清单校验失败：{key}')
    if not Path(cfg['architecture']['base_model']).is_dir():
        raise FileNotFoundError(cfg['architecture']['base_model'])
    if read_json(checkpoint / 'manifest.json').get('domain') != 'swe':
        raise ValueError('初始化checkpoint必须属于SWE；完整权重兼容性由Runtime检查')
    env = read_json(env_path)
    for key in ('python', 'dataset_name'):
        if not Path(env['grader'][key]).is_file():
            raise FileNotFoundError(env['grader'][key])
    return env


def prepare(config_path, checkpoint, root, recover_main_from=None, continuous_memory=False):
    """Copy input manifests without modifying their bytes or historical files."""
    cfg = deepcopy(read_json(config_path))
    if continuous_memory:
        cfg['memory_continuity'] = {'enabled': True, 'replica_mapping': 'same-index-v1',
                                    'credit_scope': 'current-group', 'test_bank_replica': 0}
    configure_solver(cfg)
    base = config_path.parent
    env = validate_files(cfg, base, checkpoint)
    root.mkdir(parents=True, exist_ok=False)
    for key in ('public_manifest', 'private_labels', 'duplicate_manifest'):
        destination = root / (key + '.json')
        shutil.copy2(referenced(cfg['data'][key], base), destination)
        cfg['data'][key] = str(destination)
    for section, key, name in [('environment', 'reset_audit', 'reset_audit.json'),
                               ('runtime', 'dependency_lock', 'dependency_lock.json')]:
        destination = root / name
        shutil.copy2(referenced(cfg[section][key], base), destination)
        cfg[section][key] = str(destination)
    cfg.pop('_config_dir', None)
    cfg['experiment']['name'] = root.name
    cfg['rollout'] = {'backend': 'hf'}
    cfg['optimization'].pop('rollout_correction', None)
    cfg.pop('first_task_replay', None)
    if cfg.get('first_task_replay_by_task'):
        from eur_memory_rl.data.replay_swe import validate_replay_map
        validate_replay_map(cfg, read_json(Path(cfg['data']['public_manifest'])))
    policy_loss = cfg['optimization'].setdefault('policy_loss', 'single_step_pg')
    if policy_loss not in {'single_step_pg', 'ppo'}:
        raise ValueError('Unknown optimization.policy_loss')
    cfg['generation'].update(prompt_profile='e-text-v1', use_generation_cache=policy_loss == 'single_step_pg')
    cfg['generation']['input_limits'] = {'E': 32768, 'U': 8192}
    cfg['architecture'].update(gradient_checkpointing=True, activation_offload=True, mlp_chunk_tokens=1024)
    cfg['runtime'].update(hf_attention_backend='flash-gqa', hf_pointwise_chunk_tokens=512)
    for key in ('adapter_gradient_audit', 'old_new_logprob_audit'):
        cfg['runtime'][key] = str(root / 'model_audit.json')
    cfg['output_dir'], cfg['checkpoint_dir'] = str(root / 'rl'), str(root / 'checkpoints')
    cfg['e_preprocessing']['output_dir'] = str(root / 'e_preprocessing')
    cfg['e_preprocessing']['config'].update(max_input_tokens=32768, prompt_profile='e-text-v1')
    if recover_main_from is not None:
        old = recover_main_from.resolve()
        cfg['main_recovery'] = {
            'root': str(old), 'sft_checkpoint': str(checkpoint),
            'sha256': {name: file_sha(old / name) for name in (
                'config.json', 'executions.jsonl', 'grades.jsonl', 'generations.jsonl',
                'memory_validation.jsonl', 'initialization.json', 'launch_provenance.json')},
        }
    cfg['environment']['image_manifest'] = str(root / 'environment_config.json')
    env['grader']['artifact_root'], env['swe']['output_artifact_root'] = str(root / 'official_evaluator'), str(root / 'tool_outputs')
    for name, value in [('environment_config.json', env), ('config.json', cfg), ('effective_config.json', cfg)]:
        write_json(root / name, value)
    write_json(root / 'launch_provenance.json', {
        'source_config': str(config_path), 'source_config_sha256': file_sha(config_path),
        'sft_checkpoint': str(checkpoint), 'config_fingerprint': fingerprint(cfg),
        'environment_fingerprint': fingerprint(env),
        'source': source_identity(Path(__file__).resolve().parent),
        'note': ('HF only; explicit continuous-memory fork; initialization recorded in protocol_transition.json'
                 if continuous_memory else
                 'HF only; fresh optimizer/bank/cursor; budgets and dataset preserved; declared first-task replay only'),
    })
    return cfg


def configure_solver(cfg):
    """Use the same effective Solver protocol as Runtime, before hashing."""
    if cfg.get('domain') != 'swe':
        return
    source_src = str(Path(__file__).resolve().parent / 'src')
    if source_src not in sys.path:
        sys.path.insert(0, source_src)
    from eur_memory_rl.domains.mini_swe import configure_swe_runner
    from eur_memory_rl.config import validate_solver_parallelism

    configure_swe_runner(cfg)
    validate_solver_parallelism(cfg)


def source_identity(source):
    """Record both the Git identity and exact core bytes, including uncommitted work."""
    files = [path for folder in ('src', 'scripts') for path in (source / folder).rglob('*')
             if path.is_file() and path.suffix in ('.py', '.sh') and '__pycache__' not in path.parts]
    files += [path for path in (source / 'start_memrl.py', source / 'pyproject.toml') if path.is_file()]
    result = {'core_sha256': fingerprint({str(path.relative_to(source)): file_sha(path) for path in sorted(files)})}
    try:
        result['git_commit'] = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                               check=True, capture_output=True, text=True).stdout.strip()
        result['git_dirty'] = bool(subprocess.run(['git', '-C', str(source), 'status', '--porcelain'],
                                                 check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        result['git_commit'] = None
    return result


def load_prepared(root, checkpoint=None):
    provenance = read_json(root / 'launch_provenance.json')
    cfg, env = read_json(root / 'config.json'), read_json(root / 'environment_config.json')
    if (fingerprint(cfg) != provenance['config_fingerprint'] or
            fingerprint(env) != provenance['environment_fingerprint']):
        raise ValueError('准备好的配置已变更；请从新配置创建新运行目录')
    saved = Path(provenance['sft_checkpoint'])
    if checkpoint and checkpoint.resolve() != saved:
        raise ValueError('准备目录的SFT权重与参数不一致')
    validate_files(cfg, root, saved)
    return cfg, saved


def ensure_unstarted(root):
    if any((root / name).exists() for name in ('summary.json', 'checkpoints', 'initialization.json')):
        raise ValueError('该目录已经运行过；请创建新目录。此入口不是RL断点续训入口。')


def http_json(url, payload=None, key_env=None, timeout=30):
    headers = {'Content-Type': 'application/json'}
    if key_env:
        headers['Authorization'] = 'Bearer ' + os.environ[key_env]
    request = Request(url, data=None if payload is None else json.dumps(payload).encode(), headers=headers)
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def check_environment(cfg, exact=True):
    result = http_json(cfg['environment']['base_url'].rstrip('/') + '/health', timeout=5)
    if result.get('service') != 'eur-environment':
        raise ValueError('端口监听者不是新版MemRL环境服务')
    if exact and result.get('config_fingerprint') != fingerprint(read_json(Path(cfg['environment']['image_manifest']))):
        raise ValueError('环境服务配置与本轮不匹配；请使用本轮environment_config.json启动')
    if cfg.get('global', {}).get('main_parallel_workers', 1) > 1:
        if result.get('concurrency_protocol') != 'per-execution-v1':
            raise ValueError('旧环境服务不支持独立任务并行；请用新源码启动本轮环境服务')
    return result


def environment_smoke(cfg, base):
    manifest = read_json(referenced(cfg['data']['public_manifest'], base))
    task = next(task for episode in manifest['episodes'] for task in episode['tasks'])
    url, key = cfg['environment']['base_url'].rstrip('/'), cfg['environment'].get('api_key_env')
    result = http_json(url + '/reset', {
        'domain': 'swe', 'execution_id': 'handoff-check-' + datetime.now().strftime('%Y%m%d%H%M%S%f'),
        'task_id': task['identity']['sample_id'], 'seed': cfg['seed'], 'public_payload': task['public_payload'],
    }, key, timeout=240)
    session = result['session_id']
    try:
        if result['state_hash'] != task['public_payload']['expected_initial_state_hash']:
            raise ValueError('初始环境hash不一致')
        observation = http_json(url + '/step', {
            'session_id': session, 'action': {'command': 'git rev-parse HEAD; git status --porcelain'},
            'timeout_seconds': 30,
        }, key, timeout=60)['observation']
        if observation['exit_code'] != 0:
            raise ValueError('环境命令失败')
        final = http_json(url + '/finalize', {'session_id': session, 'answer': ''}, key, timeout=60)
        if final['artifact']['patch']:
            raise ValueError('只读环境检查意外产生补丁')
        return {'task_id': task['identity']['sample_id'], 'initial_state_matches': True,
                'command_exit_code': 0, 'patch_empty': True, 'official_grading_executed': False}
    finally:
        http_json(url + '/close', {'session_id': session}, key, timeout=30)


def run_checks(cfg, base, checkpoint, require_environment=True, exact_environment=True, smoke=False):
    checks = []

    def check(name, action, required=True):
        try:
            checks.append({'name': name, 'ok': True, 'detail': action(), 'required': required})
        except Exception as exc:
            checks.append({'name': name, 'ok': False, 'detail': f'{type(exc).__name__}: {exc}', 'required': required})

    check('files_and_manifest_hashes', lambda: bool(validate_files(cfg, base, checkpoint)))
    check('dependencies', lambda: {name: importlib.metadata.version(name) for name in
                                  ('torch', 'transformers', 'peft', 'fastapi', 'uvicorn', 'causal-conv1d')})
    configure_solver(cfg)
    from eur_memory_rl.config import validate_solver_parallelism

    check('solver_parallelism', lambda: validate_solver_parallelism(cfg))
    if cfg.get('domain') == 'swe':
        from eur_memory_rl.domains.mini_swe import check_worker

        check('mini_swe_agent', lambda: check_worker(cfg['consumer']['mini_swe']['python']))

    def command(argv):
        return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=30).stdout.strip()

    check('gpu', lambda: command(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader']))
    check('docker', lambda: command(['docker', 'info', '--format', '{{.ServerVersion}}']))

    def images():
        env = read_json(referenced(cfg['environment']['image_manifest'], base))
        missing = []
        for task, image in env['swe']['allowed_images'].items():
            result = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, timeout=15)
            if result.returncode:
                missing.append(task)
        if missing and cfg['environment'].get('readiness_mode') == 'per-group-v1':
            manifest = read_json(referenced(cfg['data']['public_manifest'], base))
            first = next(episode for episode in manifest['episodes'] if episode['split'] == 'train')
            required = {task['identity']['query_id'] or task['identity']['sample_id'] for task in first['tasks']}
            main_ids = set(required)
            required.update(probe['identity']['query_id'] or probe['identity']['sample_id']
                            for probe in manifest['probes']
                            if set(probe['probe_certificate']['source_task_ids']) & main_ids)
            if required & set(missing):
                raise ValueError('首组镜像缺失：' + ', '.join(sorted(required & set(missing))))
            return {'available': len(env['swe']['allowed_images']) - len(missing),
                    'deferred_until_group_gate': missing, 'first_group_available': True}
        if missing:
            raise ValueError('本地缺少镜像：' + ', '.join(missing))
        return {'available': len(env['swe']['allowed_images'])}

    check('docker_images', images)

    def solver(base_url=None):
        result = http_json((base_url or cfg['consumer']['base_url']).rstrip('/') + '/chat/completions', {
            'model': cfg['consumer']['profile']['model'], 'messages': [{'role': 'user', 'content': 'Reply only OK.'}],
            'max_tokens': 8, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False},
        }, cfg['consumer'].get('api_key_env'), timeout=60)
        content = result['choices'][0]['message'].get('content')
        if not content:
            raise ValueError('Solver正文为空')
        return {'content': content}

    def embedding():
        result = http_json(cfg['retrieval']['base_url'].rstrip('/') + '/embeddings', {
            'model': cfg['retrieval']['served_model'], 'input': ['MemRL deployment check'],
        }, cfg['retrieval'].get('api_key_env'), timeout=60)
        vector = result['data'][0]['embedding']
        norm = math.sqrt(sum(x*x for x in vector))
        if len(vector) != cfg['retrieval']['dimension'] or not all(math.isfinite(x) for x in vector) or norm == 0:
            raise ValueError('Embedding维度、有限值或范数不合格')
        return {'dimension': len(vector), 'norm': norm}

    check('solver_api', solver)
    for index, endpoint in enumerate(cfg.get('local_solver_endpoints', []), 1):
        check(f'solver_worker_{index}_api', lambda endpoint=endpoint: solver(endpoint))
    check('embedding_api', embedding)
    check('environment_api', lambda: check_environment(cfg, exact_environment), require_environment)
    if smoke:
        check('environment_reset_step_finalize_close', lambda: environment_smoke(cfg, base))
    return {'ready': all(row['ok'] for row in checks if row['required']), 'checks': checks}


def subprocess_environment(source):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(source / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    for key, value in [('HF_HUB_OFFLINE', '1'), ('TRANSFORMERS_OFFLINE', '1'),
                       ('PYTORCH_ALLOC_CONF', 'expandable_segments:True')]:
        env.setdefault(key, value)
    return env


@contextmanager
def environment_service(cfg, root, source, start):
    process, log = None, None
    try:
        if start:
            url = urlsplit(cfg['environment']['base_url'])
            if url.hostname != '127.0.0.1' or not url.port:
                raise ValueError('自动启动仅支持127.0.0.1及显式端口')
            with socket.socket() as probe:
                occupied = probe.connect_ex((url.hostname, url.port)) == 0
            if occupied:
                check_environment(cfg)
            else:
                log = (root / 'environment.log').open('a', encoding='utf-8')
                command = [sys.executable, '-m', 'eur_memory_rl', 'serve-environments',
                           str(root / 'environment_config.json'), '--host', url.hostname, '--port', str(url.port)]
                process = subprocess.Popen(command, cwd=source, env=subprocess_environment(source),
                                           stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'环境服务退出；查看{root / "environment.log"}')
                    try:
                        check_environment(cfg)
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f'环境启动超时；查看{root / "environment.log"}')
                        time.sleep(0.25)
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log:
            log.close()


@contextmanager
def gpu_lock(device):
    import fcntl
    directory = RESULTS / 'runs/.locks'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ('.launcher-' + device.replace(':', '-') + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('另一个统一入口正在使用该GPU；请等待其结束') from exc
        yield


def main():
    p = argparse.ArgumentParser(description='MemRL纯HF入口；默认仅启动验收，不做题或更新参数。')
    p.add_argument('--source-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--config', type=Path, help='完整配置；默认源码包configs/coding_hf.json')
    p.add_argument('--sft-checkpoint', type=Path, help='默认当前coding的E文本SFT权重')
    p.add_argument('--recover-main-from', type=Path, help='从未更新的同策略运行恢复首个完整主组；局部分支全部重跑')
    p.add_argument('--full-suite', action='store_true', help='完成全部训练组后重建只读记忆库并统一测试')
    p.add_argument('--max-updates', type=int, help='本次累计完成指定组数后退出；尚未完成全部组时不运行测试')
    p.add_argument('--resume-checkpoint', type=Path, help='配合--prepared-root --train，从同协议完整RL checkpoint恢复优化器和游标')
    directories = p.add_mutually_exclusive_group()
    directories.add_argument('--root', type=Path, help='新目录；已有目录拒绝使用')
    directories.add_argument('--prepared-root', type=Path, help='复用仅准备过的目录；不用于RL断点续训')
    p.add_argument('--start-environment', action='store_true', help='启动/复用匹配的环境服务；结束后仅停止自己启动的服务')
    p.add_argument('--report', type=Path, help='--check的JSON报告路径')
    p.add_argument('--environment-smoke', action='store_true', help='配合--check，创建一个临时题目容器做只读检查')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true', help='正式任务流和参数更新')
    mode.add_argument('--startup-only', action='store_true', help='启动验收（也是默认行为）')
    mode.add_argument('--prepare-only', action='store_true', help='仅整理配置，不加载GPU模型')
    mode.add_argument('--check', action='store_true', help='文件/依赖/GPU/Docker/实际模型API检查，不加载训练模型')
    mode.add_argument('--serve-environment', action='store_true', help='前台运行准备目录的环境服务')
    a = p.parse_args()
    if a.max_updates is not None and (not a.train or a.max_updates < 1):
        p.error('--max-updates需要--train及正整数')
    if a.full_suite and not a.train:
        p.error('--full-suite需要--train')
    if a.resume_checkpoint and (not a.prepared_root or not a.train or a.recover_main_from):
        p.error('--resume-checkpoint需要--prepared-root --train，不能与主轨迹恢复混用')
    if a.environment_smoke and not a.check:
        p.error('--environment-smoke仅用于--check')
    if a.start_environment and (a.prepare_only or a.serve_environment):
        p.error('--start-environment不能与--prepare-only/--serve-environment同时使用')
    if a.serve_environment and not a.prepared_root:
        p.error('--serve-environment需要--prepared-root')
    if a.check and (a.root or (a.start_environment and not a.prepared_root)):
        p.error('--check不创建目录；启动环境请先准备目录并指定--prepared-root')
    source = a.source_dir.resolve()
    runner = source / 'scripts/run_hf_orm.py'
    if not runner.is_file():
        p.error(f'找不到源码入口：{runner}')
    checkpoint = Path(a.sft_checkpoint).resolve() if a.sft_checkpoint else None
    root = a.prepared_root.resolve() if a.prepared_root else None
    if root:
        if a.recover_main_from:
            p.error('--recover-main-from只能用于创建新目录')
        if a.config:
            p.error('--prepared-root已确定配置，不能再指定--config')
        cfg, checkpoint = load_prepared(root, checkpoint)
    else:
        config = (a.config or source / 'configs/coding_hf.json').resolve()
        checkpoint = checkpoint or DEFAULT_SFT
        cfg = read_json(config)
    if a.check:
        with environment_service(cfg, root, source, a.start_environment):
            report = run_checks(cfg, root or config.parent, checkpoint, exact_environment=bool(root),
                                smoke=a.environment_smoke)
        if a.report:
            a.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(a.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report['ready'] else 1)
    if root is None:
        root = (a.root or RESULTS / 'runs/memrl' / ('hf-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
        cfg = prepare(config, checkpoint, root, a.recover_main_from)
    print(f'独立运行目录：{root}', flush=True)
    if a.prepare_only:
        return
    if a.serve_environment:
        url = urlsplit(cfg['environment']['base_url'])
        command = [sys.executable, '-m', 'eur_memory_rl', 'serve-environments',
                   str(root / 'environment_config.json'), '--host', url.hostname, '--port', str(url.port)]
        raise SystemExit(subprocess.call(command, cwd=source, env=subprocess_environment(source)))
    if not a.resume_checkpoint:
        ensure_unstarted(root)
    else:
        resume = a.resume_checkpoint.resolve()
        if not resume.is_relative_to(root / 'checkpoints'):
            raise ValueError('RL恢复仅允许本运行目录的完整checkpoint')
        manifest = read_json(resume / 'manifest.json')
        if manifest['iteration'] < 1 or file_sha(resume / 'state.pt') != manifest['state_sha256']:
            raise ValueError('需要完整且校验通过的RL checkpoint')
        source_now = source_identity(source)
        source_before = read_json(root / 'launch_provenance.json')['source']
        if source_now['core_sha256'] != source_before['core_sha256']:
            raise ValueError('源码已变更；不得把新源码静默混入原RL协议恢复')
    command = [sys.executable, '-u', str(runner), '--root', str(root)]
    if a.resume_checkpoint:
        command += ['--resume-checkpoint', str(resume)]
    else:
        command += ['--sft-checkpoint', str(checkpoint), '--sft-weights-only']
    if not a.train:
        command.append('--startup-only')
    if a.full_suite:
        command.append('--full-suite')
    if a.max_updates is not None:
        command += ['--max-updates', str(a.max_updates)]
    command_name = ('resume-command-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.json'
                    if a.resume_checkpoint else 'launch_command.json')
    write_json(root / command_name, {'command': command, 'train': a.train})
    with gpu_lock(cfg['architecture']['device']), environment_service(cfg, root, source, a.start_environment):
        report = run_checks(cfg, root, checkpoint, require_environment=a.train)
        write_json(root / 'service_checks.json', report)
        if not report['ready']:
            raise RuntimeError(f'依赖或服务检查失败；查看{root / "service_checks.json"}')
        print(f'启动中；进度查看：{root / "launcher.log"}', flush=True)
        with (root / 'launcher.log').open('a' if a.resume_checkpoint else 'w', encoding='utf-8') as log:
            result = subprocess.run(command, env=subprocess_environment(source), cwd=source,
                                    stdout=log, stderr=subprocess.STDOUT)
    print(f'退出码：{result.returncode}；日志：{root / "launcher.log"}', flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
