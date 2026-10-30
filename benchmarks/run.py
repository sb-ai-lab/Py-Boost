#!/usr/bin/env python3
"""Run a frozen GPU suite against one or more Py-Boost source checkouts."""

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

if __package__:
    from .protocol import (ROOT, SCHEMA_VERSION, comparison_errors, digest_json,
                           implementation_hash, load_suite, make_data, quality, summarize)
    from .gpu import RssPeak, memory_tracker
else:
    from protocol import (ROOT, SCHEMA_VERSION, comparison_errors, digest_json,
                          implementation_hash, load_suite, make_data, quality, summarize)
    from gpu import RssPeak, memory_tracker


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def source_info(source):
    source = Path(source).resolve()
    if not (source / 'py_boost' / '__init__.py').is_file():
        raise ValueError(f'Not a Py-Boost checkout: {source}')
    paths = ['py_boost', 'pyproject.toml']
    if git(source, 'status', '--porcelain', '--untracked-files=all', '--', *paths):
        raise ValueError(f'Commit library changes before benchmarking: {source}')
    digest = hashlib.sha256()
    for name in git(source, 'ls-files', '--', *paths).splitlines():
        digest.update(name.encode())
        digest.update((source / name).read_bytes())
    return {'path': str(source), 'commit': git(source, 'rev-parse', 'HEAD'),
            'library_hash': digest.hexdigest()}


def environment(cp, np, psutil, device):
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('Exactly one visible CUDA device is required')
    cp.zeros(1).sum().item()  # A real allocation and kernel; no CPU fallback.
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name']
    gpu_details = subprocess.check_output(
        ['nvidia-smi', '-i', device,
         '--query-gpu=uuid,driver_version,pci.bus_id,memory.total,power.limit,clocks.max.sm',
         '--format=csv,noheader,nounits'], text=True).strip()
    if '\n' in gpu_details:
        raise RuntimeError('GPU selector resolved to more than one physical device')
    uuid, driver_version, pci_bus_id, memory_mib, power_limit, max_sm_clock = [
        part.strip() for part in gpu_details.split(',')]
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        text=True)
    busy = [line for line in processes.splitlines()
            if line.split(',')[0].strip() == uuid and line.split(',')[1].strip() != str(os.getpid())]
    if busy:
        raise RuntimeError(f'GPU is in use by other compute processes: {busy}')
    packages = {}
    for distribution in importlib.metadata.distributions():
        package = distribution.metadata['Name']
        if package and package.lower().replace('_', '-') in {
            'numpy', 'scipy', 'numba', 'llvmlite', 'scikit-learn', 'joblib', 'pandas',
            'cupy', 'cupy-cuda11x', 'cupy-cuda12x', 'cupy-cuda13x', 'psutil',
            'treelite', 'treelite-runtime', 'onnx', 'onnxruntime', 'ujson',
        }:
            packages[package.lower().replace('_', '-')] = distribution.version
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in
                    Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), cpu)
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'cpu': cpu, 'cpu_threads': int(os.environ['NUMBA_NUM_THREADS']),
            'host_ram_bytes': psutil.virtual_memory().total, 'gpu_count': 1,
            'gpu': {'name': name.decode() if isinstance(name, bytes) else name,
                    'uuid': uuid, 'vram_bytes': props['totalGlobalMem'],
                    'driver_version': driver_version, 'pci_bus_id': pci_bus_id,
                    'physical_vram_bytes': int(float(memory_mib) * 2**20),
                    'power_limit_watts': power_limit, 'max_sm_clock_mhz': max_sm_clock,
                    'multiprocessors': props['multiProcessorCount'],
                    'memory_bus_width_bits': props.get('memoryBusWidth'),
                    'compute_capability': [props['major'], props['minor']]},
            'cuda_driver': cp.cuda.runtime.driverGetVersion(),
            'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(), 'packages': packages}


def worker(job):
    import numpy as np
    import cupy as cp
    import psutil

    logging.disable(logging.CRITICAL)
    cp.cuda.Device(0).use()
    metadata = environment(cp, np, psutil, job['device'])
    if job['phase'] == 'metadata':
        return {'status': 'ok', 'environment': metadata}
    sys.path.insert(0, job['source']['path'])
    import py_boost
    expected = Path(job['source']['path'])
    if Path(py_boost.__file__).resolve().parents[1] != expected:
        raise RuntimeError(f'Imported the wrong library: {py_boost.__file__}')
    if source_info(expected) != job['source']:
        raise RuntimeError('Library source changed after the run started')
    case, suite = job['case'], job['suite']
    X, y, valid, test_X, test_y, data_hash = make_data(case, suite['data_seed'])
    params = dict(suite['common_params'], **case['params'])
    params.update(loss=case['loss'], seed=suite['model_seed'])
    if case['model'] == 'SketchBoost':
        params['sketch_params'] = suite['sketch_params'][params['sketch_method']]
    else:
        params['target_splitter'] = 'Single'

    class Iterations(py_boost.Callback):
        count = 0

        def after_iteration(self, info):
            self.count += 1
            return False

    def new_model():
        counter = Iterations()
        model = getattr(py_boost, case['model'])(**params, callbacks=[counter])
        return model, counter

    def cleanup(model):
        model.to_cpu()
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def sync():
        cp.cuda.Device(0).synchronize()

    for _ in range(suite['warmups']):
        warm, _ = new_model()
        warm.fit(X, y, eval_sets=valid)
        warm.predict(test_X)
        sync()
        cleanup(warm)
        del warm
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    def measure(phase):
        fit_times, predict_times, measurements = [], [], []
        repeats = job['repeats'] if phase == 'timing' else 1
        for _ in range(repeats):
            model, counter = new_model()
            if phase == 'timing':
                sync()
                start = time.perf_counter()
                model.fit(X, y, eval_sets=valid)
                sync()
                fit_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                prediction = model.predict(test_X)
                sync()
                predict_times.append(time.perf_counter() - start)
            else:
                with memory_tracker(cp) as tracker:
                    with RssPeak(psutil.Process()) as rss_fit:
                        model.fit(X, y, eval_sets=valid)
                        sync()
                    fit_peak = tracker.peak
                    tracker.reset_peak()
                    with RssPeak(psutil.Process()) as rss_predict:
                        prediction = model.predict(test_X)
                        sync()
                    memory = {'fit_cupy_peak_bytes': fit_peak,
                              'predict_cupy_peak_bytes': tracker.peak,
                              'fit_rss_peak_bytes': rss_fit.peak,
                              'predict_rss_peak_bytes': rss_predict.peak,
                              'fit_rss_baseline_bytes': rss_fit.baseline}
            if counter.count != params['ntrees']:
                raise RuntimeError(f'Expected {params["ntrees"]} iterations, got {counter.count}')
            if prediction.shape != (case['test'], case['outputs']):
                raise RuntimeError(f'Unexpected prediction shape: {prediction.shape}')
            score = quality(case['task'], test_y, prediction)
            if not np.isfinite(score['value']):
                raise RuntimeError('Non-finite holdout metric')
            measurements.append({'trained_iterations': counter.count,
                                 'retained_trees': len(model.models), 'quality': score})
            cleanup(model)
            del model, prediction
        result = {'status': 'ok', 'environment': metadata, 'data_hash': data_hash,
                  'params': params, 'measurements': measurements,
                  'quality': dict(name=score['name'], **summarize([m['quality']['value'] for m in measurements])),
                  'imported_library': py_boost.__file__}
        if phase == 'timing':
            result.update(fit_seconds=summarize(fit_times), predict_seconds=summarize(predict_times))
        else:
            result.update(memory)
        return result


    result = {'status': 'running', 'environment': metadata, 'data_hash': data_hash}
    for phase in ['timing', 'memory']:
        try:
            result[phase] = measure(phase)
        except Exception:
            result[phase] = {'status': 'error', 'error': traceback.format_exc()}
            result['status'] = 'error'
            return result
        write_json(job['result_path'], result)
    result['status'] = 'ok'
    return result

def invoke(python, output, job, timeout):
    output = Path(output)
    request, result, log = output.with_suffix('.job.json'), output.with_suffix('.json'), output.with_suffix('.log')
    job['result_path'] = str(result)
    write_json(request, job)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = job['device']
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
        env[key] = str(job['suite']['cpu_threads'])
    env['PYTHONHASHSEED'] = '0'
    env['PYTHONPATH'] = job['source']['path']
    try:
        with log.open('w') as stream:
            process = subprocess.run(
                [python, str(Path(__file__).resolve()), '_worker', str(request), str(result)],
                stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=timeout,
                cwd=job['source']['path'])
        if result.exists():
            record = json.loads(result.read_text())
        else:
            record = {'status': 'error', 'error': f'Worker exited {process.returncode} without a result'}
        if process.returncode and record.get('status') == 'ok':
            record = {'status': 'error', 'error': f'Worker exited {process.returncode}'}
    except subprocess.TimeoutExpired:
        record = json.loads(result.read_text()) if result.exists() else {}
        record.update(status='error', error=f'Timeout after {timeout} seconds')
    record['log'] = str(log)
    return record


def run(args):
    suite = load_suite()
    sources = [source_info(p) for p in args.source]
    if len({s['commit'] for s in sources}) != len(sources):
        raise ValueError('Each source must have a different commit')
    selected = args.case or [c['id'] for c in suite['cases']]
    if set(selected) - {c['id'] for c in suite['cases']}:
        raise ValueError('Unknown case ID')
    if args.repeats < 1 or args.timeout < 1:
        raise ValueError('Repeats and timeout must be positive')
    if ',' in args.device or not args.device:
        raise ValueError('Select exactly one GPU index or UUID')
    cases = [c for c in suite['cases'] if c['id'] in selected]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output directory must be empty; preserve previous measurements')
    reports, directories = [], []
    for source in sources:
        directory = output / source['commit'][:12]
        (directory / 'raw').mkdir(parents=True)
        report = {'schema_version': SCHEMA_VERSION, 'benchmark_id': suite['benchmark_id'],
                  'implementation_hash': implementation_hash(), 'suite_hash': digest_json(suite),
                  'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'source': source, 'case_ids': [c['id'] for c in cases],
                  'full_suite': len(cases) == len(suite['cases']),
                  'repeats': args.repeats, 'warmups': suite['warmups'],
                  'suite': suite, 'cases': {}, 'complete': False}
        job = dict(source=source, phase='metadata', device=args.device, suite=suite)
        probe = invoke(args.python, directory / 'raw' / 'metadata', job, args.timeout)
        if probe['status'] != 'ok':
            report['error'] = probe
            write_json(directory / 'results.json', report)
            raise RuntimeError(f'GPU/environment preflight failed; see {probe["log"]}')
        report['environment'] = probe['environment']
        reports.append(report)
        directories.append(directory)
        write_json(directory / 'results.json', report)
    for index, case in enumerate(cases):
        # One fresh process per version/case; timing and memory are separate phases.
        order = list(range(len(sources)))
        if index % 2:
            order.reverse()
        for i in order:
            print(f'{sources[i]["commit"][:12]} {case["id"]}', flush=True)
            job = dict(source=sources[i], phase='case', device=args.device, suite=suite,
                       case=case, repeats=args.repeats)
            record = invoke(args.python, directories[i] / 'raw' / case['id'], job, args.timeout)
            if record['status'] == 'ok' and record['environment'] != reports[i]['environment']:
                record.update(status='error', error='Environment changed during the run')
            reports[i]['cases'][case['id']] = dict(spec=case, **record)
            write_json(directories[i] / 'results.json', reports[i])
    for report, directory in zip(reports, directories):
        report['complete'] = all(c['status'] == 'ok' for c in report['cases'].values())
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(directory / 'results.json', report)
    if len(reports) == 2:
        (output / 'comparison.md').write_text(render_comparison(*reports))
    return 0 if all(r['complete'] for r in reports) else 1


def render_comparison(base, head):
    errors = comparison_errors(base, head)
    lines = [f'# {base.get("benchmark_id", "GPU benchmark")}', '',
             f'Base: `{base["source"]["commit"]}`; candidate: `{head["source"]["commit"]}`.', '']
    if errors:
        return '\n'.join(lines + ['**Not comparable:**'] + [f'- {e}' for e in errors]) + '\n'
    env = base['environment']
    gpu = env['gpu']
    capability = '.'.join(str(part) for part in gpu['compute_capability'])
    def cuda_version(value):
        return f'{value // 1000}.{value % 1000 // 10}'
    lines += [f'**GPU count: {env.get("gpu_count", "not recorded")}**; versions and cases execute sequentially.',
              f'GPU: **{gpu["name"]}**; physical VRAM: **{gpu.get("physical_vram_bytes", gpu["vram_bytes"]) / 2**30:.1f} GiB**; CUDA-visible: **{gpu["vram_bytes"] / 2**30:.1f} GiB**; compute capability: **{capability}**.',
              f'UUID: `{gpu["uuid"]}`; PCI bus: `{gpu.get("pci_bus_id", "not recorded")}`.',
              f'SMs: {gpu.get("multiprocessors", "not recorded")}; memory bus: {gpu.get("memory_bus_width_bits", "not recorded")} bits; NVIDIA driver: {gpu.get("driver_version", "not recorded")}.',
              f'Power limit: {gpu.get("power_limit_watts", "not recorded")} W; max SM clock: {gpu.get("max_sm_clock_mhz", "not recorded")} MHz.',
              f'CUDA driver API: {cuda_version(env["cuda_driver"])}; CUDA runtime: {cuda_version(env["cuda_runtime"])}.', '',
              'K is the full target dimension; k is the sketch width used for structure search.',
              '`proj`: k projected components; `rand`/`filter`: k draws with replacement (at most k distinct outputs);',
              '`topk`: k distinct outputs. Full GradientBoosting uses all K outputs. Predictions always have K outputs.', '',
              'Times are warmed wall-clock medians. GPU memory is peak live CuPy allocations,',
              'including the private fit pool; CUDA context, pool cache and external library allocations are excluded.',
              'Quality is a holdout metric (lower is better). These are measured changes, not statistical verdicts.', '',
              '| Case | Outputs K | Sketch k | Fit s (base → PR) | Δ fit | Predict s (base → PR) | Δ predict | Fit MiB (base → PR) | Δ memory | Quality (base → PR) |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    notes = []
    for case_id in base['case_ids']:
        a, b = base['cases'][case_id], head['cases'][case_id]
        def pair(section, key, scale=1):
            x, y = a[section][key], b[section][key]
            if isinstance(x, dict):
                x, y = x['median'], y['median']
            delta = f'{(y / x - 1) * 100:+.1f}%' if x else 'n/a'
            return f'{x / scale:.4f} → {y / scale:.4f}', delta
        fit, df = pair('timing', 'fit_seconds')
        pred, dp = pair('timing', 'predict_seconds')
        mem, dm = pair('memory', 'fit_cupy_peak_bytes', 2**20)
        qa = a['timing']['quality']['median']
        qb = b['timing']['quality']['median']
        spec = a['spec']
        sketch = spec.get('params', {}).get('sketch_method')
        sketch_width = f'{spec["params"]["sketch_outputs"]} ({sketch})' if sketch else f'all {spec["outputs"]}'
        lines.append(f'| {case_id} | {spec["outputs"]} | {sketch_width} | {fit} | {df} | {pred} | {dp} | {mem} | {dm} | {qa:.6g} → {qb:.6g} |')
        ra = [m['retained_trees'] for m in a['timing']['measurements']]
        rb = [m['retained_trees'] for m in b['timing']['measurements']]
        if len(set(ra + rb)) > 1:
            notes.append(f'Retained tree counts differ for `{case_id}`: {ra} vs {rb}; interpret prediction timing with care.')
    return '\n'.join(lines + [''] + notes) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('run', help='Run one or more clean source checkouts')
    command.add_argument('--source', action='append', required=True, help='Git checkout; repeat to compare versions')
    command.add_argument('--output', required=True)
    command.add_argument('--device', required=True, help='One physical GPU index or UUID')
    command.add_argument('--python', default=sys.executable, help='Interpreter containing the pinned dependencies')
    command.add_argument('--repeats', type=int, default=load_suite()['repeats'])
    command.add_argument('--case', action='append', help='Diagnostic subset; omit for the full suite')
    command.add_argument('--timeout', type=int, default=900, help='Seconds per worker, including warmup')
    command = commands.add_parser('compare', help='Compare two archived results.json files')
    command.add_argument('base')
    command.add_argument('candidate')
    command.add_argument('--output', required=True)
    command = commands.add_parser('_worker', help=argparse.SUPPRESS)
    command.add_argument('job')
    command.add_argument('result')
    args = parser.parse_args()
    if args.command == '_worker':
        try:
            result = worker(json.loads(Path(args.job).read_text()))
            write_json(args.result, result)
            return 0 if result['status'] == 'ok' else 1
        except Exception:
            write_json(args.result, {'status': 'error', 'error': traceback.format_exc()})
            traceback.print_exc()
            return 1
    if args.command == 'compare':
        base, head = [json.loads(Path(p).read_text()) for p in [args.base, args.candidate]]
        Path(args.output).write_text(render_comparison(base, head))
        return 2 if comparison_errors(base, head) else 0
    return run(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
