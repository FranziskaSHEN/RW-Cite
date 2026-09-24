"""Audit, run and evaluate native CE+GAT on existing retrospective381 assets."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.ewm_top10.rwcite_contract import (
    audit, digest, indexed, load_config, read_rows, safe_output, validate_predictions, write_json,
)


def merge(output, queries, corpus, windows):
    """Refuse partial, foreign or candidate-mismatched results before scoring."""
    from experiments.ewm_top10.common import write_jsonl
    predictions = []
    for mode in ('fixed400', 'endtoend400'):
        directory = output / 'rows' / mode
        actual = {p.stem for p in directory.glob('*.json')}
        if actual != set(queries):
            raise ValueError(f'{mode}: missing={len(set(queries)-actual)} extra={len(actual-set(queries))}')
        rows = []
        for qid, q in queries.items():
            row = json.loads((directory / f'{qid}.json').read_text(encoding='utf-8'))
            expected = windows[qid]['ranked_ids'] if mode == 'fixed400' else row['window_ids']
            validate_predictions(row, q, corpus, expected)
            if mode == 'endtoend400' and (len(expected) > 400 or not set(expected) <= set(row['universe_ids'])):
                raise ValueError('end-to-end window escaped its retrieved universe')
            rows.append(row)
        path = output / 'predictions' / f'rw_cite_{mode}.jsonl'
        write_jsonl(path, rows)
        predictions.append(path)
    return predictions


def template():
    names = ('release_code', 'test', 'corpus', 'fixed400', 'baseline_train', 'baseline_dev',
             'graph', 'ce', 'gat', 'struct', 'scibert', 'bge', 'retrieval_embeddings',
             'training_provenance', 'masked_sources')
    return {'schema': 1, 'evaluation_protocol': 'retrospective381', 'seed': 42,
            'cpu_threads_per_worker': 4,
            'assets': {n: {'path': f'REPLACE/{n}', 'sha256': None} for n in names},
            'recipe': {'window': 400, 'ce_max_length': 256, 'ce_max_sents': 3, 'gat_chunk': 0,
                       'gat_alpha': 0.4, 'rrf_k': 20, 'rrf_weight': 0.4},
            'universe_env': {'RR_CAND_M': 200, 'RR_PRF_K': 10, 'RR_PRF_M': 80,
                             'RR_BRIDGE_SEEDS': 120, 'RR_HUB_CAP': 400, 'RR_TOP_DENSE': 60,
                             'RR_MIN_OD_DIRECT': 2, 'RR_PRED_MIN_OD': 2, 'RR_HOP2_K': 100}}


def runtime_versions():
    versions = {'python': sys.version}
    for name in ('torch', 'transformers', 'numpy', 'scipy', 'networkx', 'pyarrow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    return versions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('template', 'hashes', 'audit', 'run', 'merge'))
    ap.add_argument('--config', required=True)
    ap.add_argument('--out')
    ap.add_argument('--gpus', default='4,5,6,7')
    ap.add_argument('--max-queries', type=int, default=0, help='smoke only; keep a separate output directory')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve()
    if args.action == 'template':
        safe_output(root, config_path)
        if config_path.exists():
            raise ValueError('refusing to overwrite existing config')
        write_json(config_path, template())
        print(f'Edit asset paths and reviewed training provenance: {config_path}')
        return
    c = load_config(config_path)
    if args.action == 'hashes':
        safe_output(root, config_path)
        for spec in c['assets'].values():
            spec['sha256'] = digest(spec['path'])
        write_json(config_path, c)
        print('Recorded current fingerprints only; this is NOT a training/leakage audit.')
        return
    report = audit(c)
    print(json.dumps(report, indent=2), flush=True)
    if args.action == 'audit':
        return
    if not args.out:
        raise ValueError('--out required')
    output = safe_output(root, args.out)
    # Avoid clobbering models/datasets or any previously completed baseline run.
    for spec in c['assets'].values():
        p = Path(spec['path']).resolve()
        if p == output or output in p.parents or p in output.parents:
            raise ValueError('output overlaps an input artifact')
    if args.max_queries < 0 or args.max_queries > 381:
        raise ValueError('max-queries must be 0..381')
    gpus = args.gpus.split(',')
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('--gpus must list distinct nonnegative device indices')
    # Bind resume to all inputs, code, settings and smoke/full cohort size.
    code = {
        p.name: digest(p)
        for p in (
            Path(__file__),
            Path(__file__).with_name('rwcite_contract.py'),
            Path(__file__).with_name('rwcite_worker.py'),
        )
    }
    signature = {'config': c, 'code': code, 'max_queries': args.max_queries,
                 'runtime_versions': runtime_versions()}
    receipt = output / 'run_identity.json'
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.running'
    lock.mkdir()  # A second launcher must not share these output files.
    workers = []
    streams = []
    try:
        if receipt.exists():
            if json.loads(receipt.read_text(encoding='utf-8')) != signature:
                raise ValueError('output belongs to different inputs/code/settings; use a NEW run directory')
        else:
            # Launcher log directory may already exist, but data rows may not.
            if (output / 'rows').exists() or (output / 'predictions').exists():
                raise ValueError('unidentified prior outputs; refusing reuse')
            write_json(receipt, signature)
        write_json(output / 'audit.json', report)
        paths = {k: Path(v['path']) for k, v in c['assets'].items()}
        qs = indexed(read_rows(paths['test']), 'query_id')
        if args.max_queries:
            qs = dict(list(qs.items())[:args.max_queries])
        cs = indexed(read_rows(paths['corpus']), 'paper_id')
        ws = indexed(read_rows(paths['fixed400']), 'query_id')
        if args.action == 'run':
            n = min(len(gpus), len(qs))
            for shard in range(n):
                log_path = output / 'worker_logs' / f'shard{shard}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open('a', encoding='utf-8')
                streams.append(log)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[shard], PYTHONUNBUFFERED='1',
                           PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                worker = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('rwcite_worker.py')),
                    '--config', str(config_path), '--output', str(output),
                    '--audit-sha256', digest(output / 'audit.json'), '--shard', str(shard), '--shards', str(n),
                    '--max-queries', str(args.max_queries)], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                workers.append(worker)
                print(f'shard={shard}/{n} GPU={gpus[shard]} PID={worker.pid} log={log_path}', flush=True)
            while any(w.poll() is None for w in workers):
                if any(w.poll() not in (None, 0) for w in workers):
                    raise RuntimeError('worker failed; inspect worker_logs; completed query rows are retained')
                time.sleep(1)
            if any(w.returncode != 0 for w in workers):
                raise RuntimeError('worker failed; partial results will not be evaluated')
        # Catch accidentally edited/replaced inputs during a long GPU run before
        # publishing a summary against a different corpus, gold set or checkpoint.
        for name, spec in c['assets'].items():
            if digest(spec['path']) != spec['sha256']:
                raise RuntimeError(f'{name} changed while running; results will not be published')
        predictions = merge(output, qs, cs, ws)
        eval_test = paths['test']
        if args.max_queries:
            from experiments.ewm_top10.common import write_jsonl
            eval_test = output / 'smoke_queries.jsonl'
            write_jsonl(eval_test, qs.values())
        subprocess.run([sys.executable, '-m', 'experiments.ewm_top10.evaluate',
            '--test', str(eval_test), '--corpus', str(paths['corpus']), '--predictions',
            *map(str, predictions), '--out-dir', str(output / 'metrics')], cwd=root, check=True)
        write_json(output / 'complete.json', {'n_queries': len(qs), 'seed': c.get('seed', 42),
                   'smoke': bool(args.max_queries), 'predictions': list(map(str, predictions)),
                   'recall_at_1000': 'not reported: only the structural Top400 is CE+GAT scored'})
        print(f'COMPLETE: {output / "metrics"} ({len(qs)} queries)', flush=True)
    finally:
        for w in workers:
            if w.poll() is None:
                w.terminate()
        for w in workers:
            try:
                w.wait(timeout=15)
            except subprocess.TimeoutExpired:
                w.kill()
                w.wait()
        for stream in streams:
            stream.close()
        lock.rmdir()


if __name__ == '__main__':
    main()
