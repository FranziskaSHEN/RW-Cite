"""Isolated worker for auditing a specified RW-Cite release package."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

try:
    from .rwcite_contract import (
        canonical_graph,
        canonical_id,
        digest,
        indexed,
        load_config,
        month,
        read_rows,
        temporal_view,
        validate_predictions,
        write_json,
    )
except ImportError:  # direct script execution
    from rwcite_contract import (
        canonical_graph,
        canonical_id,
        digest,
        indexed,
        load_config,
        month,
        read_rows,
        temporal_view,
        validate_predictions,
        write_json,
    )


def native_imports(release_package):
    if any(name == 'rwcite' or name.startswith('rwcite.') for name in sys.modules):
        raise RuntimeError('native worker must start in a fresh process')
    # Whitelist runtime settings instead of inheriting an old training shell.
    for name in list(os.environ):
        if name.startswith(('RR_', 'GAT_')):
            os.environ.pop(name)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      RWCITE_READ_ONLY_ASSETS='1')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(release_package).resolve().parent))
    import rwcite
    if Path(rwcite.__file__).resolve().parent != Path(release_package).resolve():
        raise RuntimeError('wrong rwcite version imported')


def load_retrieval(path, required_ids):
    import numpy as np
    if Path(path).suffix == '.parquet':
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=['paper_id', 'embedding'])
        ids = table['paper_id'].to_pylist()
        vectors = np.asarray(table['embedding'].to_pylist(), dtype=np.float32)
    else:
        with np.load(path, allow_pickle=False) as z:
            ids = z['ids'].tolist()
            vectors = np.asarray(z['embeddings'], dtype=np.float32)
    ids = [canonical_id(x) for x in ids]
    if len(ids) != len(set(ids)) or vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError('invalid/duplicate retrieval embedding rows')
    if not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) < 1e-12).any():
        raise ValueError('nonfinite/zero retrieval embeddings')
    by = dict(zip(ids, vectors))
    if not set(required_ids) <= set(by):
        raise ValueError('retrieval embeddings do not cover the frozen corpus; no on-demand fallback')
    # Do not import the native retrieval service: it can update external caches.
    # Native service preserves parquet/map insertion order, including tie order.
    ordered = [pid for pid in ids if pid in required_ids]
    matrix = np.stack([by[p] for p in ordered])
    return ordered, matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def make_pack(ids, X, graph):
    from rwcite.gat.preprocess import build_adjacencies, build_neighbors_k
    from rwcite.gat.train import GatTrainPack
    p = GatTrainPack.__new__(GatTrainPack)
    p.ids, p.id_to_i, p.X = ids, {pid: i for i, pid in enumerate(ids)}, X
    adjs = build_adjacencies(graph, p.id_to_i)
    p.adj_in_full, p.adj_out_full, p.adj_cocite_full = adjs['in'], adjs['out'], adjs['cocite']
    # Recompute cocitation witnesses and neighbour selection after masking.
    nbr = build_neighbors_k(adjs['in'], adjs['cocite'], X)
    p.nbr_k, p.nbr_pool = nbr['neighbors_k32'], nbr['neighbors_pool64']
    p.use_foldout, p.exact_target_b, p.expand_hop = False, True, 0
    p.hop_budget, p.l4_train_k, p.l4_eval_k = 64, 32, 32
    p.fold_of, p.fold_adjs = {}, {}
    allowed_idx = {p.id_to_i[pid] for pid in graph}
    selected = set(int(i) for i in p.nbr_pool.ravel() if i >= 0)
    if not selected <= allowed_idx:
        raise RuntimeError('temporally unavailable GAT neighbour')
    return p


def score_candidates(q, candidates, graph, universe, pack, raw_X, ce, gat, device):
    import numpy as np
    import torch
    from rwcite.gat.fuse_ce import _fuse_scores
    from rwcite.gat.train import score_query_chunked
    from rwcite.ranker.rr_ranker import extract_features
    from rwcite.ranker.rr_ranker_ce import _query_text
    from rwcite.ranker.rr_ranker_ce_sent import cand_text_with_sents, collect_cite_sentences
    qid = q['query_id']
    qidx = pack.id_to_i[qid]
    qvec = raw_X[qidx]
    support = universe.get('support') or {}
    native_cands = {cand['id']: cand for cand in universe.get('universe', [])}
    node_by = {p: p for p in graph}
    texts, phi, evidence = [], [], []
    for cand in candidates:
        pid = cand['id']
        cv = raw_X[pack.id_to_i[pid]]
        sim = float(np.dot(qvec, cv) / ((np.linalg.norm(qvec) + 1e-8) * (np.linalg.norm(cv) + 1e-8)))
        # Native universe uses a 700-character abstract for lexical features.
        # CE still receives the full corpus abstract for its no-evidence fallback.
        feature_cand = native_cands.get(pid, dict(cand, abstract=cand.get('abstract', '')[:700]))
        phi.append(extract_features(title=q['title'], abstract=q.get('abstract', ''), cand=feature_cand,
                   support=support, dense_rank=universe.get('dense_rank') or {},
                   focused_hubs=universe.get('focused_hubs') or set(), graph=graph,
                   node_by=node_by, emb_sim=sim,
                   max_support=max(support.values(), default=1.0), query_id=qid, mask_citers={qid}))
        texts.append(cand_text_with_sents(cand, graph=graph, exclude_citer=qid, max_sents=3, short=True))
        evidence.append(collect_cite_sentences(graph, pid, exclude_citer=qid, max_sents=3))
    sc = np.asarray(ce.score_pairs(_query_text(q['title'], q.get('abstract', '')), texts,
                                   batch_size=32), dtype=np.float32)
    if len(sc) != len(candidates) or not np.isfinite(sc).all():
        raise RuntimeError('invalid CE scores; no sentinel/fallback ranking permitted')
    with torch.inference_mode():
        result = score_query_chunked(gat, pack, query_global=qidx, query_id=qid,
                   cand_globals=np.asarray([pack.id_to_i[c['id']] for c in candidates]),
                   n_cands=len(candidates), phi_np=np.asarray(phi, dtype=np.float32),
                   device=device, rng=np.random.default_rng(0), chunk_size=0,
                   ce_arr=None, use_checkpoint=False)
    sg = result['scores'].detach().float().cpu().numpy()
    if len(sg) != len(candidates) or not np.isfinite(sg).all():
        raise RuntimeError('invalid GAT scores')
    fused = _fuse_scores(
        sg=sg,
        sc=sc,
        alpha=0.4,
        fuse_mode='score_rank',
        rrf_k=20,
        rrf_w=0.4,
    )
    order = fused.argsort()[::-1]
    return {'query_id': qid, 'ranked_ids': [candidates[i]['id'] for i in order],
            'scores': fused[order].tolist(), 'ce_scores': sc[order].tolist(),
            'gat_scores': sg[order].tolist(),
            'candidate_evidence': [{'paper_id': c['id'], 'text': t, 'sentences': s}
                                   for c, t, s in zip(candidates, texts, evidence)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--audit-sha256', required=True)
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--shards', type=int, required=True)
    ap.add_argument('--max-queries', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    c = load_config(args.config)
    paths = {k: Path(v['path']) for k, v in c['assets'].items()}
    output = Path(args.output).resolve()
    if digest(output / 'audit.json') != args.audit_sha256:
        raise RuntimeError('audit changed after launch')
    identity_path = output / 'run_identity.json'
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        if c != identity['config'] or args.max_queries != identity['max_queries']:
            raise RuntimeError('configuration changed after the coordinator audit')
    if not 0 <= args.shard < args.shards or args.max_queries < 0:
        raise ValueError('invalid shard/smoke size')
    native_imports(paths['release_code'])
    os.environ.update({k: str(v) for k, v in c['universe_env'].items()})
    import networkx as nx
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from rwcite.gat.preprocess import encode_scibert, apply_whiten
    from rwcite.gat.model import GatMvpConfig, GatMvpModel
    from rwcite.ranker.rr_ranker import load_ranker, rank_universe, FEATURE_NAMES
    from rwcite.ranker.rr_ranker_ce import CERanker
    from rwcite.ranker.rr_universe import build_rr_universe

    torch.manual_seed(c.get('seed', 42))
    torch.set_num_threads(int(c.get('cpu_threads_per_worker', 4)))
    device = torch.device(args.device)
    queries = indexed(read_rows(paths['test']), 'query_id')
    corpus = indexed(read_rows(paths['corpus']), 'paper_id')
    windows = indexed(read_rows(paths['fixed400']), 'query_id')
    masks = set(json.loads(paths['masked_sources'].read_text(encoding='utf-8')))
    graph = canonical_graph(nx.read_gexf(paths['graph']), corpus, queries)
    all_ids = sorted(graph)
    global_index = {pid: i for i, pid in enumerate(all_ids)}
    retrieval_ids, matrix = load_retrieval(paths['retrieval_embeddings'], corpus)
    print(f'worker={args.shard}/{args.shards} graph={len(graph)} edges={graph.number_of_edges()}', flush=True)
    raw_X = encode_scibert(graph, all_ids, model_dir=paths['scibert'], device=str(device))
    if 'whitening' in paths:
        with np.load(paths['whitening'], allow_pickle=False) as w:
            X = apply_whiten(raw_X, {'mu': w['mu'], 'W': w['W']})
    else:
        X = raw_X
    if not np.isfinite(X).all():
        raise ValueError('invalid node embeddings/whitening')
    ckpt = torch.load(paths['gat'], map_location='cpu', weights_only=True)
    cfg = ckpt.get('cfg')
    if not isinstance(cfg, dict) or set(cfg) - set(GatMvpConfig.__dataclass_fields__):
        raise ValueError('unsupported GAT checkpoint configuration')
    gat = GatMvpModel(GatMvpConfig(**cfg))
    if gat.cfg.use_ce_feat or gat.cfg.struct_only or gat.cfg.text_dim != X.shape[1] or gat.cfg.neighbor_k != 32:
        raise ValueError('checkpoint does not match independent CE+E4 GAT recipe')
    for flag in ('use_l1_sem', 'use_multi_channel', 'use_text_inject', 'use_gate', 'use_l4', 'use_struct_residual'):
        if not getattr(gat.cfg, flag):
            raise ValueError(f'expected full E4 GAT, but {flag} is disabled')
    if gat.cfg.fixed_alpha:
        raise ValueError('expected learned graph attention, not the fixed-alpha ablation')
    gat.load_state_dict(ckpt['model'], strict=True)
    gat.to(device).eval()
    ce = CERanker.from_pretrained(paths['ce'], device=str(device), max_length=256)
    ce.model.eval()
    struct = load_ranker(str(paths['struct']))
    if struct is None or struct.feature_names != FEATURE_NAMES or not np.isfinite(struct.sd).all() or (struct.sd <= 0).any():
        raise ValueError('missing/incompatible structural model; no heuristic fallback')
    tokenizer = AutoTokenizer.from_pretrained(paths['bge'], local_files_only=True)
    embedder = AutoModel.from_pretrained(paths['bge'], local_files_only=True).to(device).eval()
    if matrix.shape[1] != embedder.config.hidden_size:
        raise ValueError('retrieval index does not match BGE dimension')

    selected = list(queries.values())
    if args.max_queries:
        selected = selected[:args.max_queries]
    selected = [q for i, q in enumerate(selected) if i % args.shards == args.shard]
    # Month grouping allows each worker to reuse only its own temporally rebuilt pack.
    selected.sort(key=lambda q: (q['published_at'], q['query_id']))
    pack, visible, active_month, month_raw = None, None, None, None
    for ordinal, original_q in enumerate(selected, 1):
        qid = original_q['query_id']
        files = {mode: output / 'rows' / mode / f'{qid}.json' for mode in ('fixed400', 'endtoend400')}
        if all(p.exists() for p in files.values()):
            print(f'[{ordinal}/{len(selected)}] {qid} resume: both rows already saved', flush=True)
            continue
        # Gold and reference sentences never enter the scoring API.
        q = {k: original_q.get(k, '') for k in ('query_id', 'title', 'abstract', 'published_at')}
        cutoff = month(q['published_at'])
        if active_month != cutoff:
            prep = time.perf_counter()
            visible = temporal_view(graph, cutoff, masks)
            # Adjacency indices and RNG seeds must not depend on future nodes.
            # Append this month's isolated query nodes AFTER the eligible nodes.
            month_ids = sorted(visible) + sorted(qid for qid, row in queries.items()
                                                 if month(row['published_at']) == cutoff)
            positions_x = [global_index[pid] for pid in month_ids]
            month_raw = raw_X[positions_x]
            pack = make_pack(month_ids, X[positions_x], visible)
            active_month = cutoff
            print(f'month={cutoff} visible={len(visible)} edges={visible.number_of_edges()} rebuilt_s={time.perf_counter()-prep:.1f}', flush=True)
        tick = time.perf_counter()
        allowed = set(visible) & set(corpus)
        positions = np.asarray([i for i, pid in enumerate(retrieval_ids) if pid in allowed], dtype=np.int64)
        submatrix = matrix[positions]
        def search(text, limit):
            inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.inference_mode():
                v = embedder(**{k: t.to(device) for k, t in inputs.items()}).last_hidden_state[:, 0, :].float()
                v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()[0]
            sims = submatrix @ v
            order = np.argsort(-sims)[:limit]
            return [{'id': retrieval_ids[positions[i]], 'title': corpus[retrieval_ids[positions[i]]]['title'],
                     'score': float(sims[i])} for i in order]
        universe = build_rr_universe(q['title'], q['abstract'], visible, search_fn=search, exclude_id=qid)
        if not {x['id'] for x in universe['universe']} <= allowed:
            raise RuntimeError('universe contains unavailable papers')
        coarse = rank_universe(title=q['title'], abstract=q['abstract'], universe_pack=universe,
                   graph=visible, n=400, emb_sims=None, model=struct, query_id=qid, mask_citers={qid})
        retrieval_ms = (time.perf_counter() - tick) * 1000
        for mode in files:
            if files[mode].exists():
                continue
            ids = windows[qid]['ranked_ids'] if mode == 'fixed400' else [x['id'] for x in coarse]
            if len(ids) < 10 or len(ids) != len(set(ids)) or not set(ids) <= allowed:
                raise RuntimeError(f'{qid}: invalid {mode} window')
            candidates = [dict(id=pid, title=corpus[pid]['title'], abstract=corpus[pid].get('abstract', '')) for pid in ids]
            start = time.perf_counter()
            row = score_candidates(q, candidates, visible, universe, pack, month_raw, ce, gat, device)
            scoring_ms = (time.perf_counter() - start) * 1000
            row.update(method=f'rw_cite_ce_gat_{mode}', latency_ms=retrieval_ms + scoring_ms,
                       scoring_ms=scoring_ms, retrieval_feature_ms=retrieval_ms,
                       latency_scope='warm query; excludes monthly graph rebuild and offline embeddings',
                       universe_ids=[x['id'] for x in universe['universe']], window_ids=ids,
                       evidence_selection='native_graph_order_stride_after_temporal_mask',
                       graph_protocol='rolling_month_strict_all_evaluation_out_edges_masked',
                       seed=c.get('seed', 42))
            validate_predictions(row, q, corpus, windows[qid]['ranked_ids'] if mode == 'fixed400' else ids)
            write_json(files[mode], row)
        print(f'[{ordinal}/{len(selected)}] {qid} saved fixed400 + endtoend400', flush=True)


if __name__ == '__main__':
    main()
