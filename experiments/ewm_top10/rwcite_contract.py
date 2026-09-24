"""Fail-closed contracts for applying CE+GAT to the frozen comparison cohort.

This module deliberately has no dependency on either version of rwcite.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def digest(path):
    path = Path(path)
    if path.is_dir():
        h = hashlib.sha256()
        files = sorted(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        if not files:
            raise ValueError(f'empty artifact directory: {path}')
        for p in files:
            h.update(p.relative_to(path).as_posix().encode())
            h.update(b'\0')
            h.update(digest(p).encode())
        return h.hexdigest()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def canonical_id(value):
    value = str(value).strip()
    for prefix in ('https://arxiv.org/abs/', 'http://arxiv.org/abs/', 'arXiv:'):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return re.sub(r'v\d+$', '', value)


def month(value):
    # The benchmark's month is authoritative. Do not invent January for a year.
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])', value):
        raise ValueError(f'expected explicit YYYY-MM, got {value!r}')
    return tuple(map(int, value.split('-')))


def indexed(rows, key):
    result = {}
    for row in rows:
        pid = canonical_id(row[key])
        if not pid or pid != row[key] or pid in result:
            raise ValueError(f'duplicate or noncanonical {key}: {row[key]!r}')
        result[pid] = row
    return result


def check_windows(queries, corpus, windows, width=400):
    if set(windows) != set(queries):
        raise ValueError('fixed windows must cover exactly the evaluation queries')
    for qid, row in windows.items():
        ids = row['ranked_ids']
        if len(ids) != width or len(set(ids)) != width:
            raise ValueError(f'{qid}: expected exactly {width} unique frozen candidates')
        cutoff = month(queries[qid]['published_at'])
        for pid in ids:
            if pid == qid or pid not in corpus or month(corpus[pid]['published_at']) >= cutoff:
                raise ValueError(f'{qid}: inadmissible frozen candidate {pid}')


def canonical_graph(raw, corpus, queries):
    """Preserve GEXF predecessor order, discard non-text node attributes.

    In-place relabelling preserves incoming adjacency order; rebuilding edges by
    source would change the release CE's order-sensitive stride selector.
    """
    import networkx as nx
    if not raw.is_directed() or raw.is_multigraph():
        raise ValueError('expected a simple directed citation graph; no implicit conversion')
    mapping = {node: canonical_id(node) for node in raw}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('graph contains colliding paper IDs after version normalization')
    # GEXF keys normally already canonical. Reject relabel chains rather than reorder.
    if any(node != pid for node, pid in mapping.items()):
        raise ValueError('graph IDs must be canonical; normalize upstream with an order audit')
    missing = set(corpus) - set(raw)
    if missing:
        raise ValueError(f'graph missing {len(missing)} benchmark corpus nodes; no silent narrowing')
    g = raw
    g.remove_nodes_from([pid for pid in g if pid not in corpus and pid not in queries])
    for pid in set(corpus) | set(queries):
        row = queries.get(pid, corpus.get(pid))
        if pid not in g:
            g.add_node(pid)
        g.nodes[pid].clear()
        g.nodes[pid].update(title=row['title'], abstract=row.get('abstract', ''),
                            published_at=row['published_at'])
    return g


def temporal_view(graph, cutoff, masked_sources):
    """Filter before degree, universe, evidence, cocitation or neighbour creation."""
    import networkx as nx
    allowed = {p for p, a in graph.nodes(data=True) if month(a['published_at']) < cutoff}
    return nx.subgraph_view(graph, filter_node=lambda p: p in allowed,
                           filter_edge=lambda u, v: u not in masked_sources)


def safe_output(root, candidate):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f'output must be a descendant of checkout: {candidate}')
    return candidate


def load_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text(encoding='utf-8'))
    for name, spec in c['assets'].items():
        p = Path(spec['path']).expanduser()
        spec['path'] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    return c


def audit(c):
    """Verify data/weight fingerprints and recorded training provenance.

    Recorded attestations are NOT a proof of training-time data flow. A reviewer
    must check the referenced training reports; the output states this limitation.
    """
    if c.get('schema') != 1 or c.get('evaluation_protocol') != 'retrospective381':
        raise ValueError('requires schema=1 and evaluation_protocol=retrospective381')
    required = {'release_code', 'test', 'corpus', 'fixed400', 'baseline_train',
                'baseline_dev', 'graph', 'ce', 'gat', 'struct', 'scibert',
                'bge', 'retrieval_embeddings', 'training_provenance', 'masked_sources'}
    if not required <= set(c['assets']):
        raise ValueError(f'missing assets: {sorted(required - set(c["assets"]))}')
    for name, spec in c['assets'].items():
        if digest(spec['path']) != spec.get('sha256'):
            raise ValueError(f'{name}: SHA256 missing or mismatched; do not guess artifact identity')
    paths = {k: Path(v['path']) for k, v in c['assets'].items()}
    qs = indexed(read_rows(paths['test']), 'query_id')
    cs = indexed(read_rows(paths['corpus']), 'paper_id')
    if len(qs) != 381:
        raise ValueError(f'expected original 381 unique evaluation queries, got {len(qs)}')
    for q in qs.values():
        month(q['published_at'])
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', q['query_id']) or q['query_id'] in ('.', '..'):
            raise ValueError('evaluation query IDs must be safe single-component filenames')
        if q['query_id'] in cs:
            for key in ('title', 'abstract', 'published_at'):
                if q.get(key, '') != cs[q['query_id']].get(key, ''):
                    raise ValueError(f'{q["query_id"]}: query/corpus text or date mismatch')
        gold = q['gold_ids']
        if not gold or len(gold) != len(set(gold)):
            raise ValueError(f'{q["query_id"]}: empty/duplicate gold')
        for pid in gold:
            if pid not in cs or month(cs[pid]['published_at']) >= month(q['published_at']):
                raise ValueError(f'{q["query_id"]}: gold incompatible with benchmark corpus/date')
    for row in cs.values():
        month(row['published_at'])
    check_windows(qs, cs, indexed(read_rows(paths['fixed400']), 'query_id'))
    train = indexed(read_rows(paths['baseline_train']), 'query_id')
    dev = indexed(read_rows(paths['baseline_dev']), 'query_id')
    cutoff = min(month(q['published_at']) for q in qs.values())
    if set(train) & (set(dev) | set(qs)) or set(dev) & set(qs):
        raise ValueError('benchmark train/development/evaluation IDs overlap')
    for row in list(train.values()) + list(dev.values()):
        if month(row['published_at']) >= cutoff:
            raise ValueError('benchmark supervision is not strictly before evaluation period')
    provenance = json.loads(paths['training_provenance'].read_text(encoding='utf-8'))
    if not provenance.get('reviewed_by') or not provenance.get('evidence_references'):
        raise ValueError('training provenance requires a reviewer and evidence references')
    for component in ('ce', 'gat', 'struct'):
        p = provenance['components'][component]
        if p.get('artifact_sha256') != c['assets'][component]['sha256']:
            raise ValueError(f'{component}: training provenance is for different weights')
        if set(p.get('train_query_ids', [])) != set(train):
            raise ValueError(f'{component}: training IDs differ from baseline train; retrain or investigate')
        if not set(p.get('selection_query_ids', [])) <= set(dev):
            raise ValueError(f'{component}: checkpoint selection used non-development queries')
        if p.get('temporal_training_audited') is not True or p.get('target_leakage_audited') is not True:
            raise ValueError(f'{component}: training graph/text safeguards have not been reviewed')
    if provenance.get('retrieval_inputs_paper_local_only') is not True:
        raise ValueError('topic embeddings need a paper-local input audit (no citation/future context)')
    if provenance.get('frozen_text_versions_audited') is not True:
        raise ValueError('paper/evidence versions must be audited; month filters cannot date revisions')
    if provenance.get('fusion_selected_without_evaluation_gold') is not True:
        raise ValueError('fusion settings require development-only provenance')
    if provenance.get('ce_evidence_recipe') != {'selection': 'graph_order_stride', 'max_sents': 3,
            'short': True, 'max_length': 256, 'sentence_preprocessing': 'as_stored_in_graph'}:
        raise ValueError('CE training evidence recipe must match the native graph-order scorer')
    if 'whitening' in paths:
        w = provenance.get('whitening', {})
        if w.get('artifact_sha256') != c['assets']['whitening']['sha256']:
            raise ValueError('whitening provenance mismatch')
        fit = set(w.get('fit_paper_ids', []))
        if not fit or not fit <= set(cs) or fit & set(qs):
            raise ValueError('whitening fit set missing or contains evaluation papers')
        if any(month(cs[p]['published_at']) >= cutoff for p in fit):
            raise ValueError('whitening was fitted using post-cutoff papers')
    elif provenance.get('gat_uses_raw_scibert') is not True:
        raise ValueError('missing audited whitening transform for this GAT')
    if c.get('recipe') != {'window': 400, 'ce_max_length': 256, 'ce_max_sents': 3,
                            'gat_chunk': 0, 'gat_alpha': 0.4, 'rrf_k': 20, 'rrf_weight': 0.4}:
        raise ValueError(
            'unexpected recipe: this runner fixes the paper-defined '
            'CE+GAT score-and-rank fusion recipe'
        )
    expected_env = {'RR_CAND_M', 'RR_PRF_K', 'RR_PRF_M', 'RR_BRIDGE_SEEDS', 'RR_HUB_CAP',
                    'RR_TOP_DENSE', 'RR_MIN_OD_DIRECT', 'RR_PRED_MIN_OD', 'RR_HOP2_K'}
    if set(c.get('universe_env', {})) != expected_env:
        raise ValueError('explicit frozen universe settings required; inherited defaults forbidden')
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in c['universe_env'].values()):
        raise ValueError('universe settings must be nonnegative integers')
    masked = json.loads(paths['masked_sources'].read_text(encoding='utf-8'))
    if not isinstance(masked, list) or not set(qs) <= set(masked):
        raise ValueError('mask must include ALL 381 evaluation source IDs, not only the release 370')
    if any(canonical_id(x) != x for x in masked):
        raise ValueError('masked sources must be canonical IDs')
    return {'status': 'passed_recorded_contracts', 'n_queries': len(qs), 'n_train': len(train),
            'n_dev': len(dev), 'training_cutoff_exclusive': '%04d-%02d' % cutoff,
            'note': 'Training assertions rely on reviewed provenance; not an automated proof.',
            'evaluation_protocol': 'retrospective381', 'assets': c['assets']}


def validate_predictions(row, q, corpus, expected_ids=None):
    ids = row['ranked_ids']
    if row['query_id'] != q['query_id'] or len(ids) < 10 or len(ids) != len(set(ids)):
        raise ValueError('invalid prediction IDs/count')
    if expected_ids is not None and set(ids) != set(expected_ids):
        raise ValueError('fixed candidate IDs changed during scoring')
    for pid in ids:
        if pid == q['query_id'] or pid not in corpus or month(corpus[pid]['published_at']) >= month(q['published_at']):
            raise ValueError('prediction contains unavailable candidate')
    for key in ('scores', 'ce_scores', 'gat_scores'):
        import math
        if len(row[key]) != len(ids) or not all(math.isfinite(x) for x in row[key]):
            raise ValueError(f'invalid {key}')
