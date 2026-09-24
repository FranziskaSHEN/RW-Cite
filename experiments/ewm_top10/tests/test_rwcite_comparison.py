"""Contract regressions; optional native integration uses an isolated process."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import networkx as nx
import pytest

# Also permits running this file in a staging directory before installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rwcite_contract import (audit, canonical_graph, check_windows, digest, indexed,
                          month, safe_output, temporal_view, validate_predictions)
from rwcite_worker import load_retrieval


def papers():
    return {'old': {'paper_id': 'old', 'title': 'Old', 'abstract': 'text', 'published_at': '2025-01'},
            'other': {'paper_id': 'other', 'title': 'Other', 'abstract': '', 'published_at': '2025-02'},
            'past': {'paper_id': 'past', 'title': 'Past', 'abstract': '', 'published_at': '2025-03'},
            'same': {'paper_id': 'same', 'title': 'Same', 'abstract': '', 'published_at': '2026-06'},
            'future': {'paper_id': 'future', 'title': 'Future', 'abstract': '', 'published_at': '2026-07'}}


def test_temporal_mask_before_evidence_and_degrees():
    cs = papers()
    q = {'query_id': 'q', 'title': 'Q', 'abstract': '', 'published_at': '2026-06'}
    g = nx.DiGraph()
    g.add_nodes_from(cs)
    for source in ('future', 'past', 'same', 'q'):
        g.add_edge(source, 'old', sentence=source)
    g.nodes['old']['cached_future_degree'] = 999
    graph = canonical_graph(g, cs, {'q': q})
    assert 'cached_future_degree' not in graph.nodes['old']
    view = temporal_view(graph, (2026, 6), {'q'})
    assert list(view.predecessors('old')) == ['past']
    assert view.in_degree('old') == 1
    assert 'q' not in view and 'same' not in view and 'future' not in view
    # Earlier evaluation queries must also remain masked as citing sources.
    view = temporal_view(graph, (2026, 7), {'q', 'past'})
    assert list(view.predecessors('old')) == ['same']
    assert graph.has_edge('q', 'old')  # view never mutates the source graph


def test_predecessor_order_is_preserved():
    cs = papers()
    g = nx.DiGraph()
    g.add_nodes_from(cs)
    for p in ('past', 'other', 'future'):
        g.add_edge(p, 'old', sentence=p)
    g = canonical_graph(g, cs, {})
    assert list(temporal_view(g, (2026, 6), set()).predecessors('old')) == ['past', 'other']


@pytest.mark.parametrize('value', ['2026', '2026-00', '2026-13', '', None, '2026-06-01'])
def test_month_never_guesses(value):
    with pytest.raises(ValueError):
        month(value)


def test_alias_collision_and_missing_nodes_fail():
    g = nx.DiGraph()
    g.add_nodes_from(['2401.00001', '2401.00001v2'])
    with pytest.raises(ValueError, match='colliding'):
        canonical_graph(g, {}, {})
    with pytest.raises(ValueError, match='missing'):
        canonical_graph(nx.DiGraph(), papers(), {})


def test_no_silent_window_replacement():
    cs = papers()
    qs = {'q': {'query_id': 'q', 'published_at': '2026-06'}}
    ws = {'q': {'ranked_ids': ['old', 'other']}}
    check_windows(qs, cs, ws, width=2)
    for ids in (['old', 'same'], ['old', 'future'], ['old', 'old'], ['old']):
        with pytest.raises(ValueError):
            check_windows(qs, cs, {'q': {'ranked_ids': ids}}, width=2)
    with pytest.raises(ValueError):
        check_windows(qs, cs, {}, width=2)


def test_predictions_validate_alignment():
    cs = {f'p{i}': {'published_at': '2025-01'} for i in range(10)}
    q = {'query_id': 'q', 'published_at': '2026-01'}
    row = {'query_id': 'q', 'ranked_ids': list(cs), 'scores': [1.]*10,
           'ce_scores': [1.]*10, 'gat_scores': [1.]*10}
    validate_predictions(row, q, cs, list(cs))
    row['ce_scores'][0] = float('nan')
    with pytest.raises(ValueError, match='ce_scores'):
        validate_predictions(row, q, cs, list(cs))
    row['ce_scores'][0] = 1
    with pytest.raises(ValueError, match='changed'):
        validate_predictions(row, q, cs, ['different']*10)


def test_path_escape_and_directory_hash(tmp_path):
    inside = tmp_path / 'checkout'
    inside.mkdir()
    assert safe_output(inside, inside / 'out') == inside / 'out'
    with pytest.raises(ValueError):
        safe_output(inside, inside / '..' / 'outside')
    with pytest.raises(ValueError):
        safe_output(inside, inside)
    (inside / 'weights').write_text('a')
    before = digest(inside)
    (inside / '__pycache__').mkdir()
    (inside / '__pycache__' / 'x.pyc').write_text('ignored')
    assert digest(inside) == before
    (inside / 'weights').write_text('b')
    assert digest(inside) != before


def test_retrieval_missing_and_duplicate_vectors_fail(tmp_path):
    import numpy as np
    p = tmp_path / 'v.npz'
    np.savez(p, ids=np.array(['a', 'b']), embeddings=np.array([[3.,4.], [1.,0.]]))
    ids, mat = load_retrieval(p, {'a', 'b'})
    assert ids == ['a', 'b']
    assert np.allclose(np.linalg.norm(mat, axis=1), 1)
    with pytest.raises(ValueError, match='cover'):
        load_retrieval(p, {'a', 'missing'})
    np.savez(p, ids=np.array(['a', 'av2']), embeddings=np.ones((2,2)))
    with pytest.raises(ValueError, match='duplicate'):
        load_retrieval(p, {'a'})


def test_parquet_preserves_native_index_order(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    import numpy as np
    p = tmp_path/'native.parquet'
    pq.write_table(pa.table({'paper_id':['b','a'], 'embedding':[[1.,0.],[0.,1.]]}), p)
    ids, mat = load_retrieval(p, {'a','b'})
    assert ids == ['b','a']
    assert np.allclose(mat, [[1,0],[0,1]])


@pytest.fixture
def audit_config(tmp_path):
    import run_rwcite_comparison as runner
    c = runner.template()
    def save(name, value, rows=False):
        path = tmp_path / name
        path.write_text(''.join(json.dumps(x)+'\n' for x in value) if rows else json.dumps(value))
        c['assets'][name] = {'path': str(path), 'sha256': digest(path)}
    corpus = [dict(paper_id=f'p{i}', title='P', abstract='', published_at='2024-01') for i in range(400)]
    qs = [dict(query_id=f'q{i}', title='Q', abstract='', published_at='2026-06', gold_ids=['p0']) for i in range(381)]
    save('test', qs, True)
    save('corpus', corpus, True)
    save('fixed400', [dict(query_id=q['query_id'], ranked_ids=[p['paper_id'] for p in corpus]) for q in qs], True)
    save('baseline_train', [dict(query_id='train', published_at='2025-01')], True)
    save('baseline_dev', [dict(query_id='dev', published_at='2025-05')], True)
    save('masked_sources', [q['query_id'] for q in qs])
    for n in list(c['assets']):
        if not Path(c['assets'][n]['path']).exists():
            save(n, {'placeholder': n})
    p = {'reviewed_by': 'synthetic test fixture, NOT a real audit', 'evidence_references': ['fixture'],
         'retrieval_inputs_paper_local_only': True, 'frozen_text_versions_audited': True,
         'fusion_selected_without_evaluation_gold': True, 'gat_uses_raw_scibert': True,
         'ce_evidence_recipe': {'selection':'graph_order_stride', 'max_sents':3, 'short':True,
                                'max_length':256, 'sentence_preprocessing':'as_stored_in_graph'},
         'components': {n: {'artifact_sha256': c['assets'][n]['sha256'], 'train_query_ids': ['train'],
                            'selection_query_ids': ['dev'], 'temporal_training_audited': True,
                            'target_leakage_audited': True} for n in ('ce', 'gat', 'struct')}}
    save('training_provenance', p)
    return c, save, p


def test_audit_passes_only_recorded_contracts(audit_config):
    c, _, _ = audit_config
    result = audit(c)
    assert result['n_queries'] == 381
    assert result['status'] == 'passed_recorded_contracts'


@pytest.mark.parametrize('component', ['ce', 'gat', 'struct'])
def test_eleven_extra_queries_cannot_have_been_trained_on(audit_config, component):
    c, save, p = audit_config
    p['components'][component]['train_query_ids'].append('q380')
    save('training_provenance', p)
    with pytest.raises(ValueError, match='training IDs'):
        audit(c)


def test_eval_model_selection_rejected(audit_config):
    c, save, p = audit_config
    p['components']['gat']['selection_query_ids'] = ['q0']
    save('training_provenance', p)
    with pytest.raises(ValueError, match='selection'):
        audit(c)


def test_ce_selector_mismatch_rejected(audit_config):
    c, save, p = audit_config
    p['ce_evidence_recipe']['selection'] = 'temporal_recent'
    save('training_provenance', p)
    with pytest.raises(ValueError, match='evidence recipe'):
        audit(c)


def test_changed_asset_hash_rejected(audit_config):
    c, _, _ = audit_config
    Path(c['assets']['ce']['path']).write_text('changed')
    with pytest.raises(ValueError, match='SHA256'):
        audit(c)


def test_only_370_masks_rejected(audit_config):
    c, save, _ = audit_config
    save('masked_sources', [f'q{i}' for i in range(370)])
    with pytest.raises(ValueError, match='ALL 381'):
        audit(c)


def test_whitening_future_fit_rejected(audit_config):
    c, save, p = audit_config
    save('whitening', {'mu': [0], 'W': [[1]]})
    p['whitening'] = {'artifact_sha256': c['assets']['whitening']['sha256'], 'fit_paper_ids': ['q0']}
    save('training_provenance', p)
    with pytest.raises(ValueError, match='whitening fit'):
        audit(c)


def test_merge_rejects_partial_and_extra_queries(tmp_path):
    from experiments.ewm_top10.run_rwcite_comparison import merge
    qs = {'q': {'query_id': 'q', 'published_at': '2026-06', 'gold_ids': ['p0']}}
    cs = {f'p{i}': {'paper_id': f'p{i}', 'published_at': '2024-01'} for i in range(10)}
    ws = {'q': {'ranked_ids': list(cs)}}
    with pytest.raises(ValueError, match='missing=1'):
        merge(tmp_path, qs, cs, ws)
    row = dict(query_id='q', ranked_ids=list(cs), scores=[1.]*10, ce_scores=[1.]*10,
               gat_scores=[1.]*10, window_ids=list(cs), universe_ids=list(cs))
    for mode in ('fixed400', 'endtoend400'):
        d = tmp_path/'rows'/mode
        d.mkdir(parents=True)
        (d/'q.json').write_text(json.dumps(row))
    assert len(merge(tmp_path, qs, cs, ws)) == 2
    (tmp_path/'rows/fixed400/foreign.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='extra=1'):
        merge(tmp_path, qs, cs, ws)


def test_query_corpus_version_mismatch_rejected(audit_config):
    c, save, _ = audit_config
    rows = [dict(paper_id=f'p{i}', title='P', abstract='', published_at='2024-01') for i in range(400)]
    rows.append(dict(paper_id='q0', title='different revision', abstract='', published_at='2026-06'))
    save('corpus', rows, True)
    with pytest.raises(ValueError, match='text or date mismatch'):
        audit(c)


def test_native_release_integration(tmp_path):
    release = os.environ.get('RWCITE_NATIVE_TEST_ROOT')
    if not release:
        pytest.skip('set RWCITE_NATIVE_TEST_ROOT to an extracted release checkout for native integration')
    subprocess.run([sys.executable, str(Path(__file__).with_name('rwcite_native_probe.py')),
                    release, str(tmp_path)], check=True, timeout=180,
                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1'))
