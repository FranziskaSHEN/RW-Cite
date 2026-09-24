"""Offline native-code regression probe, executed in a fresh interpreter."""
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rwcite_contract import canonical_graph, digest, load_config, temporal_view, write_json
from rwcite_worker import make_pack, native_imports


def main():
    release, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    native_imports(release / 'rwcite')
    import networkx as nx
    import numpy as np
    import torch
    from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizerFast
    from rwcite.gat.model import GatMvpModel, GatMvpConfig
    from rwcite.gat.fuse_ce import _fuse_scores, _rrf_from_scores
    from rwcite.ranker.rr_ranker import RankerModel, FEATURE_NAMES
    from rwcite.ranker.rr_ranker_ce_sent import collect_cite_sentences
    torch.set_num_threads(2)
    torch.manual_seed(7)

    # A future shared citer must not create an old-old cocitation link.
    cs = {p: dict(paper_id=p, title=p, abstract='', published_at=m)
          for p, m in [('a','2024-01'), ('b','2024-02'), ('later','2026-07'), ('past','2025-01')]}
    g = nx.DiGraph()
    g.add_nodes_from(cs)
    g.add_edge('later', 'a', sentence='future')
    g.add_edge('later', 'b', sentence='future')
    g.add_edge('past', 'a', sentence='past')
    graph = canonical_graph(g, cs, {})
    ids = sorted(graph)
    X = np.random.default_rng(0).normal(size=(4,16)).astype(np.float32)
    before = make_pack(ids, X, graph)
    after_graph = temporal_view(graph, (2026,6), set())
    after = make_pack(ids, X, after_graph)
    a,b,later = [ids.index(p) for p in ('a','b','later')]
    assert before.adj_cocite_full[a,b] > 0
    assert after.adj_cocite_full[a,b] == 0
    assert later not in after.nbr_pool
    assert collect_cite_sentences(after_graph, 'a') == ['past']
    # Confirm the precise native zero-based RRF convention, including constant inputs.
    sc, sg = np.array([1.,3.,2.],dtype=np.float32), np.array([3.,1.,2.],dtype=np.float32)
    rrf = _rrf_from_scores(sc, sg, rrf_k=20)
    assert np.allclose(rrf, [1/22+1/20, 1/20+1/22, 2/21])
    z = lambda x: (x-x.mean())/x.std()
    assert np.allclose(_fuse_scores(sg=sg, sc=sc, alpha=.4, fuse_mode='l0_rrf'),
                       .6*(.6*z(sc)+.4*z(sg))+.4*z(rrf), atol=1e-5)
    assert np.isfinite(_fuse_scores(sg=np.ones(3), sc=np.ones(3), alpha=.4, fuse_mode='l0_rrf')).all()

    # Small local BERTs exercise the REAL native CE, GAT, feature and fusion code.
    vocab = work/'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\npaper\nrobot\nrelated\nwork\n')
    tok = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    bert_cfg = BertConfig(vocab_size=9, hidden_size=16, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=24, max_position_embeddings=512,
                          num_labels=1)
    for name, model in [('base', BertModel(bert_cfg)), ('ce', BertForSequenceClassification(bert_cfg))]:
        model.save_pretrained(work/name)
        tok.save_pretrained(work/name)
    gc = GatMvpConfig(text_dim=16, hidden_dim=8, n_layers=1, pc_dim=4, dropout=0)
    torch.save({'cfg': vars(gc), 'model': GatMvpModel(gc).state_dict()}, work/'gat.pt')
    RankerModel(w=np.ones(10), b=0., mu=np.zeros(10), sd=np.ones(10),
                feature_names=FEATURE_NAMES).save(work/'struct.npz')
    corpus = [dict(paper_id=f'2401.{i:05d}', title='paper robot', abstract='related work',
                   published_at='2024-01') for i in range(405)]
    queries = [dict(query_id='2606.00001', title='robot', abstract='paper', published_at='2026-06', gold_ids=['2401.00000'])]
    graph = nx.DiGraph()
    for row in corpus:
        graph.add_node(row['paper_id'])
    graph.add_node(queries[0]['query_id'])
    # Query references must not propagate to the candidate window or evidence.
    graph.add_edge(queries[0]['query_id'], '2401.00000', sentence='FORBIDDEN_QUERY_SENTENCE')
    for i in range(404):
        graph.add_edge(corpus[i+1]['paper_id'], corpus[i]['paper_id'], sentence='related paper')
    nx.write_gexf(graph, work/'graph.gexf')
    for name, rows in [('corpus',corpus), ('test',queries),
                        ('fixed400',[dict(query_id=queries[0]['query_id'],ranked_ids=[p['paper_id'] for p in corpus[:400]])])]:
        (work/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(work/'mask.json', [queries[0]['query_id']])
    np.savez(work/'retrieval.npz', ids=np.array([p['paper_id'] for p in corpus]),
             embeddings=np.random.default_rng(0).normal(size=(405,16)).astype(np.float32))
    names = {'release_code':release/'rwcite','corpus':work/'corpus.jsonl','test':work/'test.jsonl',
             'fixed400':work/'fixed400.jsonl','masked_sources':work/'mask.json','graph':work/'graph.gexf',
             'scibert':work/'base','bge':work/'base','ce':work/'ce','gat':work/'gat.pt',
             'struct':work/'struct.npz','retrieval_embeddings':work/'retrieval.npz'}
    cfg = {'assets':{k:{'path':str(v)} for k,v in names.items()}, 'universe_env':{}, 'seed':7,
           'cpu_threads_per_worker':2}
    write_json(work/'config.json', cfg)
    out = work/'out'
    write_json(out/'audit.json', {'synthetic_internal_probe':True})
    write_json(out/'run_identity.json', {'config':load_config(work/'config.json'), 'max_queries':0})
    command = [sys.executable, str(Path(__file__).resolve().parents[1]/'rwcite_worker.py'),
               '--config',str(work/'config.json'),'--output',str(out),
               '--audit-sha256',digest(out/'audit.json'),'--shard','0','--shards','1','--device','cpu']
    subprocess.run(command, check=True, timeout=100)
    fixed = json.loads((out/'rows/fixed400/2606.00001.json').read_text())
    assert len(fixed['ranked_ids']) == 400 and len(set(fixed['ranked_ids'])) == 400
    assert not any('FORBIDDEN_QUERY_SENTENCE' in e['text'] for e in fixed['candidate_evidence'])
    assert '2606.00001' not in fixed['universe_ids']
    before_hash = digest(out/'rows')
    # Resume does not overwrite completed rows or repeat query scoring.
    subprocess.run(command, check=True, timeout=100)
    assert digest(out/'rows') == before_hash
    # Changing gold and the hidden query's outgoing sentence cannot change scores.
    queries[0]['gold_ids'] = ['2401.00123']
    (work/'test.jsonl').write_text(json.dumps(queries[0])+'\n')
    graph['2606.00001']['2401.00000']['sentence'] = 'DIFFERENT_HIDDEN_SENTENCE'
    nx.write_gexf(graph, work/'graph.gexf')
    out2 = work/'counterfactual'
    write_json(out2/'audit.json', {'synthetic_internal_probe':True})
    changed_command = list(command)
    changed_command[changed_command.index('--output')+1] = str(out2)
    subprocess.run(changed_command, check=True, timeout=100)
    changed = json.loads((out2/'rows/fixed400/2606.00001.json').read_text())
    for key in ('scores', 'ce_scores', 'gat_scores', 'ranked_ids', 'candidate_evidence', 'universe_ids'):
        assert changed[key] == fixed[key], key
    print('native CE+GAT inference, temporal arrays, fusion and resume: PASS')


if __name__ == '__main__':
    main()
