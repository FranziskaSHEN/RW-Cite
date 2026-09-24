import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rwcite.retrieve.utils.utils import read_yaml_file
from rwcite.retrieve.retriever.corpus_utils import build_id2topics, build_paper_list, load_merged_embeddings
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def generate_topic_level_embeddings(model, tokenizer, paper_list, tmp_id_2_abs):
    id2topics = {
        entry["paper_id"]: [entry["Level 1"], entry["Level 2"], entry["Level 3"]]
        for entry in tmp_id_2_abs['train']
    }

    for topic_level in ['Level 1', 'Level 2', 'Level 3']:
        i = 0
        batch_size = 2048
        candidate_emb_list = []
        pbar = tqdm(total=len(paper_list))
        while i < len(paper_list):
            yield i / len(paper_list) / 3 if topic_level == 'Level 1' else 0.33 + i / len(paper_list) / 3 if topic_level == 'Level 2' else 0.66 + i / len(paper_list) / 3
            paper_batch = paper_list[i:i+batch_size]
            paper_text_batch = []
            for paper_id in paper_batch:
                topics = id2topics[paper_id][int(topic_level[6])-1]
                topic_text = ''
                for t in topics:
                    topic_text += t + ','
                paper_text_batch.append(topic_text)
            inputs = tokenizer(paper_text_batch, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs.to('cuda'))
                candidate_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                candidate_embeddings = candidate_embeddings.reshape(-1, 1024)
                candidate_emb_list.append(candidate_embeddings)

                i += len(candidate_embeddings)
                pbar.update(len(candidate_embeddings))

        all_candidate_embs = torch.cat(candidate_emb_list, 0)
        
        df = pd.DataFrame({
            "paper_id": paper_list,
            "embedding": list(all_candidate_embs.numpy())
        })
        
        if not os.path.exists('datasets/topic_level_embeds'):
            os.makedirs('datasets/topic_level_embeds')

        df.to_parquet(f'datasets/topic_level_embeds/{topic_level}_emb.parquet', engine='pyarrow', compression='snappy')
        
    all_candidate_embs_L1 = torch.tensor(np.array(pd.read_parquet('datasets/topic_level_embeds/Level 1_emb.parquet')['embedding'].tolist()))
    all_candidate_embs_L2 = torch.tensor(np.array(pd.read_parquet('datasets/topic_level_embeds/Level 2_emb.parquet')['embedding'].tolist()))
    all_candidate_embs_L3 = torch.tensor(np.array(pd.read_parquet('datasets/topic_level_embeds/Level 3_emb.parquet')['embedding'].tolist()))
    all_candidate_embs = all_candidate_embs_L1 + all_candidate_embs_L2 + all_candidate_embs_L3
    
    df = pd.DataFrame({
        "paper_id": paper_list,
        "embedding": list(all_candidate_embs.numpy())
    })
    
    df.to_parquet('datasets/topic_level_embeds/arxiv_papers_embeds.parquet', engine='pyarrow', compression='snappy')



def _load_embedder(embedder_name):
    load_kwargs = {"local_files_only": True}
    try:
        tokenizer = AutoTokenizer.from_pretrained(embedder_name, **load_kwargs)
        model = AutoModel.from_pretrained(embedder_name, **load_kwargs)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(embedder_name)
        model = AutoModel.from_pretrained(embedder_name)
    return tokenizer, model.to(device="cuda", dtype=torch.float16)


def retriever(query, retrieval_nodes_path, config_path="configs/config.yaml"):
    yield 0
    config = read_yaml_file(config_path)

    # Load the model and tokenizer to generate the embeddings
    embedder_name = config['retriever']['embedder']
    tokenizer, model = _load_embedder(embedder_name)


    # Load the arXiv dataset (HF base + local supplement)
    paper_list = build_paper_list(cache_dir="datasets/arxiv_topics")
    id2topics = build_id2topics(cache_dir="datasets/arxiv_topics")
    tmp_id_2_abs = {"train": [{"paper_id": pid, "Level 1": id2topics[pid][0], "Level 2": id2topics[pid][1], "Level 3": id2topics[pid][2]} for pid in paper_list]}


    # Generate the query embeddings
    inputs = tokenizer([query], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs.to('cuda'))
        query_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

    # Generate the candidate embeddings
    # Load the embeddings from the dataset, otherwise generate the embeddings and save them
    if config['retriever']['load_arxiv_embeds']:
        merged_ids, all_candidate_embs = load_merged_embeddings(
            use_hf_embeds=True,
            cache_dir="datasets/topic_level_embeds",
        )
        if merged_ids != paper_list:
            id_to_emb = dict(zip(merged_ids, all_candidate_embs))
            all_candidate_embs = np.stack([id_to_emb[pid] for pid in paper_list])
    else:
        merged_ids, all_candidate_embs = load_merged_embeddings(
            use_hf_embeds=False,
            cache_dir="datasets/topic_level_embeds",
        )
        if merged_ids != paper_list:
            if not os.path.exists("datasets/topic_level_embeds/arxiv_papers_embeds.parquet"):
                yield from generate_topic_level_embeddings(model, tokenizer, paper_list, tmp_id_2_abs)
                merged_ids, all_candidate_embs = load_merged_embeddings(use_hf_embeds=False)
            id_to_emb = dict(zip(merged_ids, all_candidate_embs))
            all_candidate_embs = np.stack([id_to_emb[pid] for pid in paper_list])

    all_candidate_embs = np.stack(all_candidate_embs)


    # Calculate the cosine similarity between the query and all candidate embeddings
    query_embeddings = np.array(query_embeddings)
    similarity_scores = cosine_similarity(query_embeddings, all_candidate_embs)[0]


    # Sort the papers by similarity scores and select the top K papers
    id_score_list = []
    for i in range(len(paper_list)):
        id_score_list.append([paper_list[i], similarity_scores[i]])

    sorted_scores = sorted(id_score_list, key=lambda i: i[-1], reverse = True)
    top_K_paper = [sample[0] for sample in sorted_scores[:config['retriever']['num_retrievals']]]

    papers_results = {
        paper: True
        for paper in top_K_paper
    }

    with open(retrieval_nodes_path, 'w') as f:
        json.dump(papers_results, f)

    yield 1.0
