import os
import json
import openai
import numpy as np
import concurrent.futures
import tiktoken
import time
import cachetools


EMBED_QUERY_TEMPLATE_V1 = """
Instruct: Given a reference description, retrieve similar descriptions that mentioned all important information in the reference AND correctly described the task completion status specified in the reference.
Reference: {reference}
""".strip()


_JUDGE_CFG_IN_COMPUTE_SCORE = {}
_HELPER_TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
_EMBED_CACHE = cachetools.Cache(maxsize=1000)


def _get_judge_config():
    api_base = os.getenv("JUDGE_EMBED_MODEL_API_BASE")
    api_key = os.getenv("JUDGE_EMBED_MODEL_API_KEY")
    embed_model_name = os.getenv("JUDGE_EMBED_MODEL_NAME")
    max_token_to_judge = int(os.getenv("JUDGE_MAX_TOKEN_TO_JUDGE", "1024"))
    return {
        "api_base": api_base,
        "api_key": api_key,
        "embed_model_name": embed_model_name,
        "max_token_to_judge": max_token_to_judge,
    }


def _init_openai_client():
    judge_cfg = _get_judge_config()
    api_base = judge_cfg["api_base"]
    api_key = judge_cfg["api_key"]
    return openai.OpenAI(base_url=api_base, api_key=api_key)


def _parse_nsp(data_source, solution_str, parsing_metadata, max_token_to_judge):
    nsp_parse_tags = parsing_metadata["nsp_parse_tags"]
    start_tag, end_tag = nsp_parse_tags
    start_idx = solution_str.rfind(start_tag)
    end_idx = solution_str.rfind(end_tag)
    if start_idx == -1 or end_idx == -1:
        return "none"
    if start_idx >= end_idx:
        return "none"
    nsp_text = solution_str[start_idx+len(start_tag):end_idx]
    nsp_text = _HELPER_TOKENIZER.decode(
        _HELPER_TOKENIZER.encode(nsp_text)[:max_token_to_judge]
    )
    return nsp_text.strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    judge_cfg = _get_judge_config()
    judge_cfg_key = json.dumps(judge_cfg)
    if judge_cfg_key not in _JUDGE_CFG_IN_COMPUTE_SCORE:
        print(f"[compute_score] using {judge_cfg=}")
        _JUDGE_CFG_IN_COMPUTE_SCORE[judge_cfg_key] = judge_cfg

    obs_text = extra_info["obs_text"]
    obs_images = extra_info["obs_images"]
    action_text = extra_info["action_text"]
    parsing_metadata = extra_info.get("parsing_metadata", {})
    max_token_to_judge = judge_cfg["max_token_to_judge"]
    if parsing_metadata:
        solution_str = _parse_nsp(data_source, solution_str, parsing_metadata, max_token_to_judge)
    
    assert obs_images is None, \
        "multimodal observation is not supported yet"
    
    query = EMBED_QUERY_TEMPLATE_V1.format(reference=ground_truth)
    document = solution_str

    client = _init_openai_client()
    embed_model_name = judge_cfg["embed_model_name"]

    input_texts = [query, document]
    if query in _EMBED_CACHE:
        query_embedding = _EMBED_CACHE[query]
        input_texts = [document]

    try:
        response = client.embeddings.create(
            model=embed_model_name,
            input=input_texts,
            encoding_format="float",
        )
        if len(response.data) == 2:
            query_embedding = np.array(response.data[0].embedding)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            document_embedding = np.array(response.data[1].embedding)
            document_embedding = document_embedding / np.linalg.norm(document_embedding)
            _EMBED_CACHE[query] = query_embedding
        else:
            document_embedding = np.array(response.data[0].embedding)
            document_embedding = document_embedding / np.linalg.norm(document_embedding)
        sim_score = query_embedding @ document_embedding
    except Exception as e:
        print(f"[compute_score] error parsing {response=}: {e}")
        sim_score = 0.0
    reward = 1.0 if sim_score >= 0.8 else 0.0
    return reward


def _compute_single_score_wrapper(idx, data_source, solution_str, ground_truth, extra_info):
    return idx, compute_score(data_source, solution_str, ground_truth, extra_info)


def batched_compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs) -> list[float]:
    # concurrency = 16
    concurrency = 4
    _start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(len(data_sources)):
            future = executor.submit(
                _compute_single_score_wrapper,
                i, data_sources[i], solution_strs[i], ground_truths[i], extra_infos[i]
            )
            futures.append(future)
        
        results = [None] * len(futures)
        n_completed = 0
        for future in concurrent.futures.as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            n_completed += 1

            if n_completed % 100 == 0:
                elapsed_time = (time.time() - _start_time) / 60.0
                print(f"[batched_compute_score] {n_completed}/{len(futures)} completed in {elapsed_time:.2f}m")
    elapsed_time = (time.time() - _start_time) / 60.0
    print(f"[batched_compute_score] {len(futures)} completed in {elapsed_time:.2f}m")
    return results