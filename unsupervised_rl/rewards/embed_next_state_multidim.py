import json
import openai
import numpy as np
import concurrent.futures
import tiktoken
import time
import cachetools


EMBED_QUERY_TEMPLATE_V1_STATEONLY = """
Instruct: Given a reference description, retrieve similar descriptions that mentioned all important information in the given reference.
Reference: {reference}
""".strip()


EMBED_QUERY_TEMPLATE_V1_TSONLY = """
Instruct: Using the reference, find all descriptions that semantically match the reference.
Reference: {reference}
""".strip()


EMBED_QUERY_TEMPLATE_V3 = """
Instruct: Using the reference, find all descriptions that semantically match the reference.
Reference: {reference}
""".strip()


_JUDGE_CFG_IN_COMPUTE_SCORE = {}
_HELPER_TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
_EMBED_CACHE = cachetools.Cache(maxsize=1000)



def _init_openai_client(api_base=None, api_key=None):
    # could include other clients thatn openai typed in the future, if needed
    return openai.OpenAI(base_url=api_base, api_key=api_key)


def _parse_ans_from_response(data_source, solution_str, parsing_tags: tuple[str, str], max_token_to_judge: int):
    start_tag, end_tag = parsing_tags
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


def _compute_score_for_one_pair(
    judge_cfg: dict,
    query: str,
    document: str,
    threshold: float = 0.8
):
    judge_embed_model_name = judge_cfg["judge_embed_model_name"]
    judge_api_base = judge_cfg["judge_api_base"]
    judge_api_key = judge_cfg["judge_api_key"]

    client = _init_openai_client(judge_api_base, judge_api_key)

    input_texts = [query, document]
    if query in _EMBED_CACHE:
        # to prevent cache going too big, just save the query embeddings
        query_embedding = _EMBED_CACHE[query]
        input_texts = [document]
    try:
        response = client.embeddings.create(
            model=judge_embed_model_name,
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
    reward = 1.0 if sim_score >= threshold else 0.0
    return reward


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    judge_api_base=None,
    judge_api_key=None,
    judge_embed_model_name=None,
    max_token_to_judge=None,
    max_ts_token_to_judge=None,
    ts_score_weight=0.5,
    embed_query_template_name=None,
) -> float:
    judge_cfg = {
        "judge_api_base": judge_api_base,
        "judge_api_key": judge_api_key,
        "judge_embed_model_name": judge_embed_model_name,
        "max_token_to_judge": max_token_to_judge,
        "max_ts_token_to_judge": max_ts_token_to_judge,
        "embed_query_template_name": embed_query_template_name,
    }
    judge_cfg_key = json.dumps(judge_cfg)
    if judge_cfg_key not in _JUDGE_CFG_IN_COMPUTE_SCORE:
        print(f"[compute_score] using {judge_cfg=}")
        _JUDGE_CFG_IN_COMPUTE_SCORE[judge_cfg_key] = judge_cfg

    obs_text = extra_info["obs_text"]
    obs_images = extra_info["obs_images"]
    action_text = extra_info["action_text"]
    parsing_metadata = extra_info.get("parsing_metadata", {})

    state_pred_str = _parse_ans_from_response(
        data_source, solution_str, parsing_metadata['nsp_parse_tags'], max_token_to_judge
    )
    ts_pred_str = _parse_ans_from_response(
        data_source, solution_str, parsing_metadata['ts_parse_tags'], max_ts_token_to_judge
    )
    
    assert obs_images is None, \
        "multimodal observation is not supported yet"
    
    if embed_query_template_name == "v1":
        state_template = EMBED_QUERY_TEMPLATE_V1_STATEONLY
        ts_template = EMBED_QUERY_TEMPLATE_V1_TSONLY
    elif embed_query_template_name == "v3":
        # yes, same template for both
        state_template = EMBED_QUERY_TEMPLATE_V3
        ts_template = EMBED_QUERY_TEMPLATE_V3
    else:
        raise ValueError(f"unknown {embed_query_template_name=}")

    state_gt = ground_truth['observation']
    state_query = state_template.format(reference=state_gt)
    ts_gt = ground_truth['task_status']
    ts_query = ts_template.format(reference=ts_gt)
    
    state_score = _compute_score_for_one_pair(
        judge_cfg,
        state_query,
        state_pred_str,
        threshold=0.9
    )
    ts_score = _compute_score_for_one_pair(
        judge_cfg,
        ts_query,
        ts_pred_str,
        threshold=0.8
    )
    reward = (1.0 - ts_score_weight) * state_score + ts_score_weight * ts_score
    return reward


def _compute_single_score_wrapper(idx, *args, **kwargs):
    return idx, compute_score(*args, **kwargs)


def batched_compute_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    judge_api_base=None,
    judge_api_key=None,
    judge_embed_model_name=None,
    max_token_to_judge=128,
    max_ts_token_to_judge=16,
    ts_score_weight=0.5,
    embed_query_template_name=None,
    judge_api_concurrency=4,
    **kwargs
) -> list[float]:
    concurrency = int(judge_api_concurrency)
    max_token_to_judge = int(max_token_to_judge)

    _start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(len(data_sources)):
            future = executor.submit(
                _compute_single_score_wrapper,
                i, data_sources[i], solution_strs[i], ground_truths[i], extra_infos[i],
                judge_api_base=judge_api_base,
                judge_api_key=judge_api_key,
                judge_embed_model_name=judge_embed_model_name,
                max_token_to_judge=max_token_to_judge,
                max_ts_token_to_judge=max_ts_token_to_judge,
                ts_score_weight=ts_score_weight,
                embed_query_template_name=embed_query_template_name
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