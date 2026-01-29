import os
import json
import openai
import numpy as np
import concurrent.futures
import tiktoken
import time
import cachetools
import random
from enum import Enum
from rouge_score import rouge_scorer



EMBED_QUERY_TEMPLATE_V1_NOTS = """
Instruct: Given a reference description, retrieve similar descriptions that mentioned all important information in the given reference.
Reference: {reference}
""".strip()



_JUDGE_CFG_IN_COMPUTE_SCORE = {}
_HELPER_TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
_EMBED_CACHE = cachetools.Cache(maxsize=1000)


def compute_rouge_score(ground_truth: str, prediction: str, rouge_metric: str = 'rouge1') -> float:
    scorer = rouge_scorer.RougeScorer([rouge_metric], use_stemmer=True)
    scores = scorer.score(
        target=ground_truth,
        prediction=prediction
    )
    return scores[rouge_metric].fmeasure


def _init_openai_client(api_base=None, api_key=None):
    # judge_cfg = _get_judge_config()
    # api_base = judge_cfg["api_base"]
    # api_key = judge_cfg["api_key"]
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


def _has_thinking(solution_str: str) -> bool:
    start_of_think_tag = "<think>"
    end_of_think_tag = "</think>"
    start_idx = solution_str.find(start_of_think_tag)
    end_idx = solution_str.find(end_of_think_tag)
    if start_idx == -1 or end_idx == -1:
        return False
    elif start_idx >= end_idx:
        return False
    else:
        thinking_portion = solution_str[start_idx+len(start_of_think_tag):end_idx]
        n_tokens = len(_HELPER_TOKENIZER.encode(thinking_portion))
        return n_tokens > 16


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    judge_api_base=None,
    judge_api_key=None,
    judge_embed_model_name=None,
    max_token_to_judge=None,
    embed_query_template_name=None,
    threshold: float = 0.8,
    penalize_no_thinking: bool = True,
    tool_rouge_metric: str = 'rougeL',
    tool_rouge_bin: float = 0.2,
    has_think_in_prompt: bool = False,
) -> float:
    judge_cfg = {
        "judge_api_base": judge_api_base,
        "judge_api_key": judge_api_key,
        "judge_embed_model_name": judge_embed_model_name,
        "max_token_to_judge": max_token_to_judge,
        "embed_query_template_name": embed_query_template_name,
    }
    judge_cfg_key = json.dumps(judge_cfg)
    if judge_cfg_key not in _JUDGE_CFG_IN_COMPUTE_SCORE:
        print(f"[compute_score] using {judge_cfg=}")
        _JUDGE_CFG_IN_COMPUTE_SCORE[judge_cfg_key] = judge_cfg

    _debug_reward_stats = {}
    obs_text = extra_info["obs_text"]
    obs_images = extra_info["obs_images"]
    action_text = extra_info["action_text"]
    parsing_metadata = extra_info.get("parsing_metadata", {})
    if has_think_in_prompt:
        solution_str = "<think>" + solution_str

    solution_str_original = solution_str
    if parsing_metadata:
        solution_str = _parse_nsp(data_source, solution_str, parsing_metadata, max_token_to_judge)
    
    assert obs_images is None, \
        "multimodal observation is not supported yet"
    
    if embed_query_template_name == "v1_nots":
        template = EMBED_QUERY_TEMPLATE_V1_NOTS
    else:
        raise ValueError(f"unknown {embed_query_template_name=}")

    query = template.format(reference=ground_truth)
    document = solution_str

    client = _init_openai_client(judge_api_base, judge_api_key)

    input_texts = [query, document]
    if query in _EMBED_CACHE:
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

    response_type = extra_info['metadata']['response_type']
    _debug_reward_stats['response_type'] = response_type
    if response_type == "tool":
        reward = compute_rouge_score(ground_truth, solution_str, rouge_metric=tool_rouge_metric)
        _debug_reward_stats['rouge_score'] = reward
        reward = round(reward / tool_rouge_bin) * tool_rouge_bin
    else:
        _debug_reward_stats['sim_score'] = sim_score
        reward = 1.0 if sim_score >= threshold else 0.0
    _debug_reward_stats['main_reward'] = reward

    if penalize_no_thinking:
        if not _has_thinking(solution_str_original):
            # print(f"[compute_score] warning: no thinking found in response [{solution_str_original}]")
            reward -= 0.1
    _debug_reward_stats['final_reward'] = reward

    if random.random() < 0.01:
        print(f"[compute_score] debug reward stats: {_debug_reward_stats} from response [{solution_str_original}] and gt [{ground_truth}]")
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
    embed_query_template_name=None,
    threshold: float = 0.8,
    penalize_no_thinking: bool = True,
    tool_rouge_metric: str = 'rougeL',
    tool_rouge_bin: float = 0.2,
    has_think_in_prompt: bool = False,
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
                embed_query_template_name=embed_query_template_name,
                threshold=threshold,
                penalize_no_thinking=penalize_no_thinking,
                tool_rouge_metric=tool_rouge_metric,
                tool_rouge_bin=tool_rouge_bin,
                has_think_in_prompt=has_think_in_prompt,
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