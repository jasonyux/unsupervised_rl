import json
import os
import openai
import numpy as np
import concurrent.futures
import tiktoken
import time


WM_JUDGE_PROMPT_V3 = """
You are a helpful judge AI agent. Your task is to evaluate the quality of a predicted next observation in a customer service environment.

# Action proposed by an AI agent
{action_text}

# Predicted Next observation
After executing the above action, another AI agent predicted the next observation as follows:
{next_obs_desc}

# Actual Next observation
The actual next observation from the environment is as follows:
{actual_next_obs_text}

# Evaluate the predicted next observation
Now, your task is to evaluate how well the predicted next observation matches the actual next observation.
Specifically, you need to judge whether the prediction demonstrates a *genuine* understanding of the environment dynamics relevant to the task, using the actual next observation as reference.

Award points for:
- Specific and accurate environment description in the prediction that matches the actual next observation.
- Correct prediction of task completion status.
Penalize:
- Incorrect environment descriptions in the prediction that differ from the actual next observation, especially those that could impact (future) task completion.
- Missing or wrong task completion status in the prediction compared to the actual next observation.
STRONGLY penalize:
- The prediction misses key information from the actual next observation that is important for solving the task.
- Vague or generic descriptions that is consistent with the actual next observation but does not really show genuine understanding of the environment dynamics.

# Your output format
Your task is to output a JSON object in the following format:
<json>
{{
    "positive aspects": "enumerating good aspects of the agent's prediction according to the guidelines above, using the actual next observation as reference.",  # no more than 200 words
    "negative aspects": "enumerating bad aspects of the agent's prediction according to the guidelines above, using the actual next observation as reference.",  # no more than 200 words
    "overall analysis": "overall analysis weighing both aspects, and whether you think the agent has a genuine understanding of the environment dynamics.",  # no more than 50 words
    "score": 0.0-1.0 # overall score summarizing your judgement. higher the better.
}}
</json>
Directly output the JSON object. DO NOT generate anything else.
""".strip()



WM_JUDGE_PROMPT_V4 = """
You are a helpful judge AI agent. Your task is to evaluate the quality of a predicted next observation in a customer service environment.

# Action proposed by an AI agent
{action_text}

# Predicted Next observation
After executing the above action, another AI agent predicted the next observation as follows:
{next_obs_desc}

# Actual Next observation
The actual next observation from the environment is as follows:
{actual_next_obs_text}

# Evaluate the predicted next observation
Now, your task is to evaluate how well the predicted next observation matches the actual next observation.
Specifically, you need to judge whether the prediction demonstrates a *genuine* understanding of the environment dynamics relevant to the task, using the actual next observation as reference.
- If all important task-related information in the actual next observation is PRESENT in the predicted next observation AND the task completion status matches the reference in the actual next observation, assign a score of 1.0.
- Otherwise, assign a score of 0.0.

# Your output format
Your task is to output a JSON object in the following format:
<json>
{{
    "analysis": "which important task related content is present/missing in the predicted next observation, and whether the task completion status is correctly predicted.",  # no more than 200 words
    "score": 0.0 or 1.0
}}
</json>
Directly output the JSON object. DO NOT generate anything else.
""".strip()


_JUDGE_CFG_IN_COMPUTE_SCORE = {}
_HELPER_TOKENIZER = tiktoken.encoding_for_model("gpt-4o")


def _get_judge_config():
    api_base = os.getenv("JUDGE_MODEL_API_BASE")
    api_key = os.getenv("JUDGE_MODEL_API_KEY")
    judge_model_id = os.getenv("JUDGE_MODEL_NAME")
    judge_gen_kwargs = json.loads(os.getenv("JUDGE_GEN_KWARGS", "{}"))
    max_token_to_judge = int(os.getenv("JUDGE_MAX_TOKEN_TO_JUDGE", "1024"))
    return {
        "api_base": api_base,
        "api_key": api_key,
        "judge_model_id": judge_model_id,
        "judge_gen_kwargs": judge_gen_kwargs,
        "max_token_to_judge": max_token_to_judge,
    }


def _init_openai_client(api_base, api_key):
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


def _has_language_mixing(s, threshold=0.1) -> bool:
    not_ok_strings = []
    for ss in s.split():
        if not ss.isascii():
            not_ok_strings.append(ss)
    not_ok_text = ' '.join(not_ok_strings)
    not_ok_n_tokens = len(_HELPER_TOKENIZER.encode(not_ok_text, disallowed_special=()))
    total_n_tokens = len(_HELPER_TOKENIZER.encode(s, disallowed_special=()))
    if total_n_tokens == 0:
        return False
    if not_ok_n_tokens / total_n_tokens > threshold:
        return True
    return False


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
    judge_model_id='',
    judge_temperature=0.7,
    judge_top_p=0.95,
    judge_max_completion_tokens=2048,
    max_token_to_judge=128,
    judge_template_name: str = 'v3'
) -> float:
    judge_cfg = {
        "judge_api_base": judge_api_base,
        "judge_model_id": judge_model_id,
        "judge_temperature": judge_temperature,
        "judge_top_p": judge_top_p,
    }
    judge_cfg_key = str(json.dumps(judge_cfg, sort_keys=True))
    if judge_cfg_key not in _JUDGE_CFG_IN_COMPUTE_SCORE:
        print(f"[compute_score] using {judge_cfg=}")
        _JUDGE_CFG_IN_COMPUTE_SCORE[judge_cfg_key] = judge_cfg

    obs_text = extra_info["obs_text"]
    obs_images = extra_info["obs_images"]
    action_text = extra_info["action_text"]
    parsing_metadata = extra_info.get("parsing_metadata", {})
    solution_str_original = solution_str
    if parsing_metadata:
        solution_str = _parse_nsp(data_source, solution_str, parsing_metadata, max_token_to_judge)
    
    assert obs_images is None, \
        "multimodal observation is not supported yet"
    
    if judge_template_name == 'v3':
        prompt = WM_JUDGE_PROMPT_V3.format(
            action_text=action_text,
            next_obs_desc=solution_str,
            actual_next_obs_text=ground_truth,
        )
    elif judge_template_name == 'v4':
        prompt = WM_JUDGE_PROMPT_V4.format(
            action_text=action_text,
            next_obs_desc=solution_str,
            actual_next_obs_text=ground_truth,
        )
    else:
        raise ValueError(f"unknown judge_template {judge_template_name=}")

    client = _init_openai_client(judge_api_base, judge_api_key)

    try:
        response = client.chat.completions.create(
            model=judge_model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=judge_temperature,
            top_p=judge_top_p,
            max_completion_tokens=judge_max_completion_tokens,
        ).choices[0].message.content
        json_output = response.replace("<json>", "").replace("</json>", "").replace("```json", "").replace("```", "")

        rubric_data = json.loads(json_output)
        parsed_reward = float(rubric_data['score'])
        parsed_reward = np.clip(parsed_reward, 0.0, 1.0).item()
    except Exception as e:
        print(f"[compute_score] error parsing {response=}: {e}")
        parsed_reward = 0.0


    if _has_language_mixing(solution_str_original):
        # print(f"[compute_score] warning: language mixing found in response [{solution_str_original}]")
        parsed_reward -= 0.2

    if not _has_thinking(solution_str_original):
        # print(f"[compute_score] warning: no thinking found in response [{solution_str}]")
        parsed_reward -= 0.1
    return parsed_reward


def _compute_single_score_wrapper(idx, *args, **kwargs):
    return idx, compute_score(*args, **kwargs)


def batched_compute_score(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    judge_api_base=None,
    judge_api_key=None,
    judge_model_id='',
    judge_temperature=0.7,
    judge_top_p=0.95,
    judge_max_completion_tokens=2048,
    max_token_to_judge=128,
    judge_template_name: str = 'v3',
    judge_api_concurrency=4,
    **kwargs
) -> list[float]:
    concurrency = judge_api_concurrency
    _start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(len(data_sources)):
            future = executor.submit(
                _compute_single_score_wrapper,
                i, data_sources[i], solution_strs[i], ground_truths[i], extra_infos[i],
                judge_api_base=judge_api_base,
                judge_api_key=judge_api_key,
                judge_model_id=judge_model_id,
                judge_temperature=judge_temperature,
                judge_top_p=judge_top_p,
                judge_max_completion_tokens=judge_max_completion_tokens,
                max_token_to_judge=max_token_to_judge,
                judge_template_name=judge_template_name
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