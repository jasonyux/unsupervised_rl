import json
import os
import openai
import numpy as np
import concurrent.futures
import tiktoken
import time

# WM_JUDGE_PROMPT_V1 = """
# {obs_text}

# # Action proposed by an AI agent
# {action_text}

# # Predicted Next observation
# After executing the above action, another AI agent predicted the next observation as follows:
# {next_obs_desc}

# # Actual Next observation
# The actual next observation from the environment is as follows:
# {actual_next_obs_text}

# # Evaluate the predicted next observation
# Now, your task is to evaluate the correctness of the predicted next observation by comparing it to the actual next observation.
# Specifically, you need to check if the predicted next observation is consistent with the actual next observation based on the following dimensions:
# 1) **coverage**: maximum 0.2 points
# - all important task-related changes in the actual next observation is covered in the predicted next observation (+0.2 points)
# - only some of the important task-related changes in the actual next observation is covered in the predicted next observation (+0.05~0.15 point depending on the proportion of the changes covered)
# - actual next observation is completely different from the predicted next observation (assign 0 points)
# 2) **precision**: maximum 0.4 points
# - all important task-related changes in the predicted next observation correctly showed up on the actual next observation (+0.4 points)
# - only some of the predicted changes are shown on the actual next observation (+0.1~0.3 point depending on the proportion of correctly predicted changes)
# - actual next observation is completely different from the predicted next observation (assign 0 points)
# 3) **confidence**: maximum 0.2 points
# - correctly predicted changes are also confident (+0.0~0.2 point depending on the proportion of correctly predicted changes)
# - correctly predicted changes are not confident (+0.0~0.1 point depending on the proportion of correctly predicted changes)
# 4) **success**: maximum 0.2 points
# - correctly predicted that the task is successfully completed (+0.2 points)
# - correctly predicted that the task is not yet completed (+0.05 points)
# - incorrect prediction of task completion or did not predict task completion (assign 0 points)

# Note: precision, coverage, and confidence are independent dimensions.
# Note: if the prediction made a mistake/missed *unimportant* changes, ignore them and only focus on awarding points for correctly predicted important changes using the above criteria.
# Note: to calculate proportions, first enumerate all atomic facts in the actual next observation that are important to solving the task. Then, check whether the predicted next observation covered each of them. Proportions is then #correctly predicted atomic facts / #total atomic facts.

# # Your output format
# Your task is to output a JSON object in the following format:
# <json>
# {{
#     "coverage analysis": "which coverage criteria in the evaluation rules are satisfied, and how many points to assign.", # no more than 100 words
#     "coverage score": 0.0-0.4, # score for the coverage dimension
#     "precision analysis": "which precision criteria in the evaluation rules are satisfied, and how many points to assign.", # no more than 100 words
#     "precision score": 0.0-0.4, # score for the precision dimension
#     "confidence analysis": "which confidence criteria in the evaluation rules are satisfied, and how many points to assign.", # no more than 100 words
#     "confidence score": 0.0-0.2, # score for the confidence dimension
#     "success analysis": "which success criteria in the evaluation rules are satisfied, and how many points to assign.", # no more than 100 words
#     "success score": 0.0-0.2, # score for the success dimension
#     "score": 0.0-1.0 # total score; add the coverage score, precision score, confidence score, and success score
# }}
# </json>
# Directly output the JSON object. DO NOT generate anything else.
# """.strip()


WM_JUDGE_PROMPT_V1 = """
{obs_text}

# Action proposed by an AI agent
{action_text}

# Predicted Next observation
After executing the above action, another AI agent predicted the next observation as follows:
{next_obs_desc}

# Actual Next observation
The actual next observation from the environment is as follows:
{actual_next_obs_text}

# Evaluate the predicted next observation
Now, your task is to evaluate the correctness of the predicted next observation by comparing it to the actual next observation.
Specifically, you need to check if the predicted next observation is consistent with the actual next observation based on the following dimensions:
1) **coverage**: maximum 0.2 points.
- all important task-related changes in the actual next observation can be found in the predicted next observation (assign 0.2 points)
- only some of the important task-related changes in the actual next observation can be found in the predicted next observation (assign 0.05~0.15 point depending on the proportion of the changes covered)
- actual next observation is completely different from the predicted next observation (assign 0 points)
2) **specificity**: maximum 0.4 points.
- all important task related changes are described in detail in the predicted next observation (assign 0.4 points)
- some of the important task related changes are vaguely described, although it is aligned with the actual next observation (assign 0.1-0.2 point depending on the proportion of changes that are correctly described IN DETAIL)
- actual next observation is completely different from the predicted next observation (assign 0 points)
3) **confidence**: maximum 0.2 points
- correctly predicted changes are also confident (assign 0.0~0.2 point depending on the proportion of correctly predicted changes)
- correctly predicted changes are not confident (assign 0.0~0.1 point depending on the proportion of correctly predicted changes)
- one or more of the predicted changes are wrong and the agent is confident (assign 0 points)
4) **success**: maximum 0.2 points
- correctly predicted that the task is successfully completed (assign 0.2 points)
- correctly predicted that the task is not yet completed (assign 0.05 points)
- incorrect prediction of task completion or did not predict task completion (assign 0 points)

Note: coverage, specificity, and confidence are independent dimensions.
Note: to calculate proportions, first enumerate all atomic facts in the actual next observation that are important to solving the task. Then, check whether the predicted next observation covered each of them. Proportions is then #correctly predicted atomic facts / #total atomic facts.
Note: if the agent used vague/general terms to describe the changes, treat them as INCORRECT. Only award points to predictions that are detailed and specific, showing that the agent has a deep understanding of the environment dynamics.

# Your output format
Your task is to output a JSON object in the following format:
<json>
{{
    "coverage analysis": "which coverage criteria in the evaluation rules are satisfied, and how many points to assign.", # no more than 100 words
    "coverage score": 0.0-0.2, # score for the coverage dimension
    "specificity analysis": "which specificity criteria in the evaluation rules are satisfied, and how many points to assign.", # no more than 100 words
    "specificity score": 0.0-0.4, # score for the specificity dimension
    "confidence analysis": "which confidence criteria in the evaluation rules are satisfied, and how many points to assign.", # no more than 100 words
    "confidence score": 0.0-0.2, # score for the confidence dimension
    "success analysis": "which success criteria in the evaluation rules are satisfied, and how many points to assign.", # no more than 100 words
    "success score": 0.0-0.2, # score for the success dimension
    "score": 0.0-1.0 # total score; add the coverage score, specificity score, confidence score, and success score
}}
</json>
Directly output the JSON object. DO NOT generate anything else.
""".strip()


WM_JUDGE_PROMPT_V2 = """
{obs_text}

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
Specifically, you need to judge whether the prediction demonstrates a *genuine* and *deep* understanding of the environment dynamics relevant to the task, using the actual next observation as reference.

Award points for:
- Specific and accurate environment details in the prediction that are consistent with the actual next observation.
- Correct prediction of task completion status.
- Correct confidence calibration (high for correct, low for incorrect predictions).
Penalize:
- Inconsistent or incorrect environment descriptions with the actual next observation.
- Wrong task completion status.
STRONGLY penalize:
- Vague or generic descriptions that are consistent with the actual next observation but does not really show genuine or deep understanding of the environment dynamics.
- High confidence in incorrect predictions.

# Your output format
Your task is to output a JSON object in the following format:
<json>
{{
    "positive aspects": "enumerating good aspects of the agent's prediction according to the guidelines above, using the actual next observation as reference.",  # no more than 200 words
    "negative aspects": "enumerating bad aspects of the agent's prediction according to the guidelines above, using the actual next observation as reference.",  # no more than 200 words
    "overall analysis": "overall analysis weighing both aspects, and whether you think the agent has a deep understanding of the environment dynamics.",  # no more than 50 words
    "score": 0.0-1.0 # overall score summarizing your judgement. higher the better.
}}
</json>
Directly output the JSON object. DO NOT generate anything else.
""".strip()


WM_JUDGE_PROMPT_V3 = """
{obs_text}

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
- Incorrect environment descriptions in the prediction that differ from the actual next observation, especially those that cold impact (future) task completion. 
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
{obs_text}

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
    
    # prompt = WM_JUDGE_PROMPT_V1.format(
    # prompt = WM_JUDGE_PROMPT_V2.format(
    # prompt = WM_JUDGE_PROMPT_V3.format(
    prompt = WM_JUDGE_PROMPT_V4.format(
        obs_text=obs_text,
        action_text=action_text,
        next_obs_desc=solution_str,
        actual_next_obs_text=ground_truth,
    )

    client = _init_openai_client()
    judge_model_id = judge_cfg["judge_model_id"]
    judge_gen_kwargs = judge_cfg["judge_gen_kwargs"]

    try:
        response = client.chat.completions.create(
            model=judge_model_id,
            messages=[{"role": "user", "content": prompt}],
            **judge_gen_kwargs,
        ).choices[0].message.content
        json_output = response.replace("<json>", "").replace("</json>", "").replace("```json", "").replace("```", "")

        rubric_data = json.loads(json_output)
        parsed_reward = float(rubric_data['score'])
        parsed_reward = np.clip(parsed_reward, 0.0, 1.0).item()
    except Exception as e:
        print(f"[compute_score] error parsing {response=}: {e}")
        parsed_reward = 0.0
    return parsed_reward


def _compute_single_score_wrapper(idx, data_source, solution_str, ground_truth, extra_info):
    return idx, compute_score(data_source, solution_str, ground_truth, extra_info)


def batched_compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs) -> list[float]:
    concurrency = 16
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