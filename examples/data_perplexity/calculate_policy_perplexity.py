from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import argparse


def _remove_multiple_contents_per_turn(messages):
    fixed_messages = []
    for turn in messages:
        assert isinstance(turn['content'], list)
        assert len(turn['content']) == 1, f"Unexpected number of contents: {len(turn['content'])}"
        assert turn['content'][0]['type'] == 'text', f"Unexpected content type: {turn['content'][0]['type']}"
        fixed_messages.append({
            'role': turn['role'],
            'content': turn['content'][0]['text']
        })
    return fixed_messages


def tokenize_single(
    tokenizer,
    messages: list[dict],
    max_length: int,
    truncation: str = "error",
):
    # messages = _remove_multiple_contents_per_turn(messages)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    batch = tokenizer(
        text=[full_text],
        return_tensors="pt",
        padding='do_not_pad',  # this will be handled maunally later
        truncation='do_not_truncate'
    )
    input_ids = batch['input_ids'][0]
    attention_mask = batch['attention_mask'][0]

    # Create loss mask by identifying assistant responses
    loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

    # Process each message to find assistant responses
    for i, msg in enumerate(messages):
        # Get tokens for messages up to this point to find the start position
        prefix_messages = messages[: i + 1]
        prefix_msg_text = tokenizer.apply_chat_template(
            prefix_messages, tokenize=False
        )
        prefix_tokens = tokenizer(
            text=[prefix_msg_text],
            return_tensors="pt",
            padding='do_not_pad',
            truncation='do_not_truncate'
        )['input_ids'][0]
        if i > 0:
            prev_msg_text = tokenizer.apply_chat_template(
                messages[:i], tokenize=False
            )
            prev_tokens = tokenizer(
                text=[prev_msg_text],
                return_tensors="pt",
                padding='do_not_pad',
                truncation='do_not_truncate'
            )['input_ids'][0]
        else:
            prev_tokens = None

        # Calculate start and end positions
        start_pos = prev_tokens.shape[0] if prev_tokens is not None else 0
        end_pos = prefix_tokens.shape[0]

        # If this is an assistant message, set loss mask
        if msg["role"] == "assistant":
            loss_mask[start_pos:end_pos] = 1

    # Handle sequence length
    sequence_length = input_ids.shape[0]
    if sequence_length < max_length:
        # Pad sequences
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded_input_ids = torch.ones(size=(max_length - sequence_length,), dtype=input_ids.dtype) * pad_token_id
        padded_attention_mask = torch.zeros(size=(max_length - sequence_length,), dtype=attention_mask.dtype)
        padded_loss_mask = torch.zeros(size=(max_length - sequence_length,), dtype=loss_mask.dtype)

        input_ids = torch.cat((input_ids, padded_input_ids))
        attention_mask = torch.cat((attention_mask, padded_attention_mask))
        loss_mask = torch.cat((loss_mask, padded_loss_mask))
    elif sequence_length > max_length:
        if truncation == "left":
            input_ids = input_ids[-max_length :]
            attention_mask = attention_mask[-max_length :]
            loss_mask = loss_mask[-max_length :]
        elif truncation == "right":
            input_ids = input_ids[: max_length]
            attention_mask = attention_mask[: max_length]
            loss_mask = loss_mask[: max_length]
        elif truncation == "error":
            raise ValueError(f"{sequence_length=} is larger than {max_length=}")
        else:
            raise ValueError(f"Unknown truncation method {truncation}")
    
    # loss mask become labels
    labels = input_ids.clone()
    labels[loss_mask == 0] = -100
    labels[labels == tokenizer.pad_token_id] = -100  #
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    

def calculate_seq_perplexity(
    chat_message: list[dict],
    tokenizer,
    model,
    max_length: int = 2048,
):
    encoded_inputs = tokenize_single(
        tokenizer, chat_message, max_length=max_length
    )
    input_ids = encoded_inputs['input_ids'].to(model.device).unsqueeze(0)
    attention_mask = encoded_inputs['attention_mask'].to(model.device).unsqueeze(0)
    labels = encoded_inputs['labels'].to(model.device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        nll_likelihood = outputs.loss
    
    ppl = torch.exp(nll_likelihood)
    return ppl.cpu().item()

"""
Example:
python examples/data_perplexity/calculate_policy_perplexity.py \
    --dset_fpath data/sft/alfworld/solver_validation/alfworld_expert_optimal.parquet \
    --model_id Qwen/Qwen2.5-7B-Instruct
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_fpath", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()

    print(f"received args: {args}")

    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dset_fpath = args.dset_fpath
    dset = load_dataset('parquet', data_files=dset_fpath)['train']
    all_chat_messages = []
    for sample in dset:
        messages = sample['messages']
        # assumes text only model
        messages[0] = {
            'role': 'user',
            'content': messages[0]['content'][0]['text']
        }
        messages[1] = {
            'role': 'assistant',
            'content': messages[1]['content'][0]['text']
        }
        all_chat_messages.append(messages)
    
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

    ppls = []
    for chat_message in tqdm(all_chat_messages, desc="Calculating PPL"):
        ppl = calculate_seq_perplexity(chat_message, tokenizer, model, max_length=args.max_seq_len)
        ppls.append(ppl)
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Mean PPL: {np.mean(ppls):.4f}pm{np.std(ppls):.4f}")
    print(f"Min PPL: {np.min(ppls):.4f} Max PPL: {np.max(ppls):.4f}")