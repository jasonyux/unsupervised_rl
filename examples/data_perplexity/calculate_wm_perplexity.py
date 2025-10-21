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
    n_loss_tokens = (encoded_inputs['labels'] != -100).sum().item()
    n_total_tokens = (encoded_inputs['attention_mask'] != 0).sum().item()

    input_ids = encoded_inputs['input_ids'].to(model.device).unsqueeze(0)
    attention_mask = encoded_inputs['attention_mask'].to(model.device).unsqueeze(0)
    labels = encoded_inputs['labels'].to(model.device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        nll_likelihood = outputs.loss
    
    ppl = torch.exp(nll_likelihood)
    return ppl.cpu().item(), n_loss_tokens, n_total_tokens


"""
Example:
python examples/data_perplexity/calculate_wm_perplexity.py \
--dset_fpath data/state_pred/alfworld/train_alfworld_react-qwen3-235b-inst-default_w_refl-step30_hist2_temp1.0.parquet \
--max_samples 1000 \
--model_id Qwen/Qwen2.5-7B-Instruct
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_fpath", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    print(f"received args: {args}")

    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab_size = len(tokenizer)

    dset_fpath = args.dset_fpath
    dset = load_dataset('parquet', data_files=dset_fpath)['train']
    all_chat_messages = []
    max_samples = len(dset) if args.max_samples is None else min(args.max_samples, len(dset))
    for sample in dset.select(range(max_samples)):
        # assumes text only model
        messages = sample['prompt']
        answer = sample['reward_model']['ground_truth']
        messages.append({
            'role': 'assistant',
            'content': answer
        })
        all_chat_messages.append(messages)

    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

    ppls = []
    n_loss_tokens = []
    n_total_tokens = []
    for chat_message in tqdm(all_chat_messages, desc="Calculating PPL"):
        # tried batching, but it's actually not faster
        ppl, n_loss_token, n_total_token = calculate_seq_perplexity(chat_message, tokenizer, model, max_length=args.max_seq_len)
        ppls.append(ppl)
        n_loss_tokens.append(n_loss_token)
        n_total_tokens.append(n_total_token)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Mean ppl tokens: {np.mean(n_loss_tokens):.4f}pm{np.std(n_loss_tokens):.4f}")
    print(f"Mean total tokens: {np.mean(n_total_tokens):.4f}pm{np.std(n_total_tokens):.4f}")
    print(f"Mean PPL: {np.mean(ppls):.4f}pm{np.std(ppls):.4f}")
    print(f"Min PPL: {np.min(ppls):.4f} Max PPL: {np.max(ppls):.4f}")