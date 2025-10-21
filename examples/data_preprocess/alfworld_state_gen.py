# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Geometry3k dataset to parquet format
"""

import os
import datasets
import numpy as np
from verl.utils.hdfs_io import copy, makedirs
import argparse

### precompute seeds where the s_0 are GUARANTEED to be unique
### if you directly do range(134), actually many tasks/s_0 may be duplicate due to how AlfredTWEnv is implemented
TEST_ID_UNIQUE_SEEDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41,
    42, 43, 44, 45, 46, 47, 49, 51, 53, 54, 56, 60, 61, 63, 64, 65, 68, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 86, 87, 88, 90, 94, 97, 99, 100, 101, 106,
    110, 111, 115, 117, 119, 125, 129, 130, 135, 139, 141, 143, 145, 147, 151, 153, 155, 159, 160, 161,
    165, 174, 184, 186, 191, 193, 212, 213, 214, 215, 216, 221, 225, 230, 233, 239, 249, 250, 257, 262,
    267, 286, 294, 298, 301, 306, 308, 323, 330, 334, 338, 355, 363, 392, 395, 457, 463
]
TEST_OOD_UNIQUE_SEEDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 23, 24, 25, 26, 30, 32, 34, 35, 38, 40, 41, 42, 44, 48, 52, 53, 57, 59, 60,
    64, 66, 69, 72, 74, 75, 76, 77, 80, 83, 86, 87, 91, 98, 103, 107, 117, 118, 119, 127,
    130, 132, 137, 143, 147, 149, 156, 162, 166, 169, 173, 200, 205, 211, 234, 275, 482, 793
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='visual', choices=['visual', 'text'])
    parser.add_argument('--local_dir', default='~/data/verl-agent/')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_data_size', default=256, type=int)
    parser.add_argument('--val_data_size', default=256, type=int)
    parser.add_argument('--val_split', default='id', choices=['id', 'ood'], type=str)
    parser.add_argument('--train_idx_offset', default=0, type=int)

    args = parser.parse_args()
    print(f"processing data for mode: {args.mode}")
    print(f"train_idx_offset: {args.train_idx_offset}")
    args.local_dir = os.path.join(args.local_dir, args.mode)
    
    test_unique_seeds = TEST_ID_UNIQUE_SEEDS if args.val_split == 'id' else TEST_OOD_UNIQUE_SEEDS
    assert args.val_data_size <= len(test_unique_seeds), \
        f"val_data_size {args.val_data_size} is larger than the number of unique seeds {len(test_unique_seeds)}"

    data_source = 'hiyouga/geometry3k'

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train'].select(range(args.train_data_size))
    test_dataset = dataset['test'].select(range(args.val_data_size))

    instruction_following = {
        "visual": "<image>",
        "text": "",
    }

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem = example.pop('problem')
            prompt = instruction_following[args.mode]
            # answer = example.pop('answer')
            images = example.pop('images')

            if args.mode == 'visual':
                data = {
                    "data_source": args.mode,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "images": images,
                    "ability": "agent",
                    "reward_model": {"style": "rule", "ground_truth": "placeholder"},
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        'env_kwargs': {
                            'seed': idx + args.train_idx_offset if split == 'train' else test_unique_seeds[idx],
                        }
                    }
                }
            else:
                data = {
                    "data_source": args.mode,
                    "prompt": [{
                        "role": "user",
                        "content": prompt,
                    }],
                    "ability": "agent",
                    "reward_model": {"style": "rule", "ground_truth": "placeholder"},
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        'env_kwargs': {
                            'seed': idx + args.train_idx_offset if split == 'train' else test_unique_seeds[idx],
                        }
                    }
                }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, num_proc=8)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if args.train_idx_offset > 0:
        train_dataset.to_parquet(os.path.join(local_dir, f'train_alfworld_{args.train_data_size}_offset{args.train_idx_offset}.parquet'))
    else:
        train_dataset.to_parquet(os.path.join(local_dir, f'train_alfworld_{args.train_data_size}.parquet'))
    if args.val_split == 'id':
        test_dataset.to_parquet(os.path.join(local_dir, f'test_alfworld_{args.val_data_size}.parquet'))
    else:
        test_dataset.to_parquet(os.path.join(local_dir, f'test_alfworld_{args.val_data_size}_ood.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)