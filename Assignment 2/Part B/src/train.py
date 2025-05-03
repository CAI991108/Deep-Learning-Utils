# Copyright 2025 the LlamaFactory team.
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

from llamafactory.train.tuner import run_exp


def main():
    # ============= 核心参数配置 =============
    args = {
        # 数据相关配置
        # "train_data_path": "/content/LLaMA-Factory/data/disc_law_sft/train.json",
        # "eval_data_path": "/content/LLaMA-Factory/data/disc_law_sft/test.json", 
        "dataset_dir": "/content/LLaMA-Factory/data",  # 数据集根目录
        # "data_paths": {  # 多文件支持
        #     "train": "disc_law_sft/train.json",
        #     "test": "disc_law_sft/test.json"
        # },
        
        # 模型配置
        "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "template": "chinese_law",
        
        # 训练参数
        "stage": "sft",
        "finetuning_type": "lora",
        "output_dir": "outputs/law_sft",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "fp16": True,
        
        # 优化配置
        "lora_target": "q_proj,v_proj",
        "max_grad_norm": 1.0,
        "logging_steps": 10
    }
    
    # ============= 启动训练 =============
    run_exp(args)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()

if __name__ == "__main__":
    main()