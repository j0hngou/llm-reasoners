from typing import Type, Callable, Optional, Literal

import numpy as np

from reasoners.benchmark import GWEvaluator

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation

from world_model import CausalWorldModel, GWState, GWAction
from search_config import GWConfig
import utils
from causal_mappers import MLP


def node_visualizer(x: MCTSNode):
#     if not x.state:
#         return {}
#     return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}
    pass

def rap_biscuit(base_model: LanguageModel,
              prompt: dict,
              search_algo: Type[SearchAlgorithm] = MCTS,
              resume: int = 0,
              depth_limit: int = 14,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 2,
              temperature: float = 0.8,
              early_stop_base: int = 2,
              early_stop_threshold: float = 0.5,
              reward_alpha: float = 0.5,
              goal_reached_reward = 100000,
              goal_reward_default = 0,
              reward_confidence_default: float = 0.8,
              cum_reward: Callable[[list[float]], float] = np.mean,
              calc_q: Callable[[list[float]], float] = max,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              output_trace_in_each_iter: bool = True,
              aggregate: bool = True,
              crl_model_path: str = None,
              causal_mapper_path: str = None,
              nl_model_path: Optional[str] = None,
              tokenizer_path: str = None,
              autoencoder_path: str = None,
              device='cuda',
              config_file: str = "/home/john/PhD/BISCUIT/data_generation/data/gridworld_simplified_3c1b3l_noturn_noshufflecars_f/val_metadata.json",
              data_path : str = "/home/john/PhD/BISCUIT/llm-reasoners/examples/gridworld/data/step_10.pth",
              **search_algo_params):


    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, 'disable_tqdm': disable_tqdm, 'depth_limit': depth_limit, 'n_iters': 30}
    crl_model, causal_mapper, nl_model, tokenizer = utils.load_models(crl_model_path, autoencoder_path, causal_mapper_path, tokenizer_path, nl_model_path, device=device)
    world_model = CausalWorldModel(crl_model=crl_model, causal_mapper=causal_mapper, nl_model=nl_model, tokenizer=tokenizer, device=device, max_steps=depth_limit, config_file=config_file)
    config = GWConfig(base_model=base_model, prompt=prompt, batch_size=batch_size,
                      reward_alpha=reward_alpha, goal_reached_reward=goal_reached_reward,
                      goal_reward_default=goal_reward_default, config_file=config_file, world_model=world_model,)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    evaluator = GWEvaluator(config_file=config_file, data_path=data_path, init_prompt=prompt)

    accuracy = evaluator.evaluate(reasoner, num_shot=2, resume=resume, log_dir=log_dir)
    print(accuracy)


if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')


    def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama'] = 'llama-2',
             llama_ckpts: str = llama_ckpts,
             llama_2_ckpts: str = llama_2_ckpts,
             llama_size: str = '13B',
             llama_cpp_path: str = None,
             llama_cpp_n_batch: int = 512,
             hf_path: str = 'meta-llama/Llama-2-13b-hf',
             hf_peft_path: Optional[str] = None,
             hf_quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
             hf_load_awq_path: Optional[str] = None,
             exllama_model_dir: str = 'WizardMath-13B-V1.0-GPTQ',
             exllama_lora_dir: Optional[str] = None,
             exllama_mem_map: Optional[str] = None,
             batch_size: int = 1,
             prompt: str = '/home/john/PhD/BISCUIT/llm-reasoners/examples/gridworld/prompts/prompt.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
      

        with open(prompt) as f:
            prompt = json.load(f)
        if base_lm in ['llama', 'llama2']:
            import torch
            import torch.backends.cudnn
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True

        if base_lm == 'llama':
            from reasoners.lm import LlamaModel
            base_model = LlamaModel(llama_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama.cpp':
            from reasoners.lm import LlamaCppModel
            base_model = LlamaCppModel(llama_cpp_path, n_batch=llama_cpp_n_batch, n_ctx=8192)
        elif base_lm == 'llama-2':
            from reasoners.lm import Llama2Model
            base_model = Llama2Model(llama_2_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'hf':
            from reasoners.lm import HFModel
            base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=512,
                                 peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
        elif base_lm == 'exllama':
            from reasoners.lm import ExLlamaModel
            base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir, mem_map=exllama_mem_map,
                                      max_batch_size=batch_size, max_new_tokens=200, max_seq_length=3072)
        else:
            assert False, f'cannot resolve {base_lm=}'
        rap_biscuit(base_model=base_model,
                  prompt=prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  **kwargs)


    fire.Fire(main)
