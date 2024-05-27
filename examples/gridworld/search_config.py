import numpy as np
import json
import re
import reasoners.benchmark.gw_utils as utils
from world_model import GWState, GWAction
from reasoners import SearchConfig, LanguageModel


class GWConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 batch_size=2,
                 reward_alpha=0.5,
                 goal_reward_default=0.,
                 goal_reached_reward=100,
                 config_file=None,
                 world_model=None) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        self.config = json.load(open(config_file, 'r'))
        self.can_move_car = self.config['can_move_car']
        self.entities = self._load_entities()
        self.world_model = world_model

    def get_actions(self, state: GWState) -> list[GWAction]:
        cur_state = state.description
        return list(set(utils.generate_actions(cur_state, self.entities, self.config['action_types'], self.can_move_car)))

    def fast_reward(self, state: GWState, action: GWAction) -> tuple[float, dict]:
        # current_state = state.description
        
        # icl_template = self.prompt["icl_list"][state.step_idx]
        
        # inputs = icl_template.replace("<init_state>", current_state)\
        #     .replace("<goals>", utils.extract_goals(self.example, return_raw=True)).replace("<action>", "")
        # intuition = self.base_model.get_loglikelihood(inputs, [inputs + action])[0]

        # self_eval_prompt = self.prompt["self-eval"].replace("<init_state>", current_state)\
        #     .replace("<goals>", utils.extract_goals(self.example, return_raw=True)).replace("<action>", action)
        # self_eval = self.base_model.get_loglikelihood(self_eval_prompt, 
        #     [self_eval_prompt + "good"])[0]
        intuition = 0
        self_eval = 0
        goal_reached = self.world_model.step(state, action)[1]['goal_reached']
        return self.calculate_reward(intuition, self_eval, goal_reached), {'intuition': intuition, "self_eval": self_eval}

    def calculate_reward(self, intuition, self_eval, goal_reached=None):
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = goal_reached[1]
        return (intuition + self_eval) * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self, state: GWState, action: GWAction,
               intuition: float = None,
               self_eval: float = None,
               goal_reached: tuple[bool, float] = None) -> float:
        assert intuition is not None, "intuition is required to calculate reward in this search config, consider passing it in fast_reward"
        assert self_eval is not None, "self_eval is required to calculate reward in this search config, consider passing it in fast_reward"
        assert goal_reached is not None, "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
        return (self.calculate_reward(intuition, self_eval, goal_reached), 
                {'intuition': intuition, 'goal_reached': goal_reached})

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)
    
    def _load_entities(self):
        keys = self.config['flattened_causals']
        entities = []
        for key in keys:
            match = re.search(r'([a-zA-Z]+)_\((\d+,\s*\d+,\s*\d+)\)(.*)', key)
            if match:
                entity = match.group(1)
                color_str = match.group(2)
                attribute = match.group(3).replace('_', ' ').strip()
                color_tuple = tuple(map(int, color_str.split(',')))
                color_name = utils.closest_color(color_tuple)
            else:
                entity = "unknown entity"
                color_name = "unknown color"
                attribute = "unknown attribute"
            entities.append(f"{color_name} {entity}")
        return set(entities)
