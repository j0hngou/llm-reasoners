import torch
from typing import Union, Tuple, NamedTuple, Callable
import reasoners.benchmark.gw_utils as utils
from reasoners import WorldModel, LanguageModel
import copy
import json


GWAction = str


        
class GWState(NamedTuple):
    """The state of the Blocksworld.
    
    See the docstring of BlocksWorldModel for more details.
    """
    step_idx: int
    image: torch.Tensor
    description: str
    latents: torch.Tensor = None

class CausalWorldModel(WorldModel):
    def __init__(self, crl_model, causal_mapper, nl_model, tokenizer, device, max_steps=6, config_file=None):
        super().__init__()
        # self.autoencoder = autoencoder
        self.crl_model = crl_model.eval()
        self.causal_mapper = causal_mapper
        # self.causal_mapper = CausalMapper(causal_mapper.to(self.crl_model.device), cm_mean, cm_std, target_assignment)
        self.nl_model = nl_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_steps = max_steps
        self.keys = json.load(open(config_file, 'r'))['flattened_causals']

    def init_state(self, initial_image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Initialize the state with an image, encode it to latent, transform it,
        and generate the natural language description of the initial state.
        """
        initial_image = (initial_image * 2.0) - 1.0
        latents = self.crl_model.autoencoder.encoder(initial_image[None].to(self.device))
        disentangled_latents, _ = self.crl_model.flow.forward(latents)
        causal_variables = self.causal_mapper(disentangled_latents)
        description = self.map_to_language(causal_variables)
        return (disentangled_latents, description)

    @torch.no_grad()
    def step(self, state: GWState, action: str) -> Tuple[Tuple[torch.Tensor, str], dict]:
        """
        Update the state based on the action.
        """
        if state.latents is None:
            image = (state.image * 2.0) - 1.0
            current_latents = self.crl_model.autoencoder.encoder(image[None].to(self.device))
            current_latents, _ = self.crl_model.flow.forward(current_latents)
        else:
            current_latents = state.latents
        tokenized_description = self.tokenizer(action, return_token_type_ids=True, padding='max_length', max_length=64)
        input_ids = torch.tensor(tokenized_description['input_ids']).to(self.device)
        token_type_ids = torch.tensor(tokenized_description['token_type_ids']).to(self.device)
        attention_mask = torch.tensor(tokenized_description['attention_mask']).to(self.device)
        tokenized_description = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        new_latents, _ = self.crl_model.prior_t1.sample(current_latents, tokenized_description=tokenized_description, action=torch.empty(1).to(self.device))
        new_latents = new_latents.squeeze(1)
        causal_variables = self.causal_mapper(new_latents)
        if len(causal_variables) == 1 and isinstance(causal_variables, list):
            causal_variables = causal_variables[0]
        new_description = self.map_to_language(causal_variables)
        new_state = GWState(step_idx=state.step_idx + 1, image=None, description=new_description, latents=new_latents)
        return new_state, {'goal_reached' : utils.goal_check(utils.extract_goals(self.example), new_description, ignore_obstacles=True)}

    def is_terminal(self, state: Tuple[torch.Tensor, str]) -> bool:
        if utils.goal_check(utils.extract_goals(self.example), state.description, ignore_obstacles=True)[0]:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False
        # if len(state) > 0:
        #     generated_ans = ''.join([x.action for x in state])
        #     return "[invalid]" != extract_answer(generated_ans)
        # return False

    def map_to_language(self, causals: torch.Tensor) -> str:
        """
        Map the causal variables to a natural language description (TODO: using the language model.)
        """
        return utils.describe_latent(causals, self.keys)

    def init_state(self) -> GWState:
        """Initialize the world model.

        :return: the initial state
        """
        return GWState(step_idx=0, image=self.example['images'][0], description=utils.
                       extract_init_state(self.example))