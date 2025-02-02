```python
# ./multistep.py
# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def compute_lambda_returns(
    r_t: np.ndarray, discount_t: np.ndarray, v_t: np.ndarray, lambda_: np.ndarray = 1.0
):
    """Estimates a multistep truncated lambda return from a trajectory (rewritten from rlax).

    Given a a trajectory of length `T+1`, generated under some policy π, for each
    time-step `t` we can estimate a target return `G_t`, by combining rewards,
    discounts, and state values, according to a mixing parameter `lambda`.

    The parameter `lambda_`  mixes the different multi-step bootstrapped returns,
    corresponding to accumulating `k` rewards and then bootstrapping using `v_t`.

        rₜ₊₁ + γₜ₊₁ vₜ₊₁
        rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ vₜ₊₂
        rₜ₊₁ + γₜ₊₁ rₜ₊₂ + γₜ₊₁ γₜ₊₂ rₜ₊₂ + γₜ₊₁ γₜ₊₂ γₜ₊₃ vₜ₊₃

    The returns are computed recursively, from `G_{T-1}` to `G_0`, according to:

        Gₜ = rₜ₊₁ + γₜ₊₁ [(1 - λₜ₊₁) vₜ₊₁ + λₜ₊₁ Gₜ₊₁].

    In the `on-policy` case, we estimate a return target `G_t` for the same
    policy π that was used to generate the trajectory. In this setting the
    parameter `lambda_` is typically a fixed scalar factor. Depending
    on how values `v_t` are computed, this function can be used to construct
    targets for different multistep reinforcement learning updates:

        TD(λ):  `v_t` contains the state value estimates for each state under π.
        Q(λ):  `v_t = max(q_t, axis=-1)`, where `q_t` estimates the action values.
        Sarsa(λ):  `v_t = q_t[..., a_t]`, where `q_t` estimates the action values.

    In the `off-policy` case, the mixing factor is a function of state, and
    different definitions of `lambda` implement different off-policy corrections:

        Per-decision importance sampling:  λₜ = λ ρₜ = λ [π(aₜ|sₜ) / μ(aₜ|sₜ)]
        V-trace, as instantiated in IMPALA:  λₜ = min(1, ρₜ)

    Note that the second option is equivalent to applying per-decision importance
    sampling, but using an adaptive λ(ρₜ) = min(1/ρₜ, 1), such that the effective
    bootstrap parameter at time t becomes λₜ = λ(ρₜ) * ρₜ = min(1, ρₜ).
    This is the interpretation used in the ABQ(ζ) algorithm (Mahmood 2017).

    Of course this can be augmented to include an additional factor λ.  For
    instance we could use V-trace with a fixed additional parameter λ = 0.9, by
    setting λₜ = 0.9 * min(1, ρₜ) or, alternatively (but not equivalently),
    λₜ = min(0.9, ρₜ).

    Estimated return are then often used to define a td error, e.g.:  ρₜ(Gₜ - vₜ).

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/sutton/book/ebook/node74.html).

    Args:
        r_t: sequence of rewards rₜ for timesteps t in [1, T].
        discount_t: sequence of discounts γₜ for timesteps t in [1, T].
        v_t: sequence of state values estimates under π for timesteps t in [1, T].
        lambda_: mixing parameter; a scalar or a vector for timesteps t in [1, T].

    Returns:
        Multistep lambda returns.
    """
    lambda_ = np.ones_like(r_t) * lambda_

    def _body(acc, xs):
        returns, discounts, values, lambda_ = xs
        acc = returns + discounts * ((1 - lambda_) * values + lambda_ * acc)
        return acc, acc

    carry = v_t[-1]
    ys = [None] * len(r_t)
    for t in reversed(range(len(r_t))):
        carry, y = _body(carry, (r_t[t], discount_t[t], v_t[t], lambda_[t]))
        ys[t] = y

    return np.array(ys)

```

```python
# ./interface.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defining how components interface with each other."""
import logging
from typing import Type

import launchpad as lp
from launchpad.nodes.python import local_multi_processing

from oat.actors import PreferenceActor
from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.learners.base import LearnerBase
from oat.utils.ipc import PlasmaShmServer
from oat.utils.launcher import get_free_port


def get_program(
    args: OATArgs,
    learner_cls: Type[LearnerBase],
    actor_cls: Type[ActorBase] = PreferenceActor,
):
    """Define the default distributed program topology with configs."""
    program = lp.Program("online_dap")

    # Resource.
    if args.collocate:
        actor_gpus = learner_gpus = list(range(args.gpus))
    else:
        if args.gpus % 2 == 0:
            actor_gpus = list(range(args.gpus // 2))
            learner_gpus = list(range(args.gpus // 2, args.gpus))
        else:
            logging.warn(
                "Number of GPUs not divisible by 2, one GPU will be forced to collocate learner and actor."
            )
            actor_gpus = list(range(args.gpus // 2 + 1))
            learner_gpus = list(range(args.gpus // 2, args.gpus))

    logging.warn(
        f"=== GPU allocations ===\nActor: {actor_gpus}, Learner: {learner_gpus}"
    )

    # IPC.
    ipc_server = program.add_node(
        lp.CourierNode(PlasmaShmServer, size_mb=args.shm_size_mb), label="ipc_server"
    )

    # Actor.
    vllm_args = {
        "model": args.pretrain,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.vllm_gpu_ratio,
        "dtype": "bfloat16",
        "enable_prefix_caching": False,
        # "max_model_len": args.max_model_len,
    }

    actors = []
    local_resources = {}
    for i in actor_gpus:
        label = f"actor_{i}"
        actors.append(
            program.add_node(
                lp.CourierNode(actor_cls, ipc_server, vllm_args, args),
                label=label,
            )
        )
        local_resources[label] = local_multi_processing.PythonProcess(
            env={"CUDA_VISIBLE_DEVICES": str(i)}
        )

    # Learner.
    master_addr = "0.0.0.0"
    master_port = get_free_port()
    args.local_rank = 0
    label = "learner_0"
    master_learner = lp.PyClassNode(
        learner_cls,
        len(learner_gpus),
        0,
        0,
        master_addr,
        master_port,
        True,
        args,
        actors,
        ipc_server,
    )
    program.add_node(master_learner, label=label)
    local_resources[label] = local_multi_processing.PythonProcess(
        env={"CUDA_VISIBLE_DEVICES": str(learner_gpus[0])}
    )
    for i in range(1, len(learner_gpus)):
        label = f"learner_{i}"
        worker_learner = lp.PyClassNode(
            learner_cls,
            len(learner_gpus),
            i,
            i,
            master_addr,
            master_port,
            False,
            args,
            actors,
            ipc_server,
        )
        program.add_node(worker_learner, label=label)
        local_resources[label] = local_multi_processing.PythonProcess(
            env={"CUDA_VISIBLE_DEVICES": str(learner_gpus[i])}
        )

    return program, local_resources

```

```python
# ./__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

```python
# ./types.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple

import torch

Metric = Dict[str, Any]


class DAPAlgo(Enum):
    DPO = 0
    IPO = 1
    SLiC = 2
    SimPO = 3
    BNF = 4
    LR_DPO = 5


class RLAlgo(Enum):
    PPO = 100


class SFTAlgo(Enum):
    SFT = 200


@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str
    chosen_id: int = 0
    chosen_feature: torch.Tensor = None
    rejected_feature: torch.Tensor = None
    init_clash: bool = False
    loss_mask: bool = True
    is_model_data: bool = False
    info: Metric = None


@dataclass
class TrajectoryData:
    prompt: str
    prompt_ids: List[int]
    response: str
    response_ids: List[int]
    response_logprobs: List[float]
    rewards: List[float]
    loss_mask: bool = True
    info: Metric = None


class RewardData(NamedTuple):
    pair_features: torch.Tensor  # (B, 2, d)
    loss_masks: torch.Tensor  # (B,)

```

```python
# ./ever.py
import os

def create_markdown_from_files(root_dir):
    # Files/directories to ignore
    ignore_list = ['.git', '.pytest_cache', '__pycache__', 'LICENSE', 'README.md']
    
    # Dictionary to store file extensions and their corresponding language identifiers and comments
    lang_map = {
        '.py': ('python', '#'),
        '.cpp': ('cpp', '//'),
        '.h': ('cpp', '//'),
        '.js': ('javascript', '//'),
        '.html': ('html', '<!--'),
        '.css': ('css', '/*'),
        '.java': ('java', '//'),
        # Add more mappings as needed
    }
    
    markdown_content = []
    
    for root, dirs, files in os.walk(root_dir):
        # Remove ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_list]
        
        for file in files:
            if file not in ignore_list:
                file_path = os.path.join(root, file)
                # Get file extension
                _, ext = os.path.splitext(file)
                # Get language identifier and comment symbol from extension, default to 'text' and '#'
                lang_info = lang_map.get(ext, ('text', '#'))
                lang = lang_info[0]
                comment_symbol = lang_info[1]
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Create markdown block with file path as comment
                    markdown_content.append(
                        f"```{lang}\n{comment_symbol} {file_path}\n{content}\n```\n"
                    )
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Write to markdown file
    with open('files_content.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_content))

if __name__ == "__main__":
    # Use current directory as root
    create_markdown_from_files('.')
```

```python
# ./__about__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Version."""
__version__ = "0.0.6"

```

```python
# ./args.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Argument parsing."""
import math
from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from oat.types import DAPAlgo, RLAlgo, SFTAlgo


@dataclass
class OATArgs:
    """Experiment arguments."""

    """Resources."""
    # Launchpad launch type
    launch_type: str = "local_mp"
    # Number of GPUs to run the experiment.
    gpus: int = 8
    # Ratio of pre-allocated GPU memory for vLLM.
    vllm_gpu_ratio: float = 0.25
    # Actor-learner collocation.
    collocate: bool = False
    # Size of Plasma shared memory.
    shm_size_mb: int = 5000
    # Asynchronous training.
    asynchronous: bool = False

    """Training configurations."""
    # Model name.
    pretrain: str = "trl-lib/pythia-1b-deduped-tldr-sft"
    # Reference model name, defaults to pretrain if None.
    ref_pretrain: str = None
    # Critic initial model.
    critic_pretrain: str = None
    # Tokenizer name.
    tokenizer: str = ""

    # LLM alignment algorithms.
    algo: Literal[
        "DPO",
        "IPO",
        "LR_DPO",
        "SLiC",
        "SimPO",
        "BNF",
        "SFT",
        "PPO",
    ] = "DPO"
    # Set 1 for truly online algorithms; large number for offline; intermediate value for iterative.
    sync_params_every: int = 1
    # Used in KL-regularized losses.
    beta: float = 0.1
    # cDPO https://arxiv.org/pdf/2305.18290.
    label_smoothing: float = 0
    # SimPO https://arxiv.org/pdf/2405.14734.
    gamma_beta_ratio: float = 0.5
    # DPO-Positive https://arxiv.org/pdf/2402.13228.
    dpo_positive_lambda: float = 0
    # DPO + SFT loss coefficient.
    sft_weight: float = 0

    # Oracle.
    oracle: str = "pairrm"
    oracle_type: Literal["preference", "reward"] = "preference"
    oracle_batch_size: int = 1
    remote_rm_url: str = ""
    remote_rm_client_workers: int = 4
    # Sampling a Bernoulli to get the binary feedback instead of thresholding.
    bt_sample: bool = False

    # Critic.
    critic_type: Literal["ppo", "grpo"] = "ppo"

    # Epistemic reward model (for exploration).
    num_ensemble: int = 20
    enn_max_try: int = -1
    enn_lambda: float = 0.5
    learn_rm: bool = False
    rm_lr: float = 1e-3
    rm_wd: float = 5e-5
    rm_hidden_dim: int = 128
    rm_act_fn: str = "relu"
    rm_sgd_steps: int = 5
    rm_fixed_reg: bool = False
    rm_train_budget: int = -1
    rm_backbone: str = "llm-blender/PairRM-hf"
    # Learn the ERM only without updating the LLM.
    learn_rm_only: bool = False
    # Load a pre-trained RM.
    rm_pretrain: str = ""
    # Exploration strategies.
    exp_method: Literal[
        "no",
        "EnnBAITS",
        "EnnEETS",
        "EnnUncertainty",
        "EnnPassive",
    ] = "no"
    # Random sampling if the dueling responses coincide.
    exp_rnd_sample: bool = False
    # Take the top 2 best actions.
    exp_allow_second_best: bool = False
    # Enable SEA's Mixed Preference Learning (Dyna)
    model_rollout: bool = False
    max_model_data_ratio: float = 0.3
    burn_in_period: int = 5
    pure_model_based: bool = False
    # Dyna search control.
    model_data_strategy: Literal["random"] = "random"

    # Prompt dataset.
    prompt_data: str = "lkevinzc/tldr-with-sft-reference"
    input_key: str = "prompt"
    output_key: str = "output"
    train_split: str = "train"
    max_train: int = 50000
    # Maximum number of oracle queries, defaults to max_train.
    max_queries: int = -1

    # On-policy generation params.
    generate_max_length: int = 53
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: float = -1
    num_samples: int = 2

    """Evaluation configurations."""
    online_evaluation: bool = False
    best_of_n_eval: bool = False
    num_bon: int = 1
    bon_temperature: float = 0.7
    max_eval: int = 1000
    eval_split: str = "test"
    eval_batch_size: int = -1
    eval_generate_max_length: int = 200
    eval_temperature: float = 0.0
    eval_top_p: float = 0.95
    eval_top_k: float = -1
    eval_n: int = 1
    eval_steps: int = 20
    eval_query_interval: int = -1
    # Defaults to prompt_data if empty.
    eval_data: str = ""
    # Defaults to input_key if empty.
    eval_input_key: str = ""
    # Defaults to output_key if empty.
    eval_output_key: str = ""

    """Training specs."""
    save_path: str = "./oat-output"
    save_steps: int = -1
    max_save_num: int = 5
    max_save_mem: int = 1000
    logging_steps: int = 1
    num_prompt_epoch: int = 1
    train_batch_size: int = 128
    train_batch_size_per_device: int = 1
    rollout_batch_size: int = 128
    rollout_batch_size_per_device: int = 16
    pi_buffer_maxlen_per_device: int = 16
    max_epochs: int = 1
    max_sgd_steps: float = math.inf
    r_buffer_maxlen: int = 50000
    prompt_max_length: int = 1024
    max_step_adjustment: float = 1
    critic_max_step_adjustment: float = 1
    buffer_clear_every: float = math.inf
    dump_all_buffer: bool = False

    max_norm: float = 1.0
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.95
    l2: float = 0.0
    gradient_checkpointing: bool = False
    seed: int = 42
    disable_fast_tokenizer: bool = False
    local_rank: int = -1

    zero_stage: int = 2
    bf16: bool = True
    ref_offload: bool = False
    learning_rate: float = 5e-7
    critic_learning_rate: float = 9e-6
    lr_scheduler: str = "cosine_with_min_lr"
    lr_warmup_ratio: float = 0.03
    zpg: int = 1
    adam_offload: bool = False
    flash_attn: bool = True
    grad_accum_dtype: str = None
    disable_trace_cache: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str = "all-linear"
    lora_dropout: float = 0
    gradient_checkpointing_use_reentrant: bool = False

    apply_chat_template: bool = False

    """Misc."""
    # Skip the first evaluation.
    debug: bool = False
    # Random seed conditioned on time.
    rnd_seed: bool = True

    # Weights and biases logging.
    use_wb: bool = False
    wb_org: str = None
    wb_group: str = None
    wb_project: str = "oat-llm"
    wb_run_name: str = "debug"


def get_default_args(args_cls=OATArgs):
    return tyro.cli(args_cls)


def default_args_validation(args: OATArgs):
    # Validation.
    for algo_pool in [DAPAlgo, RLAlgo, SFTAlgo]:
        try:
            args.algo = getattr(algo_pool, args.algo)
            break
        except AttributeError:
            continue
    else:
        raise ValueError(f"Invalid algorithm name {args.algo}")

    if args.algo != DAPAlgo.SimPO and (
        args.ref_pretrain is None or args.ref_pretrain == ""
    ):
        args.ref_pretrain = args.pretrain
    if args.critic_pretrain is None:
        args.critic_pretrain = args.pretrain
    if args.learn_rm:
        assert args.exp_method != "no" and args.rm_pretrain == ""
    if args.learn_rm_only:
        assert args.best_of_n_eval
    if args.enn_max_try == -1:
        args.enn_max_try = args.num_ensemble
    if args.eval_batch_size == -1:
        args.eval_batch_size = args.rollout_batch_size_per_device
    if args.rm_train_budget == -1:
        args.rm_train_budget = math.inf
    if args.max_queries > 0:
        args.max_queries = min(args.max_queries, args.max_train)
    else:
        args.max_queries = args.max_train
    if args.asynchronous:
        assert not args.collocate, "async training needs to disable collocation"
    args.max_model_len = (
        args.prompt_max_length
        + max(args.generate_max_length, args.eval_generate_max_length)
        + 128
    )
    gpu_available = torch.cuda.device_count()
    assert (
        gpu_available >= args.gpus
    ), f"{gpu_available} GPUs available, but {args.gpus} required"
    return args

```

```python
# ./exploration.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import random
from dataclasses import dataclass
from typing import Dict, List

import einops
import numpy as np
import torch
import tree

from oat.args import OATArgs
from oat.rm import uncertainty
from oat.rm.backbone import RMBackbone
from oat.rm.model import RewardModel
from oat.types import Metric


@dataclass
class ExplorationResults:
    dueling_candidates: Dict[int, List[str]]
    candidate_features: torch.Tensor
    init_clash: List[bool]
    is_model_data: List[bool]
    all_rewards: torch.Tensor
    info: Metric


class ExplorerBase(abc.ABC):
    @abc.abstractmethod
    def best_of_n(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> List[str]:
        """Best-of-N generation given the reward model.

        Args:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            List[str]: A list of the best response per prompt.
        """

    @abc.abstractmethod
    def select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> ExplorationResults:
        """Select dueling responses from candidates.

        Args:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            ExplorationResults: Pair of responses per prompt (and features), M -> 2
        """

    @abc.abstractmethod
    def compare(self, candidate_features: torch.Tensor) -> torch.Tensor:
        """Compare candidates using the reward model.

        Args:
            candidate_features (torch.Tensor): (M, 2, d)

        Returns:
            torch.Tensor: (M,), 1 means the first wins
        """


class Explorer(ExplorerBase):
    def __init__(
        self, reward_model: RewardModel, rm_backbone: RMBackbone, args: OATArgs
    ) -> None:
        self.backbone = rm_backbone
        self.reward_model = reward_model

        self.max_length = 2048
        self.source_max_length = 1224
        self.backbone_bs = 8

        self.random_sampling = args.exp_rnd_sample

    def best_of_n(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> List[str]:
        """Best-of-N generation given the reward model.

        Args:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            List[str]: A list of the best response per prompt.
        """
        features = self._get_features(prompts, candidates)  # (M, N, d)
        best_response_indices = (
            self.reward_model.get_best_action(features).cpu().squeeze()
        )  # (M,)
        best_responses = [
            candidates[i][sel_idx] for i, sel_idx in enumerate(best_response_indices)
        ]
        return best_responses

    def select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ) -> ExplorationResults:
        """Select dueling responses from candidates.

        Args:
            prompts (List[str]): A list of prompt texts, M
            candidates (Dict[int, List[str]]): Lists of responses per prompt, M -> N

        Returns:
            ExplorationResults: Pair of responses per prompt (and features), M -> 2
        """
        (
            rewards,
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
        ) = self._inner_select(prompts, candidates)
        return ExplorationResults(
            dueling_candidates=dueling_candidates,
            candidate_features=(
                torch.stack(
                    [
                        features[i][selected_candidate_indices[i]]
                        for i in range(len(prompts))
                    ]
                )
            ),
            init_clash=init_clash.tolist(),
            is_model_data=[False] * len(prompts),
            all_rewards=rewards,
            info=info,
        )

    def compare(self, candidate_features: torch.Tensor) -> torch.Tensor:
        rewards = self.reward_model.get_rewards(candidate_features).mean(0)  # (M, 2, 1)
        return (rewards[:, 0] > rewards[:, 1]).squeeze().float().cpu().numpy()

    def _inner_select(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ):
        features = self._get_features(prompts, candidates)  # (M, N, d)
        rewards, first_indices, second_indices = self.reward_model.get_duel_actions(
            features
        )  # rewards: (E or 2, M, N, 1); indices: both (M, 1)

        init_clash = (second_indices == first_indices).cpu().squeeze()
        rewards_with_agreed_best = rewards[:, init_clash]
        clashed_best_indices = second_indices[init_clash]
        agreed_best_resp_std = np.mean(
            [
                torch.std(rewards_with_agreed_best[:, i, clashed_best_indices[i]]).cpu()
                for i in range(len(clashed_best_indices))
            ]
        )
        rewards_without_agreed_best = rewards[:, ~init_clash]
        not_clashed_best_indices = second_indices[~init_clash]
        not_agreed_best_resp_std = np.mean(
            [
                torch.std(
                    rewards_without_agreed_best[:, i, not_clashed_best_indices[i]]
                ).cpu()
                for i in range(len(not_clashed_best_indices))
            ]
        )
        # In the case where both responses are the same, do random sampling
        if self.random_sampling:
            N = features.shape[1]
            rnd_second_indices = torch.ones_like(second_indices) * -1
            for _ in range(3):
                # Clash prob 1 / N^3
                rand_indices = torch.randint_like(second_indices, N)
                valid_idx = (rand_indices != first_indices) * (rnd_second_indices == -1)
                rnd_second_indices[valid_idx] = rand_indices[valid_idx]
                if -1 not in rnd_second_indices:
                    break

            second_indices = torch.where(
                second_indices == first_indices, rnd_second_indices, second_indices
            )

        selected_candidate_indices = torch.cat(
            [first_indices, second_indices], dim=-1
        ).cpu()
        dueling_candidates = {}
        for i, sel_idx in enumerate(selected_candidate_indices):
            dueling_candidates[i] = [candidates[i][j] for j in sel_idx]

        info = {
            "explorer/agreed_best_resp_std": np.nan_to_num(agreed_best_resp_std),
            "explorer/not_agreed_best_resp_std": np.nan_to_num(
                not_agreed_best_resp_std
            ),
        }
        return (
            rewards,
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
        )

    def _get_features(
        self,
        prompts: List[str],
        candidates: Dict[int, List[str]],
    ):
        input_ids = []
        M = len(prompts)
        N = len(candidates[0])
        for i in range(M):
            for j in range(N):
                pair_ids = self.backbone.tokenize_pair(
                    prompt=prompts[i],
                    candidate=candidates[i][j],
                    source_max_length=self.source_max_length,
                    max_length=self.max_length,
                )
                input_ids.append(pair_ids)
        encodings = self.backbone.tokenizer.pad(
            {"input_ids": input_ids},
            return_tensors="pt",
        )

        features = []
        for ndx in range(0, M * N, self.backbone_bs):
            batch_enc = tree.map_structure(
                lambda x: x[ndx : min(ndx + self.backbone_bs, M * N)].to(
                    self.backbone.device
                ),
                encodings,
            )
            features.append(self.backbone.get_feature(**batch_enc))
        features = torch.cat(features, dim=0)  # (M*N, d)
        features = features.view(M, N, -1)
        return features


class ModelBasedExplorer(Explorer):
    """It not only explores based on Thompson sampling, but also synthesizes
    model rollout when it trusts itself to boot sample efficiency."""

    def __init__(
        self, reward_model: RewardModel, rm_backbone: RMBackbone, args: OATArgs
    ) -> None:
        super().__init__(reward_model, rm_backbone, args)
        self.count = 1
        self.burn_in_period = args.burn_in_period
        self.max_model_data_ratio = args.max_model_data_ratio
        self.model_data_selector = getattr(self, f"_{args.model_data_strategy}_select")
        self.pure_model_based = args.pure_model_based

    def _random_select(
        self,
        candidates,
        rewards,
        dueling_candidates,
        selected_candidate_indices,
        is_model_data,
    ):
        reward_margin = rewards - einops.rearrange(rewards, "e m n 1 -> e m 1 n")
        E, M, _, _ = reward_margin.shape
        random_belief_reward_margin = reward_margin[
            torch.randint(E, (M,)), torch.arange(M)
        ]  # M, N, N'
        # mean_rewards = rewards.mean(0)
        max_model_data = int(len(is_model_data) * self.max_model_data_ratio)
        is_model_data[:max_model_data] = 1
        random.shuffle(is_model_data)
        for i, imd in enumerate(is_model_data):
            if imd:
                margin_i = random_belief_reward_margin[i]
                margin_i_abs = torch.abs(margin_i)
                tr_pairs = torch.where(margin_i_abs == margin_i_abs.max())
                sel_idx = np.random.choice(len(tr_pairs[0]))  # break tie
                candidate_1, candidate_2 = tr_pairs[0][sel_idx], tr_pairs[1][sel_idx]
                if margin_i[candidate_1, candidate_2] > 0:
                    rnd_chosen, rnd_rejected = candidate_1, candidate_2
                else:
                    rnd_chosen, rnd_rejected = candidate_2, candidate_1
                dueling_candidates[i] = [
                    candidates[i][rnd_chosen],
                    candidates[i][rnd_rejected],
                ]
                selected_candidate_indices[i] = torch.tensor([rnd_chosen, rnd_rejected])
        return dueling_candidates, selected_candidate_indices, is_model_data

    def select(
        self, prompts: List[str], candidates: Dict[int, List[str]]
    ) -> ExplorationResults:
        # Select the query points using exploration strategies.
        # Be optimistic and reduce uncertainty.
        (
            rewards,  # rewards: (E, M, N, 1)
            dueling_candidates,
            features,
            selected_candidate_indices,
            init_clash,
            info,
        ) = self._inner_select(prompts, candidates)
        # Replace queries that the agent is already confident about the results.
        # Utilize uncertainty to build trust region.
        is_model_data = np.zeros(len(prompts))
        model_chosen_rewards = []
        model_rejected_rewards = []
        model_pred_prob = []
        sel_pair_ep_uct = []
        sel_prompt_ep_uct = []
        uct_mean = 0
        if self.count > self.burn_in_period:
            dueling_candidates, selected_candidate_indices, is_model_data = (
                self.model_data_selector(
                    candidates,
                    rewards,
                    dueling_candidates,
                    selected_candidate_indices,
                    is_model_data,
                )
            )
            mean_rewards = rewards.mean(0)  # (M, N, 1)
            uct = uncertainty.logits_variance(rewards)
            uct_mean = uct.mean().item()

        for i in range(len(prompts)):
            if is_model_data[i]:
                tr_chosen = selected_candidate_indices[i, 0]
                tr_rejected = selected_candidate_indices[i, 1]

                model_chosen_rewards.append(mean_rewards[i, tr_chosen].item())
                model_rejected_rewards.append(mean_rewards[i, tr_rejected].item())
                model_pred_prob.append(
                    (mean_rewards[i, tr_chosen] - mean_rewards[i, tr_rejected])
                    .sigmoid()
                    .item()
                )
                sel_pair_ep_uct.append(uct[i][tr_chosen, tr_rejected].item())
                sel_prompt_ep_uct.append(uct[i].mean().item())
            else:
                if self.pure_model_based:
                    # Disable learning.
                    dueling_candidates[i] = ["dummy", "dummy"]

        self.count += 1

        info.update(
            {
                "explorer/model_chosen_rewards": np.mean(model_chosen_rewards),
                "explorer/model_rejected_rewards": np.mean(model_rejected_rewards),
                "explorer/model_pred_prob_min": (
                    np.min(model_pred_prob) if model_pred_prob else np.nan
                ),
                "explorer/model_pred_prob_max": (
                    np.max(model_pred_prob) if model_pred_prob else np.nan
                ),
                "explorer/model_pred_prob_mean": np.mean(model_pred_prob),
                "explorer/sel_pair_ep_uct_mean": np.mean(sel_pair_ep_uct),
                "explorer/sel_pair_ep_uct_std": np.std(sel_pair_ep_uct),
                "explorer/sel_prompt_ep_uct_mean": np.std(sel_prompt_ep_uct),
                "explorer/sel_prompt_ep_uct_std": np.std(sel_prompt_ep_uct),
                "explorer/all_ep_uct_mean": uct_mean,
                "explorer/model_data_ratio": np.mean(is_model_data),
            }
        )
        return ExplorationResults(
            dueling_candidates=dueling_candidates,
            candidate_features=(
                torch.stack(
                    [
                        features[i][selected_candidate_indices[i]]
                        for i in range(len(prompts))
                    ]
                )
            ),
            init_clash=init_clash.tolist(),
            is_model_data=is_model_data.astype("bool").tolist(),
            all_rewards=rewards,
            info=info,
        )

```

```python
# ./model.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference to https://github.com/OpenRLHF/OpenRLHF.

import logging
from typing import Optional

import deepspeed
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class LLM(nn.Module):
    """Large language model interface."""

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        device_map=None,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = (
                "flash_attention_2" if use_flash_attention_2 else "eager"
            )

            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                logging.debug("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> torch.LongTensor:
        generate_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }
        sequences = self.model.generate(**generate_args)
        return sequences

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids
        )

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs={"use_reentrant": False}
    ):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()


def build_critic_cls(base_cls, base_pretrain_cls, value_head_prefix):
    class CriticModel(base_pretrain_cls):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_cls(config))

            self.value_head_prefix = value_head_prefix
            setattr(
                self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False)
            )

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(
                -1
            )

            if return_output:
                return (values, outputs)
            else:
                return values

    return CriticModel


class Critic(nn.Module):
    """Large language model interface."""

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        init_value_head=True,
        value_head_prefix="score",
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            config = AutoConfig.from_pretrained(
                pretrain_or_model, trust_remote_code=True
            )
            config._attn_implementation = (
                "flash_attention_2" if use_flash_attention_2 else "eager"
            )
            value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)

            base_class = AutoModel._model_mapping[type(config)]
            critic_cls = build_critic_cls(
                base_class, base_class.__base__, value_head_prefix
            )

            self.model = critic_cls.from_pretrained(
                pretrain_or_model,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                quantization_config=nf4_config,
                device_map=device_map,
            )

            if init_value_head:
                value_head = getattr(self.model, value_head_prefix)
                if (
                    ds_config is not None
                    and ds_config["zero_optimization"]["stage"] == 3
                ):
                    with deepspeed.zero.GatheredParameters(
                        [value_head.weight], modifier_rank=0
                    ):
                        if torch.distributed.get_rank() == 0:
                            value_head.weight.data.normal_(
                                mean=0.0, std=1 / (config.hidden_size + 1)
                            )
                else:
                    value_head.weight.data.normal_(
                        mean=0.0, std=1 / (config.hidden_size + 1)
                    )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                logging.debug("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

    def forward(self, **input):
        return self.model(**input)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs={"use_reentrant": False}
    ):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

```

```python
# ./prompts.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# These examples are from the Table 20 of CoT paper (https://arxiv.org/pdf/2201.11903.pdf).
GSM8K_EXAMPLES = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot_answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        "short_answer": "5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        "short_answer": "39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "cot_answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        "short_answer": "8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        "short_answer": "9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "cot_answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        "short_answer": "29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "cot_answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        "short_answer": "33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot_answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        "short_answer": "8",
    },
]

# These examples are from the DeepSeekMath GitHub repository (https://github.com/deepseek-ai/DeepSeek-Math/tree/main/evaluation/few_shot_prompts)
MATH_EXAMPLES = [
    {
        "question": "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "cot_answer": "The expressions inside each square root must be non-negative.\nTherefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$.\nAlso, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.\nTherefore, the domain of the expression is $\\boxed{[2,5)}$.",
        "short_answer": "[2,5)",
    },
    {
        "question": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "cot_answer": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
        "short_answer": "24",
    },
    {
        "question": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "cot_answer": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}",
        "short_answer": "16",
    },
    {
        "question": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "cot_answer": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$",
        "short_answer": "-\\frac{2}{3}",
    },
]

MATH_TEMPLATE = """[MATH_TASK] Problem:
{question}

Solution:"""

```

```python
# ./actors/__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from oat.actors.preference import PreferenceActor
from oat.actors.reward import RewardActor

__all__ = ["PreferenceActor", "RewardActor"]

```

```python
# ./actors/reward.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import logging
import time
from typing import List

import tree
import vllm

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.types import TrajectoryData


class RewardActor(ActorBase):
    """The environment is a reward oracle. In this case the problem can be formulated
    as conventional reinforcement learning or contextual bandit.

    When the reward is a trained model from human preferences, this is also known as RLHF.
    """

    def __init__(self, ipc_server, vllm_args, args: OATArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.eval_sampling_params = vllm.SamplingParams(
            n=1,
            temperature=(args.eval_temperature),
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
        )

    def extract_candidates_from_output(self, outputs, sampling_params, strip=True):
        candidates = []
        for i in range(len(outputs)):
            # for each prompt
            candidates.append([])
            for k in range(sampling_params.n):
                # for each response
                text = outputs[i].outputs[k].text
                if strip:
                    text = text.strip()
                candidates[i].append(text)
        return candidates

    def generate_and_maybe_eval(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ):
        assert self.eval_mode
        outputs = self.generate(formatted_prompts, self.eval_sampling_params)
        candidates = self.extract_candidates_from_output(
            outputs, self.eval_sampling_params
        )
        responses = []
        for j in range(self.eval_sampling_params.n):
            responses.extend([candidates[i][j] for i in range(len(prompts))])

        win_probs = None
        if references:
            logging.debug(f"Evaluating using oracle {self.oracle}")
            st = time.time()
            win_probs, _ = self.oracle.compare(
                prompts * self.eval_sampling_params.n,
                responses,
                references * self.eval_sampling_params.n,
                batch_size=self.oracle_batch_size,
                return_probs=True,
                disable_tqdm=True,
            )
            logging.debug(f"Time elapse {time.time() - st}")
        reshaped_responses = []
        for x_i in range(len(prompts)):
            reshaped_responses.append(
                [responses[y_i] for y_i in range(x_i, len(responses), len(prompts))]
            )
        reshaped_win_probs = win_probs.reshape(
            self.eval_sampling_params.n, len(prompts)
        ).transpose(1, 0)
        return reshaped_responses, reshaped_win_probs

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[TrajectoryData]:
        assert not self.eval_mode
        info = {}

        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)
        all_candidates = self.extract_candidates_from_output(
            outputs, self.sampling_params
        )
        info["actor/generate_time"] = time.time() - st

        # step 2. query for oracle reward
        st = time.time()

        rewards, oracle_info = self.oracle.get_reward(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in prompts
                )
            ),
            tree.flatten(all_candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in references
                )
            ),
        )
        rewards = rewards.reshape(len(prompts), self.sampling_params.n)

        info["actor/rewards_mean"] = rewards.mean().item()
        info["actor/rewards_std"] = rewards.std().item()
        info["actor/rewards_std_per_prompt"] = rewards.std(1).mean().item()
        info["actor/oracle_time"] = time.time() - st
        # info.update({f"oracle/{k}": v for k, v in oracle_info.items()})

        trajectory_data = [
            TrajectoryData(
                prompt=prompts[i],
                responses=all_candidates[i],
                rewards=rewards[i],
                info=info,
            )
            for i in range(len(prompts))
        ]

        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle

```

```python
# ./actors/preference.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import List

import numpy as np
import torch
import vllm

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.exploration import ExplorationResults, Explorer, ModelBasedExplorer
from oat.rm import backbone, model
from oat.types import PreferenceData


class PreferenceActor(ActorBase):
    """The environment is a preference oracle. In this case the problem can be formulated
    as preference-based reinforcement learning (PbRL) or contextual dueling bandit (CDB).
    """

    def __init__(self, ipc_server, vllm_args, args: OATArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        # ###################################
        # ####        Exploration        ####
        # ###################################
        self.learning_rm = False
        if args.exp_method == "no":
            if self.sampling_params.n > 2:
                logging.warn(
                    f"trying to sample {self.sampling_params.n} responses but "
                    "no selection mechanism is provided"
                )
        else:
            assert self.sampling_params.n > 2
            # We assume reward model-based explorer.
            rm_backbone_cls = backbone.get_cls(args.rm_backbone)
            logging.info(f"Using RM backbone {args.rm_backbone} {rm_backbone_cls}")
            self.rm_backbone = rm_backbone_cls.from_pretrained(
                args.rm_backbone, device_map="cuda:0"
            ).eval()

            explorer_cls = ModelBasedExplorer if args.model_rollout else Explorer
            self.explorer = explorer_cls(
                reward_model=getattr(model, args.exp_method)(args).cuda(),
                rm_backbone=self.rm_backbone,
                args=args,
            )

            if args.rm_pretrain:
                logging.info(f"Loading pretrained ENN from {args.rm_pretrain}")
                self.explorer.reward_model.load_state_dict(torch.load(args.rm_pretrain))
            else:
                self.learning_rm = True  # Learn RM online.
        self.model_rollout = args.model_rollout

        # ###################################
        # ####  Best-of-N for Evaluation ####
        # ###################################
        if args.best_of_n_eval:
            self.num_eval_gen = args.num_bon
        else:
            self.num_eval_gen = 1
        self.eval_sampling_params = vllm.SamplingParams(
            n=self.num_eval_gen,
            temperature=(
                args.eval_temperature
                if self.num_eval_gen == 1
                else args.bon_temperature
            ),
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
        )

    def generate(self, prompts: List[str], sampling_params: vllm.SamplingParams):
        outputs = super().generate(prompts, sampling_params)
        candidates = {}
        for i in range(len(outputs)):
            # for each prompt
            candidates[i] = []
            for k in range(sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text.strip())
        return candidates

    def generate_and_maybe_eval(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ):
        assert self.eval_mode
        candidates = self.generate(formatted_prompts, self.eval_sampling_params)

        if self.num_eval_gen > 1:
            # best of n sampling
            responses = self.explorer.best_of_n(prompts, candidates)
        else:
            responses = [candidates[i][0] for i in range(len(prompts))]

        if references:
            logging.debug(f"Evaluating using oracle {self.oracle}")
            st = time.time()
            win_probs, _ = self.oracle.compare(
                prompts,
                responses,
                references,
                batch_size=self.oracle_batch_size,
                return_probs=True,
                disable_tqdm=True,
            )
            logging.debug(f"Time elapse {time.time() - st}")
            return responses, win_probs
        return responses, None

    def online_eval(self, prompts, references, candidates):
        """Evaluate online responses."""
        win_probs_1, _ = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            references,
            batch_size=self.oracle_batch_size,
            return_probs=True,
            disable_tqdm=True,
        )
        win_probs_2, _ = self.oracle.compare(
            prompts,
            [candidates[i][1] for i in range(len(prompts))],
            references,
            batch_size=self.oracle_batch_size,
            return_probs=True,
            disable_tqdm=True,
        )
        return (win_probs_1 + win_probs_2) / 2

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[PreferenceData]:
        """Step the actor.

        Given a prompt x, K responses {y_1, ..., y_K} are sample from the behavior LLM pi_beta,
        from which 2 responses are selected to query the oracle for preference signal.
        The final constructed pair (x, y_w, y_l) is inserted into the replay buffer for learners.

        Args:
            prompts: A list of prompt texts.
            formatted_prompts: A list of chat template formatted prompt texts.
            references: A list of reference texts.
        """
        assert not self.eval_mode
        info = {}

        # step 1. generate
        st = time.time()
        all_candidates = self.generate(formatted_prompts, self.sampling_params)
        info["actor/generate_time"] = time.time() - st

        # step 2a. optional selection
        results = None
        if self.sampling_params.n > 2:
            results: ExplorationResults
            results = self.explorer.select(prompts, all_candidates)
            candidates = results.dueling_candidates
        else:
            candidates = all_candidates

        # step 2b. optional online eval
        if self.enable_online_evaluation:
            assert references is not None
            win_probs = self.online_eval(prompts, references, candidates)
            info["eval/online_win_probs"] = win_probs.mean()

        # step 3. query for oracle preference
        st = time.time()
        bt_probs, _ = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            [candidates[i][1] for i in range(len(prompts))],
            batch_size=self.oracle_batch_size,
            return_probs=True,
            disable_tqdm=True,
        )
        info["actor/first_action_win_prob"] = bt_probs.mean().item()
        info["actor/oracle_time"] = time.time() - st

        if self.args.bt_sample:
            binary_feedback = torch.bernoulli(torch.from_numpy(bt_probs)).bool().numpy()
        else:
            binary_feedback = bt_probs > 0.5

        chosen = 1 - binary_feedback

        # Model-based rollout for 1) Dyna - sample efficiency; 2) Better argmax r approximation.
        # (Mixed preference learning: Section 4.2.3 of https://arxiv.org/pdf/2411.01493)
        if self.model_rollout:
            # Record metric and overwrite label.
            model_data = np.array(results.is_model_data)
            model_rollout_correct = chosen[model_data] == 0
            model_rollout_acc = np.sum(model_rollout_correct) / (
                np.sum(model_data) + 1e-8
            )
            model_rollout_win_prob = np.nan_to_num(bt_probs[model_data].mean())
            info["eval/model_rollout_acc"] = model_rollout_acc
            info["eval/model_rollout_win_prob"] = model_rollout_win_prob

        rejected = 1 - chosen

        same_response = [
            candidates[i][chosen[i]] == candidates[i][rejected[i]]
            for i in range(len(prompts))
        ]

        if self.learning_rm:
            # Measure the internal RM accuracy
            pred_first_win, _ = self.explorer.compare(results.candidate_features)
            candidate_features = results.candidate_features.cpu()
            correct = pred_first_win == binary_feedback
            info["eval/rm_acc"] = correct.mean().item()

        if results is not None:
            info.update(results.info)

        chosen_responses = [candidates[i][chosen[i]] for i in range(len(prompts))]
        rejected_responses = [candidates[i][rejected[i]] for i in range(len(prompts))]

        preference_data = [
            PreferenceData(
                prompt=prompts[i],
                chosen_id=chosen[i],
                chosen_response=chosen_responses[i],
                rejected_response=rejected_responses[i],
                chosen_feature=(
                    candidate_features[i][chosen[i]] if self.learning_rm else None
                ),
                rejected_feature=(
                    candidate_features[i][rejected[i]] if self.learning_rm else None
                ),
                init_clash=results.init_clash[i] if self.learning_rm else False,
                loss_mask=not same_response[i],
                is_model_data=results.is_model_data[i] if self.learning_rm else False,
                info=info,
            )
            for i in range(len(prompts))
        ]

        handle = self.ipc_client.serialize_ipc(preference_data)
        return handle

```

```python
# ./actors/base.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import logging
import time
from typing import List, Union

import torch
import tree
import vllm

from oat import oracles
from oat.args import OATArgs
from oat.rm import model
from oat.types import PreferenceData, TrajectoryData
from oat.utils.distributed import WorkerWrap, torch_type_codec
from oat.utils.ipc import PlasmaShmClient

logging.getLogger("vllm").setLevel(logging.ERROR)


class ActorBase(abc.ABC):
    """Actor handles the interaction between the agent and the environment."""

    def __init__(self, ipc_server, vllm_args, args: OATArgs) -> None:
        self.args = args
        self.eval_mode = False
        self.generate_mode = False

        # Measuring the **online** performance
        self.enable_online_evaluation = args.online_evaluation

        self.ipc_client = PlasmaShmClient(ipc_server)

        # ###################################
        # ####      vLLM Generation      ####
        # ###################################
        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            n=args.num_samples,
        )

        self.__vllm_version__ = vllm.__version__

        assert self.__vllm_version__ >= "0.4.1", "Upgrade to vLLM >= 0.4.1"
        assert (
            self.sampling_params.n >= 2
        ), "need to sample at least 2 responses per prompt"

        vllm.worker.worker.Worker = WorkerWrap
        vllm_args.update({"seed": time.time_ns() % 2**32})
        self.llm = vllm.LLM(**vllm_args)
        self.tokenizer = self.llm.get_tokenizer()
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        # ###################################
        # ####     Feedback Oracles      ####
        # ###################################
        oracle_cls = oracles.get_cls(args.oracle)
        logging.info(f"Using reward oracle {args.oracle} {oracle_cls}")
        self.oracle = (
            oracle_cls(
                reward_model_path=args.oracle,
                tokenizer_path=args.pretrain,
                remote_rm_url=args.remote_rm_url,  # Only for remote RM.
                max_workers=args.remote_rm_client_workers,  # Only for remote RM.
            )
            if oracle_cls is not None
            else None
        )
        self.oracle_batch_size = args.oracle_batch_size

    def generate(self, prompts: List[str], sampling_params: vllm.SamplingParams):
        self.generate_mode = True
        if self.tokenizer.bos_token:
            # lstrip bos_token because vllm will add it.
            prompts = [p.lstrip(self.tokenizer.bos_token) for p in prompts]
        outputs = self.llm.generate(
            prompts, sampling_params=sampling_params, use_tqdm=False
        )
        if self.tokenizer.bos_token:
            # make sure vllm added bos_token.
            assert self.tokenizer.bos_token_id in outputs[0].prompt_token_ids

        self.generate_mode = False
        return outputs

    @abc.abstractmethod
    def generate_and_maybe_eval(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ):
        """
        1) Generate responses for given prompts;
        2) Optionally evaluate the win rate over references based on the oracle reward model.
        """

    @abc.abstractmethod
    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[Union[PreferenceData, TrajectoryData]]:
        """Step the actor.

        Given a prompt x, K responses {y_1, ..., y_K} are sample from the behavior LLM pi_beta,
        from which some responses are selected to query the oracle for feedback signal.
        The final constructed pair (x, y_w, y_l) is inserted into the replay buffer for learners.

        Args:
            prompts: A list of prompt texts.
            formatted_prompts: A list of chat template formatted prompt texts.
            references: A list of reference texts.
        """

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        self._model_update_group = (
            self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address,
                master_port,
                rank_offset,
                world_size,
                group_name,
                backend,
            )
        )

    def is_generating(self):
        return self.generate_mode

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self._stop_remote_worker_execution_loop()
        return self.llm.llm_engine.model_executor.driver_worker.update_weight(
            name, dtype, shape, empty_cache
        )

    def update_rm(self, name, dtype, shape):
        assert self.learning_rm
        dtype = torch_type_codec(dtype)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)
        params_dict = dict(self.explorer.reward_model.named_parameters())
        model.default_weight_loader(params_dict[name], weight)
        del weight

    def notify_eval_start(self, eval=True):
        """Temporarily cache the current behavior policy weights to CPU."""
        if eval:
            self.eval_mode = True
        logging.debug("Start offloading...")
        st = time.time()
        self.cache_model_state = tree.map_structure(
            lambda x: x.cpu(), self.model.state_dict()
        )
        logging.debug(f"Finished offloading in {time.time() - st} seconds")

    def notify_eval_done(self, eval=True):
        """Load cached behavior policy weights to GPU."""
        if eval:
            assert self.eval_mode
        logging.debug("Start loading from cpu...")
        st = time.time()
        self.model.load_state_dict(self.cache_model_state)
        logging.debug(f"Finished loading in {time.time() - st} seconds")
        if eval:
            self.eval_mode = False

    def _stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__vllm_version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()

```

```python
# ./rm/__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

```python
# ./rm/uncertainty.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import einops
import torch


def kl_divergence(rewards: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Epistemic uncertainty measured by model disagreement between ensembles.

    Calculates KL divergence between individual distribution predictions and
    the Bernoulli mixture distribution.

    Args:
        rewards (torch.Tensor): Reward prediction (logits), (E, M, N, 1)

    Returns:
        torch.Tensor: Uncertainty, (M, N, N')
    """
    E = rewards.shape[0]
    p = bradley_terry_prob_with_temp(
        rewards,
        einops.rearrange(rewards, "e m n 1 -> e m 1 n"),
        temperature=temperature,
    )  # (E, M, N, N')

    p_mean = p.mean(dim=0, keepdim=True)
    pc = 1 - p
    pc_mean = 1 - p_mean

    component_p = p * torch.log(p / p_mean)
    component_pc = pc * torch.log(pc / pc_mean)

    repeat_p_mean = p_mean.repeat(E, 1, 1, 1)
    repeat_pc_mean = pc_mean.repeat(E, 1, 1, 1)
    kl = torch.where(
        repeat_p_mean == 1,
        component_p,
        torch.where(repeat_pc_mean == 1, component_pc, component_p + component_pc),
    ).mean(dim=0)

    # Avoid numerical errors.
    nan_idx = torch.isnan(kl)
    kl_T = kl.transpose(-1, -2)
    kl[nan_idx] = kl_T[nan_idx]
    return (kl + kl_T) / 2


def logits_variance(rewards: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Computes variance of pre-sigmoid logits."""
    del temperature
    pref_logits = rewards - einops.rearrange(
        rewards, "e m n 1 -> e m 1 n"
    )  # (E, M, N, N')
    return pref_logits.var(dim=0)


def probabilities_variance(
    rewards: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    prob = bradley_terry_prob_with_temp(
        rewards,
        einops.rearrange(rewards, "e m n 1 -> e m 1 n"),
        temperature=temperature,
    )
    return prob.var(dim=0)


def bernoulli_variance(rewards: torch.Tensor):
    prob = bradley_terry_prob_with_temp(
        rewards,
        einops.rearrange(rewards, "e m n 1 -> e m 1 n"),
        temperature=1.0,
    ).mean(0)
    return prob * (1 - prob)


def bradley_terry_prob_with_temp(scores_1, score_2, temperature=1.0):
    return 1 / (1 + torch.exp(-(scores_1 - score_2) / temperature))

```

```python
# ./rm/backbone.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    SequenceClassifierOutput,
)


def get_cls(model_name: str):
    if "pairrm" in model_name.lower():
        return DebertaV2PairRM
    if "deberta" in model_name.lower():
        return DebertaV2Vanilla
    return PythiaPretrained


class RMBackbone(abc.ABC):
    tokenizer: AutoTokenizer
    source_prefix: str
    cand_prefix: str

    def tokenize_pair(
        self, prompt: str, candidate: str, source_max_length: int, max_length: int
    ):
        source_ids = self.tokenizer.encode(
            self.source_prefix + prompt,
            max_length=source_max_length,
            truncation=True,
        )
        candidate_max_length = max_length - len(source_ids)
        candidate_ids = self.tokenizer.encode(
            self.cand_prefix + candidate,
            max_length=candidate_max_length,
            truncation=True,
        )
        return source_ids + candidate_ids

    def postprocess(self, outputs, input_ids: torch.Tensor):
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand_idxs = torch.where(input_ids == self.cand_prefix_id)
        cand_encs = encs[cand_idxs[0], cand_idxs[1], :]

        # reduce
        source_cand_encs = torch.cat([source_encs, cand_encs], dim=-1)
        return source_cand_encs.detach()

    def preprocess(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        #  <source_prefix_id>...<sep><cand_prefix_id>...<sep>
        if self.source_prefix_id is not None:
            assert all(
                [
                    self.source_prefix_id in input_ids[i]
                    for i in range(input_ids.shape[0])
                ]
            ), "<source> id not in input_ids"
        if self.cand_prefix_id is not None:
            assert all(
                [self.cand_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
            ), "<candidate> id not in input_ids"

        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        return input_ids, attention_mask

    @torch.no_grad
    def get_feature(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """Get the feature \phi(s, a) in a singleton form."""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        input_ids, attention_mask = self.preprocess(input_ids, attention_mask)

        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        return self.postprocess(outputs, input_ids)


class CustomBackbone(RMBackbone):
    @classmethod
    def from_pretrained(cls, model_name, device_map):
        inst = cls(model_name).to(device_map)
        return inst

    @property
    def device(self):
        return self.pretrained_model.device


class PythiaPretrained(CustomBackbone, nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = self.pretrained_model.config
        self.source_prefix_id = None
        self.cand_prefix_id = None

        self.eval()

    def tokenize_pair(
        self, prompt: str, candidate: str, source_max_length: int, max_length: int
    ):
        del source_max_length
        tokens = self.tokenizer.encode(
            prompt + candidate,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
        return tokens

    @torch.no_grad
    def get_feature(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        input_ids, attention_mask = self.preprocess(input_ids, attention_mask)

        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        return self.postprocess(outputs, attention_mask)

    def postprocess(self, outputs, attention_mask: torch.Tensor):
        encs = outputs.hidden_states[-1]
        last_pos = attention_mask.sum(-1).long() - 1
        batch_idx = torch.arange(len(encs), device=encs.device)
        return encs[batch_idx, last_pos, :].detach()


class DebertaV2Vanilla(CustomBackbone, nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = self.pretrained_model.config
        self.source_prefix_id = 1
        self.source_prefix = "[CLS]"
        self.cand_prefix_id = 2
        self.cand_prefix = "[SEP]"

        self.eval()

    def tokenize_pair(
        self, prompt: str, candidate: str, source_max_length: int, max_length: int
    ):
        source_ids = self.tokenizer.encode(
            self.source_prefix + prompt,
            max_length=source_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        candidate_max_length = max_length - len(source_ids)
        candidate_ids = self.tokenizer.encode(
            self.cand_prefix + candidate,
            max_length=candidate_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        return source_ids + candidate_ids


class DebertaV2PairRM(RMBackbone, DebertaV2PreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.n_tasks = config.n_tasks
        self.drop_out = config.drop_out

        self.pretrained_model = DebertaV2Model(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)

        self.sep_token_id = config.sep_token_id  # to add
        self.source_prefix_id = config.source_prefix_id  # to add
        self.source_prefix = "<|source|>"  # to add
        self.cand_prefix_id = config.cand_prefix_id
        self.cand_prefix = "<|candidate|>"

        # Initialize weights and apply final processing
        self.post_init()
        self.eval()

```

```python
# ./rm/model.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import random
from typing import Any, Dict, Tuple

import einops
import torch
import torch.nn.functional as F
from torch import nn, optim

from oat.args import OATArgs
from oat.rm import uncertainty
from oat.rm.networks import EnsembleModel
from oat.utils.buffer import UniformBuffer


class RewardModel(abc.ABC, nn.Module):

    train_bs = 128
    infer_bs = 128

    @abc.abstractclassmethod
    def get_metrics(cls):
        """Get learning metrics."""

    @abc.abstractmethod
    def get_duel_actions(
        self, features: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Get dueling actions based on rewards of given features.

        Args:
            features (torch.Tensor): (M, N, d)

        Returns:
            Tuple[torch.LongTensor]: rewards, first and second indices [(E or 2, M, N, 1), (M, 1), (M, 1)]
        """

    @abc.abstractmethod
    def get_best_action(self, features: torch.Tensor) -> torch.LongTensor:
        """Get Best-of-N action based on rewards of given features.

        Args:
            features (torch.Tensor): (M, N, d)

        Returns:
            torch.LongTensor: (M, 1)
        """

    @abc.abstractmethod
    def learn(self, buffer: UniformBuffer) -> Dict[str, Any]:
        """Learn the reward model based on preference data."""

    @abc.abstractmethod
    def get_rewards(self, features: torch.Tensor) -> torch.Tensor:
        """Compute rewards."""


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self,
        chosen_reward: torch.Tensor,
        reject_reward: torch.Tensor,
        mask: torch.Tensor,
        margin: torch.Tensor = None,
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return (loss * mask).mean()


class EnnEETS(RewardModel):
    """E&E Thompson Sampling based on ensemble."""

    @classmethod
    def get_metrics(cls):
        return {
            "train/rm/loss_rew": 0,
            "train/rm/loss_reg": 0,
            "train/rm/chosen_rewards": 0,
            "train/rm/rejected_rewards": 0,
            "train/rm/lambda": 0,
        }

    def __init__(self, args: OATArgs) -> None:
        super().__init__()
        assert args.enn_max_try <= args.num_ensemble

        self.model = EnsembleModel(
            encoding_dim=getattr(
                args, "encoding_dim", 2048
            ),  # Fixed due to PairRM's backbone
            num_ensemble=args.num_ensemble,
            hidden_dim=args.rm_hidden_dim,
            activation=args.rm_act_fn,
        )
        self.model.init()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.rm_lr, weight_decay=args.rm_wd
        )
        self.reg_lambda = args.enn_lambda
        self.max_resample = args.enn_max_try
        self.allow_second_best = args.exp_allow_second_best
        self.sgd_steps = args.rm_sgd_steps
        self.loss_fn = PairWiseLoss()

    @torch.no_grad
    def get_rewards(self, features: torch.Tensor) -> torch.Tensor:
        M, N, _ = features.shape
        E = self.model.num_ensemble
        features = einops.rearrange(features, "m n d -> (m n) d")
        rewards = []
        for ndx in range(0, len(features), self.infer_bs):
            batch_feat = features[ndx : min(ndx + self.infer_bs, len(features))]
            batch_feat = batch_feat[None, :, :].repeat([E, 1, 1])
            rewards.append(self.model(batch_feat))
        rewards = torch.cat(rewards, dim=1)  # (E, M*N, 1)
        rewards = rewards.view(E, M, N, 1)
        return rewards

    @torch.no_grad
    def get_best_action(self, features: torch.Tensor) -> torch.LongTensor:
        rewards = self.get_rewards(features)  # (E, M, N, 1)
        avg_rewards = rewards.mean(0)  # (M, N, 1)
        best_actions = avg_rewards.argmax(dim=1)  # (M, 1)
        return best_actions

    @torch.no_grad
    def get_duel_actions(
        self, features: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        rewards = self.get_rewards(features)
        E = rewards.shape[0]
        best_actions = rewards.argmax(dim=2)  # (E, M, 1)
        # sample without replacement
        s1 = list(range(E))
        random.shuffle(s1)
        first_actions = best_actions[s1[0]]
        second_actions = torch.ones_like(first_actions) * -1
        for actions in best_actions[s1[1 : self.max_resample]]:
            valid_idx = (actions != first_actions) * (second_actions == -1)
            second_actions[valid_idx] = actions[valid_idx]
            if -1 not in second_actions:
                break
        if self.allow_second_best:
            second_best_actions = rewards.argsort(dim=2)[..., -2, :]
            for actions in second_best_actions[s1[: self.max_resample]]:
                valid_idx = (actions != first_actions) * (second_actions == -1)
                second_actions[valid_idx] = actions[valid_idx]
                if -1 not in second_actions:
                    break
        second_actions = torch.where(
            second_actions == -1, first_actions, second_actions
        )
        return rewards, first_actions, second_actions

    def learn(self, buffer: UniformBuffer) -> Dict[str, Any]:
        total_num_queries = buffer.total_num_queries
        for _ in range(self.sgd_steps):
            batch = buffer.sample(self.train_bs)
            if batch is None:
                return self.get_metrics()
            pair_feats = batch.pair_features.view(2 * self.train_bs, -1)
            batch_inp = pair_feats[None, :, :].repeat([self.model.num_ensemble, 1, 1])
            scores = self.model(batch_inp)
            scores = scores.view(self.model.num_ensemble, self.train_bs, 2, 1)
            chosen_scores, rejected_scores = scores[..., 0, :], scores[..., 1, :]
            loss_rew = self.loss_fn(
                chosen_scores, rejected_scores, batch.loss_masks[None]
            )
            loss_reg = (
                self.reg_lambda
                * self.train_bs
                / total_num_queries
                * self.model.regularization()
            )
            self.optimizer.zero_grad()
            (loss_rew + loss_reg).backward()
            self.optimizer.step()

        return {
            "train/rm/loss_rew": loss_rew.detach(),
            "train/rm/loss_reg": loss_reg.detach(),
            "train/rm/chosen_rewards": chosen_scores.mean().detach(),
            "train/rm/rejected_rewards": rejected_scores.mean().detach(),
            "train/rm/lambda": self.reg_lambda * self.train_bs / total_num_queries,
        }


class EnnUncertainty(EnnEETS):
    """Pure exploration based on ensemble."""

    def __init__(self, args: OATArgs) -> None:
        super().__init__(args)
        self.uct_fn = uncertainty.logits_variance

    @torch.no_grad
    def get_duel_actions(self, features: torch.Tensor) -> Tuple[torch.LongTensor]:
        rewards = self.get_rewards(features)  # (E, M, N, 1)
        _, M, N, _ = rewards.shape
        pref_uncertainty = self.uct_fn(rewards)
        flatten_idx = pref_uncertainty.view(M, -1).argmax(-1)
        first_actions = flatten_idx // N
        second_actions = flatten_idx % N
        return rewards, first_actions.view(M, 1), second_actions.view(M, 1)


class EnnBAITS(EnnEETS):
    """BAI Thompson Sampling based on ensemble."""

    def __init__(self, args: OATArgs) -> None:
        super().__init__(args)
        self.uct_fn = uncertainty.logits_variance

    @torch.no_grad
    def get_duel_actions(
        self, features: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        rewards = self.get_rewards(features)  # (E, M, N, 1)
        E, M, _, _ = rewards.shape
        best_actions = rewards.argmax(dim=2)  # (E, M, 1)
        # sample without replacement
        s1 = list(range(E))
        random.shuffle(s1)
        first_actions = best_actions[s1[0]]

        pref_uncertainty = self.uct_fn(rewards)

        second_actions = torch.stack(
            [pref_uncertainty[i][first_actions[i]].argmax() for i in range(M)], dim=0
        ).view(M, 1)

        return rewards, first_actions, second_actions


class EnnPassive(EnnEETS):
    """Learning RM but not for sampling, only for BoN generation."""

    @torch.no_grad
    def get_duel_actions(self, features: torch.Tensor) -> Tuple[torch.LongTensor]:
        M, _, _ = features.shape  # M, N, d
        rewards = self.get_rewards(features)
        first_actions = torch.zeros((M, 1), device=features.device).long()
        second_actions = torch.ones((M, 1), device=features.device).long()
        return rewards, first_actions, second_actions


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise

```

```python
# ./rm/networks.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deep networks."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t
            )
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, encoding_dim, hidden_dim=128, activation="relu") -> None:
        super(MLPModel, self).__init__()
        self.hidden_size = hidden_dim
        self.output_dim = 1

        self.nn1 = nn.Linear(encoding_dim, hidden_dim)
        self.nn2 = nn.Linear(hidden_dim, hidden_dim)
        self.nn_out = nn.Linear(hidden_dim, self.output_dim)

        self.apply(init_weights)

        if activation == "swish":
            self.activation = Swish()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.nn1(encoding))
        x = self.activation(self.nn2(x))
        score = self.nn_out(x)
        return score

    def init(self):
        self.init_params = self.get_params().data.clone()
        if torch.cuda.is_available():
            self.init_params = self.init_params.cuda()

    def regularization(self):
        """Prior towards independent initialization."""
        return ((self.get_params() - self.init_params) ** 2).mean()


class EnsembleFC(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int,
        bias: bool = True,
    ) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(
            torch.Tensor(ensemble_size, in_features, out_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b


class EnsembleModel(MLPModel):
    def __init__(
        self, encoding_dim, num_ensemble, hidden_dim=128, activation="relu"
    ) -> None:
        super().__init__(encoding_dim, hidden_dim, activation)
        self.num_ensemble = num_ensemble
        self.nn1 = EnsembleFC(encoding_dim, hidden_dim, num_ensemble)
        self.nn2 = EnsembleFC(hidden_dim, hidden_dim, num_ensemble)
        self.nn_out = EnsembleFC(hidden_dim, self.output_dim, num_ensemble)
        self.apply(init_weights)

    def init(self):
        self.init_params = self.get_params().data.clone()
        if torch.cuda.is_available():
            self.init_params = self.init_params.cuda()

    def regularization(self):
        """Prior towards independent initialization."""
        return ((self.get_params() - self.init_params) ** 2).mean()

```

```python
# ./collectors/__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from oat.collectors.asynchronous import AsyncFeedbackCollector
from oat.collectors.base import FeedbackCollector

__all__ = ["AsyncFeedbackCollector", "FeedbackCollector"]

```

```python
# ./collectors/asynchronous.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import List, Union

import torch

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.collectors.base import FeedbackCollector
from oat.types import PreferenceData, TrajectoryData
from oat.utils.ipc import PlasmaShmClient


class AsyncFeedbackCollector(FeedbackCollector):
    def __init__(
        self, args: OATArgs, actors: List[ActorBase], ipc_client: PlasmaShmClient
    ) -> None:
        self.args = args
        self.actors = actors
        self.ipc_client = ipc_client
        self.prev_fut = None

    def collect_feedback(
        self,
        prompts: Union[str, List[str]],
        formatted_prompts: List[str],
        refs: Union[str, List[str]],
    ):
        # generate response & get feedback
        st_time = time.time()

        if self.prev_fut is not None:
            handle = self.prev_fut.result()
            feedback_data: List[Union[PreferenceData, TrajectoryData]] = (
                self.ipc_client.deserialize_ipc(handle)
            )
        else:
            feedback_data = None

        rank = torch.distributed.get_rank()
        actor = self.actors[rank % len(self.actors)]
        if self.args.online_evaluation:
            handle_fut = actor.futures.step(prompts, formatted_prompts, refs)
        else:
            handle_fut = actor.futures.step(prompts, formatted_prompts)

        self.prev_fut = handle_fut

        actor_time = time.time() - st_time

        if feedback_data is not None:
            metrics = self.get_metrics(actor_time, feedback_data)
        else:
            metrics = {}

        return feedback_data, metrics

```

```python
# ./collectors/base.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import List, Union

import Levenshtein
import numpy as np
import torch
import tree

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.types import PreferenceData, TrajectoryData
from oat.utils.ipc import PlasmaShmClient


class FeedbackCollector:
    def __init__(
        self, args: OATArgs, actors: List[ActorBase], ipc_client: PlasmaShmClient
    ) -> None:
        self.args = args
        self.actors = actors
        self.ipc_client = ipc_client

    def get_metrics(
        self,
        actor_time: float,
        feedback_data: List[Union[PreferenceData, TrajectoryData]],
    ):
        metric = {
            "actor/total_time": actor_time,
        }
        if isinstance(feedback_data[0], PreferenceData):
            metric.update(
                {
                    "actor/chosen_avg_str_len": np.mean(
                        [len(p.chosen_response) for p in feedback_data]
                    ),
                    "actor/rejected_avg_str_len": np.mean(
                        [len(p.rejected_response) for p in feedback_data]
                    ),
                    "actor/init_clash_ratio": np.mean(
                        [p.init_clash for p in feedback_data]
                    ),
                    "actor/loss_mask": np.mean([p.loss_mask for p in feedback_data]),
                    "actor/pair_edit_dist": np.mean(
                        [
                            Levenshtein.distance(p.chosen_response, p.rejected_response)
                            for p in feedback_data
                        ]
                    ),
                    "actor/chosen_id": np.mean([p.chosen_id for p in feedback_data]),
                }
            )
        elif isinstance(feedback_data[0], TrajectoryData):
            metric.update(
                {
                    "actor/generate_avg_str_len": np.mean(
                        [len(t.response) for t in feedback_data]
                    )
                }
            )
        else:
            raise ValueError("Invalid feedback data type.")

        mean_info = tree.map_structure(
            lambda *x: np.mean(x), *[p.info for p in feedback_data]
        )
        metric.update(mean_info)

        return metric

    def collect_feedback(
        self,
        prompts: Union[str, List[str]],
        formatted_prompts: List[str],
        refs: Union[str, List[str]],
    ):
        # generate response & get feedback
        st_time = time.time()

        rank = torch.distributed.get_rank()
        actor = self.actors[rank % len(self.actors)]
        if self.args.online_evaluation:
            handle = actor.step(prompts, formatted_prompts, refs)
        else:
            handle = actor.step(prompts, formatted_prompts)
        feedback_data: List[Union[PreferenceData, TrajectoryData]] = (
            self.ipc_client.deserialize_ipc(handle)
        )

        actor_time = time.time() - st_time
        return feedback_data, self.get_metrics(actor_time, feedback_data)

```

```python
# ./utils/data.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import random
from typing import Callable, List, Tuple

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from oat.types import PreferenceData, TrajectoryData
from oat.utils.deepspeed import DeepspeedStrategy


def get_tokenizer(pretrain, model=None, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrain, trust_remote_code=True, use_fast=use_fast
    )
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def load_data_from_disk_or_hf(data_name):
    if os.path.exists(data_name):
        return datasets.load_from_disk(data_name)
    return datasets.load_dataset(data_name)


def get_datasets(tokenizer, strategy, eval_only=False):
    args = strategy.args
    if not eval_only or args.eval_data == "":
        prompt_dataset = load_data_from_disk_or_hf(args.prompt_data)
        prompts_data = prompt_dataset[args.train_split].select(
            range(min(args.max_train, len(prompt_dataset[args.train_split])))
        )
    if args.eval_data:
        strategy.print(f"loading eval data {args.eval_data}")
        if "@" in args.eval_data:
            name, path = args.eval_data.split("@")
        else:
            name, path = None, args.eval_data
        eval_dataset = datasets.load_dataset(path, name, trust_remote_code=True)
    else:
        # Share the same dataset but use different split.
        eval_dataset = prompt_dataset

    eval_prompts_data = eval_dataset[args.eval_split].select(
        range(min(args.max_eval, len(eval_dataset[args.eval_split])))
    )
    if not eval_only:
        prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_key=args.input_key,
            output_key=args.output_key,
            apply_chat_template=args.apply_chat_template,
            get_reference=True,
        )
    else:
        prompts_dataset = None
    eval_prompts_dataset = PromptDataset(
        eval_prompts_data,
        tokenizer,
        strategy,
        input_key=args.eval_input_key or args.input_key,
        output_key=args.eval_output_key or args.output_key,
        apply_chat_template=args.apply_chat_template,
        get_reference=True,
    )
    return prompts_dataset, eval_prompts_dataset


def shard_buffer(
    dataset,
    rank: int,
    num_replicas: int,
    seed: int,
    shuffle=True,
    drop_last=True,
):
    if drop_last and len(dataset) % num_replicas != 0:
        # Ensure each rank receives the same amount of data.
        num_samples = math.ceil((len(dataset) - num_replicas) / num_replicas)
    else:
        num_samples = math.ceil(len(dataset) / num_replicas)
    total_size = num_samples * num_replicas
    indices = list(range(len(dataset)))
    if shuffle:
        # deterministically shuffle based on seed
        random.Random(seed).shuffle(indices)
    if not drop_last:
        padding_size = total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            dataset += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
    else:
        indices = indices[:total_size]
    assert len(indices) == total_size
    indices = indices[rank:total_size:num_replicas]
    assert len(indices) == num_samples
    return [dataset[i] for i in indices]


def pad_to_length(tensor, length, pad_value, dim=-1):
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value
                * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def extract_assistant_content(conversation: List[dict]):
    assert len(conversation) == 2
    for msg in conversation:
        if msg["role"] == "assistant":
            return msg["content"]
    raise ValueError("No assistant content found")


def _preprocess_preference_data(
    data: PreferenceData,
    apply_chat_template=None,
) -> Tuple[str, str, str, bool]:
    if apply_chat_template:
        prompt = {"content": data.prompt, "role": "user"}
        chosen = {"content": data.chosen_response, "role": "assistant"}
        rejected = {"content": data.rejected_response, "role": "assistant"}
        chosen = apply_chat_template([prompt, chosen], tokenize=False)
        rejected = apply_chat_template([prompt, rejected], tokenize=False)

        prompt = apply_chat_template(
            [prompt], tokenize=False, add_generation_prompt=True
        )
        chosen = chosen[len(prompt) :]
        rejected = rejected[len(prompt) :]
    else:
        prompt = data.prompt
        chosen = data.chosen_response
        rejected = data.rejected_response

    return prompt, chosen, rejected, data.loss_mask, data.chosen_id


class PromptDataset(Dataset):
    """Dataset for processing prompts."""

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_key,
        output_key=None,
        apply_chat_template=False,
        get_reference=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.get_reference = get_reference
        self.prompt_max_length = strategy.args.prompt_max_length

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
        if get_reference:
            assert output_key is not None

        self.raw_prompts = []
        self.processed_prompts = []
        self.references = []

        def preprocess_data(data, input_key="input", apply_chat_template=None) -> str:
            if apply_chat_template:
                prompt = apply_chat_template(
                    [{"content": data[input_key], "role": "user"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = data[input_key]
            if get_reference:
                return data[input_key], prompt, data[output_key]
            return data[input_key], prompt

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            if get_reference:
                prompt, processed_prompt, reference = preprocess_data(
                    data, input_key, apply_chat_template
                )
                self.references.append(reference)
            else:
                prompt, processed_prompt = preprocess_data(
                    data, input_key, apply_chat_template
                )
            if len(tokenizer(processed_prompt)["input_ids"]) <= self.prompt_max_length:
                self.processed_prompts.append(processed_prompt)
                self.raw_prompts.append(prompt)

    def __len__(self):
        return len(self.raw_prompts)

    def __getitem__(self, idx):
        if self.get_reference:
            return (
                self.processed_prompts[idx],
                self.raw_prompts[idx],
                self.references[idx],
            )
        return self.processed_prompts[idx], self.raw_prompts[idx]


class PreferenceDataset(Dataset):
    def __init__(
        self,
        buffer: List[PreferenceData],
        tokenizer: Callable,
        prompt_max_length: int,
        generate_max_length: int,
        strategy: DeepspeedStrategy,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.chosen_responses = []
        self.rejected_responses = []
        self.prompt_ids_lens = []
        self.loss_masks = []
        self.chosen_ids = []

        self.tokenizer = tokenizer
        self.strategy = strategy
        self.prompt_max_length = prompt_max_length
        self.generate_max_length = generate_max_length

        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            strategy.print("Applying chat template...")
            apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(
                self.strategy.args, "tokenizer_chat_template", None
            )
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        for data in tqdm(
            buffer,
            disable=not self.strategy.is_rank_0(),
            desc="Constructing preference dataset",
        ):
            prompt, chosen, rejected, loss_mask, chosen_id = (
                _preprocess_preference_data(
                    data,
                    apply_chat_template,
                )
            )
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.prompt_max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            if prompt_ids_len >= self.prompt_max_length - 2:
                logging.warn("Masking samples with too long prompts")
                loss_mask = True

            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.chosen_responses.append(chosen)
            self.rejected_responses.append(rejected)
            self.loss_masks.append(loss_mask)
            self.chosen_ids.append(chosen_id)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt, chosen, rejected = (
            self.prompts[idx],
            self.chosen_responses[idx],
            self.rejected_responses[idx],
        )
        extra = {
            "prompt_ids_lens": self.prompt_ids_lens[idx],
            "loss_masks": self.loss_masks[idx],
            "chosen_ids": self.chosen_ids[idx],
        }  # Modify collate_fn below as well.

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.prompt_max_length + self.generate_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        rejected = (prompt + rejected).rstrip("\n")
        if not rejected.endswith(self.tokenizer.eos_token):
            rejected += " " + self.tokenizer.eos_token
        rejected_token = self.tokenizer(
            rejected,
            max_length=self.prompt_max_length + self.generate_max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # Avoid EOS_token truncation.
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        rejected_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        rejected_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            rejected_token["input_ids"],
            rejected_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        rejected_ids = []
        rejected_masks = []
        extras = {"prompt_ids_lens": [], "loss_masks": [], "chosen_ids": []}
        for chosen_id, chosen_mask, rejected_id, rejected_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            rejected_ids.append(rejected_id)
            rejected_masks.append(rejected_mask)
            extras["prompt_ids_lens"].append(extra["prompt_ids_lens"])
            extras["loss_masks"].append(extra["loss_masks"])
            extras["chosen_ids"].append(extra["chosen_ids"])

        padding_side = "right"
        chosen_ids = zero_pad_sequences(
            chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id
        )
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        rejected_ids = zero_pad_sequences(
            rejected_ids, side=padding_side, value=self.tokenizer.pad_token_id
        )
        rejected_masks = zero_pad_sequences(rejected_masks, side=padding_side)
        return chosen_ids, chosen_masks, rejected_ids, rejected_masks, extras


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        buffer: List[TrajectoryData],
        tokenizer: Callable,
        strategy: DeepspeedStrategy,
        **_,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        # Storing training data.
        self.trajectories = []

        for i in tqdm(
            range(len(buffer)),
            disable=not strategy.is_rank_0(),
            desc="Constructing ppo dataset",
        ):
            trajectory_ids = list(buffer[i].prompt_ids) + list(buffer[i].response_ids)
            self.trajectories.append(
                {
                    "input_ids": torch.tensor(trajectory_ids),
                    "attention_mask": torch.ones(len(trajectory_ids)),
                    "action_ids": buffer[i].response_ids,
                    "rewards": buffer[i].rewards,
                    "loss_mask": buffer[i].loss_mask,
                    "prompt_ids_lens": len(buffer[i].prompt_ids),
                    "action_logprobs": buffer[i].response_logprobs,
                }
            )

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def collate_fn(self, item_list):
        batch_trajectories = {
            "input_ids": [],
            "action_ids": [],
            "attention_mask": [],
            "rewards": [],
            "loss_masks": [],
            "prompt_ids_lens": [],
            "action_logprobs": [],
        }
        for t in item_list:
            batch_trajectories["input_ids"].append(t["input_ids"])
            batch_trajectories["attention_mask"].append(t["attention_mask"])
            batch_trajectories["rewards"].append(t["rewards"])
            batch_trajectories["loss_masks"].append(t["loss_mask"])
            batch_trajectories["prompt_ids_lens"].append(t["prompt_ids_lens"])
            batch_trajectories["action_logprobs"].append(t["action_logprobs"])
            batch_trajectories["action_ids"].append(t["action_ids"])

        padding_side = "right"
        batch_trajectories["input_ids"] = zero_pad_sequences(
            batch_trajectories["input_ids"],
            side=padding_side,
            value=self.tokenizer.pad_token_id,
        )
        batch_trajectories["attention_mask"] = zero_pad_sequences(
            batch_trajectories["attention_mask"],
            side=padding_side,
        )
        return batch_trajectories

```

```python
# ./utils/__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

```python
# ./utils/deepspeed.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference to https://github.com/OpenRLHF/OpenRLHF.

import logging
import os
import random
import shutil
import time
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from pprint import pprint
from typing import List, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from oat.args import OATArgs
from oat.model import LLM

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


def get_strategy(args):
    if args.rnd_seed:
        logging.info("Using randomly generated seed")
        args.seed = time.time_ns() % 2**32
    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        train_batch_size_per_device=getattr(args, "train_batch_size_per_device", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return args, strategy


def get_train_ds_config(
    offload,
    adam_offload=True,
    stage=2,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=False,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "allgather_bucket_size": 1e9,
        "reduce_bucket_size": 1e9,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {
            "grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"
        },
    }


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 1000,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=[
        "bias",
        "layer_norm.weight",
        "layernorm.weight",
        "norm.weight",
        "ln_f.weight",
    ],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


class DeepspeedStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """

    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        train_batch_size_per_device=1,
        train_batch_size=1,
        zero_stage=2,
        bf16=True,
        args: OATArgs = None,
    ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.train_batch_size_per_device = train_batch_size_per_device
        self.bf16 = bf16
        self.seed = seed
        self.max_norm = max_norm
        self.adam_offload = getattr(args, "adam_offload", False)
        self.zpg = getattr(args, "zpg", 1)
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", "fp32")
        # disable_trace_cache
        self.disable_trace_cache = getattr(args, "disable_trace_cache", False)

        self.is_rlhf = False
        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = (
            self.train_batch_size // self.train_batch_size_per_device // self.world_size
        )

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, LLM):
            model = model.model
        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def backward(
        self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs
    ) -> None:
        if isinstance(model, LLM):
            model = model.model
        model.backward(loss)

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        if isinstance(model, LLM):
            model = model.model
        model.step()

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
    ):
        # DDP only mode, replay buffers on each rank are different.
        if sampler is None:
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
            )

        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, LLM):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert (
                    len(arg) == 3
                ), f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._ds_init_train_model(*arg))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(self, model, optim, scheduler):
        is_wrapped = isinstance(model, LLM)
        ds_config = self.get_ds_train_config(is_wrapped)

        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_wrapped else model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            args={"local_rank": self.args.local_rank},
            dist_init_required=True,
        )
        if is_wrapped:
            model.model = engine
        else:
            model = engine

        return model, optim, scheduler

    def get_ds_train_config(self, is_wrapped):
        # DS Config
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=self.stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            disable_trace_cache=self.disable_trace_cache,
        )

        ds_config["train_micro_batch_size_per_gpu"] = self.train_batch_size_per_device
        train_batch_size = self.train_batch_size
        # corner case for ptx loss (backward twice)
        # if self.is_rlhf and is_wrapped and self.args.pretrain_data is not None:
        #     train_batch_size *= 2
        ds_config["train_batch_size"] = train_batch_size

        return ds_config

    def _ds_init_eval_model(self, model):
        is_wrapped = isinstance(model, LLM)
        ds_config = self.get_ds_eval_config(offload=getattr(model, "_offload", False))

        engine, *_ = deepspeed.initialize(
            model=model.model if is_wrapped else model,
            args={"local_rank": self.args.local_rank},
            config=ds_config,
            dist_init_required=True,
        )
        if is_wrapped:
            model.model = engine
        else:
            model = engine
        return model

    def get_ds_eval_config(self, offload=False):
        # DS Config
        ds_config = get_eval_ds_config(
            offload=offload, stage=self.stage if self.stage == 3 else 0, bf16=self.bf16
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.train_batch_size_per_device
        ds_config["train_batch_size"] = self.train_batch_size

        return ds_config

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            data = param.data.to(device)
                            param_ema.data.copy_(
                                (1 - beta) * data + beta * param_ema.data
                            )
                        else:
                            # TODO: use prefiltering for efficiency
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            with deepspeed.zero.GatheredParameters(
                                params_to_fetch, enabled=len(params_to_fetch) > 0
                            ):
                                data = param.data.to(device)
                                param_ema.data.copy_(
                                    (1 - beta) * data + beta * param_ema.data
                                )

    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_model(
        self,
        model: nn.Module,
        tokenizer,
        output_dir,
        tag,
        max_num=3,
        max_mem=1000,
        **kwargs,
    ) -> None:
        if self.is_rank_0():
            save_dir = os.path.join(output_dir, tag)
            os.makedirs(save_dir, exist_ok=True)

            # max hard drive space limit
            MAX_SIZE = max_mem * 1024 * 1024 * 1024

            while True:
                # Get all subdirectory and modification time
                subdirs = [
                    (
                        os.path.join(output_dir, d),
                        os.path.getmtime(os.path.join(output_dir, d)),
                    )
                    for d in os.listdir(output_dir)
                    if os.path.isdir(os.path.join(output_dir, d))
                ]
                # Sort by modification time, oldest first
                subdirs.sort(key=lambda x: x[1])
                # Calculate the total size of all sub -directory
                total_size = 0
                for subdir, _ in subdirs:
                    for dirpath, dirnames, filenames in os.walk(subdir):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)

                # If the number of subdire directors is greater than equal to max_num or the total size is greater than max_mem, the oldest Checkpoint is deleted
                if len(subdirs) > max_num or total_size > MAX_SIZE:
                    oldest_dir, _ = subdirs[0]  # The oldest directory
                    if os.path.exists(oldest_dir):  # Ensure that the directory exists
                        shutil.rmtree(oldest_dir)  # Delete directory
                        self.print(
                            f"Deleted oldest ckpt {oldest_dir}"
                        )  # The standard print function is used here
                else:
                    break

        dist.barrier()

        # save model weights for ZeRO2/3
        model_to_save = self._unwrap_model(model)

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(
                params_to_fetch, enabled=len(params_to_fetch) > 0
            ):
                vv = v.data.cpu()
                if self.is_rank_0():
                    output_state_dict[k] = vv

        if self.is_rank_0():
            state_dict = model_to_save.state_dict()

            # copy named_buffers with `persistent=True`
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False):
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(save_dir, **kwargs)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(save_dir, "adapter_model.bin"),
                    )
            else:
                # save model
                model_to_save.save_pretrained(
                    save_dir, state_dict=output_state_dict, **kwargs
                )

            # save config
            output_config_file = os.path.join(save_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            # save tokenizer
            tokenizer.save_pretrained(save_dir)

            # for models not in AutoModel, copy python module files
            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for filename in os.listdir(train_from_model_path):
                    if filename.endswith(".py"):
                        shutil.copy(
                            os.path.join(train_from_model_path, filename),
                            os.path.join(save_dir, filename),
                        )

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(
                data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM
            )
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [
                torch.zeros_like(data).to(torch.cuda.current_device())
                for _ in range(self.world_size)
            ]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [
                torch.zeros_like(data).to(torch.cuda.current_device())
                for _ in range(self.world_size)
            ]
            dist.gather(
                data.to(torch.cuda.current_device()), ret if self.is_rank_0() else None
            )
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def broadcast(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.broadcast(v)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"
            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            dist.broadcast(data, 0)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def debug(self, *msg):
        if self.args.debug:
            print(*msg)

    def print(self, *msg):
        if self.is_rank_0():
            print("\n")
            print(*msg)
            print("\n")

    def pprint(self, *msg):
        if self.is_rank_0():
            print("\n")
            pprint(*msg)
            print("\n")

    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank()

    def save_ckpt(
        self,
        model,
        save_dir,
        tag=None,
        max_num=3,
        max_mem=1000,
        client_state={},
        save_latest=True,
    ):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        if self.is_rank_0():
            # Check and create the directory
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            # max hard drive space limit
            MAX_SIZE = max_mem * 1024 * 1024 * 1024

            while True:
                # Get all subdirectory and modification time
                subdirs = [
                    (
                        os.path.join(save_dir, d),
                        os.path.getmtime(os.path.join(save_dir, d)),
                    )
                    for d in os.listdir(save_dir)
                    if os.path.isdir(os.path.join(save_dir, d))
                ]
                # Sort by modification time, oldest first
                subdirs.sort(key=lambda x: x[1])
                # Calculate the total size of all sub -directory
                total_size = 0
                for subdir, _ in subdirs:
                    for dirpath, dirnames, filenames in os.walk(subdir):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)

                # If the number of subdire directors is greater than equal to max_num or the total size is greater than max_mem, the oldest Checkpoint is deleted
                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir, _ = subdirs[0]  # The oldest directory
                    if os.path.exists(oldest_dir):  # Ensure that the directory exists
                        shutil.rmtree(oldest_dir)  # Delete directory
                        self.print(
                            f"Deleted oldest ckpt {oldest_dir}"
                        )  # The standard print function is used here
                else:
                    break

        dist.barrier()
        model.save_checkpoint(
            save_dir, tag=tag, client_state=client_state, save_latest=save_latest
        )

    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        # basic ckpt: reuse deepspeed.DeepSpeedEngine.load_checkpoint
        return model.load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
        )


class DummyStrategy:
    def __init__(self, args) -> None:
        self.args = args

    def print(self, *args):
        print(*args)

    def is_rank_0(self):
        return True

```

```python
# ./utils/buffer.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import tree

from oat.types import RewardData


class UniformBuffer(object):
    def __init__(self, max_len: int):
        self._max_len = int(max_len)
        self._storage = None
        self._n = 0
        self._idx = 0

    def extend(self, batch: RewardData):
        if self._storage is None:
            sample_batch = tree.map_structure(lambda t: t[0], batch)
            self._storage = tree.map_structure(
                lambda t: torch.empty(
                    (self._max_len,) + t.shape, dtype=t.dtype, device=t.device
                ),
                sample_batch,
            )

        num_steps = len(batch.pair_features)
        indices = torch.arange(self._idx, self._idx + num_steps) % self._max_len
        tree.map_structure(lambda a, x: assign(a, indices, x), self._storage, batch)
        self._idx = (self._idx + num_steps) % self._max_len
        self._n = min(self._n + num_steps, self._max_len)

    def sample(self, batch_size: int) -> RewardData:
        if batch_size > self._n:
            return None
        start_indices = np.random.choice(self._n, batch_size, replace=False)
        base_idx = 0 if self._n < self._max_len else self._idx
        all_indices = (start_indices + base_idx) % self._max_len
        return tree.map_structure(lambda a: a[all_indices], self._storage)

    def get_all(self) -> RewardData:
        all_indices = np.arange(self.size)
        return tree.map_structure(lambda a: a[all_indices], self._storage)

    @property
    def size(self):
        return self._n


def assign(a, i, x):
    a[i] = x

```

```python
# ./utils/slicer.py
# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slice LLM outputs into reasoning steps."""

from typing import List


def get_slicer(name: str):
    if "gsm8k" in name:
        return slice_gsm8k
    else:
        raise NotImplementedError


def slice_gsm8k(
    solution: str, delimiter: str = "\n", answer_prefix: str = "\n#### "
) -> List[int]:
    """
    Reference to VinePPO codebase: https://github.com/McGill-NLP/VinePPO.

    Args:
        solution: The solution text.

    Returns:
        A list of indices where each index corresponds to the start of a reasoning step.
        Example:
        >>> solution = '...'
        >>> indices = slice_gsm8k(solution)
        >>> steps = [solution[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]
    """

    if answer_prefix is None:
        sol_without_answer, answer = solution, None
    else:
        try:
            solution_parts = solution.split(answer_prefix)
            if len(solution_parts) < 2:
                sol_without_answer, answer = solution, None
            else:
                sol_without_answer, answer = solution_parts
        except Exception:
            print(solution)
            raise

    steps = sol_without_answer.split(delimiter)

    # Merge first empty steps to the first non-empty step
    first_non_empty_step_idx = None
    for i, step in enumerate(steps):
        if step.strip() != "":
            first_non_empty_step_idx = i
            break

    if first_non_empty_step_idx is not None and first_non_empty_step_idx > 0:
        new_first_step = delimiter.join(steps[: first_non_empty_step_idx + 1])

        steps = [new_first_step] + steps[first_non_empty_step_idx + 1 :]

    if answer is not None:
        # We want to merge the last step with the answer

        # Find last non-empty step index
        last_non_empty_step_idx = None
        for i in range(len(steps) - 1, -1, -1):
            if steps[i].strip() != "":
                last_non_empty_step_idx = i
                break

        if last_non_empty_step_idx is None:
            # Then it means the entire solution is a single step
            last_non_empty_step_idx = 0

        new_last_step = delimiter.join(steps[last_non_empty_step_idx:])
        # Also merge the last step with the answer
        new_last_step = f"{new_last_step}{answer_prefix}{answer}"
        steps = steps[:last_non_empty_step_idx] + [new_last_step]

    reconstructed_solution = delimiter.join(steps)
    assert reconstructed_solution == solution, f"{reconstructed_solution} != {solution}"

    # Find the indices of the reasoning steps
    indices = [0]
    for i, step in enumerate(steps):
        if i == 0:
            indices.append(indices[-1] + len(step))
        else:
            indices.append(indices[-1] + len(step) + len(delimiter))

    assert indices[-1] == len(solution), f"{indices[-1]} != {len(solution)}"

    return indices

```

```python
# ./utils/distributed.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference to https://github.com/OpenRLHF/OpenRLHF.

import errno
import logging
import socket
from datetime import timedelta
from typing import Any, Optional, Union

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from vllm.worker.worker import Worker

# !!! IMPORTANT NOTE !!!(liuzc)
# torch.dtype cannot be passed through lp's rpc due to segmentation fault; use string instead.
_torch_type_decode = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}
_torch_type_encode = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
}


def torch_type_codec(dtype_or_str):
    if isinstance(dtype_or_str, torch.dtype):
        return _torch_type_encode[dtype_or_str]
    elif isinstance(dtype_or_str, str):
        return _torch_type_decode[dtype_or_str]
    else:
        raise ValueError(f"Invalid dtype or str: {dtype_or_str}")


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        pg_options=pg_options,
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    print(
        f"init_process_group: init_method={init_method}, backend={backend}, "
        + f"rank={rank}, world_size={world_size}, group_name={group_name}"
    )
    return pg


class WorkerWrap(Worker):
    def init_process_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Init torch process group for model weights update"""
        assert (
            torch.distributed.is_initialized()
        ), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logging.info(
            f"init_process_group: master_address={master_address}, master_port={master_port}, "
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )
        return self._model_update_group

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (learner model)"""
        dtype = torch_type_codec(dtype)

        # if torch.distributed.get_rank() == 0:
        #     print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert (
            dtype == self.model_config.dtype
        ), f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        if empty_cache:
            torch.cuda.empty_cache()


def node_ip_address_from_perspective(address: str = "8.8.8.8:53"):
    """IP address by which the local node can be reached *from* the `address`.

    Args:
        address: The IP address and port of any known live service on the
            network you care about.

    Returns:
        The IP address by which the local node can be reached from the address.
    """
    ip_address, port = address.split(":")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This command will raise an exception if there is no internet
        # connection.
        s.connect((ip_address, int(port)))
        node_ip_address = s.getsockname()[0]
    except OSError as e:
        node_ip_address = "127.0.0.1"
        # [Errno 101] Network is unreachable
        if e.errno == errno.ENETUNREACH:
            try:
                # try get node ip address from host name
                host_name = socket.getfqdn(socket.gethostname())
                node_ip_address = socket.gethostbyname(host_name)
            except Exception:
                pass
    finally:
        s.close()

    return node_ip_address

```

```python
# ./utils/ipc.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference to https://github.com/mosecorg/mosec."""

import logging
import multiprocessing as mp
import pickle
import subprocess
import time
from typing import Any

import launchpad as lp
from pyarrow import plasma  # type: ignore

DataID = bytes


class PlasmaShmServer:
    def __init__(self, size_mb: int = 5):
        self._size_mb = size_mb
        self._terminated = False
        self._shm_path = ""

    def get_shm_path(self):
        return self._shm_path

    def halt(self):
        self._terminated = True

    def _start_plasma_server(self, size_mb):
        with plasma.start_plasma_store(plasma_store_memory=size_mb * 1000 * 1000) as (
            shm_path,
            shm_process,
        ):
            self._shm_path = shm_path
            while not self._terminated:
                time.sleep(3)
                code = None
                if isinstance(shm_process, mp.Process):
                    code = shm_process.exitcode
                elif isinstance(shm_process, subprocess.Popen):
                    code = shm_process.poll()

                if code is not None:
                    logging.warn(f"Plasma daemon process error {code}")
                    break

    def run(self):
        self._start_plasma_server(self._size_mb)
        lp.stop()


class PlasmaShmClient:
    """Plasma shared memory client."""

    _plasma_client = None

    def __init__(self, server: PlasmaShmServer) -> None:
        self.server = server

    def _get_client(self):
        """Get the plasma client. This will create a new one if not exist."""

        if not self._plasma_client:
            path = self.server.get_shm_path()
            if not path:
                raise RuntimeError("plasma path no found")
            self._plasma_client = plasma.connect(path)
        return self._plasma_client

    def serialize_ipc(self, data: Any) -> DataID:
        """Save the data to the plasma server and return the id."""
        client = self._get_client()
        object_id = client.put(pickle.dumps(data))
        return object_id.binary()

    def deserialize_ipc(self, data: DataID) -> Any:
        """Get the data from the plasma server and delete it."""
        client = self._get_client()
        object_id = plasma.ObjectID(bytes(data))
        obj = pickle.loads(client.get(object_id))
        client.delete((object_id,))
        return obj

```

```python
# ./utils/launcher.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import socket


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    ip, port = sock.getsockname()
    sock.close()
    return port


class DistributedLauncher:
    def __init__(
        self, world_size, rank, local_rank, master_addr, master_port, is_master=False
    ) -> None:
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr
        self._master_port = master_port
        if is_master:
            self._master_port = self.bind()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        os.environ["LOCAL_RANK"] = str(0)

    def bind(self):
        with socket.socket() as sock:
            sock.bind((self._master_addr, self._master_port))
            return sock.getsockname()[1]

```

```python
# ./utils/ops.py
# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch


def disable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None
) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(
    values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(
    values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True
) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

```

```python
# ./oracles/__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from oat.oracles.gpt import GPTJudgeOracle
from oat.oracles.gsm8k import GSM8KOracle
from oat.oracles.pair import PairRMOracle
from oat.oracles.remote.client import RemoteRMOracle


def get_cls(model_name: str):
    if "pairrm" in model_name.lower():
        return PairRMOracle
    if "gpt" in model_name.lower():
        return GPTJudgeOracle
    if "remote" in model_name.lower():
        return RemoteRMOracle
    if "gsm8k" == model_name.lower():
        return GSM8KOracle
    # Return None if specified oracle is not implemented in oat;
    # in this case users need to define their own oracle.
    return None

```

```python
# ./oracles/pair.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Tuple

import llm_blender
import torch

from oat.oracles.base import PreferenceOracleBase
from oat.types import Metric


class PairRMOracle(PreferenceOracleBase):
    def __init__(self, **_) -> None:
        super().__init__()
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        logits = self.blender.compare(
            inputs,
            candidates_A,
            candidates_B,
            batch_size=batch_size,
            return_logits=True,
            disable_tqdm=disable_tqdm,
        )
        probs = torch.from_numpy(logits).sigmoid().numpy()
        if return_probs:
            return probs, {}
        else:
            return probs > 0.5, {}

```

```python
# ./oracles/gsm8k.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, Tuple

import regex as re
import torch

from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric

FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
)


class GSM8KOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the GSM8K task."""

    def __init__(self, use_original_format: bool = False, **_) -> None:
        super().__init__()
        self.use_original_format = use_original_format

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        del inputs, batch_size
        predicted_answers = []
        rewards = []

        for resp, ref in zip(responses, references):
            answer_candidate = self._extract_predicted_answer_from_text(resp)
            predicted_answers.append(answer_candidate)
            grading_res = self._grade_answer(answer_candidate, ref)
            rewards.append(float(grading_res))

        return torch.tensor(rewards), {"predicted_answers": predicted_answers}

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info

    def _extract_predicted_answer_from_text(self, text: str) -> Optional[str]:
        if self.use_original_format:
            # Extract the final answer based on ####
            if "####" not in text:
                return None
            parts = text.split("####")
            assert len(parts) >= 2
            return parts[-1].strip()

        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
        if len(pred_answer) == 0:
            return None
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip().rstrip(".")
            return pred_answer

    def _grade_answer(self, pred_answer: str, gt_answer: str) -> bool:
        if pred_answer is None:
            return False
        return (
            pred_answer.strip().replace(",", "").lower()
            == gt_answer.replace(",", "").strip().lower()
        )

```

```python
# ./oracles/countdown.py
# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Util functions borrowed from TinyZero: https://github.com/Jiayi-Pan/TinyZero."""
import json
import re
from typing import Any, List, Tuple

import regex as re
import torch

from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    ground_truth = json.loads(ground_truth)
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    do_print = True

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


class CountdownOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the MATH task."""

    def __init__(self) -> None:
        super().__init__()

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        del inputs, batch_size
        rewards = []

        for resp, ref in zip(responses, references):
            r = compute_score(resp, ref)
            rewards.append(r)

        return torch.tensor(rewards), {}

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info

```

```python
# ./oracles/base.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, List, Tuple

import torch

from oat.types import Metric


class PreferenceOracleBase(abc.ABC):
    @abc.abstractmethod
    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Provide oracle preference feedback.

        Args:
            inputs (List[str]): List of input strings.
            candidates_A (List[str]): List of candidate A strings.
            candidates_B (List[str]): List of candidate B strings
            batch_size (int, optional): Batch size. Defaults to 4.
            disable_tqdm (bool, optional): Print progress. Defaults to False.

        Returns:
            List[Any]:
                - List[float], logits as confidence that A is better than B.
                    >0 means A is better than B, <0 means B is better than A
                - List[bool], True if A is better than B, False otherwise
            Metric: Extra information from the oracle.
        """


class RewardOracleBase(abc.ABC):
    @abc.abstractmethod
    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        """Provide oracle reward feedback.

        Args:
            inputs (List[str]): List of input strings.
            responses (List[str]): List of response strings.
            references (List[str]): List of references strings.
            batch_size (int, optional): Batch size. Defaults to 4.

        Returns:
            torch.Tensor: Rewards.
            Metric: Extra information from the oracle.
        """

```

```python
# ./oracles/gpt.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import logging
import os
import threading
import time
import traceback
from typing import Any, List, Sequence, Tuple
from warnings import warn

import numpy as np
from openai import OpenAI
from scipy.special import logsumexp

from oat.oracles.base import PreferenceOracleBase
from oat.types import Metric

DEFAULT_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''


class GPTJudgeOracle(PreferenceOracleBase):
    def __init__(
        self,
        reward_model_path: str,
        shuffle_order: bool = True,
        max_workers: int = 4,
        max_retry: int = 10,
        **_,
    ) -> None:
        super().__init__()
        self.client = OpenAI(
            api_key=os.environ.get("OAI_KEY"),
            base_url=os.environ.get("OAI_URL"),
        )
        self.model = reward_model_path
        self.shuffle_order = shuffle_order
        self.invalid_count = 0
        self.max_workers = max_workers
        self.max_retry = max_retry
        self.mutex = threading.Lock()
        self.template = DEFAULT_PAIRWISE_SYSTEM_PROMPT

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        del batch_size, disable_tqdm

        completions = list(zip(candidates_A, candidates_B))

        # Shuffle the order of the completions to avoid positional bias
        if self.shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(inputs))
            completions = [
                pair[::-1] if flip else pair
                for flip, pair in zip(flip_mask, completions)
            ]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.template.format(
                prompt=prompt, response0=candidates[0], response1=candidates[1]
            )
            messages = [{"role": "user", "content": content}]

            wait_time = 1
            for _ in range(self.max_retry):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1,
                        logprobs=True,
                        top_logprobs=5,
                        temperature=0,
                    )
                    break
                except Exception as e:
                    warn(f"OpenAI API failure: {e} {traceback.format_exc()}")
                    time.sleep(wait_time)
                    wait_time *= 1.3
            else:
                raise RuntimeError("OpenAI API error!")

            first_win_prob = logprob_parser(
                completion, numerator_token="0", denominator_tokens=["0", "1"]
            )
            if np.isnan(first_win_prob):
                logging.warn("Invalid win prob!")
                with self.mutex:
                    self.invalid_count += 1
                return np.random.uniform(0, 1)
            else:
                return first_win_prob

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            first_win_probs = list(executor.map(get_rank, inputs, completions))

        # Flip back the ranks to the original order if needed
        if self.shuffle_order:
            first_win_probs = [
                first_win_probs[i] if not flip else 1 - first_win_probs[i]
                for i, flip in enumerate(flip_mask)
            ]
        first_win_probs = np.array(first_win_probs)
        if return_probs:
            return first_win_probs, {}
        else:
            return first_win_probs > 0.5, {}


def logprob_parser(
    completion: dict,
    numerator_token: str,
    denominator_tokens: Sequence[str],
) -> float:
    top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
    map_tokens_to_logprobs = {
        t.token: t.logprob
        for t in top_logprobs
        if t.token in denominator_tokens + [numerator_token]
    }
    missing = float("-inf")
    if len(map_tokens_to_logprobs) == 0:
        return np.nan

    baseline_logprob = map_tokens_to_logprobs.get(numerator_token, missing)
    denominator_logprob = logsumexp(
        [map_tokens_to_logprobs.get(t, missing) for t in denominator_tokens]
    )

    out_logprob = baseline_logprob - denominator_logprob
    probability = np.exp(out_logprob)
    return probability

```

```python
# ./oracles/remote/server.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from typing import List

import torch
import tyro
from mosec import Runtime, Server, Worker
from mosec.mixin import TypedMsgPackMixin
from msgspec import Struct
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Request(Struct, kw_only=True):
    batch_prompt: List[str]
    batch_response: List[str]  # For reward oracle.
    batch_candidates: List[List[str]]  # For preference oracle.


class Response(Struct, kw_only=True):
    batch_score: List[float]  # For reward oracle.
    batch_first_win_prob: List[float]  # For preference oracle.


MODEL_CONFIGS = {
    "Skywork/Skywork-Reward-Llama-3.1-8B": {
        "attn_implementation": "flash_attention_2",
        "num_labels": 1,
    },
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2": {
        "attn_implementation": "flash_attention_2",
        "num_labels": 1,
    },
    "Skywork/Skywork-Reward-Gemma-2-27B-v0.2": {
        "attn_implementation": "flash_attention_2",
        "num_labels": 1,
    },
}


class RewardModel(TypedMsgPackMixin, Worker):
    def __init__(self):
        super().__init__()
        self.model_name = os.environ.get("RM_MODEL_NAME")
        configs = MODEL_CONFIGS.get(self.model_name, {})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **configs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.example = Request(
            batch_prompt=[
                "What is the range of the numeric output of a sigmoid node in a neural network?"
            ],
            batch_response=[],
            batch_candidates=[
                [
                    "The output of a sigmoid node is bounded between -1 and 1.",
                    "The output of a sigmoid node is bounded between 0 and 1.",
                ]
            ],
        )  # To warmup: do one forward pass to allocate GPU memory

    def forward(self, request: Request) -> Response:
        assert self.max_batch_size == 1

        batch_msg = []
        if request.batch_candidates:
            # Rank two candidates.
            batch_msg1 = []
            batch_msg2 = []
            num_data = len(request.batch_prompt)
            for i, prompt in enumerate(request.batch_prompt):
                msg1 = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": request.batch_candidates[i][0]},
                ]
                msg2 = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": request.batch_candidates[i][1]},
                ]
                batch_msg1.append(msg1)
                batch_msg2.append(msg2)
            batch_msg = batch_msg1 + batch_msg2
        elif request.batch_response:
            # Score a given response.
            for i, prompt in enumerate(request.batch_prompt):
                msg = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": request.batch_response[i]},
                ]
                batch_msg.append(msg)

        pair = self.tokenizer.apply_chat_template(batch_msg, tokenize=False)
        pair = self.tokenizer(pair, return_tensors="pt", padding=True).to(
            self.model.device
        )
        with torch.no_grad():
            logits = self.model(**pair).logits.cpu().float().squeeze()

        if request.batch_candidates:
            batch_scores_1 = logits[:num_data]
            batch_scores_2 = logits[num_data:]
            # Apply BT model.
            batch_first_win_prob = (batch_scores_1 - batch_scores_2).sigmoid().tolist()
            batch_score = []
        elif request.batch_response:
            batch_first_win_prob = []
            batch_score = logits.tolist()
        return Response(
            batch_score=batch_score, batch_first_win_prob=batch_first_win_prob
        )


@dataclass
class ServerArgs:
    remote_rm_model: str = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    max_wait_time: int = 10
    cuda_devices: str = "all"
    multi_gpu: bool = False


if __name__ == "__main__":
    args = tyro.cli(ServerArgs)

    if args.multi_gpu:
        NUM_DEVICE = 1
        devices = [",".join([str(i) for i in range(torch.cuda.device_count())])]
    else:
        if args.cuda_devices == "all":
            NUM_DEVICE = torch.cuda.device_count()
            devices = list(range(NUM_DEVICE))
        else:
            devices = args.cuda_devices.split(",")
            NUM_DEVICE = len(devices)

    def _prepare_env(cid: int) -> dict:
        return {
            "CUDA_VISIBLE_DEVICES": str(cid),
            "RM_MODEL_NAME": args.remote_rm_model,
        }

    server = Server()
    runtime = Runtime(
        worker=RewardModel,
        num=NUM_DEVICE,
        max_batch_size=1,
        env=[_prepare_env(x) for x in devices],
        max_wait_time=args.max_wait_time,
        timeout=10,
    )
    server.register_runtime(
        {
            "/get_feedback": [runtime],
        }
    )
    server.run()

```

```python
# ./oracles/remote/__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

```python
# ./oracles/remote/client.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import logging
import threading
import time
from http import HTTPStatus
from typing import Any, List, Tuple
from warnings import warn

import httpx
import msgspec
import numpy as np
import torch
import tree

from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric

logging.getLogger("httpx").setLevel(logging.WARNING)


class RemoteRMOracle(PreferenceOracleBase, RewardOracleBase):
    def __init__(
        self,
        remote_rm_url: str,
        max_workers: int = 4,
        max_retry: int = 10,
        **_,
    ) -> None:
        super().__init__()
        self.client = httpx.Client(
            base_url=remote_rm_url,
        )
        self.invalid_count = 0
        self.max_workers = max_workers
        self.max_retry = max_retry
        self.mutex = threading.Lock()

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> List[Any]:
        del disable_tqdm

        completions = list(zip(candidates_A, candidates_B))

        batched_prompts = []
        batched_completions = []
        num = len(inputs)
        for ndx in range(0, num, batch_size):
            batched_prompts.append(inputs[ndx : min(ndx + batch_size, num)])
            batched_completions.append(completions[ndx : min(ndx + batch_size, num)])

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(bp, bc):
            wait_time = 1
            for _ in range(self.max_retry):
                try:
                    resp = self.client.post(
                        "/get_feedback",
                        content=msgspec.msgpack.encode(
                            {
                                "batch_prompt": bp,
                                "batch_response": [],  # Leave this empty to query preference feedback.
                                "batch_candidates": bc,
                            }
                        ),
                        timeout=5,
                    )
                    if resp.status_code == HTTPStatus.OK:
                        result = msgspec.msgpack.decode(resp.content)
                        batch_first_win_prob = result["batch_first_win_prob"]
                        break
                    else:
                        raise RuntimeError(f"{resp.status_code}, {resp.content}")
                except Exception as e:
                    warn(f"Remote RM server failure: {e}")
                    time.sleep(wait_time)
                    wait_time *= 1.3
            else:
                raise RuntimeError("Remote RM server error!")

            return batch_first_win_prob

        # Call the remote server concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            batch_first_win_prob = list(
                executor.map(get_rank, batched_prompts, batched_completions)
            )

        first_win_probs = np.array(tree.flatten(batch_first_win_prob))
        if return_probs:
            return first_win_probs, {}
        else:
            return first_win_probs > 0.5, {}

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        del references  # Not considering reference answer for now.

        batched_prompts = []
        batched_completions = []
        num = len(inputs)
        for ndx in range(0, num, batch_size):
            batched_prompts.append(inputs[ndx : min(ndx + batch_size, num)])
            batched_completions.append(responses[ndx : min(ndx + batch_size, num)])

        # Define a function to get the score for a single prompt, will be called concurrently
        def get_score(bp, br):
            wait_time = 1
            for _ in range(self.max_retry):
                try:
                    resp = self.client.post(
                        "/get_feedback",
                        content=msgspec.msgpack.encode(
                            {
                                "batch_prompt": bp,
                                "batch_response": br,
                                "batch_candidates": [],  # Leave this empty to query score feedback.
                            }
                        ),
                        timeout=5,
                    )
                    if resp.status_code == HTTPStatus.OK:
                        result = msgspec.msgpack.decode(resp.content)
                        batch_score = result["batch_score"]
                        break
                    else:
                        raise RuntimeError(f"{resp.status_code}, {resp.content}")
                except Exception as e:
                    warn(f"Remote RM server failure: {e}")
                    time.sleep(wait_time)
                    wait_time *= 1.3
            else:
                raise RuntimeError("Remote RM server error!")

            return batch_score

        # Call the remote server concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            batch_score = list(
                executor.map(get_score, batched_prompts, batched_completions)
            )
        return torch.tensor(tree.flatten(batch_score)), {}

```

```python
# ./learners/offline.py
# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time

import launchpad as lp
from tqdm import tqdm

from oat.learners.base import LearnerBase


class OfflineLearner(LearnerBase):
    def run(self):
        self._init(self.args, self.actors)

        self.steps = 0
        self.start_time = time.time()

        self.actor_info = {}
        bs = self.args.rollout_batch_size_per_device

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True)

        self.steps = 1
        self.gradient_update_st = time.time()
        for p_ep in range(self.args.num_prompt_epoch):
            progress_bar = tqdm(
                range(len(self.all_buffer) // bs),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
            )
            for ndx in range(0, len(self.all_buffer), bs):
                # Directly fetch from pre-loaded buffer instead of collecting preference data online.
                self.pi_buffer.extend(
                    self.all_buffer[ndx : min(ndx + bs, len(self.all_buffer))]
                )
                self.prompt_consumed += bs
                self.query_step += bs

                if self.steps % self.update_interval == 0:
                    train_info = self.learn(self.steps // self.update_interval)

                    self.eval_and_log(train_info)

                progress_bar.update()
                self.steps += 1
            self.prompt_epoch = p_ep + 1
            # Reorder data for another epoch.
            random.Random(self.args.seed + p_ep).shuffle(self.all_buffer)

        self.eval_and_log(train_info, eval=True, save=True)

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            try:
                lp.stop()
            except AssertionError:
                pass

```

```python
# ./learners/__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from oat.learners.dap import DAPLearner
from oat.learners.dap_with_rm import DAPwRMLearner
from oat.learners.offline import OfflineLearner
from oat.learners.offline_dap import OfflineDAPLearner
from oat.learners.rl import RLLearner
from oat.learners.sft import OfflineSFTLearner, SFTLearner

__all__ = [
    "DAPLearner",
    "DAPwRMLearner",
    "OfflineDAPLearner",
    "OfflineSFTLearner",
    "RLLearner",
    "SFTLearner",
    "OfflineLearner",
]

```

```python
# ./learners/dap.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Direct optimizer - DAP: Direct Alignment from Preferences."""

import time
from typing import List, Tuple

import numpy as np
import torch
import tree
from torch import distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.learners.base import LearnerBase
from oat.learners.loss import BNFLoss, DPOLoss, SimPOLoss
from oat.model import LLM
from oat.types import DAPAlgo, PreferenceData, SFTAlgo
from oat.utils.data import PreferenceDataset, pad_to_length
from oat.utils.ops import disable_dropout


class DAPLearner(LearnerBase):
    """Direct Alignment from Preference (DAP) learning."""

    def _init(self, args: OATArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)

        if self.algo not in [DAPAlgo.SimPO, SFTAlgo.SFT]:
            self.strategy.print("Running reference-based algorithm... (DPO, IPO, etc.)")
            assert args.ref_pretrain, "Reference model must be non-empty"
            self.ref_model = LLM(
                args.ref_pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                ds_config=self.strategy.get_ds_eval_config(offload=args.ref_offload),
            )
            disable_dropout(self.ref_model)
        else:
            self.strategy.print(
                f"Running reference-free algorithm... ({self.algo.name})"
            )

        # prepare models/optimizers...
        if self.algo not in [DAPAlgo.SimPO, SFTAlgo.SFT]:
            ((self.model, self.optimizer, self.scheduler), self.ref_model) = (
                self.strategy.prepare(
                    (self.model, self.optimizer, self.scheduler),
                    self.ref_model,
                    is_rlhf=True,
                )
            )
        else:
            (self.model, self.optimizer, self.scheduler) = self.strategy.prepare(
                (self.model, self.optimizer, self.scheduler),
                is_rlhf=True,
            )
            self.ref_model = None

        if self.algo in [DAPAlgo.DPO, DAPAlgo.LR_DPO, DAPAlgo.IPO, DAPAlgo.SLiC]:
            self.loss = DPOLoss(
                beta=args.beta,
                label_smoothing=args.label_smoothing,
                dpo_positive_lambda=args.dpo_positive_lambda,
                len_reg_alpha=args.len_reg_alpha,
                sft_weight=args.sft_weight,
                dap_algo=self.algo,
            )
        elif self.algo == DAPAlgo.SimPO:
            self.loss = SimPOLoss(
                args.beta, args.gamma_beta_ratio, args.label_smoothing
            )
        elif self.algo == DAPAlgo.BNF:
            self.loss = BNFLoss()
        else:
            assert self.algo in SFTAlgo, "Invalid DAP Algorithm"

        self.dataset_builder = PreferenceDataset
        dist.barrier()

    def process_feedback_data(self, data_list: List[PreferenceData]):
        self.query_step += np.sum([not p.is_model_data for p in data_list])
        for pref in data_list:
            self.pi_buffer.append(pref)
            if self.args.dump_all_buffer:
                c = pref.chosen_response
                r = pref.rejected_response
                self.all_buffer.append(
                    PreferenceData(
                        prompt=pref.prompt,
                        chosen_response=c,
                        rejected_response=r,
                        same=c == r,
                    )
                )

    def learn(self, learning_round: int):
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.args.prompt_max_length,
            self.args.generate_max_length,
            self.strategy,
        )
        if learning_round == 1:
            self.strategy.print("Training example")
            self.strategy.print(dataset[0])

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.train_batch_size_per_device,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        for epoch in range(self.args.max_epochs):
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )
            acc_mean = []
            loss_mean = []
            chosen_rewards = []
            rejected_rewards = []
            reward_margin = []
            learn_batch_time = []

            self.model.train()
            st = time.time()
            for data in dataloader:
                if local_sgd_steps > self.args.max_sgd_steps:
                    break
                infos = self.learning_step(data)

                # metrics
                loss = infos.pop("loss")
                chosen_reward = infos.pop("chosen_reward")
                rejected_reward = infos.pop("rejected_reward")
                chosen_rewards.append(chosen_reward.mean().item())
                rejected_rewards.append(rejected_reward.mean().item())
                acc_mean.append((chosen_reward > rejected_reward).float().mean().item())
                loss_mean.append(loss.cpu().item())
                reward_margin.append((chosen_reward - rejected_reward).mean().item())

                step_bar.update()
                self.global_step += 1
                if self.global_step % self.strategy.accumulated_gradient == 0:
                    learn_batch_time.append(time.time() - st)
                    self.gradient_update_elapse = time.time() - self.gradient_update_st
                    st = time.time()
                    self.gradient_update_st = time.time()
                    self.policy_sgd_step += 1
                    local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "epoch": epoch + 1,
            "chosen_reward": np.mean(chosen_rewards),
            "rejected_reward": np.mean(rejected_rewards),
            "acc_mean": np.mean(acc_mean),
            "loss_mean": np.mean(loss_mean),
            "reward_margin": np.mean(reward_margin),
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        return train_info

    def learning_step(self, data):
        device = torch.cuda.current_device()
        chosen_ids, c_mask, rejected_ids, r_mask, extra = data
        chosen_ids = chosen_ids.squeeze(1).to(device)
        c_mask = c_mask.squeeze(1).to(device)
        rejected_ids = rejected_ids.squeeze(1).to(device)
        r_mask = r_mask.squeeze(1).to(device)

        prompt_id_lens = extra["prompt_ids_lens"]
        loss_masks = torch.tensor(extra["loss_masks"]).float().to(device)

        if self.algo == DAPAlgo.BNF:

            policy_logps, policy_entropy, token_masks = self.concatenated_forward(
                self.model, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
            )

            with torch.no_grad():
                ref_logps, _, _ = self.concatenated_forward(
                    self.ref_model,
                    chosen_ids,
                    c_mask,
                    rejected_ids,
                    r_mask,
                    prompt_id_lens,
                )
            # BNFLoss
            preference_loss, chosen_reward, rejected_reward = self.loss(
                policy_logps,
                policy_entropy,
                ref_logps,
                token_masks,
                loss_masks,
                chosen_ids.shape,
            )

        else:
            chosen_logps, rejected_logps, _, token_masks = self.concatenated_forward(
                self.model, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
            )

            if self.ref_model is not None:
                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps, _, _ = (
                        self.concatenated_forward(
                            self.ref_model,
                            chosen_ids,
                            c_mask,
                            rejected_ids,
                            r_mask,
                            prompt_id_lens,
                        )
                    )
                # DPOLoss
                preference_loss, chosen_reward, rejected_reward = self.loss(
                    chosen_logps,
                    rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    loss_masks,
                    token_masks,
                )
            else:
                # SimPOLoss
                preference_loss, chosen_reward, rejected_reward = self.loss(
                    chosen_logps, rejected_logps, loss_masks
                )

        loss = preference_loss
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

        infos = {
            "loss": loss.detach(),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
        }
        return infos

    def concatenated_forward(
        self, model, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
    ):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks)
        all_logits = output["logits"]

        if self.algo != DAPAlgo.BNF:

            all_logps, token_masks = self.get_batch_logps(
                all_logits,
                input_ids,
                att_masks,
                prompt_id_lens,
                average_log_prob=self.algo
                in [DAPAlgo.SimPO, DAPAlgo.IPO, DAPAlgo.LR_DPO],
            )
            chosen_logps = all_logps[: chosen_ids.shape[0]]
            rejected_logps = all_logps[chosen_ids.shape[0] :]
            aux_loss = output.aux_loss if "aux_loss" in output else []

            return (
                chosen_logps,
                rejected_logps,
                aux_loss,
                token_masks,
            )

        else:

            all_logps, entropy, token_masks = self.get_batch_logps(
                all_logits,
                input_ids,
                att_masks,
                prompt_id_lens,
            )

            return all_logps, entropy, token_masks

    def concatenated_inputs(
        self, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
    ):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        max_length = max(chosen_ids.shape[1], rejected_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(rejected_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat(
            (
                pad_to_length(c_mask, max_length, 0),
                pad_to_length(r_mask, max_length, 0),
            ),
            dim=0,
        )
        return inputs_ids, att_masks, prompt_id_lens * 2

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
        prompt_id_lens: List[int],
        average_log_prob: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Get masked sum/avg log probabilities."""
        all_logp, target_logps, completion_masks = super().get_batch_logps(
            logits, labels, attention_mask, prompt_id_lens
        )
        if self.algo != DAPAlgo.BNF:
            length = completion_masks.sum(-1)
            if average_log_prob:
                return (target_logps * completion_masks).sum(
                    -1
                ) / length, completion_masks
            else:
                return (target_logps * completion_masks).sum(-1), completion_masks
        else:
            entropy = (all_logp.exp().detach() * all_logp).sum(
                -1
            ) - target_logps.exp().detach() * target_logps
            return (
                target_logps * completion_masks,
                entropy * completion_masks,
                completion_masks,
            )

```

```python
# ./learners/dap_with_rm.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List

import torch
import torch.distributed as dist

from oat.learners.dap import DAPLearner
from oat.rm import model
from oat.types import PreferenceData, RewardData
from oat.utils.buffer import UniformBuffer
from oat.utils.distributed import torch_type_codec


class DAPwRMLearner(DAPLearner):
    """Collocated DAP and reward model learning."""

    def _init(self, args, actors) -> None:
        super()._init(args, actors)
        self.rm = None
        self.learn_rm_only = args.learn_rm_only
        self.fixed_reg = args.rm_fixed_reg
        self.train_budget = args.rm_train_budget

        assert args.exp_method != "no" and args.rm_pretrain == ""
        rm_cls = getattr(model, args.exp_method)
        if self.strategy.is_rank_0():
            self.rm: model.RewardModel = rm_cls(args).to(torch.cuda.current_device())
            self.r_buffer = UniformBuffer(args.r_buffer_maxlen)
        self.train_rm_info = rm_cls.get_metrics()

    def process_preference_data(self, data_list: List[PreferenceData], raw_prompts):
        super().process_preference_data(data_list, raw_prompts)
        c_feats = torch.stack([data.chosen_feature for data in data_list]).unsqueeze(
            dim=1
        )
        r_feats = torch.stack([data.rejected_feature for data in data_list]).unsqueeze(
            dim=1
        )
        pair_feats = torch.cat([c_feats, r_feats], dim=1).to(
            torch.cuda.current_device()
        )  # (micro_b, 2, d)
        same_masks = torch.tensor([data.same for data in data_list]).to(
            torch.cuda.current_device()
        )  # (micro_b,)
        model_data_masks = torch.tensor([data.is_model_data for data in data_list]).to(
            torch.cuda.current_device()
        )  # (micro_b,)

        all_pair_feats = self.strategy.gather(pair_feats)
        all_same_masks = self.strategy.gather(same_masks)
        all_model_data_masks = self.strategy.gather(model_data_masks)
        if self.rm:
            self.r_buffer.extend(
                RewardData(
                    pair_features=all_pair_feats,
                    loss_masks=1 - (all_same_masks | all_model_data_masks).float(),
                )
            )

    def learn(self, learning_round):
        train_info = {}
        # NOTE Put reward learning after policy learning otherwise program gets stuck.
        if not self.learn_rm_only:
            train_info.update(super().learn(learning_round))
        train_info.update(self._reward_learning())
        return train_info

    def get_misc_info(self) -> Dict[str, Any]:
        info = super().get_misc_info()
        r_buffer_len = 0
        if self.rm:
            r_buffer_len = self.r_buffer.size
        info.update({"r_buffer_len": self.strategy.all_reduce(r_buffer_len, "max")})
        return info

    def sync_params_to_actors(self):
        """Additionally sync reward model params."""
        # Sync RM.
        if self.rm:
            for name, param in self.rm.named_parameters():
                shape = param.shape
                futs = [
                    actor.futures.update_rm(
                        name,
                        dtype=torch_type_codec(param.dtype),
                        shape=shape,
                    )
                    for actor in self.actors
                ]
                dist.broadcast(param.data, 0, group=self._model_update_group)
                _ = [fut.result() for fut in futs]

        dist.barrier()

        if not self.learn_rm_only:
            # Sync policy.
            super().sync_params_to_actors()

    def _reward_learning(self):
        total_num_queries = self.strategy.all_reduce(self.query_step, "sum")
        if self.rm and total_num_queries < self.train_budget:
            if self.fixed_reg:
                total_num_queries = self.rm.train_bs
            self.r_buffer.total_num_queries = total_num_queries
            train_rm_info = self.rm.learn(self.r_buffer)
            assert self.train_rm_info.keys() == train_rm_info.keys()
            self.train_rm_info = train_rm_info
        dist.barrier()
        self.strategy.broadcast(self.train_rm_info)
        return self.train_rm_info

```

```python
# ./learners/loss.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from oat.types import DAPAlgo


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(
        self,
        beta: float,
        label_smoothing: float = 0.0,
        len_reg_alpha: float = 0.0,
        dpo_positive_lambda: float = 0.0,
        sft_weight: float = 0.0,
        dap_algo=DAPAlgo.DPO,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.dpo_positive_lambda = dpo_positive_lambda
        self.sft_weight = sft_weight
        self.len_reg_alpha = len_reg_alpha
        self.dap_algo = dap_algo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        loss_masks: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.dap_algo == DAPAlgo.IPO:
            losses = (
                logits - 1 / (2 * self.beta)
            ) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        elif self.dap_algo == DAPAlgo.SLiC:
            losses = torch.relu(1 - self.beta * logits)
        else:
            if self.len_reg_alpha > 0:
                y_length = token_masks.sum(-1)
                length_diff = (
                    y_length[: len(y_length) // 2] - y_length[len(y_length) // 2 :]
                )
                # Eq. 9 https://arxiv.org/pdf/2403.19159; Length Reg in loss.
                logits += self.len_reg_alpha / self.beta * length_diff

            if self.dpo_positive_lambda > 0:
                # Eq. 3 https://arxiv.org/pdf/2402.13228; mitigates chosen prob decreasing issue.
                logits -= self.dpo_positive_lambda * torch.relu(
                    reference_chosen_logps - policy_chosen_logps
                )
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            # Add SFT loss.
            if self.sft_weight > 0:
                losses -= (
                    self.sft_weight
                    * policy_chosen_logps
                    / token_masks.sum(-1)[: (len(token_masks) // 2)]
                )

        loss = (losses * loss_masks).mean()
        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return loss, chosen_rewards, rejected_rewards


class SimPOLoss(nn.Module):
    def __init__(
        self,
        beta: float,
        gamma_beta_ratio: float,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.gamma_beta_ratio = gamma_beta_ratio
        self.loss_type = loss_type
        assert loss_type in (
            "sigmoid",
            "hinge",
        ), f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        loss_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        logits = pi_logratios - self.gamma_beta_ratio
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise RuntimeError

        loss = (losses * loss_masks).mean()
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()

        return loss, chosen_rewards, rejected_rewards


class BNFLoss(nn.Module):
    """
    BNF Loss
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        policy_logps: torch.Tensor,
        policy_entropy: torch.Tensor,
        ref_logps: torch.Tensor,
        token_masks: torch.Tensor,
        loss_masks: torch.Tensor,
        chosen_shape: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        token_count = torch.min(token_masks.sum(-1, keepdims=True))
        target_response = torch.clamp(
            policy_logps.exp() / (ref_logps.exp() + 1e-6),
            max=torch.ones_like(policy_logps),
        )
        target_other = (1 - target_response + 1e-6) / (1 - policy_logps.exp() + 1e-6)
        logp = (
            target_response.detach() * policy_logps
            + target_other.detach() * policy_entropy
        )

        logp = logp * token_masks
        logp_sum = logp.sum(-1, keepdims=True) / token_count

        rewards = policy_logps.sum(-1, keepdims=True) - ref_logps.sum(-1, keepdims=True)

        chosen_logp_sum = logp_sum[: chosen_shape[0]]
        rejected_logp_sum = logp_sum[chosen_shape[0] :]

        chosen_rewards = rewards[: chosen_shape[0]]
        rejected_rewards = rewards[chosen_shape[0] :]

        losses = -(chosen_logp_sum - rejected_logp_sum)

        loss = (losses * loss_masks).mean()

        return loss, chosen_rewards, rejected_rewards

```

```python
# ./learners/offline_dap.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import List

import pandas as pd
import torch
import torch.distributed as dist
import tree
from torch.utils.data import DataLoader
from tqdm import tqdm

from oat.learners.dap import DAPLearner
from oat.learners.offline import OfflineLearner
from oat.types import PreferenceData
from oat.utils.data import (
    extract_assistant_content,
    get_datasets,
    load_data_from_disk_or_hf,
    shard_buffer,
)


class OfflineDAPLearner(OfflineLearner, DAPLearner):

    def prepare_data(self, strategy, tokenizer):
        """Load offline preference data into the buffer instead of using online generated data."""
        args = self.args
        if args.preference_data:
            data = load_data_from_disk_or_hf(args.preference_data)[args.train_split]
            all_shards = []
            drop_cnt = 0
            for item in tqdm(
                data, desc="loading preference data", disable=not strategy.is_rank_0()
            ):
                format_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": item[args.prompt_key]}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if len(format_prompt) >= args.prompt_max_length:
                    drop_cnt += 1
                    continue  # drop too long prompts
                chosen = item[args.chosen_key]
                reject = item[args.rejected_key]
                if args.extract_content:
                    chosen = extract_assistant_content(chosen)
                    reject = extract_assistant_content(reject)
                all_shards.append(
                    PreferenceData(
                        prompt=item[args.prompt_key],
                        chosen_response=chosen,
                        rejected_response=reject,
                        chosen_id=0,
                        chosen_feature=None,
                        rejected_feature=None,
                        init_clash=False,
                        loss_mask=True,
                        is_model_data=False,
                        info={},
                    )
                )
            logging.info(f"[Dataset] Dropped {drop_cnt} samples with too long prompts.")

            all_shards = all_shards[: args.max_train]
            self.all_buffer: List[PreferenceData] = shard_buffer(
                all_shards,
                dist.get_rank(),
                dist.get_world_size(),
                args.seed,
                shuffle=True,
                drop_last=True,
            )
        else:
            # Load pre-dumped data.
            assert os.path.exists(args.offline_buffer_path)
            all_shards = pd.read_pickle(args.offline_buffer_path)
            self.all_buffer: List[PreferenceData] = list(
                all_shards[torch.distributed.get_rank()]
            )
        self.prompts_dataset = tree.flatten(
            all_shards
        )  # needed to calculate lr scheduler
        self.prompts_dataloader = None
        if args.eval_steps > 0:
            _, self.eval_prompts_dataset = get_datasets(
                tokenizer, strategy, eval_only=True
            )
            self.eval_prompts_dataloader = DataLoader(
                self.eval_prompts_dataset,
                batch_size=strategy.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )

```

```python
# ./learners/sft.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SFT optimizer for imitation learning."""

import torch

from oat.learners.dap import DAPLearner
from oat.learners.offline_dap import OfflineDAPLearner


class SFTLearner(DAPLearner):
    """Policy learning via supervised learning.

    We reuse the dap learner and take `chosen` as the target.
    """

    def learning_step(self, data):
        device = torch.cuda.current_device()
        chosen_ids, c_mask, _, _, extra = data
        chosen_ids = chosen_ids.squeeze(1).to(device)
        c_mask = c_mask.squeeze(1).to(device)

        loss = self.model_forward(self.model, chosen_ids, c_mask, extra)
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

        infos = {
            "loss": loss.detach(),
            "chosen_reward": torch.zeros(1),
            "rejected_reward": torch.zeros(1),
        }
        return infos

    def model_forward(self, model, input_ids, att_masks, extra):
        prompt_id_lens = extra["prompt_ids_lens"]

        output = model(input_ids, attention_mask=att_masks)
        all_logits = output["logits"]
        all_logps, _ = self.get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=True
        )
        sft_loss = -all_logps.mean()  # average across examples
        return sft_loss


class OfflineSFTLearner(SFTLearner, OfflineDAPLearner):
    """Offline learning."""

```

```python
# ./learners/rl.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL optimizer."""

import math
import time
from typing import List

import numpy as np
import torch
import tree
from torch import distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.learners.base import LearnerBase
from oat.model import LLM, Critic
from oat.types import RLAlgo, TrajectoryData
from oat.utils.data import TrajectoryDataset
from oat.utils.ops import disable_dropout


class RLLearner(LearnerBase):
    """Policy learning through RL algorithms."""

    def _init(self, args: OATArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        assert self.algo in RLAlgo

        if self.algo == RLAlgo.PPO:
            # Reference policy for regularization.
            self.strategy.print("Running KL-regularized algorithm...")
            assert args.ref_pretrain, "Reference model must be non-empty"
            self.ref_model = LLM(
                args.ref_pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
            )
            disable_dropout(self.ref_model)

            # prepare models/optimizers...
            ((self.model, self.optimizer, self.scheduler), self.ref_model) = (
                self.strategy.prepare(
                    (self.model, self.optimizer, self.scheduler),
                    self.ref_model,
                    is_rlhf=True,
                )
            )
        else:
            self.strategy.print("Running reference-free algorithm...")
            (self.model, self.optimizer, self.scheduler) = self.strategy.prepare(
                (self.model, self.optimizer, self.scheduler),
                is_rlhf=True,
            )
            self.ref_model = None

        if args.critic_type == "ppo":
            self.strategy.print("Learning critic online...")
            self.critic = Critic(
                args.critic_pretrain,
                use_flash_attention_2=args.flash_attn,
                bf16=args.bf16,
                load_in_4bit=args.load_in_4bit,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules,
                ds_config=self.strategy.get_ds_train_config(is_wrapped=True),
            )
            disable_dropout(self.critic)
            if args.gradient_checkpointing:
                self.critic.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": args.gradient_checkpointing_use_reentrant
                    }
                )
            self.critic_optimizer = self.strategy.create_optimizer(
                self.critic,
                lr=args.critic_learning_rate,
                betas=(args.adam_beta_1, args.adam_beta_2),
                weight_decay=args.l2,
            )
            max_steps_to_schedule = self.max_steps * args.critic_max_step_adjustment
            scheduler_specific_kwargs = {}
            if args.lr_scheduler not in ["polynomial"]:
                scheduler_specific_kwargs["min_lr"] = args.learning_rate * 0.1
            self.critic_scheduler = get_scheduler(
                args.lr_scheduler,
                self.critic_optimizer,
                num_warmup_steps=math.ceil(
                    max_steps_to_schedule * args.lr_warmup_ratio
                ),
                num_training_steps=max_steps_to_schedule,
                scheduler_specific_kwargs=scheduler_specific_kwargs,
            )

            (self.critic, self.critic_optimizer, self.critic_scheduler) = (
                self.strategy.prepare(
                    (self.critic, self.critic_optimizer, self.critic_scheduler),
                    is_rlhf=True,
                )
            )
        else:
            self.critic = None

        self.dataset_builder = TrajectoryDataset
        dist.barrier()

    def process_feedback_data(self, data_list: List[TrajectoryData]):
        self.query_step += len(data_list)
        for trajectory in data_list:
            self.pi_buffer.append(trajectory)
            if self.args.dump_all_buffer:
                self.all_buffer.append(
                    TrajectoryData(
                        prompt=trajectory.prompt,
                        response=trajectory.response,
                        rewards=trajectory.rewards,
                    )
                )

    def learn(self, learning_round: int):
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.args.prompt_max_length,
            self.args.generate_max_length,
            self.strategy,
        )
        if learning_round == 1:
            self.strategy.print("Training example")
            self.strategy.print(dataset[0])

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.train_batch_size_per_device,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        for epoch in range(self.args.max_epochs):
            step_bar = tqdm(
                range(len(dataloader)),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )
            learn_batch_time = []

            self.model.train()
            st = time.time()
            for data in dataloader:
                if local_sgd_steps > self.args.max_sgd_steps:
                    break
                infos = self.learning_step(data)

                step_bar.update()
                self.global_step += 1
                if self.global_step % self.strategy.accumulated_gradient == 0:
                    learn_batch_time.append(time.time() - st)
                    self.gradient_update_elapse = time.time() - self.gradient_update_st
                    st = time.time()
                    self.gradient_update_st = time.time()
                    self.policy_sgd_step += 1
                    local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "epoch": epoch + 1,
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        return train_info

```

```python
# ./learners/base.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import math
import os
import socket
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union
from warnings import warn

import deepspeed
import launchpad as lp
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
import vllm
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler

from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.collectors import AsyncFeedbackCollector, FeedbackCollector
from oat.model import LLM
from oat.types import PreferenceData, TrajectoryData
from oat.utils.data import get_datasets, get_tokenizer
from oat.utils.deepspeed import get_strategy
from oat.utils.distributed import (
    init_process_group,
    node_ip_address_from_perspective,
    torch_type_codec,
)
from oat.utils.ipc import PlasmaShmClient, PlasmaShmServer
from oat.utils.launcher import DistributedLauncher
from oat.utils.ops import disable_dropout


class LearnerBase(abc.ABC, DistributedLauncher):
    """Learner updates the LLM policy from preference data collected by actors."""

    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str,
        master_port: str,
        is_master: bool,
        args: OATArgs,
        actors: List[ActorBase],
        ipc_server: PlasmaShmServer,
    ) -> None:
        super().__init__(
            world_size, rank, local_rank, master_addr, master_port, is_master
        )
        self.args = args
        self.actors = actors
        self.ipc_server = ipc_server

    def _init(self, args: OATArgs, actors: List[ActorBase]) -> None:
        args, strategy = get_strategy(args)
        strategy.setup_distributed()

        # ---------- Model related ----------
        # init policy model
        self.model = LLM(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            # ds_config=strategy.get_ds_train_config(is_wrapped=True),
        )
        disable_dropout(self.model)
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant
                }
            )
        # load tokenizer
        tokenizer_path = args.tokenizer if args.tokenizer else args.pretrain
        strategy.print("Loading tokenizer from:", tokenizer_path)

        self.tokenizer = get_tokenizer(
            tokenizer_path,
            self.model.model,
            "left",
            use_fast=not args.disable_fast_tokenizer,
        )
        strategy.print("chat template:", self.tokenizer.chat_template)

        # ---------- Data related ----------
        # prepare buffer
        self.pi_buffer = deque(maxlen=args.pi_buffer_maxlen_per_device)
        self.all_buffer = deque(maxlen=int(1e9))
        # prepare (eval) prompts dataloader
        self.prepare_data(strategy, self.tokenizer)

        strategy.print("Prompt dataset example:")
        strategy.print(self.prompts_dataset[0])
        strategy.print("Prompt dataset len:", len(self.prompts_dataset))

        self.eval_input_key = args.eval_input_key or args.input_key
        self.eval_output_key = args.eval_output_key or args.output_key

        # ---------- Optimizer related ----------
        self.optimizer = strategy.create_optimizer(
            self.model,
            lr=args.learning_rate,
            betas=(args.adam_beta_1, args.adam_beta_2),
            weight_decay=args.l2,
        )
        num_policy_sgd_steps_per_episodes = int(
            len(self.prompts_dataset) * args.max_epochs // args.train_batch_size
        )
        self.max_steps = math.ceil(
            args.num_prompt_epoch * num_policy_sgd_steps_per_episodes
        )
        max_steps_to_schedule = self.max_steps * args.max_step_adjustment

        scheduler_specific_kwargs = {}
        if args.lr_scheduler not in ["polynomial"]:
            scheduler_specific_kwargs["min_lr"] = args.learning_rate * 0.1
        self.scheduler = get_scheduler(
            args.lr_scheduler,
            self.optimizer,
            num_warmup_steps=math.ceil(max_steps_to_schedule * args.lr_warmup_ratio),
            num_training_steps=max_steps_to_schedule,
            scheduler_specific_kwargs=scheduler_specific_kwargs,
        )
        strategy.print(
            f"num_policy_sgd_steps_per_episodes={num_policy_sgd_steps_per_episodes}; max_steps={max_steps_to_schedule}"
        )

        # prepare collector, which communicates with actors
        if actors:
            if self.args.asynchronous:
                self.collector = AsyncFeedbackCollector(
                    args, actors, PlasmaShmClient(self.ipc_server)
                )
            else:
                self.collector = FeedbackCollector(
                    args, actors, PlasmaShmClient(self.ipc_server)
                )
        else:
            strategy.print("No actors or feedback collector in offline mode.")

        exp_name = args.wb_run_name + "_" + datetime.now().strftime("%m%dT%H:%M:%S")
        self.save_path = os.path.join(args.save_path, exp_name)
        if strategy.is_rank_0():
            os.makedirs(self.save_path, exist_ok=True)

        # logger
        self._wandb = None
        if strategy.args.use_wb and strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wb)
            wandb.init(
                entity=args.wb_org,
                project=args.wb_project,
                group=args.wb_group,
                name=exp_name,
                config=args.__dict__,
                reinit=True,
            )

        self.algo = args.algo
        self.strategy = strategy
        self.update_interval = args.rollout_batch_size // (
            strategy.world_size * args.rollout_batch_size_per_device
        )

        self.global_step = 0
        self.pi_beta_version = 0
        self.policy_sgd_step = 0
        self.query_step = 0
        self.prompt_consumed = 0
        self.prompt_epoch = 0
        self.gradient_update_elapse = np.nan

        # Log summary of the learner
        strategy.print(self.model)
        strategy.print(self.optimizer)
        strategy.print(self.scheduler)
        strategy.pprint(vars(args))
        strategy.print(f"Update interval = {self.update_interval}")

        # prepare parameter syncing to actors (reference to openrlhf)
        #
        # For ZeRO-1/2:
        #   1. Broadcast parameters from rank 0 to all vllm engines
        # For ZeRO-3:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all vllm engines
        if actors and strategy.is_rank_0():
            master_addr = node_ip_address_from_perspective()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            world_size = len(actors) + 1
            backend = "nccl"
            if vllm.__version__ > "0.4.2":
                backend = "gloo"
                warn(f"Using gloo backend for vLLM version {vllm.__version__}")
            futs = [
                actor.futures.init_process_group(
                    master_addr,
                    master_port,
                    i + 1,
                    world_size,
                    "oat",
                    backend=backend,
                )
                for i, actor in enumerate(actors)
            ]
            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="oat",
            )
            _ = [fut.result() for fut in futs]

        dist.barrier()

    def prepare_data(self, strategy, tokenizer):
        self.prompts_dataset, self.eval_prompts_dataset = get_datasets(
            tokenizer, strategy
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
        )
        self.eval_prompts_dataloader = DataLoader(
            self.eval_prompts_dataset,
            batch_size=strategy.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def run(self):
        self._init(self.args, self.actors)

        self.steps = 0
        early_stop = False
        self.start_time = time.time()

        self.actor_info = {}

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True, save=False)

        self.steps = 1
        self.gradient_update_st = time.time()
        for p_ep in range(self.args.num_prompt_epoch):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(p_ep)
                self.strategy.print(f"Set DistributedSampler at epoch {p_ep}")
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
            )

            for processed_prompts, raw_prompts, refs in self.prompts_dataloader:
                if early_stop:
                    break
                feedback_data, self.actor_info = self.collector.collect_feedback(
                    raw_prompts, processed_prompts, refs
                )
                del raw_prompts, processed_prompts

                if feedback_data is None:
                    # Asynchronous prefilling, data is stored in collector's buffer.
                    continue
                self.prompt_consumed += len(feedback_data)

                self.process_feedback_data(feedback_data)

                if self.steps % self.update_interval == 0:
                    train_info = self.learn(self.steps // self.update_interval)

                    self.eval_and_log(train_info)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        self.sync_params_to_actors()

                    if (
                        self.steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                progress_bar.update()
                self.steps += 1

                if self.get_current_query() > self.args.max_queries:
                    early_stop = True

            self.prompt_epoch = p_ep + 1

        self.eval_and_log(train_info, eval=True, save=True)

        if self.args.dump_all_buffer:  # For debug purpose.
            if not self.strategy.is_rank_0():
                dist.gather_object(self.all_buffer)
            else:
                gather_all_buffer = [None] * self.strategy.world_size
                dist.gather_object(self.all_buffer, gather_all_buffer)
                pd.to_pickle(
                    gather_all_buffer, os.path.join(self.save_path, "all_buffer.pkl")
                )

        if self.strategy.is_rank_0():
            self._wandb.finish() if self._wandb else None
            lp.stop()

    @abc.abstractmethod
    def process_feedback_data(
        self, data_list: List[Union[PreferenceData, TrajectoryData]]
    ):
        """Process collected feedback data, e.g., adding it to buffer."""

    @abc.abstractmethod
    def learn(self, learning_round: int):
        """Agent learning given the current data in the buffer."""

    @abc.abstractmethod
    def learning_step(self, data):
        """Agent learning step."""

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
        prompt_id_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            all_logp: all log prob of shape (batch_size, sequence_length, vocab_size)
            target_logps: target log prob of shape (batch_size, sequence_length)
            completion_masks: mask=True if it is completion's token, shape (batch_size, sequence_length)
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        completion_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(completion_masks, prompt_id_lens):
            mask[:source_len] = False
        completion_masks = completion_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[completion_masks == False] = 0

        all_logp = logits.log_softmax(-1)
        target_logps = torch.gather(all_logp, dim=2, index=labels.unsqueeze(2)).squeeze(
            2
        )

        return all_logp, target_logps, completion_masks

    def get_misc_info(self) -> Dict[str, Any]:
        return {
            "pi_beta_version": self.pi_beta_version,
            "global_step": self.global_step,
            "policy_sgd_step": self.policy_sgd_step,
            "pi_buffer_len": len(self.pi_buffer),
            "elapse": time.time() - self.start_time,
            "update_interval": self.update_interval,
            "prompt_epoch": self.prompt_epoch,
            "gradient_update_elapse": self.gradient_update_elapse,
        }

    def get_current_query(self):
        return self.strategy.all_reduce(self.query_step, op="sum")

    def _should_do(self, interval_steps):
        if interval_steps <= 0:
            return False
        if not hasattr(self, "_pending_eval"):
            self._pending_eval = False

        do_eval = self.steps % interval_steps == 0
        if not (do_eval or self._pending_eval):
            return False
        else:
            if do_eval and not hasattr(self, "last_eval_query_step"):
                self.last_eval_query_step = self.get_current_query()
                return True
            query_step_elapse = self.get_current_query() - self.last_eval_query_step
            if query_step_elapse < self.args.eval_query_interval:
                self._pending_eval = True
                return False
            self._pending_eval = False
            self.last_eval_query_step = self.get_current_query()
            return True

    def eval_and_log(self, train_info, eval=False, save=False):
        # eval
        eval_info = {}
        if (self.args.eval_steps > 0 and eval) or self._should_do(self.args.eval_steps):
            eval_info = self.evaluate(self.eval_prompts_dataloader, self.steps)

        # save
        if (self.args.save_steps > 0 and save) or (
            self.steps > 0 and self._should_do(self.args.save_steps)
        ):
            self.strategy.save_model(
                self.model,
                self.tokenizer,
                os.path.join(self.save_path, "saved_models"),
                tag="step_{:05d}".format(self.steps),
                max_num=self.args.max_save_num,
                max_mem=self.args.max_save_mem,
            )

        # logs
        if eval_info or self.steps % self.args.logging_steps == 0:
            misc_info = self.get_misc_info()
            last_lr = self.scheduler.get_last_lr()[0]
            misc_info["lr"] = last_lr

            misc_info = {
                "misc/%s" % k: v
                for k, v in {
                    **misc_info,
                }.items()
            }
            logs_dict = {**train_info, **eval_info, **self.actor_info, **misc_info}
            logs_dict = self.strategy.all_reduce(logs_dict)
            logs_dict.update(
                self.strategy.all_reduce(
                    {
                        "misc/query_step": self.query_step,
                        "misc/prompt_consumed": self.prompt_consumed,
                    },
                    op="sum",
                )
            )

            if self.strategy.is_rank_0():
                if self.pi_buffer:
                    self.strategy.print(np.random.choice(self.pi_buffer))
                self.strategy.pprint(logs_dict)
                if self._wandb is not None:
                    self._wandb.log(logs_dict)

    def evaluate(self, dataloader, steps):
        self.strategy.print(f"Start generating evaluation responses at step {steps}")
        st_time = time.time()
        # 1) Let Actors cache the current behavior policy.
        if self.strategy.is_rank_0():
            done = [actor.futures.notify_eval_start() for actor in self.actors]
            _ = [d.result() for d in done]

        # 2) Push the latest policy for fast vLLM generation.
        dist.barrier()
        self._broadcast_to_vllm()

        # 3) Generate and process results
        win_rate = 0
        win_rate_prob = 0
        response_len = 0
        if self.strategy.is_rank_0():
            processed_prompts = []
            prompts = []
            responses = []
            response_lens = []
            references = []
            futs = []
            win_probs = []
            wins = []
            progress_bar = tqdm(range(len(dataloader)), desc="Evaluating")
            for i, (batch_processed_prompts, batch_prompts, refs) in enumerate(
                dataloader
            ):
                processed_prompts.extend(batch_processed_prompts)
                prompts.extend(batch_prompts)
                references.extend(refs)

                actor = self.actors[i % len(self.actors)]
                fut = actor.futures.generate_and_maybe_eval(
                    batch_prompts, batch_processed_prompts, refs
                )
                futs.append(fut)
                if len(futs) == len(self.actors) or i == len(dataloader) - 1:
                    for fut in futs:
                        resp, win_prob = fut.result()
                        responses.extend(resp)
                        wins.extend(win_prob > 0.5)
                        win_probs.extend(win_prob)
                    futs.clear()
                progress_bar.update()

            eval_res_path = os.path.join(self.save_path, "eval_results")
            os.makedirs(eval_res_path, exist_ok=True)
            pd.DataFrame(
                {
                    self.eval_input_key: prompts,
                    "output": responses,
                    f"format_{self.eval_input_key}": processed_prompts,
                    "reference": references,
                    "generator": self.args.wb_run_name,
                }
            ).to_json(
                os.path.join(eval_res_path, f"{steps}.json"),
                orient="records",
                indent=4,
            )
            win_rate = np.mean(wins).item()
            win_rate_prob = np.mean(win_probs).item()
            response_len = np.mean(tree.map_structure(lambda x: len(x), responses))

        win_rate = self.strategy.broadcast(win_rate)
        win_rate_prob = self.strategy.broadcast(win_rate_prob)
        response_len = self.strategy.broadcast(response_len)

        # 4) Recover Actors' original behavior policy.
        if self.strategy.is_rank_0():
            done = [actor.futures.notify_eval_done() for actor in self.actors]
            _ = [d.result() for d in done]

        return {
            "eval/rm_win_rate": win_rate,
            "eval/rm_win_rate_prob": win_rate_prob,
            "eval/elapse": time.time() - st_time,
            "eval/response_str_len": response_len,
        }

    def sync_params_to_actors(self):
        self._broadcast_to_vllm()
        self.pi_beta_version += 1

    def _broadcast_to_vllm(self):
        if self.args.asynchronous:
            # Pooling util generation finishes.
            while True:
                time.sleep(0.1)
                actors_busy = [actor.is_generating() for actor in self.actors]
                if not any(actors_busy):
                    break
        model = self.model.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # Fire all vllm engines for broadcast
            if self.strategy.is_rank_0():
                shape = (
                    param.shape
                    if self.strategy.args.zero_stage != 3
                    else param.ds_shape
                )
                futs = [
                    actor.futures.update_weight(
                        name,
                        dtype=torch_type_codec(param.dtype),
                        shape=shape,
                        empty_cache=count == num_params,
                    )
                    for actor in self.actors
                ]

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters(
                [param], enabled=self.strategy.args.zero_stage == 3
            ):
                if self.strategy.is_rank_0():
                    dist.broadcast(param.data, 0, group=self._model_update_group)
                    _ = [fut.result() for fut in futs]

```

```python
# ./experiment/__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

```python
# ./experiment/run_offline_lp.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline alignment with online vLLM evaluation."""

from dataclasses import dataclass

import launchpad as lp

from oat.actors import PreferenceActor, RewardActor
from oat.args import OATArgs, default_args_validation, get_default_args
from oat.interface import get_program
from oat.learners import OfflineDAPLearner, OfflineSFTLearner
from oat.types import DAPAlgo


@dataclass
class OfflineArgs(OATArgs):
    """Offline DAP from a preference dataset arguments."""

    preference_data: str = ""
    extract_content: bool = False
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    offline_buffer_path: str = "./data/buffer.pkl"


def main(args: OATArgs):
    learner_cls = OfflineDAPLearner if args.algo in DAPAlgo else OfflineSFTLearner
    actor_cls = PreferenceActor if args.oracle_type == "preference" else RewardActor
    program, local_resources = get_program(args, learner_cls, actor_cls)
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args = get_default_args(OfflineArgs)
    args = default_args_validation(args)
    main(args)

```

```python
# ./experiment/run_offline.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline alignment from fixed dataset."""

from dataclasses import dataclass

from oat.args import OATArgs, default_args_validation, get_default_args
from oat.learners import OfflineDAPLearner, OfflineSFTLearner
from oat.types import DAPAlgo


@dataclass
class OfflineArgs(OATArgs):
    """Offline DAP from a preference dataset arguments."""

    preference_data: str = "HuggingFaceH4/ultrafeedback_binarized"
    extract_content: bool = (
        True  # Enable when chosen / reject key contains conversation-style data
    )
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"
    eval_steps: int = -1


def main(args):
    cls = OfflineDAPLearner if args.algo in DAPAlgo else OfflineSFTLearner

    def __init__(self, args):
        # Hack to discard DistributedLauncher and use deepspeed launcher.
        self.args = args
        self.actors = []
        self.ipc_server = None

    cls.__init__ = __init__
    learner = cls(args=args)
    learner.run()


if __name__ == "__main__":
    args = get_default_args(OfflineArgs)
    args = default_args_validation(args)
    main(args)

```

```python
# ./experiment/main.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import launchpad as lp

from oat.actors.preference import PreferenceActor
from oat.args import OATArgs, default_args_validation, get_default_args
from oat.interface import get_program
from oat.learners import DAPLearner, DAPwRMLearner


def main(args: OATArgs):
    if args.learn_rm:
        learner_cls = DAPwRMLearner
    else:
        learner_cls = DAPLearner
    program, local_resources = get_program(
        args, learner_cls=learner_cls, actor_cls=PreferenceActor
    )
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args = get_default_args()
    args = default_args_validation(args)
    main(args)

```

```python
# ./experiment/run_offline_ppo.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from oat.algorithms.ppo import OfflinePPOLearner, PPOArgs
from oat.args import default_args_validation, get_default_args


def run_ppo(args):
    cls = OfflinePPOLearner

    def __init__(self, args):
        # Hack to discard DistributedLauncher and use deepspeed launcher.
        self.args = args
        self.actors = []
        self.ipc_server = None

    cls.__init__ = __init__
    learner = cls(args=args)
    learner.run()


if __name__ == "__main__":
    args = get_default_args(PPOArgs)

    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.
    args.reward_key = "final_reward"  # Debugging purpose.

    args = default_args_validation(args)
    run_ppo(args)

```

```python
# ./experiment/run_xpo.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import launchpad as lp

from oat.algorithms.xpo import XPOActor, XPOArgs, XPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program


def run_xpo(args):
    program, local_resources = get_program(args, XPOLearner, XPOActor)
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args = get_default_args(XPOArgs)
    args = default_args_validation(args)
    run_xpo(args)

```

```python
# ./experiment/run_ppo.py
# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import launchpad as lp

from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program


def run_ppo(args: PPOArgs):
    learner_cls = PPOLearner
    actor_cls = PPOActor
    program, local_resources = get_program(args, learner_cls, actor_cls)
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args: PPOArgs = get_default_args(PPOArgs)

    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.

    args = default_args_validation(args)
    run_ppo(args)

```

```python
# ./experiment/run_apl.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import launchpad as lp

from oat.algorithms.apl import APLActor, APLArgs, APLLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program


def run_apl(args):
    program, local_resources = get_program(args, APLLearner, APLActor)
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args = get_default_args(APLArgs)
    args = default_args_validation(args)
    if args.apl_pref_certainty_only:
        args.num_samples = 2
    run_apl(args)

```

```python
# ./algorithms/rft.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import time
from typing import List

import pandas as pd
import tree

from oat.actors import PreferenceActor
from oat.args import OATArgs
from oat.learners import SFTLearner
from oat.types import PreferenceData


class RESTLearner(SFTLearner):
    """Simply SFT."""


class RESTActor(PreferenceActor):
    """Inherit PreferenceActor but we only make use of `chosen`."""

    def __init__(self, ipc_server, vllm_args, args: OATArgs) -> None:
        assert args.oracle in ["gsm8k"], f"Oracle {args.oracle} is not supported"
        super().__init__(ipc_server, vllm_args, args)

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[PreferenceData]:
        assert not self.eval_mode
        info = {}

        # step 1. generate
        st = time.time()
        all_candidates = self.generate(formatted_prompts, self.sampling_params)
        info["actor/generate_time"] = time.time() - st

        flatten_prompts = list(
            itertools.chain.from_iterable(
                itertools.repeat(x, self.sampling_params.n) for x in prompts
            )
        )
        flatten_responses = tree.flatten(all_candidates)
        flatten_references = list(
            itertools.chain.from_iterable(
                itertools.repeat(x, self.sampling_params.n) for x in references
            )
        )

        # step 2. verify
        flatten_rewards, oracle_info = self.oracle.get_reward(
            flatten_prompts,
            flatten_responses,
            flatten_references,
        )
        rewards = flatten_rewards.reshape(len(prompts), self.sampling_params.n)

        info["actor/pass@k"] = (rewards.sum(-1) > 0).float().mean().item()
        info["actor/rewards_mean"] = rewards.mean().item()
        info["actor/oracle_time"] = time.time() - st

        # step 3. filter
        trajectories = {"prompt": [], "chosen": []}
        for x, y, is_correct in zip(
            flatten_prompts, flatten_responses, flatten_rewards
        ):
            if is_correct == 1:
                trajectories["prompt"].append(x)
                trajectories["chosen"].append(y)

        # Drop duplicates based on exact match.
        # Fancier methods (e.g., similarity-based) can be done here.
        trajectories = (
            pd.DataFrame(trajectories).drop_duplicates(ignore_index=True).to_dict()
        )
        unique_count = len(trajectories["prompt"])

        info["actor/unique_trajectories"] = unique_count

        filtered_data = []
        for i in range(unique_count):
            filtered_data.append(
                PreferenceData(
                    prompt=trajectories["prompt"][i],
                    chosen_response=trajectories["chosen"][i],
                    rejected_response="",  # Nothing.
                    info=info,
                )
            )

        handle = self.ipc_client.serialize_ipc(filtered_data)
        return handle

```

```python
# ./algorithms/__init__.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

```python
# ./algorithms/ppo.py
# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal Policy Optimization."""

import gc
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import tree
import vllm
from torch.utils.data import DataLoader
from tqdm import tqdm

from oat.actors import RewardActor
from oat.actors.base import ActorBase
from oat.args import OATArgs
from oat.learners import OfflineLearner, RLLearner
from oat.types import TrajectoryData
from oat.utils.data import (
    TrajectoryDataset,
    get_datasets,
    load_data_from_disk_or_hf,
    shard_buffer,
)
from oat.utils.ops import masked_mean, masked_whiten

"""PPO (https://arxiv.org/abs/1707.06347) with additional KL regularization."""


@dataclass
class PPOArgs(OATArgs):
    num_ppo_epochs: int = field(
        default=2,
        metadata={"help": "Number of epochs to train."},
    )
    mini_train_batch_size_per_device: int = field(
        default=1,
        metadata={"help": "Mini batch size."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards."},
    )
    kl_penalty_coef: float = field(
        default=0,
        metadata={"help": "KL coefficient for pseudo rewards."},
    )
    non_stop_penalty: float = field(
        default=0,
        metadata={"help": "Penalty for responses not containing eos."},
    )
    reward_scale: float = field(
        default=1.0,
        metadata={"help": "Scaling the environment rewards."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    vf_coef: float = field(
        default=1.0,
        metadata={"help": "Value function coefficient."},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Clip range for the value function."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor."},
    )
    lam: float = field(
        default=1.0,
        metadata={"help": "Lambda value for GAE."},
    )


class PPOActor(RewardActor):
    def __init__(self, ipc_server, vllm_args, args: PPOArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            stop=["\n\nQuestion", "\n\nProblem"],
            n=args.num_samples,
            logprobs=2,
        )
        self.eval_sampling_params = vllm.SamplingParams(
            n=args.eval_n,
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
            stop=["\n\nQuestion", "\n\nProblem"],
        )

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[TrajectoryData]:
        assert not self.eval_mode
        info = {}

        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)

        candidates = []
        prompt_token_ids = []
        no_eos = []
        response_ids = []
        response_logprobs = []
        for i in range(len(outputs)):
            # for each prompt
            prompt_token_ids.append(outputs[i].prompt_token_ids)
            candidates.append([])
            response_logprobs.append([])
            response_ids.append([])
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)
                no_eos.append(
                    self.tokenizer.eos_token_id not in outputs[i].outputs[k].token_ids
                )
                token_ids = outputs[i].outputs[k].token_ids
                logps = outputs[i].outputs[k].logprobs
                logps = [item[token_ids[i]].logprob for i, item in enumerate(logps)]

                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                # print(outputs[i].outputs[k].text)
                # if no_eos[-1]:
                #     print(outputs[i].outputs[k].token_ids)
                #     print(outputs[i].outputs[k].text)

        info["actor/generate_time"] = time.time() - st

        # step 2. verify
        rewards, _ = self.oracle.get_reward(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in prompts
                )
            ),
            tree.flatten(candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in references
                )
            ),
        )
        rewards = rewards.reshape(len(prompts), -1)
        no_eos = np.array(no_eos).reshape(len(prompts), -1)

        info["actor/minibatch_accuracy"] = rewards.mean()
        info["actor/no_eos_count"] = no_eos.sum()
        info["actor/num_data"] = rewards.numel()

        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                reward = rewards[i][j]
                reward += self.args.non_stop_penalty if no_eos[i][j] else 0
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                trajectory_data.append(
                    TrajectoryData(
                        prompt=prompt,
                        prompt_ids=prompt_token_ids[i],
                        response=candidates_per_prompt[j],
                        response_ids=response_ids[i][j],
                        response_logprobs=response_logprobs[i][j],
                        rewards=dense_rewards,
                        info=info,
                    )
                )
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle


class PPOLearner(RLLearner):
    def _init(self, args: OATArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.dataset_builder = TrajectoryDataset

    def learn(self, learning_round: int):
        torch.cuda.empty_cache()
        dist.barrier()
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.strategy,
        )
        if learning_round == 1:
            self.strategy.print("Training example")
            self.strategy.print(dataset[0])

        # Load all buffered data, and PPO will iterate through inner loops.
        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        local_sgd_steps = 0
        step_bar = tqdm(
            range(len(dataloader)),
            desc="Train steps",
            disable=not self.strategy.is_rank_0(),
        )
        learn_batch_time = []

        self.model.train()
        if self.critic is not None:
            self.critic.train()
        st = time.time()
        for data in dataloader:
            if local_sgd_steps > self.args.max_sgd_steps:
                break
            infos = self.learning_step(data)
            self.policy_sgd_step += (
                len(dataset)
                * self.args.num_ppo_epochs
                / self.args.train_batch_size_per_device
                / self.strategy.accumulated_gradient
            )
            learn_batch_time.append(time.time() - st)
            step_bar.update()

            self.global_step += 1
            if self.global_step % self.strategy.accumulated_gradient == 0:

                self.gradient_update_elapse = time.time() - self.gradient_update_st
                st = time.time()
                self.gradient_update_st = time.time()

                local_sgd_steps += 1

        torch.cuda.empty_cache()
        dist.barrier()

        train_info = {
            "learning_round": learning_round,
            "learn_batch_time": np.mean(learn_batch_time),
            **tree.map_structure(lambda x: x.cpu().float().mean().item(), infos),
        }
        train_info = {
            "train/%s" % k: v
            for k, v in {
                **train_info,
            }.items()
        }
        return train_info

    def compute_ppo_advantages(
        self, rewards, input_ids, att_mask, response_masks, batch_inds
    ):
        all_values = []

        with torch.no_grad():
            for i in range(
                0, len(input_ids), self.args.mini_train_batch_size_per_device
            ):
                ## Forward critic network.
                batch_values = self.critic(
                    input_ids=input_ids[batch_inds], attention_mask=att_mask[batch_inds]
                )
                batch_value_masks = att_mask[batch_inds].clone()[:, 1:]
                batch_value_masks = torch.concat(
                    [
                        batch_value_masks,
                        torch.zeros(len(batch_value_masks), 1, device=att_mask.device),
                    ],
                    axis=1,
                )
                batch_values = (batch_values * batch_value_masks)[:, :-1]
                all_values.append(batch_values)
        values = torch.cat(all_values)

        # Compute gae (for policy learning) and return (for critic learning); vectorize later.
        advantages = torch.zeros_like(rewards)
        for i in range(len(advantages)):
            action_inds = torch.where(response_masks[i])[0]
            lastgaelam = 0
            for t in reversed(action_inds):
                nextvalues = values[i, t + 1] if t < action_inds[-1] else 0.0
                delta = rewards[i, t] + self.args.gamma * nextvalues - values[i, t]
                lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
                advantages[i, t] = lastgaelam

        returns = advantages + values
        advantages = masked_whiten(advantages, response_masks)

        return advantages, returns, values

    def compute_grpo_advantages(self, rewards, response_masks):
        rewards = rewards.sum(-1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.args.num_samples).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.args.num_samples, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.args.num_samples, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = masked_whiten(advantages.unsqueeze(-1), response_masks)
        return advantages

    def compute_sil_advantages(self, rewards):
        pass

    def learning_step(self, trajectory):
        args: PPOArgs = self.args
        infos = {}
        device = torch.cuda.current_device()
        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        final_rewards = (
            torch.tensor([r[-1] for r in trajectory["rewards"]])
            .to(device)
            .reshape(-1, 1)
        ).float() * args.reward_scale
        prompt_id_lens = trajectory["prompt_ids_lens"]
        action_logprobs = [
            torch.tensor(lp).to(device) for lp in trajectory["action_logprobs"]
        ]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]

        # self.strategy.print(f"learn data size {input_ids.shape}")

        indices = torch.arange(
            response_masks.size(1), device=response_masks.device
        ).expand_as(response_masks)
        masked_indices = torch.where(
            response_masks, indices, torch.full_like(indices, -1)
        )
        eos_indices = masked_indices.max(dim=1).values

        # Forward old models.
        all_ref_logps = []
        with torch.no_grad():
            for i in range(0, len(input_ids), args.mini_train_batch_size_per_device):
                batch_inds = torch.arange(i, i + args.mini_train_batch_size_per_device)
                ## 1) Policy log probabilities are directly from actors.
                ## 2) Reference.
                batch_ref_logits = self.ref_model(
                    input_ids[batch_inds], attention_mask=att_mask[batch_inds]
                )["logits"].float()
                batch_ref_logits /= args.temperature
                batch_ref_logps = self.get_batch_logps(
                    batch_ref_logits,
                    input_ids[batch_inds],
                    response_masks[batch_inds],
                )
                all_ref_logps.append(batch_ref_logps)

        ref_logps = torch.cat(all_ref_logps)
        logps = torch.zeros_like(ref_logps)
        for i in range(len(logps)):
            logps[i, torch.where(response_masks[i])[0]] = action_logprobs[i]

        # Combine final reward and kl penalty as rewards.
        kl_rewards = -args.kl_penalty_coef * (logps - ref_logps) * response_masks
        rewards = kl_rewards.clone()
        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()

        if self.args.critic_type == "ppo":
            advantages, returns, values = self.compute_ppo_advantages(
                rewards, input_ids, att_mask, response_masks, batch_inds
            )
        elif self.args.critic_type == "grpo":
            advantages = self.compute_grpo_advantages(rewards, response_masks)
        elif self.args.critic_type == "sil":
            advantages = self.compute_sil_advantages(rewards, response_masks)

        del all_ref_logps
        torch.cuda.empty_cache()
        gc.collect()

        # Compute losses and update models for multiple PPO epochs.
        stats = defaultdict(list)
        for _ in range(args.num_ppo_epochs):
            batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(input_ids), args.mini_train_batch_size_per_device):
                mini_batch_inds = batch_inds[
                    b_st : b_st + args.mini_train_batch_size_per_device
                ]
                mb_advantage = advantages[mini_batch_inds]
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_ref_logps = ref_logps[mini_batch_inds]
                mb_loss_masks = loss_masks[mini_batch_inds]

                if self.args.critic_type == "ppo":
                    mb_return = returns[mini_batch_inds]
                    mb_values = values[mini_batch_inds]

                # Policy learning.
                logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ].float()
                logits /= args.temperature
                new_logps = self.get_batch_logps(
                    logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                logprobs_diff = new_logps - mb_logps
                ratio = torch.exp(logprobs_diff)

                self.strategy.print(mb_advantage.shape, ratio.shape)

                pg_losses = -mb_advantage * ratio
                pg_losses2 = -mb_advantage * torch.clamp(
                    ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                )
                pg_loss_max = torch.max(pg_losses, pg_losses2)

                stats["ratio_max"].append(ratio.detach().max().item())
                stats["ratio_min"].append(ratio.detach().min().item())

                pg_loss = masked_mean(pg_loss_max, mb_response_masks, axis=1)
                pg_loss = (pg_loss * mb_loss_masks).mean()
                loss = pg_loss
                if args.beta > 0:
                    # k3 kl: http://joschu.net/blog/kl-approx.html.
                    log_ratio = mb_ref_logps - new_logps
                    kl = torch.exp(log_ratio) - log_ratio - 1
                    kl = torch.clamp(
                        kl * mb_response_masks,
                        min=0,
                        max=10,
                    )
                    reg_loss = args.beta * kl.sum(dim=1)
                    reg_loss = (reg_loss * mb_loss_masks).mean()
                    loss += reg_loss
                    infos["reg_loss"] = reg_loss.detach()

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                infos["pg_loss"] = pg_loss.detach()

                if self.args.critic_type == "ppo":
                    # torch.cuda.empty_cache()
                    # gc.collect()

                    # Critic learning.
                    value_pred = self.critic(
                        input_ids=mb_input_ids, attention_mask=mb_att_mask
                    )[:, :-1]

                    value_pred_clipped = torch.clamp(
                        value_pred,
                        mb_values - args.cliprange_value,
                        mb_values + args.cliprange_value,
                    )
                    vf_losses1 = torch.square(value_pred - mb_return)
                    vf_losses2 = torch.square(value_pred_clipped - mb_return)
                    vf_loss_max = torch.max(vf_losses1, vf_losses2)
                    vf_loss = 0.5 * masked_mean(vf_loss_max, mb_response_masks, axis=1)
                    critic_loss = args.vf_coef * (vf_loss * mb_loss_masks).mean()

                    self.strategy.backward(
                        critic_loss, self.critic, self.critic_optimizer
                    )
                    self.strategy.optimizer_step(
                        self.critic_optimizer, self.critic, self.critic_scheduler
                    )
                    infos["critic_loss"] = critic_loss.detach()
                    infos["vf_clipfrac"] = masked_mean(
                        (vf_losses2 > vf_losses1).float(), mb_response_masks
                    ).detach()

        infos.update(
            {f"{k}_nan": torch.tensor(stats[k]).isnan().sum() for k in stats.keys()}
        )
        infos.update(
            {f"{k}_inf": torch.tensor(stats[k]).isinf().sum() for k in stats.keys()}
        )
        infos["ratio_max"] = torch.tensor(stats["ratio_max"]).max()
        infos["ratio_min"] = torch.tensor(stats["ratio_min"]).min()

        return infos

    def get_completion_mask(
        self,
        attention_mask: torch.LongTensor,
        prompt_id_lens: List[int],
    ):
        completion_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(completion_masks, prompt_id_lens):
            mask[:source_len] = False
        completion_masks = completion_masks
        return completion_masks

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        completion_masks: torch.LongTensor,
    ) -> torch.Tuple[torch.Tensor]:
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        # dummy token; we'll ignore the losses on these tokens later
        labels[completion_masks == False] = 0

        all_logp = logits.log_softmax(-1)
        target_logps = torch.gather(all_logp, dim=2, index=labels.unsqueeze(2)).squeeze(
            2
        )

        return target_logps


class OfflinePPOLearner(OfflineLearner, PPOLearner):
    def prepare_data(self, strategy, tokenizer):
        """Construct offline RL dataset."""
        args: PPOArgs = self.args
        data = load_data_from_disk_or_hf(args.prompt_data)[args.train_split]
        all_shards = []
        for item in tqdm(data, desc="loading data", disable=not strategy.is_rank_0()):
            all_shards.append(
                TrajectoryData(
                    prompt=item[args.input_key],
                    responses=[item[args.output_key]],  # accept a list
                    rewards=[[item[args.reward_key]]],  # accept a list
                    info={},
                )
            )

        self.all_buffer: List[TrajectoryData] = shard_buffer(
            all_shards,
            dist.get_rank(),
            dist.get_world_size(),
            args.seed,
            shuffle=True,
            drop_last=True,
        )
        self.prompts_dataset = tree.flatten(
            all_shards
        )  # needed to calculate lr scheduler
        self.prompts_dataloader = None
        if args.eval_steps > 0:
            _, self.eval_prompts_dataset = get_datasets(
                tokenizer, strategy, eval_only=True
            )
            self.eval_prompts_dataloader = DataLoader(
                self.eval_prompts_dataset,
                batch_size=strategy.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )

```

```python
# ./algorithms/xpo.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XPO: https://arxiv.org/pdf/2405.21046."""

from dataclasses import dataclass

import torch
import vllm

from oat.actors import PreferenceActor
from oat.args import OATArgs
from oat.learners.dap import DAPLearner
from oat.types import DAPAlgo


@dataclass
class XPOArgs(OATArgs):
    """Exploratory preference optimization arguments."""

    xpo_alpha: float = 5e-6
    xpo_offload_actor_ref: bool = False


class XPOActor(PreferenceActor):
    """Sample one response from llm and another from ref_llm."""

    def __init__(self, ipc_server, vllm_args, args: XPOArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.sampling_params.n = 1  # one for each llm
        self.offload_ref_model = args.xpo_offload_actor_ref

        if not self.offload_ref_model:
            self.ref_llm = vllm.LLM(**vllm_args)
        else:
            self.ref_llm = None
            self.cache_ref_model_state = {
                k: v.cpu() for k, v in self.model.named_parameters()
            }

    def generate(self, prompts: base.List[str], sampling_params: vllm.SamplingParams):
        if self.eval_mode:
            return super().generate(prompts, sampling_params)

        assert sampling_params.n == 1
        candidates = {}

        for llm in [self.llm, self.ref_llm]:
            if llm is not None:
                outputs = llm.generate(
                    prompts, sampling_params=sampling_params, use_tqdm=False
                )
            else:
                # Cache current llm's weights, load ref_llm for infer and restore
                # original llm's weights.
                self.notify_eval_start(eval=False)
                self.model.load_state_dict(self.cache_ref_model_state)
                outputs = self.llm.generate(
                    prompts, sampling_params=sampling_params, use_tqdm=False
                )
                self.notify_eval_done(eval=False)
            for i in range(len(outputs)):
                # for each prompt
                if i not in candidates:
                    candidates[i] = []
                candidates[i].append(outputs[i].outputs[0].text.strip())

        return candidates


class XPOLearner(DAPLearner):
    """Additional optimism loss term: log(\pi(y_ref|x))."""

    def _init(self, args: XPOArgs, actors) -> None:
        super()._init(args, actors)
        assert self.algo == DAPAlgo.DPO and self.ref_model is not None
        self.xpo_alpha = args.xpo_alpha

    def learning_step(self, data):
        device = torch.cuda.current_device()
        chosen_ids, c_mask, rejected_ids, r_mask, extra = data
        chosen_ids = chosen_ids.squeeze(1).to(device)
        c_mask = c_mask.squeeze(1).to(device)
        rejected_ids = rejected_ids.squeeze(1).to(device)
        r_mask = r_mask.squeeze(1).to(device)

        prompt_id_lens = extra["prompt_ids_lens"]
        loss_masks = 1 - torch.tensor(extra["same_masks"]).float().to(device)

        chosen_logps, rejected_logps, _ = self.concatenated_forward(
            self.model, chosen_ids, c_mask, rejected_ids, r_mask, prompt_id_lens
        )
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, _ = (
                self.concatenated_forward(
                    self.ref_model,
                    chosen_ids,
                    c_mask,
                    rejected_ids,
                    r_mask,
                    prompt_id_lens,
                )
            )
        preference_loss, chosen_reward, rejected_reward = self.loss(
            chosen_logps,
            rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            loss_masks,
        )

        # `chosen` indicates the original sampling source:
        # 0 - rejected_ids are from the ref policy
        # 1 - chosen_ids are from the ref policy
        chosen = torch.tensor(extra["chosen_ids"]).to(device)
        ref_logps = torch.where(chosen == 0, rejected_logps, chosen_logps)
        optimism_loss = (ref_logps * loss_masks).mean()

        loss = preference_loss + self.xpo_alpha * optimism_loss
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

        infos = {
            "loss": loss.detach(),
            "optimism_loss": optimism_loss.detach(),
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
        }
        return infos

```

```python
# ./algorithms/apl.py
# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""APL: https://arxiv.org/pdf/2402.08114.

Due to its design of using LLM as the reward model, we have to make the actor-
learner interface more complicated. We first generate responses and estimate
the entropy in actor, then compute the implicit reward margin in learner, and
finally get oracle feedback in actor.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import launchpad as lp
import Levenshtein
import numpy as np
import torch
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from vllm.outputs import RequestOutput

from oat.actors import PreferenceActor
from oat.args import OATArgs
from oat.learners.dap import DAPLearner
from oat.model import LLM
from oat.types import Metric, PreferenceData
from oat.utils.data import zero_pad_sequences
from oat.utils.ipc import DataID, PlasmaShmClient


@dataclass
class APLArgs(OATArgs):
    """Active preference learning arguments."""

    # Fig 2b and Fig 5 both show this variant is better than random,
    # while Fig 2b shows the learning is not robust with entropy.
    apl_pref_certainty_only: bool = False


class APLActor(PreferenceActor):
    """Sample a large batch and filter with entropy and reward margin."""

    def __init__(self, ipc_server, vllm_args, args: APLArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.sampling_params.logprobs = 1

    def generate_and_entropy_filter(self, prompts: List[str]) -> DataID:
        assert not self.eval_mode
        # Generate.
        outputs = self.llm.generate(
            prompts, sampling_params=self.sampling_params, use_tqdm=False
        )

        ent_filtered_indices = None
        if not self.args.apl_pref_certainty_only:
            # Predictive entropy estimation.
            entropy_estimations = []
            for output in outputs:
                entropy = 0
                for resp_output in output.outputs:
                    entropy += resp_output.cumulative_logprob
                entropy /= len(output.outputs)
                entropy_estimations.append(entropy)
            ent_filtered_indices = np.argsort(entropy_estimations)[
                -self.args.pi_buffer_maxlen_per_device :
            ]  # Online and on-policy; as stated in their Appendix D.
            outputs = [outputs[i] for i in ent_filtered_indices]

        handle = self.ipc_client.serialize_ipc([outputs, ent_filtered_indices])
        return handle

    def query_oracle(self, handle: DataID):
        assert not self.eval_mode
        info = dict()
        prompts, candidates = self.ipc_client.deserialize_ipc(handle)
        bt_probs = self.oracle.compare(
            prompts,
            [candidates[i][0] for i in range(len(prompts))],
            [candidates[i][1] for i in range(len(prompts))],
            return_probs=True,
            disable_tqdm=True,
        )

        if self.args.bt_sample:
            binary_feedback = torch.bernoulli(torch.from_numpy(bt_probs)).bool().numpy()
        else:
            binary_feedback = bt_probs > 0.5
        chosen = 1 - binary_feedback
        rejected = 1 - chosen

        same_response = [
            candidates[i][chosen[i]] == candidates[i][rejected[i]]
            for i in range(len(prompts))
        ]

        preference_data = [
            PreferenceData(
                prompt=prompts[i],
                chosen_id=chosen[i],
                chosen_response=candidates[i][chosen[i]],
                rejected_response=candidates[i][rejected[i]],
                chosen_feature=None,
                rejected_feature=None,
                init_clash=False,
                same=same_response[i],
                is_model_data=False,
                info=info,
            )
            for i in range(len(prompts))
        ]

        metric = {
            "actor/chosen_avg_str_len": np.mean(
                [len(p.chosen_response) for p in preference_data]
            ),
            "actor/rejected_avg_str_len": np.mean(
                [len(p.rejected_response) for p in preference_data]
            ),
            "actor/init_clash_ratio": np.mean([p.init_clash for p in preference_data]),
            "actor/same_response_ratio": np.mean([p.same for p in preference_data]),
            "actor/pair_edit_dist": np.mean(
                [
                    Levenshtein.distance(p.chosen_response, p.rejected_response)
                    for p in preference_data
                ]
            ),
            "actor/model_data_ratio": np.mean(
                [p.is_model_data for p in preference_data]
            ),
            "actor/chosen_id": np.mean([p.chosen_id for p in preference_data]),
            "actor/first_action_win_prob": bt_probs.mean().item(),
        }

        handle = self.ipc_client.serialize_ipc([preference_data, metric])
        return handle


class APLLearner(DAPLearner):
    def run(self):
        """Overriding the learner run loop for APL."""
        self.ipc_client = PlasmaShmClient(self.ipc_server)
        self._init(self.args, self.actors)

        self.steps = 0
        self.start_time = time.time()

        self.actor_info = {}

        if not self.strategy.args.debug:
            self.eval_and_log({}, eval=True, save=False)

        self.steps = 1
        self.gradient_update_st = time.time()
        for p_ep in range(self.args.num_prompt_epoch):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(p_ep)
                self.strategy.print(f"Set DistributedSampler at epoch {p_ep}")
            progress_bar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Prompt epoch [{p_ep + 1}/{self.args.num_prompt_epoch}]",
                disable=not self.strategy.is_rank_0(),
            )

            for processed_prompts, raw_prompts, refs in self.prompts_dataloader:
                # ###################### #
                # (BEGIN) Logic for APL  #
                # ###################### #

                # APL Algo 1, Line 7-8: generate response & (optionally) filter by entropy.
                st_time = time.time()
                rank = torch.distributed.get_rank()
                actor: APLActor = self.actors[rank % len(self.actors)]
                handle = actor.generate_and_entropy_filter(processed_prompts)
                outputs: List[RequestOutput]
                outputs, ent_filtered_indices = self.ipc_client.deserialize_ipc(handle)

                # APL Algo 1, Line 8-9: get implicit reward margin and select pairs.
                output_info1 = f"({len(outputs)},{len(outputs[0].outputs)})"
                if not self.args.apl_pref_certainty_only:
                    # Keep all filtered prompts; select response pair.
                    processed_prompts = [
                        processed_prompts[i] for i in ent_filtered_indices
                    ]
                    raw_prompts = [raw_prompts[i] for i in ent_filtered_indices]
                    candidates, info = implicit_reward_filtering_response_only(
                        self.model,
                        self.ref_model,
                        self.tokenizer,
                        outputs,
                    )
                else:
                    # Select the (x, y, y') triplet.
                    processed_prompts, raw_prompts, candidates, info = (
                        implicit_reward_filtering_triplet(
                            processed_prompts,
                            raw_prompts,
                            self.model,
                            self.ref_model,
                            self.tokenizer,
                            outputs,
                            self.args.pi_buffer_maxlen_per_device,
                        )
                    )
                output_info2 = f"({len(processed_prompts)},{len(candidates[0])})"
                # APL Algo 1, Line 10: query oracle RM.
                handle = actor.query_oracle(
                    self.ipc_client.serialize_ipc([processed_prompts, candidates])
                )
                preference_data: List[PreferenceData]
                preference_data, self.actor_info = self.ipc_client.deserialize_ipc(
                    handle
                )
                self.actor_info.update(
                    {
                        "actor/generate_time": time.time() - st_time,
                        **info,
                    }
                )

                # ###################### #
                #   (END) Logic for APL  #
                # ###################### #

                self.prompt_consumed += len(refs)
                self.query_step += np.sum(
                    [not p.is_model_data for p in preference_data]
                )
                self.process_preference_data(preference_data, raw_prompts)

                if self.steps % self.update_interval == 0:
                    train_info = self.learn(self.steps // self.update_interval)

                    self.eval_and_log(train_info)

                    if (
                        self.steps // self.update_interval
                    ) % self.args.sync_params_every == 0:
                        self.sync_params_to_actors()

                    if (
                        self.steps // self.update_interval
                    ) % self.args.buffer_clear_every == 0:
                        self.pi_buffer.clear()

                progress_bar.update()
                self.steps += 1

        self.eval_and_log(train_info, eval=True, save=True)

        if self.strategy.is_rank_0():
            self._wandb.finish()
            lp.stop()


@torch.no_grad
def implicit_reward_filtering_response_only(
    policy_model: LLM,
    ref_model: LLM,
    tokenizer: PreTrainedTokenizer,
    outputs: List[RequestOutput],
) -> Tuple[List[str], Dict[str, List[str]], Metric]:
    """Select the response pair that gives the largest implicit reward margin."""
    candidates = {}

    avg_margins = []
    selected_margins = []
    for i, output in enumerate(outputs):
        # for each prompt
        prompt_response_ids = [
            torch.tensor(output.prompt_token_ids + list(o.token_ids))
            for o in output.outputs
        ]
        prompt_response_masks = [torch.ones_like(ids) for ids in prompt_response_ids]

        prompt_response_ids = zero_pad_sequences(
            prompt_response_ids, side="right", value=tokenizer.pad_token_id
        )
        prompt_response_masks = zero_pad_sequences(prompt_response_masks, side="right")

        prompt_response_ids = prompt_response_ids.cuda()
        prompt_response_masks = prompt_response_masks.cuda()

        logprobs = compute_logp(
            policy_model,
            prompt_response_ids,
            prompt_response_masks,
            len(output.prompt_token_ids),
        )

        logprobs_ref = compute_logp(
            ref_model,
            prompt_response_ids,
            prompt_response_masks,
            len(output.prompt_token_ids),
        )
        M = len(prompt_response_ids)
        implicit_rewards = logprobs - logprobs_ref
        # NOTE the above will be zero until the policy is updated, need to avoid
        # selecting the same response during argmax, so subtract identity.
        reward_margins = torch.abs(
            implicit_rewards.view(M, 1) - implicit_rewards.view(1, M)
        ) - torch.eye(M, device=implicit_rewards.device)

        max_idx = reward_margins.argmax()
        pair_indices = [max_idx // M, max_idx % M]
        candidates[i] = [output.outputs[j].text for j in pair_indices]

        avg_margins.append(reward_margins.mean().cpu().item())
        selected_margins.append(reward_margins.max().cpu().item())

    return (
        candidates,
        {
            "actor/avg_margins": np.mean(avg_margins),
            "actor/selected_margins": np.mean(selected_margins),
        },
    )


@torch.no_grad
def implicit_reward_filtering_triplet(
    processed_prompts: List[str],
    raw_prompts: List[str],
    policy_model: LLM,
    ref_model: LLM,
    tokenizer: PreTrainedTokenizer,
    outputs: List[RequestOutput],
    num_keep: int,
) -> Tuple[List[str], Dict[str, List[str]], Metric]:
    """Select the response pair that gives the largest implicit reward margin."""
    scores = []

    for output in outputs:
        # for each prompt
        prompt_response_ids = [
            torch.tensor(output.prompt_token_ids + o.token_ids) for o in output.outputs
        ]
        assert len(prompt_response_ids) == 2, len(prompt_response_ids)
        prompt_response_masks = [torch.ones_like(ids) for ids in prompt_response_ids]

        prompt_response_ids = zero_pad_sequences(
            prompt_response_ids, side="right", value=tokenizer.pad_token_id
        )
        prompt_response_masks = zero_pad_sequences(prompt_response_masks, side="right")

        prompt_response_ids = prompt_response_ids.cuda()
        prompt_response_masks = prompt_response_masks.cuda()

        logprobs = compute_logp(
            policy_model,
            prompt_response_ids,
            prompt_response_masks,
            len(output.prompt_token_ids),
        )

        logprobs_ref = compute_logp(
            ref_model,
            prompt_response_ids,
            prompt_response_masks,
            len(output.prompt_token_ids),
        )
        implicit_rewards = logprobs - logprobs_ref
        scores.append(torch.abs(implicit_rewards[0] - implicit_rewards[1]).cpu().item())

    scores = np.array(scores)
    top_indices = np.argsort(scores)[-num_keep:].tolist()

    processed_prompts = [processed_prompts[idx] for idx in top_indices]
    raw_prompts = [raw_prompts[idx] for idx in top_indices]
    candidates = {
        i: [outputs[idx].outputs[0].text.strip(), outputs[idx].outputs[1].text.strip()]
        for i, idx in enumerate(top_indices)
    }
    info = {
        "actor/avg_scores": scores.mean(),
        "actor/selected_scores": scores[top_indices].mean(),
    }

    return (
        processed_prompts,
        raw_prompts,
        candidates,
        info,
    )


@torch.no_grad
def compute_logp(model, prompt_response_ids, prompt_response_masks, prompt_len: int):
    model_output = model(prompt_response_ids, attention_mask=prompt_response_masks)
    all_logits = model_output["logits"]
    prompt_id_lens = [prompt_len] * len(prompt_response_masks)
    return get_batch_logps(
        all_logits,
        prompt_response_ids,
        prompt_response_masks,
        prompt_id_lens,
        average_log_prob=False,
    )


@torch.no_grad
def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask,
    prompt_id_lens,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    loss_masks = attention_mask.clone().bool()
    # mask prompts
    for mask, source_len in zip(loss_masks, prompt_id_lens):
        mask[:source_len] = False
    loss_masks = loss_masks[:, 1:]

    # dummy token; we'll ignore the losses on these tokens later
    labels[loss_masks == False] = 0
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
    else:
        return (per_token_logps * loss_masks).sum(-1)

```
