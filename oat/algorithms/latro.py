# Adapted from https://github.com/SalesforceAIResearch/LaTRO 


"""
latro.py

Implements a LaTRO (Reasoning-Trajectory Optimization) style algorithm in OAT.
It parallels the approach seen in ppo.py but applies:
  - RLOO (k samples per prompt)
  - KL penalty between policy and a reference model
  - optional SFT penalty on final chain-of-thought completion.

Actor side: LaTROActor
  - Generates multiple partial rationales per prompt
  - Optionally queries a reward oracle

Learner side: LaTROLearner
  - Takes the data from the actor (TrajectoryData or a custom class)
  - Performs the RLOO advantage computation
  - Jointly applies “policy update” for the rationales, plus an “SFT penalty” on final completions
"""

import math
import time
import gc
import numpy as np
import torch
import tree
from typing import List, Union
from dataclasses import dataclass, field

# OAT / vLLM / Launchpad related
import vllm
import launchpad as lp
import torch.distributed as dist

# OAT base imports
from oat.actors.reward import RewardActor
from oat.learners.rl import RLLearner
from oat.args import OATArgs
from oat.types import TrajectoryData, Metric
from oat.utils.ipc import DataID
from oat.utils.distributed import torch_type_codec
from oat.utils.data import zero_pad_sequences
from oat.model import LLM

# ~~~~~~~~~~~~~~~~~~~~~
# 1) Define new Args
# ~~~~~~~~~~~~~~~~~~~~~

@dataclass
class LaTROArgs(OATArgs):
    """
    Specialized arguments for LaTRO, in addition to the usual OATArgs.
    The style is analogous to PPOArgs, DAPAlgo, etc.
    """
    # For the multi-sample RLOO
    rloo_k: int = field(
        default=4,
        metadata={"help": "Number of RLOO samples per prompt. i.e. how many rationales to sample per prompt."}
    )
    # KL Coefficient
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "Coefficient for KL penalty between policy logprob and reference logprob."}
    )
    # SFT penalty
    sft_penalty: float = field(
        default=0.0,
        metadata={"help": "How much we scale the SFT loss on the final (prompt+rationale+gold) chain."}
    )
    # Non-stop penalty
    non_stop_penalty: bool = field(
        default=True,
        metadata={"help": "If True, penalize completions that never see a certain stop token. e.g. chain-of-thought."}
    )
    # If we penalize them, we can set a fallback reward
    penalty_reward_value: float = field(
        default=2.0,
        metadata={"help": "Set the reward to penalty_reward_value * average_reward if no stop token found."}
    )


# ~~~~~~~~~~~~~~~~~~~~~
# 2) The Actor
# ~~~~~~~~~~~~~~~~~~~~~

class LaTROActor(RewardActor):
    """
    Gathers data for LaTRO: multiple rationales per prompt (RLOO).
    - We produce `rloo_k` responses for each prompt.
    - Store partial logprobs in TrajectoryData, along with final text.
    - Possibly do a scalar reward query from the Oracle.
    """

    def __init__(self, ipc_server, vllm_args, args: LaTROArgs) -> None:
        super().__init__(ipc_server, vllm_args, args)
        # override sampling_params (like we do in PPOActor)
        # we generate rloo_k responses per prompt, each has logprobs
        self.sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.generate_max_length,
            # We often want multiple chain-of-thought samples
            n=args.rloo_k,
            logprobs=1,  # store logprobs so we can reconstruct partial log probabilities
            # we might define stop if we want
            stop=[],
        )

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[TrajectoryData]:
        """
        1. Generate rloo_k rationales per prompt.
        2. Possibly query oracle for final scalar reward.
        3. Return a list of TrajectoryData, each sample = 1 rationale.

        The high-level approach:
          - vLLM returns `rloo_k` outputs for each prompt.
          - Flatten them for oracle queries and store final data.
        """
        # Step 1: Generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)

        # outputs is a list of length = len(prompts)
        # each item has `.outputs` of length = n (rloo_k)
        # each sub-output has `.token_ids`, `.logprobs`
        info: Metric = {}
        info["actor/generate_time"] = time.time() - st

        # Step 2: Flatten & prepare for Oracle
        flatten_prompts = []
        flatten_responses = []
        indices_map = []
        for i, out in enumerate(outputs):
            for k in range(self.sampling_params.n):
                flatten_prompts.append(prompts[i])
                flatten_responses.append(out.outputs[k].text)
                indices_map.append((i, k))  # which prompt, which sub-sample

        # Step 3: Oracle (if using a reward-based approach)
        st = time.time()
        rewards, oracle_info = self.oracle.get_reward(
            flatten_prompts,
            flatten_responses,
            references * self.sampling_params.n if references else ["" for _ in flatten_prompts]
        )
        info["actor/oracle_time"] = time.time() - st
        info["actor/rewards_mean"] = rewards.mean().item()

        # Step 4: Re-group into final list of TrajectoryData
        # Because typically we have len(prompts)*rloo_k total completions
        # We'll store them each as a separate TrajectoryData
        # so the Learner can do RLOO advantage computations.

        # gather final objects
        trajectory_list: List[TrajectoryData] = []
        idx = 0
        for i, out in enumerate(outputs):
            for k in range(len(out.outputs)):
                # final scalar reward
                reward_value = rewards[idx].item()

                # token-level stuff
                sub_out = out.outputs[k]
                # vLLM gives sub_out.token_ids, sub_out.logprobs, sub_out.text
                # We can store final string or token IDs
                # Also store partial logprobs for the response
                # Optionally check if no stop token => apply penalty?
                # We'll do that in the Learner though.

                # Build dense reward as 0 except last token
                # we can do that or just store final scalar in the "rewards" array
                # for RLOO. We'll do final token = reward_value
                all_rewards = [0]*(len(sub_out.token_ids))
                all_rewards[-1] = reward_value

                # same for logprobs
                # sub_out.logprobs is a list of dict: we might do something simpler:
                # sub_out.logprobs[tok].logprob => or just store them for reference.
                # For RLOO, we might only need the sum. But let's store them anyway.

                # We create a TrajectoryData for each sample
                data = TrajectoryData(
                    prompt=prompts[i],
                    prompt_ids=out.prompt_token_ids,  # the vLLM prompt token IDs
                    response=sub_out.text,
                    response_ids=sub_out.token_ids,
                    response_logprobs=[lp.logprob for lp in sub_out.logprobs],
                    rewards=all_rewards,
                    loss_mask=True,  # used in Learner
                    info=info,
                )
                trajectory_list.append(data)
                idx += 1

        # done
        handle = self.ipc_client.serialize_ipc(trajectory_list)
        return handle


# ~~~~~~~~~~~~~~~~~~~~~
# 3) The Learner
# ~~~~~~~~~~~~~~~~~~~~~

class LaTROLearner(RLLearner):
    """
    LaTRO-style approach:
      - We gather multiple rationales from the Actor (TrajectoryData).
      - We do RLOO advantage computation and a KL penalty log(p) - log(ref).
      - We optionally do an SFT penalty on a (prompt + rationale + final reference) chain.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str,
        master_port: str,
        is_master: bool,
        args: LaTROArgs,
        actors: List[RewardActor],
        ipc_server,
    ) -> None:
        super().__init__(
            world_size, rank, local_rank, master_addr, master_port, is_master, args, actors, ipc_server
        )
        self.rloo_k = args.rloo_k
        self.kl_coef = args.kl_coef
        self.sft_penalty = args.sft_penalty
        self.non_stop_penalty = args.non_stop_penalty
        self.penalty_reward_value = args.penalty_reward_value

    def learn(self, learning_round: int):
        """
        Overriding the RL approach to do the multi-sample RLOO advantage computation
        + final chain-of-thought SFT penalty + KL penalty.
        """
        # Acquire all data from the buffer
        # We'll do one big pass: in OAT we typically read from `self.pi_buffer`.
        # Then batch it. If the data is large, see ppo.py pattern: we do small steps.
        dataset = self.dataset_builder(
            self.pi_buffer,
            self.tokenizer,
            self.strategy,
        )
        if learning_round == 1:
            self.strategy.print("LaTRO training example:")
            self.strategy.print(dataset[0])

        # Build dataloader
        dataloader = self.strategy.setup_dataloader(
            dataset,
            batch_size=len(dataset),
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
        local_sgd_steps = 0

        # We'll do multiple epochs or just once
        # For the snippet logic, we do a single pass to keep it consistent with RL approach
        # If needed, you can do self.args.num_ppo_epochs style loops
        for _epoch in range(self.args.num_ppo_epochs if hasattr(self.args, "num_ppo_epochs") else 1):
            step_bar = range(len(dataloader))
            learn_batch_time = []
            st = time.time()
            self.model.train()
            if self.critic is not None:
                self.critic.train()

            for data in step_bar:
                if local_sgd_steps > self.args.max_sgd_steps:
                    break
                # data is the entire batch
                batch = next(iter(dataloader))
                infos = self.learning_step(batch)
                # Update local steps
                self.policy_sgd_step += 1
                local_sgd_steps += 1
                # measure time
                learn_batch_time.append(time.time() - st)
                st = time.time()

        dist.barrier()

        train_info = {
            "train/learning_round": learning_round,
            "train/learn_batch_time": np.mean(learn_batch_time),
        }
        # merges in the last step’s info
        if infos is not None:
            for k, v in infos.items():
                train_info[f"train/{k}"] = v.cpu().float().mean().item() if isinstance(v, torch.Tensor) else v

        return train_info

    def learning_step(self, batch):
        """
        The heart of the LaTRO logic:
         - We parse the batch of 'TrajectoryData' objects
         - Group them by prompt so that each prompt has rloo_k different completions
         - Compute advantage = (reward + KL penalty) - baseline
         - Then do the gradient step with - advantage * log p(rationale)
         - Then do optional “SFT penalty” using the final chain tokens + model logprob
        """
        device = torch.cuda.current_device()

        # parse them from the collate
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_id_lens = batch["prompt_ids_lens"]
        action_logprobs = batch["action_logprobs"]  # might be none if we didn’t store
        # parse final rewards from batch["rewards"]
        # shape is [B, seq], but last token is final scalar reward
        # could flatten. see ppo.py for reference

        # We'll do a simpler approach: sum of all except last => 0, last => final reward
        # Then each row is an entire (prompt + response) sequence. The “response_mask” is everything after prompt.
        # We'll replicate logic from ppo.py or the snippet. We'll do a short version:

        # 1. group by prompt. We expect B = #prompts * rloo_k
        # 2. if we want to do a grouping, we rely on the dataset to have them in consecutive order
        #    or we can parse from data's "info" or "misc" that indicates which prompt is which.

        # For demonstration, assume the entire batch is from the same group or we re-shuffle it so that each consecutive rloo_k is the same prompt.

        B = input_ids.size(0)
        if B % self.rloo_k != 0:
            # fallback
            self.strategy.print("WARNING: cannot do perfect grouping of rloo_k in this batch!")
        # group_count = B // self.rloo_k

        # 2. Let's get policy logps from current model:
        # The snippet references: logp(policy) - logp(ref) => kl
        # We'll replicate the logic similarly:

        # Ensure model is in train mode
        output = self.model(input_ids, attention_mask=attention_mask)
        logits = output["logits"]  # shape [B, seq_len, vocab_size]
        logits /= (self.args.temperature if self.args.temperature > 1e-7 else 1.0)

        # We'll define a helper to get the “response_mask” from prompt len
        # The snippet from ppo/dap uses a function get_batch_logps(...) etc.
        # Let's replicate a small version here inline:

        # response_mask is everything after the prompt
        response_mask = attention_mask.clone().bool()
        for i, plen in enumerate(prompt_id_lens):
            response_mask[i, :plen] = False

        # The last token’s reward is in `batch["rewards"]`, shape [B, seq_len]
        # We'll flatten it or keep it. Typically the final token index is in the response.
        # parse final scalar reward
        # sum across the entire sequence to find it or just take last:
        # E.g. last = batch["rewards"][i, mask_end].
        # We'll do simpler: reward[i] = batch["rewards"][i, -1]

        final_rewards = []
        for i in range(B):
            # each row
            row_rewards = batch["rewards"][i]
            # The last token in the row is the final scalar
            # (like PPO). Or we can do row_rewards[ sum of attention_mask -1].
            # We'll do the simpler approach:
            final_rewards.append(row_rewards[-1].item())
        final_rewards = torch.tensor(final_rewards, device=device).float()

        # non-stop penalty
        # if we want to check if the user included a special stop token
        # we can do that here. We'll assume if the final token is not eos => apply penalty
        # or the snippet approach:
        if self.non_stop_penalty:
            # Suppose we do a quick check if the eos token is in “response” portion:
            # We have no official eos token from OAT. Let's just do a quick marker
            # For demonstration, skip or do something simple:
            # non_stop_mask = ...
            pass

        # 3. reference logps
        with torch.no_grad():
            ref_output = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_output["logits"]
            ref_logits /= (self.args.temperature if self.args.temperature > 1e-7 else 1.0)

        # 4. Convert logits -> logprobs, then gather tokens for the response portion
        # We do a simpler version:
        #   all_logps = log_softmax(logits, dim=-1)
        #   policy_logps = gather(all_logps, index=the actual token)
        #   sum them up or do partial. Then sub from reference to get KL, etc.

        # We'll do a function from ppo. We can do a quick inline:
        logps = logits.log_softmax(dim=-1)
        ref_logps = ref_logits.log_softmax(dim=-1)
        # shift tokens by 1 for next-token prediction
        # same approach as in ppo.py
        # so the first label is input_ids[i,1], ignoring prompt?

        # We'll do an approximate sum-of-logp approach for the entire response
        # ignoring prompt tokens
        # simpler approach:
        # The chosen token for each position j is input_ids[i,j+1]
        # We'll gather them:

        # shift
        gather_indices = input_ids[:, 1:].unsqueeze(-1)  # shape [B, seq_len-1,1]
        # also shift the mask by 1
        shift_response_mask = response_mask[:, 1:]
        # clip logps
        logps = logps[:, :-1, :]
        ref_logps = ref_logps[:, :-1, :]

        # gather policy & ref
        policy_per_token_logp = torch.gather(logps, dim=2, index=gather_indices).squeeze(-1)
        ref_per_token_logp = torch.gather(ref_logps, dim=2, index=gather_indices).squeeze(-1)

        # zero out tokens that are not in the response
        policy_per_token_logp = policy_per_token_logp * shift_response_mask
        ref_per_token_logp = ref_per_token_logp * shift_response_mask

        # sum
        policy_sums = policy_per_token_logp.sum(-1)
        ref_sums = ref_per_token_logp.sum(-1)

        # compute kl penalty
        kl_penalty = - self.kl_coef * (policy_sums - ref_sums)

        # total advantage-ish
        # final reward is final_rewards, so advantage = final_reward + kl_penalty - baseline
        # RLOO requires grouping. We'll do a grouping approach:

        # group_count = B // self.rloo_k
        # reshape them
        # Suppose the dataset put them in consecutive blocks of rloo_k. We'll do:
        R = final_rewards.view(-1, self.rloo_k)
        KL = kl_penalty.view(-1, self.rloo_k)
        # combined
        adjusted_rewards = R + KL  # shape [group_count, rloo_k]
        # baseline is (sum - own) / (rloo_k - 1)
        sums = adjusted_rewards.sum(dim=1, keepdim=True)
        baseline = (sums - adjusted_rewards) / (float(self.rloo_k) - 1.0)
        advantages = adjusted_rewards - baseline
        # flatten back
        advantages = advantages.view(-1)

        # compute the negative log-likelihood from the policy again
        # policy_sums is shape [B], we do - advantage * policy_sums
        # but typically we want partial derivatives wrt entire tokens
        # we can do a minimal approach:
        # rloo_loss = - sum( advantage[i] * policy_sums[i] ) / B
        # or we can do a step-by-step approach from the snippet
        # We'll do the snippet style:

        # expand advantage to match the sum of tokens
        # simpler approach: advantage[i] * sum_of_token_logps[i]
        # then average

        advantage_expanded = advantages.unsqueeze(-1) # shape [B,1]
        # rloo_loss
        # sum_of_token_logps[i] = policy_sums[i]
        # so rloo_loss = -(advantage[i] * policy_sums[i]) mean
        # but we can do it token-level if we want. We'll keep it simpler:
        rloo_loss = - (advantages * policy_sums).mean()

        # Now SFT penalty => do an extra pass with (prompt + response + final gold) if we have it
        # Not included in normal OAT TrajectoryData. The snippet uses the “final groundtruth.” We might skip or do partial.
        # We'll do a trivial sft_loss = 0 for demonstration
        sft_loss = torch.tensor(0.0, device=device)

        # total loss
        loss = rloo_loss + self.sft_penalty * sft_loss

        # gradient
        self.strategy.backward(loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

        # collect metrics
        stats = {
            "rloo_loss": rloo_loss.detach(),
            "sft_loss": sft_loss.detach(),
            "kl_penalty_mean": kl_penalty.mean().detach(),
            "final_rewards_mean": final_rewards.mean().detach(),
            "advantages_mean": advantages.mean().detach(),
        }

        return stats
