import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        batch_cached = getattr(self, "_jetspec_batch_runtime", None)
        if batch_cached is not None:
            batch_cached[1].close()
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        batch_cached = getattr(self, "_jetspec_batch_runtime", None)
        if batch_cached is not None:
            if batch_cached[1].requests:
                raise RuntimeError("ordinary generation cannot interleave live JetSpec batch requests")
            if not batch_cached[1]._closed:
                batch_cached[1].release_idle_scratch()
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

    def generate_jetspec(
        self,
        prompt: str | list[int],
        draft_model: str,
        *,
        max_tokens: int = 32,
        tree_depth: int = 15,
        tree_width: int = 7,
        tree_budget: int = 63,
        tree_backend: str = "dense",
        qualification_rounds: tuple[int, ...] = (),
        record_tree_layout: bool = False,
        return_rounds: bool = True,
    ) -> dict:
        """Opt-in, single-request greedy JetSpec correctness runtime.

        The normal scheduler/paged-attention ``generate`` path is unchanged.  Phase 1
        intentionally rejects TP, CUDA graphs, batching and prefix-cache reuse.
        """
        if self.model_runner.world_size != 1:
            raise ValueError("generate_jetspec Phase 1 requires tensor_parallel_size=1")
        if not self.model_runner.enforce_eager:
            raise ValueError("generate_jetspec Phase 1 requires enforce_eager=True")
        if tree_backend not in ("dense", "paged"):
            raise ValueError("tree_backend must be 'dense' or 'paged'")
        if self.scheduler.waiting or self.scheduler.running:
            raise RuntimeError("generate_jetspec requires an idle single-request scheduler")
        batch_cached = getattr(self, "_jetspec_batch_runtime", None)
        if batch_cached is not None and batch_cached[1].requests:
            raise RuntimeError("legacy JetSpec cannot interleave live packed requests")
        from nanovllm.speculative.jetspec.runtime import JetSpecRuntime

        cache_key = (draft_model, int(tree_depth), int(tree_width), int(tree_budget))
        cached = getattr(self, "_jetspec_runtime", None)
        if cached is None or cached[0] != cache_key:
            runtime = JetSpecRuntime(
                target=self.model_runner.model,
                tokenizer=self.tokenizer,
                draft_model=draft_model,
                tree_depth=tree_depth,
                tree_width=tree_width,
                tree_budget=tree_budget,
                kv_pool=self.model_runner.kv_cache,
                block_manager=self.scheduler.block_manager,
                block_size=self.model_runner.block_size,
            )
            self._jetspec_runtime = (cache_key, runtime)
        else:
            runtime = cached[1]
        return runtime.generate(
            prompt,
            max_new_tokens=max_tokens,
            tree_backend=tree_backend,
            qualification_rounds=qualification_rounds,
            record_tree_layout=record_tree_layout,
            return_rounds=return_rounds,
        )

    def get_jetspec_batch_runtime(self, draft_model: str, *, tree_depth: int = 15,
                                 tree_width: int = 7, max_tree_budget: int = 63):
        """Get the idle-scheduler packed runner for explicit create/step/finish.

        The runner owns scratch; each request owns its canonical state and Draft
        cache. This opt-in API is not wired into the ordinary serving scheduler.
        """
        if self.model_runner.world_size != 1 or not self.model_runner.enforce_eager:
            raise ValueError("packed JetSpec currently requires TP=1 and eager execution")
        if self.scheduler.waiting or self.scheduler.running:
            raise RuntimeError("packed JetSpec requires an idle ordinary scheduler")
        from nanovllm.speculative.jetspec.batch_runtime import JetSpecBatchRuntime
        key = (draft_model, int(tree_depth), int(tree_width), int(max_tree_budget))
        cached = getattr(self, "_jetspec_batch_runtime", None)
        if cached is not None and (cached[0] != key or cached[1]._closed):
            if cached[1].requests:
                raise RuntimeError("cannot replace a packed runner with live requests")
            cached[1].close()
            cached = None
        if cached is None:
            legacy = getattr(self, "_jetspec_runtime", None)
            head = legacy[1].head if legacy is not None and legacy[0][0] == draft_model else None
            config = self.model_runner.config
            runtime = JetSpecBatchRuntime(
                target=self.model_runner.model, tokenizer=self.tokenizer,
                draft_model=draft_model, kv_pool=self.model_runner.kv_cache,
                block_manager=self.scheduler.block_manager, block_size=self.model_runner.block_size,
                tree_depth=tree_depth, tree_width=tree_width, max_tree_budget=max_tree_budget,
                max_verify_tokens=config.max_num_batched_tokens, max_model_len=config.max_model_len,
                head=head,
            )
            self._jetspec_batch_runtime = (key, runtime)
        else:
            runtime = cached[1]
        return runtime

    def generate_jetspec_batch(self, prompts, draft_model: str, *, max_tokens=32,
                              tree_budgets=63, tree_depth: int = 15, tree_width: int = 7,
                              ignore_eos: bool = False, return_rounds: bool = True) -> dict:
        """Greedy eager packed ragged tree verification with shared scratch."""
        runtime = self.get_jetspec_batch_runtime(
            draft_model, tree_depth=tree_depth, tree_width=tree_width,
            max_tree_budget=max(63, max(tree_budgets) if not isinstance(tree_budgets, int) else tree_budgets),
        )
        return runtime.generate_batch(prompts, max_new_tokens=max_tokens,
                                      tree_budgets=tree_budgets, ignore_eos=ignore_eos,
                                      return_rounds=return_rounds)
