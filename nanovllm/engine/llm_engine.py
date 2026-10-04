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
        if not hasattr(self, "model_runner"):
            return
        serving = getattr(self, "_jetspec_scheduler", None)
        if serving is not None:
            serving.close()
        batch_cached = getattr(self, "_jetspec_batch_runtime", None)
        if batch_cached is not None:
            batch_cached[1].close()
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def configure_jetspec(self, draft_model: str, *, tree_depth: int = 15,
                         tree_width: int = 7, max_tree_budget: int = 63,
                         default_tree_budget: int = 63, max_admissions_per_step: int = 2,
                         max_prefill_tokens: int | None = None,
                         enable_chunked_prefill: bool = True, prefill_chunk_size: int = 256,
                         optimization: str = "serving"):
        """Select the greedy Continuous Batching adapter for add/step/cancel.

        Like nano-vLLM's ordinary engine this is a synchronous event loop, not
        an HTTP server or a concurrent GPU executor. Arrivals/cancellation are
        processed between steps; one packed Target verify executes per decode.
        Chunked initial/recompute prefill runs between decode rounds. Its token
        budget is separate from the packed verification query budget.
        """
        serving = getattr(self, "_jetspec_scheduler", None)
        if serving is not None:
            if not serving.is_finished():
                raise RuntimeError("cannot reconfigure JetSpec with pending serving requests/events")
            self.disable_jetspec()
        if not 1 <= default_tree_budget <= max_tree_budget:
            raise ValueError("default tree budget must fit the configured maximum")
        if optimization not in ("serving", "debug"):
            raise ValueError("JetSpec optimization must be 'serving' or 'debug'")
        runtime = self.get_jetspec_batch_runtime(draft_model, tree_depth=tree_depth,
            tree_width=tree_width, max_tree_budget=max_tree_budget)
        if runtime.requests or getattr(runtime, "prefills", {}):
            raise RuntimeError("cannot enter serving mode with explicit packed requests")
        optimized = optimization == "serving"
        runtime.configure_optimizations(lightweight=optimized, batched_draft=optimized,
                                        feature_storage=optimized)
        from nanovllm.engine.jetspec_scheduler import JetSpecScheduler
        config = self.model_runner.config
        self._jetspec_scheduler = JetSpecScheduler(runtime, config.max_num_seqs,
            max_verify_tokens=config.max_num_batched_tokens,
            max_admissions_per_step=max_admissions_per_step,
            max_prefill_tokens=max_prefill_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            prefill_chunk_size=prefill_chunk_size)
        self._jetspec_default_tree_budget = default_tree_budget
        self.last_step_info = None
        return self._jetspec_scheduler

    def disable_jetspec(self):
        """Return to ordinary serving; retain resident immutable Draft weights."""
        serving = getattr(self, "_jetspec_scheduler", None)
        if serving is not None:
            if not serving.is_finished():
                raise RuntimeError("cannot disable JetSpec with pending serving requests/events")
            serving.close()
            self._jetspec_scheduler = None
        cached = getattr(self, "_jetspec_batch_runtime", None)
        if cached is not None and not cached[1]._closed:
            cached[1].release_idle_scratch()
            if not cached[1].requests and not getattr(cached[1], "prefills", {}):
                cached[1].configure_optimizations()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams,
                    *, tree_budget: int | None = None, request_id: str | int | None = None):
        serving = getattr(self, "_jetspec_scheduler", None)
        if serving is not None:
            if sampling_params.temperature != 0:
                raise ValueError("JetSpec Continuous Batching currently supports greedy temperature=0 only")
            return serving.add_request(prompt, max_new_tokens=sampling_params.max_tokens,
                tree_budget=self._jetspec_default_tree_budget if tree_budget is None else tree_budget,
                ignore_eos=sampling_params.ignore_eos, request_id=request_id)
        if tree_budget is not None or request_id is not None:
            raise ValueError("tree_budget/custom request_id require configure_jetspec")
        cached = getattr(self, "_jetspec_batch_runtime", None)
        if cached is not None:
            if cached[1].requests or getattr(cached[1], "prefills", {}):
                raise RuntimeError("ordinary serving cannot interleave explicit packed requests")
            if not cached[1]._closed:
                cached[1].release_idle_scratch()
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def cancel_request(self, request_id):
        serving = getattr(self, "_jetspec_scheduler", None)
        if serving is not None:
            return serving.cancel(request_id)
        seq = self.scheduler.cancel(request_id)
        if seq is None:
            return None
        return {"request_id": request_id, "kind": "cancelled", "cancelled": True,
                "token_ids": list(seq.completion_token_ids)}

    def step(self):
        serving = getattr(self, "_jetspec_scheduler", None)
        if serving is not None:
            try:
                record = serving.step()
            except BaseException:
                self.last_step_info = serving.last_step
                raise
            self.last_step_info = record
            outputs = [(e["request_id"], e["token_ids"]) for e in record["events"]
                       if e["kind"] in ("finished", "cancelled", "error")]
            emitted = sum(len(e["token_ids"]) for e in record["events"] if e["kind"] == "tokens")
            return outputs, -emitted
        if self.scheduler.is_finished():
            self.last_step_info = {"events": [], "verification": None}
            return [], 0
        seqs, is_prefill = self.scheduler.schedule()
        old_lengths = {s.seq_id: s.num_completion_tokens for s in seqs}
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        events = []
        for seq in seqs:
            delta = seq.completion_token_ids[old_lengths[seq.seq_id]:]
            if delta:
                events.append({"request_id": seq.seq_id, "kind": "tokens", "token_ids": delta})
            if seq.is_finished:
                events.append({"request_id": seq.seq_id, "kind": "finished",
                               "token_ids": list(seq.completion_token_ids)})
        self.last_step_info = {"events": events, "verification": None, "is_prefill": is_prefill,
            "waiting_count": len(self.scheduler.waiting), "running_count": len(self.scheduler.running),
            "capacity": {"allocator_used_blocks": len(self.scheduler.block_manager.used_block_ids),
                         "reserved_kv_slots": len(self.scheduler.block_manager.used_block_ids) * self.scheduler.block_size,
                         "live_kv_slots": sum(s.num_cached_tokens for s in
                                              list(self.scheduler.running) + list(self.scheduler.waiting))}}
        return outputs, num_tokens

    def is_finished(self):
        serving = getattr(self, "_jetspec_scheduler", None)
        if serving is not None:
            return serving.is_finished()
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        """Blocking convenience wrapper; live/backpressured serving uses step().

        JetSpec terminal errors and cancellation are not successful outputs.
        On failure, cancel only this invocation's requests, preserve the error
        report, and keep the configured adapter available for a later call.
        """
        serving = getattr(self, "_jetspec_scheduler", None)
        if serving is not None and not serving.is_finished():
            raise RuntimeError("generate requires an idle JetSpec serving queue; use step for live arrivals")
        batch_cached = getattr(self, "_jetspec_batch_runtime", None)
        if batch_cached is not None and serving is None:
            if batch_cached[1].requests or getattr(batch_cached[1], "prefills", {}):
                raise RuntimeError("ordinary generation cannot interleave live JetSpec batch requests")
            if not batch_cached[1]._closed:
                batch_cached[1].release_idle_scratch()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("one sampling_params entry is required per prompt")
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        request_ids = []
        outputs = {}
        failure = None
        prefill_throughput = decode_throughput = 0.
        try:
            for prompt, sp in zip(prompts, sampling_params):
                request_ids.append(self.add_request(prompt, sp))
            while not self.is_finished():
                t = perf_counter()
                output, num_tokens = self.step()
                if serving is not None:
                    record = self.last_step_info
                    failures = [event for event in record["events"]
                                if event["kind"] in ("error", "cancelled")]
                    if failures:
                        reasons = "; ".join(f"request {e['request_id']!r}: {e['kind']}: {e['reason']}"
                                            for e in failures)
                        raise RuntimeError(f"JetSpec generation failed: {reasons}")
                    if record["blocked"]:
                        raise RuntimeError("JetSpec generation blocked: " + record["blocked_reason"] +
                                           "; use add_request/step to retry external KV backpressure")
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
            return [{"text": self.tokenizer.decode(outputs[request_id]),
                     "token_ids": outputs[request_id]} for request_id in request_ids]
        except BaseException as exception:
            failure = exception
            if serving is not None:
                # The queue was idle on entry and this API is synchronous.
                # Reconcile our tickets without hiding the original exception.
                # Also cover an add that registered a ticket before raising.
                owned_ids = dict.fromkeys(request_ids + list(serving.requests))
                for request_id in owned_ids:
                    if request_id not in serving.requests:
                        continue
                    try:
                        self.cancel_request(request_id)
                    except BaseException as cleanup_error:
                        exception.add_note(f"generate cleanup for {request_id!r} failed: {cleanup_error}")
                try:
                    serving.drain_events()
                    if not serving.runtime.requests and not getattr(serving.runtime, "prefills", {}):
                        serving.runtime.release_idle_scratch()
                except BaseException as cleanup_error:
                    exception.add_note(f"generate final cleanup failed: {cleanup_error}")
                if serving.requests:
                    exception.add_note(f"generate retains requests after failed cleanup: {list(serving.requests)!r}")
            raise
        finally:
            try:
                pbar.close()
            except BaseException as close_error:
                if failure is None:
                    raise
                failure.add_note(f"generate progress-bar cleanup failed: {close_error}")

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
        if getattr(self, "_jetspec_scheduler", None) is not None:
            raise RuntimeError("disable Continuous Batching before using legacy JetSpec")
        batch_cached = getattr(self, "_jetspec_batch_runtime", None)
        if batch_cached is not None and (batch_cached[1].requests or getattr(batch_cached[1], "prefills", {})):
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
        if getattr(self, "_jetspec_scheduler", None) is not None:
            raise RuntimeError("explicit packed API cannot interleave JetSpec Continuous Batching")
        if self.scheduler.waiting or self.scheduler.running:
            raise RuntimeError("packed JetSpec requires an idle ordinary scheduler")
        from nanovllm.speculative.jetspec.batch_runtime import JetSpecBatchRuntime
        key = (draft_model, int(tree_depth), int(tree_width), int(max_tree_budget))
        cached = getattr(self, "_jetspec_batch_runtime", None)
        if cached is not None and (cached[0] != key or cached[1]._closed):
            if cached[1].requests or getattr(cached[1], "prefills", {}):
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
