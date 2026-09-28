import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


def _reference_rms_norm(x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
    """Eager HF Qwen3 RMSNorm order for the oracle-compatible dense seam."""
    input_dtype = x.dtype
    hidden = x.to(torch.float32)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + norm.eps)
    return norm.weight * hidden.to(input_dtype)


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

    def forward_dense(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Correctness-first SDPA seam used only by the opt-in JetSpec runtime.

        Physical K/V rows stay in flat tree order while ``positions`` carries the
        logical depth-based RoPE positions.  Only the newly produced K/V is returned;
        the caller decides which provisional rows become committed.
        """
        # Keep this correctness seam numerically aligned with the HF oracle.  The
        # regular nano path intentionally uses one packed GEMM, while HF evaluates
        # q/k/v as three independent linears; their BF16 accumulation can differ
        # enough to reorder close draft candidates.
        weight = self.qkv_proj.weight
        bias = self.qkv_proj.bias
        q = F.linear(hidden_states, weight[:self.q_size], None if bias is None else bias[:self.q_size])
        k = F.linear(
            hidden_states,
            weight[self.q_size:self.q_size + self.kv_size],
            None if bias is None else bias[self.q_size:self.q_size + self.kv_size],
        )
        v = F.linear(
            hidden_states,
            weight[self.q_size + self.kv_size:],
            None if bias is None else bias[self.q_size + self.kv_size:],
        )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = _reference_rms_norm(q, self.q_norm)
            k = _reference_rms_norm(k, self.k_norm)
        # HF Qwen3 casts its cos/sin table to the query dtype before RoPE.
        # nano's regular path computes this operation in FP32, so reproduce the
        # reference BF16 operation order only inside this opt-in dense seam.
        cos, sin = self.rotary_emb.cos_sin_cache[positions].chunk(2, dim=-1)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        def apply_hf_rope(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

        q, k = apply_hf_rope(q), apply_hf_rope(k)
        new_key_value = (k, v)
        if past_key_value is not None:
            k_all = torch.cat((past_key_value[0], k), dim=0)
            v_all = torch.cat((past_key_value[1], v), dim=0)
        else:
            k_all, v_all = k, v
        groups = self.num_heads // self.num_kv_heads
        q_sdpa = q.transpose(0, 1).unsqueeze(0)
        k_sdpa = k_all.transpose(0, 1).unsqueeze(0)
        v_sdpa = v_all.transpose(0, 1).unsqueeze(0)
        enable_gqa = groups > 1 and attention_mask is None
        if groups > 1 and not enable_gqa:
            batch, heads, seq_len, dim = k_sdpa.shape
            k_sdpa = k_sdpa[:, :, None, :, :].expand(
                batch, heads, groups, seq_len, dim
            ).reshape(batch, heads * groups, seq_len, dim)
            v_sdpa = v_sdpa[:, :, None, :, :].expand(
                batch, heads, groups, seq_len, dim
            ).reshape(batch, heads * groups, seq_len, dim)
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            attention_mask = torch.where(
                attention_mask,
                torch.zeros((), dtype=q.dtype, device=q.device),
                torch.full((), torch.finfo(q.dtype).min, dtype=q.dtype, device=q.device),
            )
        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=(None if attention_mask is None else attention_mask.view(1, 1, *attention_mask.shape)),
            dropout_p=0.0,
            is_causal=attention_mask is None,
            scale=self.scaling,
            enable_gqa=enable_gqa,
        )
        out = out.squeeze(0).transpose(0, 1).contiguous()
        return self.o_proj(out.flatten(1, -1)), new_key_value

    def forward_paged_tree(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        node_slots: torch.Tensor,
        logical_slots: torch.Tensor,
        qq_bias: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Reference-numeric QKV plus JetSpec-style paged tree attention."""
        weight = self.qkv_proj.weight
        bias = self.qkv_proj.bias
        q = F.linear(hidden_states, weight[:self.q_size], None if bias is None else bias[:self.q_size])
        k = F.linear(
            hidden_states,
            weight[self.q_size:self.q_size + self.kv_size],
            None if bias is None else bias[self.q_size:self.q_size + self.kv_size],
        )
        v = F.linear(
            hidden_states,
            weight[self.q_size + self.kv_size:],
            None if bias is None else bias[self.q_size + self.kv_size:],
        )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = _reference_rms_norm(q, self.q_norm)
            k = _reference_rms_norm(k, self.k_norm)
        cos, sin = self.rotary_emb.cos_sin_cache[positions].chunk(2, dim=-1)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

        def apply_hf_rope(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

        q, k = apply_hf_rope(q), apply_hf_rope(k)
        # The nano pool allocates 256-token blocks.  Present the same underlying
        # storage as 16-token sub-pages to the official JetSpec kernel, matching
        # its validated Qwen3 launch contract without copying any K/V bytes.
        kernel_block_size = 16
        k_pages = k_pool.view(-1, kernel_block_size, self.num_kv_heads, self.head_dim)
        v_pages = v_pool.view(-1, kernel_block_size, self.num_kv_heads, self.head_dim)
        blocks = torch.div(node_slots, kernel_block_size, rounding_mode="floor").long()
        offsets = torch.remainder(node_slots, kernel_block_size).long()
        k_pages[blocks, offsets] = k
        v_pages[blocks, offsets] = v

        from nanovllm.speculative.jetspec.paged_backend import paged_tree_attention

        n = int(q.shape[0])
        total = int(logical_slots.numel())
        table_width = (total + kernel_block_size - 1) // kernel_block_size
        block_table = torch.zeros((1, table_width), dtype=torch.int32, device=q.device)
        cu = torch.tensor([0, n], dtype=torch.int32, device=q.device)
        seq_lens = torch.tensor([total], dtype=torch.int32, device=q.device)
        starts = torch.zeros((1,), dtype=torch.int32, device=q.device)
        lens = torch.tensor([total], dtype=torch.int32, device=q.device)
        out = paged_tree_attention(
            q,
            k_pages,
            v_pages,
            block_table,
            cu,
            seq_lens,
            qq_bias,
            self.scaling,
            self.num_heads // self.num_kv_heads,
            kernel_block_size,
            logical_slots.view(1, -1),
            starts,
            lens,
        )
        return self.o_proj(out.flatten(1, -1))


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x

    def forward_dense(self, x):
        """HF-compatible split projections for the JetSpec correctness seam."""
        width = self.gate_up_proj.output_sizes[0]
        weight = self.gate_up_proj.weight
        bias = self.gate_up_proj.bias
        gate = F.linear(x, weight[:width], None if bias is None else bias[:width])
        up = F.linear(x, weight[width:], None if bias is None else bias[width:])
        return self.down_proj(F.silu(gate) * up)


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_dense(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Unlike nano's regular fused-residual path, keep the materialized BF16
        # residual stream at every layer to match the HF reference operation order.
        residual_stream = hidden_states
        normed = _reference_rms_norm(hidden_states, self.input_layernorm)
        attn_output, new_key_value = self.self_attn.forward_dense(
            positions, normed, past_key_value, attention_mask
        )
        hidden_states = residual_stream + attn_output
        residual_stream = hidden_states
        hidden_states = self.mlp.forward_dense(
            _reference_rms_norm(hidden_states, self.post_attention_layernorm)
        )
        hidden_states = residual_stream + hidden_states
        return hidden_states, None, new_key_value, hidden_states

    def forward_paged_tree(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        node_slots: torch.Tensor,
        logical_slots: torch.Tensor,
        qq_bias: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = _reference_rms_norm(hidden_states, self.input_layernorm)
        hidden_states = self.self_attn.forward_paged_tree(
            positions,
            normed,
            k_pool,
            v_pool,
            node_slots,
            logical_slots,
            qq_bias,
            block_size,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp.forward_dense(
            _reference_rms_norm(hidden_states, self.post_attention_layernorm)
        )
        return residual + hidden_states


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_dense(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None,
        attention_mask: torch.Tensor | None,
        target_layer_ids: list[int] | tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor | None]:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        new_key_values = []
        tapped = []
        tap_set = set(int(i) for i in target_layer_ids)
        for layer_id, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[layer_id]
            hidden_states, residual, new_kv, post_hidden = layer.forward_dense(
                positions, hidden_states, residual, past, attention_mask
            )
            new_key_values.append(new_kv)
            if layer_id in tap_set:
                tapped.append(post_hidden)
        hidden_states = _reference_rms_norm(hidden_states, self.norm)
        target_hidden = torch.cat(tapped, dim=-1) if tapped else None
        return hidden_states, new_key_values, target_hidden

    def forward_paged_tree(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_pool: torch.Tensor,
        node_slots: torch.Tensor,
        logical_slots: torch.Tensor,
        qq_bias: torch.Tensor,
        block_size: int,
        target_layer_ids: list[int] | tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = self.embed_tokens(input_ids)
        tapped = []
        tap_set = set(int(i) for i in target_layer_ids)
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer.forward_paged_tree(
                positions,
                hidden_states,
                kv_pool[0, layer_id],
                kv_pool[1, layer_id],
                node_slots,
                logical_slots,
                qq_bias,
                block_size,
            )
            if layer_id in tap_set:
                tapped.append(hidden_states)
        hidden_states = _reference_rms_norm(hidden_states, self.norm)
        target_hidden = torch.cat(tapped, dim=-1) if tapped else None
        return hidden_states, target_hidden


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
