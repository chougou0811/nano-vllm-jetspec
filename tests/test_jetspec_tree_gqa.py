"""Small CPU launch-policy checks; GPU sanity lives in the three-shape probe."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_jetspec_tree_dot import FakeTensor, Launch
from nanovllm.speculative.jetspec import tree_gqa, target_graph


class GroupedTreeLaunchTests(unittest.TestCase):
    def test_grid_is_total_q_based_not_capture_time_max_ragged_count(self):
        # Deliberately no node_counts_host: device cu/counts drive ragged mapping.
        meta=SimpleNamespace(total_queries=141,num_requests=3,block_size=256,
            prefix_lens=None,node_counts=None,cu_seqlens_q=None,tree_slots=None,
            qq_bias=None,qq_bias_offsets=None,block_tables=FakeTensor((3,16)))
        q,k,v=FakeTensor((141,32,128)),FakeTensor((20,256,8,128)),FakeTensor((20,256,8,128))
        k.stride=lambda:(256*8*128,8*128,128,1)
        v.stride=lambda:(256*8*128,8*128,128,1)
        kernel=Launch()
        with patch.object(tree_gqa,'_validate',return_value=q),patch.object(tree_gqa,'_packed_tree_gqa_dot',kernel):
            self.assertIs(tree_gqa.packed_tree_attention_gqa(q,k,v,meta,.125,4),q)
        self.assertEqual(kernel.grid,(38,8))
        for key,value in dict(M=16,Q_TILE=4,GROUPS=4,TILE=64,PRUNE=True,PREFIX_SCALAR=True,FP32_P=False).items():
            self.assertEqual(kernel.kwargs[key],value)

    def test_gqa_graph_key_does_not_alias_legacy_dispatch(self):
        meta=SimpleNamespace(total_queries=141,prefix_lengths=(0,255,257),qq_bias=SimpleNamespace(numel=lambda:7149))
        self.assertNotEqual(target_graph.graph_signature(meta,prefix_backend=False),
                            target_graph.graph_signature(meta,prefix_backend=False,gqa_backend=True))

    def test_runtime_new_policy_keeps_rope_and_invalidates_old_graph(self):
        from nanovllm.speculative.jetspec.batch_runtime import JetSpecBatchRuntime
        from unittest.mock import Mock
        runtime=object.__new__(JetSpecBatchRuntime)
        runtime._closed=False;runtime._active_transaction=None
        runtime.requests={};runtime.prefills={};runtime._target_kernels='fused_rope'
        graph=Mock();runtime._target_graph=graph
        layer=SimpleNamespace(self_attn=SimpleNamespace())
        runtime.target=SimpleNamespace(model=SimpleNamespace(layers=[layer]))
        runtime.configure_optimizations(target_kernels='fused_rope_gqa',target_execution='cuda_graph')
        graph.close.assert_called_once()
        self.assertIsNone(runtime._target_graph)
        self.assertTrue(layer.self_attn._jetspec_tree_gqa)
        self.assertTrue(layer.self_attn._jetspec_tree_fusion)


if __name__=='__main__':
    unittest.main()
