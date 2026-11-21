// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "quasar_nanobind.hpp"

#include <nanobind/nanobind.h>

#include "ttnn/operations/experimental/quasar/pad/pad_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/tilize/tilize_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/move/move_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/ccl/all_gather/all_gather_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/untilize_with_unpadding/untilize_with_unpadding_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/slice/slice_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/transpose/transpose_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/reshard/reshard_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/pool_generic/generic_pools_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/conv2d/conv2d_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/matmul/matmul_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/binary/binary_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/fold/fold_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/to_memory_config/to_memory_config_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/untilize/untilize_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/tilize_with_val_padding/tilize_with_val_padding_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/reallocate/reallocate_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/to_device/to_device_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/padded_slice/padded_slice_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/slice_write/slice_write_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/typecast/typecast_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/sharded_to_interleaved/sharded_to_interleaved_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/interleaved_to_sharded/interleaved_to_sharded_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/softmax/softmax_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/batch_norm/batch_norm_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/transformer/transformer_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/adaptive_pool/adaptive_pools_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/cnn/convert_to_chw/convert_to_chw_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/cnn/convert_to_hwc/convert_to_hwc_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/conv1d/conv1d_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/conv3d/conv3d_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/conv_transpose2d/conv_transpose2d_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/attn_res_block_mix/attn_res_block_mix_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/attn_res_block_stats/attn_res_block_stats_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/attn_res_gather_softmax/attn_res_gather_softmax_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/attn_res_weighted_reduce_nc/attn_res_weighted_reduce_nc_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/combine/combine_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/combine_fabric2d/combine_fabric2d_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/compressor_state_exchange/compressor_state_exchange_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/csa_compressor/csa_compressor_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/dispatch/dispatch_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/dispatch_fabric2d/dispatch_fabric2d_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/extract/extract_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/hybrid_routed_expert_ffn/hybrid_routed_expert_ffn_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/inbound_socket_service_sync/inbound_socket_service_sync_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/insert/insert_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/masked_bincount/masked_bincount_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/mhc_split_sinkhorn/mhc_split_sinkhorn_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/moe_fused_swiglu/moe_fused_swiglu_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/moe_grouped_topk/moe_grouped_topk_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/moe_hash_gate/moe_hash_gate_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/moe_padding_config/moe_padding_config_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/offset_cumsum/offset_cumsum_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/outbound_socket_service_sync/outbound_socket_service_sync_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/pack_scaled_fp8_kv_cache/pack_scaled_fp8_kv_cache_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/per_token_cast_back/per_token_cast_back_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/per_token_cast_to_fp8/per_token_cast_to_fp8_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/post_combine_reduce/post_combine_reduce_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/rotary_embedding_indexed/rotary_embedding_indexed_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/routed_expert_ffn/routed_expert_ffn_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/unified_routed_expert_ffn/unified_routed_expert_ffn_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/update_padded_kv_cache/update_padded_kv_cache_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/deepseek_prefill/zero_padded_kv_cache/zero_padded_kv_cache_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/grid_sample/grid_sample_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/high_bw_all_gather/high_bw_all_gather_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/indexer_score/indexer_score_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/kda/kda_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/ring_attention_all_gather_async/ring_attention_all_gather_async_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/rotate/rotate_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/topk_large_indices/topk_large_indices_nanobind.hpp"
#include "ttnn/operations/experimental/quasar/upsample/upsample_nanobind.hpp"

namespace ttnn::operations::experimental::quasar {

void bind_quasar(nb::module_& mod) {
    auto m_quasar = mod.def_submodule("quasar", "Quasar (metal 2.0) operations");

    // Data-movement ops (host namespace ttnn::operations::experimental::quasar::detail).
    detail::bind_pad(m_quasar);
    detail::bind_tilize(m_quasar);
    detail::bind_move(m_quasar);
    detail::bind_all_gather(m_quasar);
    detail::bind_untilize_with_unpadding(m_quasar);
    detail::bind_slice(m_quasar);
    detail::bind_slice_descriptor(m_quasar);
    detail::bind_transpose(m_quasar);
    detail::bind_reshard(m_quasar);

    // conv2d.
    detail::bind_conv2d(m_quasar);

    // pool (host namespace ttnn::operations::pool::quasar).
    ttnn::operations::pool::quasar::py_module(m_quasar);

    // matmul (its own py_module binds matmul/linear/addmm/sparse_matmul + program-config classes).
    matmul::py_module(m_quasar);

    // binary front-end (add/subtract/multiply/... -> quasar binary_ng device op).
    binary::py_module(m_quasar);

    // fold (compositional data-movement op).
    detail::bind_fold_operation(m_quasar);

    // to_memory_config (dispatches to quasar reshard / interleaved_to_sharded / sharded_to_interleaved).
    detail::bind_to_memory_config(m_quasar);

    // reshape (enum exported first so it is available as a python default-arg type).
    bind_reshape_enum(m_quasar);
    bind_reshape_view(m_quasar);

    // untilize.
    detail::bind_untilize(m_quasar);

    // tilize_with_val_padding / tilize_with_zero_padding.
    detail::bind_tilize_with_val_padding(m_quasar);
    detail::bind_tilize_with_zero_padding(m_quasar);

    // to_layout (composite; dispatches to quasar tilize/untilize/tilize_with_val_padding/...).
    detail::bind_to_layout(m_quasar);

    // reallocate (thin wrapper over quasar move).
    detail::bind_reallocate(m_quasar);

    // to_device (thin host->device transfer wrapper).
    detail::bind_to_device(m_quasar);

    // padded_slice + slice_write (RM Metal-2 ports; the DRAM-slicing read/write-back used by conv2d on Quasar).
    detail::bind_padded_slice(m_quasar);
    detail::bind_slice_write(m_quasar);

    // typecast (dtype conversion).
    detail::bind_typecast(m_quasar);

    // sharded_to_interleaved / interleaved_to_sharded (standalone; to_memory_config also dispatches to these).
    detail::bind_sharded_to_interleaved(m_quasar);
    detail::bind_interleaved_to_sharded(m_quasar);

    // normalization (Quasar-native softmax last-dim, batch_norm inference).
    detail::bind_softmax(m_quasar);
    detail::bind_batch_norm(m_quasar);

    // transformer (nested ttnn.experimental.quasar.transformer submodule; sdpa_decode for now).
    transformer::bind_transformer(m_quasar);

    // ---- Quasar copies of the convolutions / sdpa / ds_prefill codeowner ops ----
    // conv1d / conv_transpose2d (compositional wrappers over the quasar conv2d / halo / move).
    conv1d_impl::bind_conv1d(m_quasar);
    conv_transpose2d_impl::bind_conv_transpose2d(m_quasar);

    // pool family (grid_sample / rotate / upsample).
    grid_sample::bind_grid_sample(m_quasar);
    rotate::py_module(m_quasar);
    upsample::py_module(m_quasar);

    // ring_attention_all_gather_async.
    ring_attention_all_gather_async::bind_ring_attention_all_gather_async(m_quasar);

    // experimental ops (cnn, conv3d, adaptive_pool, high_bw_all_gather, topk_large_indices, indexer_score, kda,
    // deepseek_prefill).
    cnn::detail::bind_convert_to_chw(m_quasar);
    cnn::detail::bind_convert_to_hwc(m_quasar);
    conv3d::detail::bind_conv3d(m_quasar);
    adaptive_pool::bind_adaptive_avg_pool2d_operation(m_quasar);
    adaptive_pool::bind_adaptive_max_pool2d_operation(m_quasar);
    deepseek_prefill::masked_bincount::detail::bind_experimental_masked_bincount_operation(m_quasar);
    high_bw_all_gather::detail::bind_experimental_high_bw_all_gather_operation(m_quasar);
    deepseek_prefill::combine_fabric2d::detail::bind_experimental_combine_fabric2d_operation(m_quasar);
    deepseek_prefill::dispatch_fabric2d::detail::bind_experimental_dispatch_fabric2d_operation(m_quasar);
    deepseek_prefill::offset_cumsum::detail::bind_experimental_offset_cumsum_operation(m_quasar);
    deepseek_prefill::moe_padding_config::detail::bind_moe_padding_config(m_quasar);
    deepseek_prefill::detail::bind_outbound_socket_service_sync(m_quasar);
    deepseek_prefill::mhc_split_sinkhorn::detail::bind_experimental_mhc_split_sinkhorn_operation(m_quasar);
    deepseek_prefill::detail::bind_post_combine_reduce(m_quasar);
    deepseek_prefill::moe_grouped_topk::detail::bind_moe_grouped_topk(m_quasar);
    deepseek_prefill::moe_hash_gate::detail::bind_moe_hash_gate(m_quasar);
    deepseek_prefill::attn_res_block_mix::detail::bind_attn_res_block_mix(m_quasar);
    deepseek_prefill::attn_res_block_stats::detail::bind_attn_res_block_stats(m_quasar);
    deepseek_prefill::attn_res_gather_softmax::detail::bind_attn_res_gather_softmax(m_quasar);
    deepseek_prefill::attn_res_weighted_reduce_nc::detail::bind_attn_res_weighted_reduce_nc(m_quasar);
    deepseek_prefill::pack_scaled_fp8_kv_cache::detail::bind_pack_scaled_fp8_kv_cache(m_quasar);
    deepseek_prefill::per_token_cast_to_fp8::detail::bind_experimental_per_token_cast_to_fp8_operation(m_quasar);
    deepseek_prefill::per_token_cast_back::detail::bind_experimental_per_token_cast_back_operation(m_quasar);
    deepseek_prefill::detail::bind_inbound_socket_service_sync(m_quasar);
    topk_large_indices::detail::bind_topk_large_indices(m_quasar);
    indexer_score::detail::bind_indexer_score(m_quasar);
    kda::detail::bind_kda(m_quasar);
    deepseek_prefill::detail::bind_compressor_state_exchange(m_quasar);
    deepseek_prefill::detail::bind_csa_compressor(m_quasar);
    deepseek_prefill::detail::bind_dispatch(m_quasar);
    deepseek_prefill::detail::bind_combine(m_quasar);
    deepseek_prefill::detail::bind_routed_expert_ffn(m_quasar);
    deepseek_prefill::detail::bind_unified_routed_expert_ffn(m_quasar);
    deepseek_prefill::detail::bind_moe_fused_swiglu(m_quasar);
    deepseek_prefill::detail::bind_hybrid_routed_expert_ffn(m_quasar);
    deepseek_prefill::detail::bind_extract(m_quasar);
    deepseek_prefill::detail::bind_insert(m_quasar);
    deepseek_prefill::detail::bind_update_padded_kv_cache(m_quasar);
    deepseek_prefill::detail::bind_zero_padded_kv_cache(m_quasar);
    deepseek_prefill::detail::bind_rotary_embedding_indexed(m_quasar);

    // NOTE: halo has no python binding (internal device backend). The binary_ng device op is exposed through the
    // binary front-end (binary::py_module -> add/subtract/multiply/...), not a direct binary_ng binding.
}

}  // namespace ttnn::operations::experimental::quasar
