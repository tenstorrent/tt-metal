// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_lib_bindings.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/nlp_tms/nlp_tms.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt::tt_metal::detail
{
    void TensorModuleCustomAndBMMOPs( py::module & m_tensor)
    {
        // Custom Generic NLP TMs
        // This op should support arbitrary B and S divisible by 32 on DRAM; on L1, might error out due to space
        m_tensor.def("nlp_create_qkv_heads_falcon7b", &nlp_create_qkv_heads_falcon7b,
            py::arg().noconvert(), py::arg("output_mem_config") = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, R"doc(
            Shuffles [B, 1, S, 4672] fused qkv matrix into 3 heads with shapes [B, 71, S, 64], [B, 1, S, 64], and [B, 1, S, 64].
        )doc");
        m_tensor.def("nlp_kv_cache_load_slice", &nlp_kv_cache_load_slice,
            py::arg("input").noconvert(), py::arg("seq_len_start"), py::arg("seq_len_end"), R"doc(
            Unpad TT INTERLEAVED, TILE layout Tensor into a height sharded tensor. Typically used to unpad the KV cache from [B,n_heads,max_seq_length,head_dim] (or [n_heads,B,max_seq_length,head_dim]) into [B,n_heads,S,head_dim] (or [n_heads,B,S,head_dim]), where S = seq_len_end-seq_len_start. seq_len_start and seq_len_end are the start and end of the sequence length to unpad, and must be multiples of 32.
            Returns an output tensor that is height sharded on B x n_heads corees (note the B and n_heads dims are interchangeable), where each shard is [S, head_dim].
        )doc");
    }

}
