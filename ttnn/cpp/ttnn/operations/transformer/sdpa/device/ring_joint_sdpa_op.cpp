// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_joint_sdpa_op.hpp"

#include "ring_joint_sdpa_program_factory.hpp"
#include "ttnn/run_operation.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_op.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::transformer {

void RingJointScaledDotProductAttention::validate(const std::vector<Tensor>& input_tensors) const {
    TT_FATAL(
        input_tensors.size() == 8,
        "Must have 6 SDPA input tensors (Q, K, V, joint_Q, joint_K, joint_V) and 4 AllGather input tensors.");

    const auto& input_tensor_q = input_tensors.at(0);
    const auto& input_tensor_k = input_tensors.at(1);
    const auto& input_tensor_v = input_tensors.at(2);
    const auto& joint_tensor_q = input_tensors.at(3);
    const auto& joint_tensor_k = input_tensors.at(4);
    const auto& joint_tensor_v = input_tensors.at(5);
    const auto& persistent_output_buffer_k = input_tensors.at(6);
    const auto& persistent_output_buffer_v = input_tensors.at(7);

    const std::vector<Tensor> sdpa_input_tensors = {
        input_tensor_q,
        persistent_output_buffer_k,
        persistent_output_buffer_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v};
    const std::vector<Tensor> ring_gather_input_tensors = {
        input_tensor_k,
        input_tensor_v,
    };
    const std::vector<std::optional<Tensor>> ring_gather_output_tensors = {
        persistent_output_buffer_k,
        persistent_output_buffer_v,
    };

    this->all_gather_struct.validate_with_output_tensors(ring_gather_input_tensors, ring_gather_output_tensors);

    // Check that SDPA coregrid does not overlap with AllGather coregrid
    TT_FATAL(this->program_config.has_value(), "Program config must be provided");
    TT_FATAL(
        this->ccl_core_grid_offset.y >= this->program_config.value().compute_with_storage_grid_size.y,
        "SDPA coregrid overlaps with AllGather coregrid");

    // Validate joint strategy is 'rear'
    TT_FATAL(this->joint_strategy == "rear", "Joint strategy must be 'rear'. Got: {}", this->joint_strategy);

    // Validate all tensors have the same dtype
    const auto dtype = input_tensor_q.get_dtype();
    for (const auto& tensor : sdpa_input_tensors) {
        TT_FATAL(
            tensor.get_dtype() == dtype,
            "All tensors must have the same dtype. Expected {}, got {}",
            dtype,
            tensor.get_dtype());
    }

    // Get shapes
    const auto& q_shape = input_tensor_q.get_logical_shape();
    const auto& k_shape = persistent_output_buffer_k.get_logical_shape();
    const auto& v_shape = persistent_output_buffer_v.get_logical_shape();
    const auto& joint_q_shape = joint_tensor_q.get_logical_shape();
    const auto& joint_k_shape = joint_tensor_k.get_logical_shape();
    const auto& joint_v_shape = joint_tensor_v.get_logical_shape();

    // Validate storage types and buffers
    for (auto& tensor : sdpa_input_tensors) {
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to Joint SDPA need to be on device");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to Joint SDPA need to be allocated in buffers on device");
        TT_FATAL(tensor.get_layout() == Layout::TILE, "Inputs to Joint SDPA must be tilized");
        TT_FATAL(
            tensor.get_dtype() == DataType::BFLOAT16 || tensor.get_dtype() == DataType::BFLOAT8_B,
            "Inputs to Joint SDPA must be BF16 or BF8");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to Joint SDPA need to be in DRAM");
    }

    // Validate input shapes match
    const auto B = q_shape[0];
    const auto NQH = q_shape[1];
    const auto NKH = k_shape[1];
    const auto N_local = q_shape[2];
    const auto N_global = k_shape[2];
    const auto L = joint_q_shape[2];
    const auto DH = q_shape[3];

    TT_FATAL(
        k_shape[0] == B && v_shape[0] == B && joint_q_shape[0] == B && joint_k_shape[0] == B && joint_v_shape[0] == B,
        "Batch sizes must match. Got Q: {}, K: {}, V: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
        B,
        k_shape[0],
        v_shape[0],
        joint_q_shape[0],
        joint_k_shape[0],
        joint_v_shape[0]);

    // Validate head dimensions match
    TT_FATAL(
        k_shape[3] == DH && v_shape[3] == DH && joint_q_shape[3] == DH && joint_k_shape[3] == DH &&
            joint_v_shape[3] == DH,
        "Head dimensions must match. Got Q: {}, K: {}, V: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
        DH,
        k_shape[3],
        v_shape[3],
        joint_q_shape[3],
        joint_k_shape[3],
        joint_v_shape[3]);

    TT_FATAL(
        v_shape[1] == NKH && joint_q_shape[1] == NQH && joint_k_shape[1] == NKH && joint_v_shape[1] == NKH,
        "Num heads must match. Got Q: {}, K: {}, V: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
        NQH,
        NKH,
        v_shape[1],
        joint_q_shape[1],
        joint_k_shape[1],
        joint_v_shape[1]);

    TT_FATAL(
        v_shape[2] == N_global,
        "V sequence length must be equal to global sequence length. Got V: {}, global sequence length: {}",
        v_shape[2],
        N_global);

    TT_FATAL(
        N_global == N_local * this->ring_size,
        "Global sequence length must be equal to local sequence length times ring size. Got global sequence length: "
        "{}, local sequence length: {}, ring size: {}",
        N_global,
        N_local,
        this->ring_size);

    TT_FATAL(
        this->logical_n <= N_global,
        "Logical sequence length must be less than or equal to global sequence length. Got logical sequence length: "
        "{}, global sequence length: {}",
        this->logical_n,
        N_global);

    TT_FATAL(
        joint_k_shape[2] == L && joint_v_shape[2] == L,
        "Joint sequence length must match. Got joint_K: {}, joint_V: {}",
        joint_k_shape[2],
        joint_v_shape[2]);

    // Check shapes based on ring
    TT_FATAL(
        q_shape[2] * this->ring_size == k_shape[2],
        "Q sequence length times ring size must be equal to K sequence length. Got Q: {}, K: {}, ring_size: {}",
        q_shape[2],
        k_shape[2],
        this->ring_size);
    TT_FATAL(
        k_shape[2] == v_shape[2],
        "K sequence length must be equal to V sequence length. Got K: {}, V: {}",
        k_shape[2],
        v_shape[2]);

    TT_FATAL(NQH == NKH, "Q num_heads must be equal to K num_heads. Got Q: {}, K: {}", NQH, NKH);

    // Validate chunk sizes if program config is provided
    auto q_chunk_size = this->get_q_chunk_size();
    auto k_chunk_size = this->get_k_chunk_size();

    TT_FATAL(
        q_chunk_size % tt::constants::TILE_WIDTH == 0,
        "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
        q_chunk_size,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        k_chunk_size % tt::constants::TILE_WIDTH == 0,
        "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
        k_chunk_size,
        tt::constants::TILE_WIDTH);

    TT_FATAL(
        N_local % q_chunk_size == 0,
        "Local sequence length must be divisible by q_chunk_size. Got N_local: {}, q_chunk_size: {}",
        N_local,
        q_chunk_size);
    TT_FATAL(
        N_local % k_chunk_size == 0,
        "Local sequence length must be divisible by k_chunk_size. Got N_local: {}, k_chunk_size: {}",
        N_local,
        k_chunk_size);

    // Validate padding: Only the sequence dimension may be padded
    auto validate_padding = [](const Tensor& tensor) {
        const auto& logical_shape = tensor.get_logical_shape();
        const auto& padded_shape = tensor.get_padded_shape();
        TT_FATAL(logical_shape[0] == padded_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == padded_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == padded_shape[3], "Padding is not supported on the head_dim dimension");
    };

    for (const auto& tensor : sdpa_input_tensors) {
        validate_padding(tensor);
    }
}

std::uint32_t RingJointScaledDotProductAttention::get_q_chunk_size() const {
    return this->program_config ? this->program_config->q_chunk_size : 32;
}

std::uint32_t RingJointScaledDotProductAttention::get_k_chunk_size() const {
    return this->program_config ? this->program_config->k_chunk_size : 32;
}

std::vector<TensorSpec> RingJointScaledDotProductAttention::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    auto& input = input_tensors.at(0);
    auto& joint_input = input_tensors.at(3);
    auto lse_shape = input.get_logical_shape();
    lse_shape[3] = 1;
    lse_shape[2] = input.get_padded_shape()[2] + joint_input.get_padded_shape()[2];

    return {
        TensorSpec(
            input.get_logical_shape(), TensorLayout(input.get_dtype(), PageConfig(Layout::TILE), output_mem_config)),
        TensorSpec(
            joint_input.get_logical_shape(),
            TensorLayout(joint_input.get_dtype(), PageConfig(Layout::TILE), output_mem_config)),
        TensorSpec(lse_shape, TensorLayout(input.get_dtype(), PageConfig(Layout::TILE), output_mem_config))};
}

operation::MeshWorkloadWithCallbacks RingJointScaledDotProductAttention::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::Hash RingJointScaledDotProductAttention::compute_program_hash(
    const std::vector<Tensor>& input_tensors) const {
    return tt::tt_metal::operation::hash_operation<RingJointScaledDotProductAttention>(
        input_tensors,
        this->joint_strategy,
        this->scale,
        this->logical_n,
        this->ring_size,
        this->compute_kernel_config,
        this->program_config,
        this->ccl_core_grid_offset,
        this->all_gather_struct.compute_program_hash(
            {input_tensors.at(1), input_tensors.at(2)}) /*all_gather input tensors*/
    );
}

operation::ProgramWithCallbacks RingJointScaledDotProductAttention::create_program_at(
    const MeshCoordinate& coord, const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    log_debug(tt::LogOp, "DEBUG: create_program_at is called");
    auto mesh_device = input_tensors[0].mesh_device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(coord) : input_tensors[0].device();
    std::vector<IDevice*> devices_to_use = {};
    // User specified the cluster-axis. Derive devices based on the current coordinate
    // and the cluster-axis.
    const auto& mesh_view = input_tensors[0].mesh_device()->get_view();
    devices_to_use = (this->all_gather_struct.cluster_axis.value() == 0) ? mesh_view.get_devices_on_column(coord[1])
                                                                         : mesh_view.get_devices_on_row(coord[0]);

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < this->all_gather_struct.ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (this->all_gather_struct.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(this->all_gather_struct.ring_size - 1);
            }
            if (i != this->all_gather_struct.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (this->all_gather_struct.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& joint_tensor_q = input_tensors.at(3);
    auto& joint_tensor_k = input_tensors.at(4);
    auto& joint_tensor_v = input_tensors.at(5);
    auto& persistent_output_buffer_k = input_tensors.at(6);
    auto& persistent_output_buffer_v = input_tensors.at(7);
    auto& output_tensor = output_tensors.at(0);
    auto& joint_output_tensor = output_tensors.at(1);
    auto& lse_output_tensor = output_tensors.at(2);

    tt::tt_metal::Program program{};

    auto scale = this->scale;
    if (not scale.has_value()) {
        scale = 1.0f / std::sqrt(static_cast<float>(input_tensor_q.get_logical_shape()[-1]));
    }

    std::size_t q_chunk_size = this->get_q_chunk_size();
    std::size_t k_chunk_size = this->get_k_chunk_size();

    std::optional<detail::RingSDPAFusedOpSignaler> sdpa_fused_op_signaler = detail::RingSDPAFusedOpSignaler();

    auto [num_targets_forward, num_targets_backward, dynamic_alternate] = ccl::get_forward_backward_configuration(
        this->all_gather_struct.ring_size, device_index, this->all_gather_struct.topology);

    // This is how ring_joint_sdpa expects the number of forward and backward writes
    uint32_t forward_writes_expected, backward_writes_expected;
    if (this->all_gather_struct.topology == ttnn::ccl::Topology::Linear) {
        forward_writes_expected = num_targets_backward;
        backward_writes_expected = num_targets_forward;
    } else {
        TT_FATAL(this->all_gather_struct.topology == ttnn::ccl::Topology::Ring, "Topology must be Linear or Ring");
        forward_writes_expected = num_targets_forward - 1;
        backward_writes_expected = num_targets_backward - 1;
    }
    // Minimally use matmul fused op signaler
    sdpa_fused_op_signaler->init_all_gather(
        this->all_gather_struct.ring_size, device_index, forward_writes_expected, backward_writes_expected);

    auto ring_joint_sdpa_program = detail::ring_joint_sdpa(
        program,
        input_tensor_q,
        persistent_output_buffer_k,
        persistent_output_buffer_v,
        joint_tensor_q,
        joint_tensor_k,
        joint_tensor_v,
        output_tensor,
        joint_output_tensor,
        lse_output_tensor,
        this->logical_n,
        scale,
        q_chunk_size,
        k_chunk_size,
        this->ring_size,
        this->compute_kernel_config,
        this->program_config,
        sdpa_fused_op_signaler);

    const auto ring_attention_callback = ring_joint_sdpa_program.override_runtime_arguments_callback;

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> all_gather_fused_op_signaler =
        ttnn::experimental::ccl::AllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        sdpa_fused_op_signaler->fused_op_receiver_cores_noc,
        sdpa_fused_op_signaler->fused_op_receiver_signal_semaphores,
        sdpa_fused_op_signaler->fused_op_signaler_mode);

    std::vector<Tensor> all_gather_input_tensors = {
        input_tensor_k,
        input_tensor_v,
    };
    std::vector<Tensor> all_gather_output_tensors = {
        persistent_output_buffer_k,
        persistent_output_buffer_v,
    };
    auto all_gather_program = ring_attention_all_gather_async_multi_core_with_workers_helper(
        ring_joint_sdpa_program.program,  // Must pass ring_joint_sdpa's program
        all_gather_input_tensors,
        target_device,
        forward_device,
        backward_device,
        all_gather_output_tensors,
        this->all_gather_struct.dim,
        this->all_gather_struct.num_links,
        this->all_gather_struct.ring_size,
        device_index,
        this->all_gather_struct.topology,
        this->all_gather_struct.semaphore,
        this->all_gather_struct.sub_device_id,
        all_gather_fused_op_signaler,
        this->ccl_core_grid_offset);

    const auto all_gather_callback = all_gather_program.override_runtime_arguments_callback;

    auto override_runtime_args = [all_gather_callback, ring_attention_callback](
                                     const void* operation,
                                     Program& program,
                                     const std::vector<Tensor>& input_tensors,
                                     const std::vector<std::optional<const Tensor>>& optional_input_tensors,
                                     const std::vector<Tensor>& output_tensors) {
        auto& input_tensor_q = input_tensors.at(0);
        auto& input_tensor_k = input_tensors.at(1);
        auto& input_tensor_v = input_tensors.at(2);
        auto& joint_tensor_q = input_tensors.at(3);
        auto& joint_tensor_k = input_tensors.at(4);
        auto& joint_tensor_v = input_tensors.at(5);
        auto& persistent_output_buffer_k = input_tensors.at(6);
        auto& persistent_output_buffer_v = input_tensors.at(7);
        auto& output_tensor = output_tensors.at(0);
        auto& joint_output_tensor = output_tensors.at(1);
        auto& lse_output_tensor = output_tensors.at(2);

        const RingAttentionAllGatherAsync* all_gather_operation =
            &(static_cast<const RingJointScaledDotProductAttention*>(operation)->all_gather_struct);
        all_gather_callback.value()(
            all_gather_operation,
            program,
            {input_tensor_k, input_tensor_v},                        /*input_tensors*/
            {},                                                      /*optional_input_tensors*/
            {persistent_output_buffer_k, persistent_output_buffer_v} /*output_tensors*/
        );

        ring_attention_callback.value()(
            operation,
            program,
            {input_tensor_q,
             persistent_output_buffer_k,
             persistent_output_buffer_v,
             joint_tensor_q,
             joint_tensor_k,
             joint_tensor_v},                                       /*input_tensors*/
            {},                                                     /*optional_input_tensors*/
            {output_tensor, joint_output_tensor, lse_output_tensor} /*output_tensors*/
        );
    };

    all_gather_program.override_runtime_arguments_callback = override_runtime_args;
    return all_gather_program;
}

}  // namespace ttnn::operations::transformer
