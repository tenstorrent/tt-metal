// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_subblock_utils.hpp"

#include <bit>
#include <climits>
#include <cmath>
#include <filesystem>
#include <string>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

// Metal 2.0 port of joint SDPA. The host-side shape/parallelization math is unchanged from the
// legacy create_descriptor; only the resource declaration is expressed as a ProgramSpec.
ttnn::device_operation::ProgramArtifacts JointSDPADeviceOperation::JointSDPAProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& args = operation_attributes;
    auto& output_tensors = tensor_return_value;

    // Metal 2.0 named resource handles (declared as LOCALS for unity-build hygiene).
    const DFBSpecName Q_DFB{"q_in"};
    const DFBSpecName K_DFB{"k_in"};
    const DFBSpecName V_DFB{"v_in"};
    const DFBSpecName MASK_DFB{"mask"};
    const DFBSpecName SCALE_DFB{"scale"};
    const DFBSpecName COL_IDENTITY_DFB{"col_identity"};
    const DFBSpecName QK_IM_DFB{"qk_im"};
    const DFBSpecName OUT_IM_A_DFB{"out_im_a"};
    const DFBSpecName OUT_IM_B_DFB{"out_im_b"};
    const DFBSpecName MAX_A_DFB{"max_a"};
    const DFBSpecName MAX_B_DFB{"max_b"};
    const DFBSpecName SUM_A_DFB{"sum_a"};
    const DFBSpecName SUM_B_DFB{"sum_b"};
    const DFBSpecName EXP_MAX_DIFF_DFB{"exp_max_diff"};
    const DFBSpecName OUT_DFB{"out"};

    const TensorParamName Q_IN_TENSOR{"q_in_tensor"};
    const TensorParamName K_IN_TENSOR{"k_in_tensor"};
    const TensorParamName V_IN_TENSOR{"v_in_tensor"};
    const TensorParamName JOINT_Q_TENSOR{"joint_q_tensor"};
    const TensorParamName JOINT_K_TENSOR{"joint_k_tensor"};
    const TensorParamName JOINT_V_TENSOR{"joint_v_tensor"};
    const TensorParamName OUT_TENSOR{"out_tensor"};
    const TensorParamName JOINT_OUT_TENSOR{"joint_out_tensor"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    const Tensor& input_tensor_q = tensor_args.input_q;
    const Tensor& input_tensor_k = tensor_args.input_k;
    const Tensor& input_tensor_v = tensor_args.input_v;
    const Tensor& joint_tensor_q = tensor_args.joint_q;
    const Tensor& joint_tensor_k = tensor_args.joint_k;
    const Tensor& joint_tensor_v = tensor_args.joint_v;
    const Tensor& output_tensor = output_tensors[JOINT_SDPA_OUTPUT_IDX];
    const Tensor& joint_output_tensor = output_tensors[JOINT_SDPA_JOINT_OUTPUT_IDX];

    std::size_t q_chunk_size = args.get_q_chunk_size();
    std::size_t k_chunk_size = args.get_k_chunk_size();

    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& joint_q_shape = joint_tensor_q.logical_shape();
    const uint32_t B = q_shape[0], NH = q_shape[1], N = q_shape[2], DH = q_shape[3];
    const uint32_t L = joint_q_shape[2];

    // Calculate padded sequence length
    const uint32_t padded_Nq = tt::round_up(N, q_chunk_size);
    const uint32_t padded_Nk = tt::round_up(N, k_chunk_size);
    const uint32_t padded_Lq = tt::round_up(L, q_chunk_size);
    const uint32_t padded_Lk = tt::round_up(L, k_chunk_size);

    const uint32_t padded_Nqt = padded_Nq / TILE_HEIGHT;
    const uint32_t padded_Nkt = padded_Nk / TILE_HEIGHT;
    const uint32_t padded_Lqt = padded_Lq / TILE_HEIGHT;
    const uint32_t padded_Lkt = padded_Lk / TILE_HEIGHT;

    // Find unpadded sequence lengths in tiles
    const uint32_t valid_Nt = tt::div_up(N, TILE_HEIGHT);
    const uint32_t valid_Lt = tt::div_up(L, TILE_HEIGHT);

    // Compute kernel operates on concatenated Q and K
    const uint32_t cat_Sq = padded_Nq + padded_Lq;
    const uint32_t cat_Sk = padded_Nk + padded_Lk;

    const uint32_t cat_Skt = cat_Sk / TILE_HEIGHT;
    const uint32_t DHt = DH / TILE_WIDTH;

    /*
    For non-causal case we must provide a padded mask if the K sequence length has been padded
    Note that we dont have this issue in non-causal case if Q is padded, since those pad tokens
    don't affect attention of unpadded tokens.
    In causal case, the causal mask takes care of masking K pad tokens.
    */
    const bool use_joint_mask = (padded_Nk != N) || (padded_Lk != L);

    const uint32_t Sq_chunk_t = q_chunk_size / TILE_HEIGHT;
    const uint32_t Sk_chunk_t = k_chunk_size / TILE_HEIGHT;
    const uint32_t q_num_chunks = cat_Sq / q_chunk_size;
    const uint32_t k_num_chunks = cat_Sk / k_chunk_size;

    IDevice* device = input_tensor_q.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);

    CoreCoord grid_size = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                          : device->compute_with_storage_grid_size();
    bool exp_approx_mode =
        args.program_config.has_value()
            ? (args.program_config->exp_approx_mode.has_value() ? args.program_config->exp_approx_mode.value() : true)
            : true;

    auto core_grid = CoreRangeSet(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));
    uint32_t num_cores = grid_size.x * grid_size.y;

    TT_FATAL(
        num_cores <= device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y,
        "Provided grid must not contain more cores than the device. Got {} cores, expected at most {} cores.",
        num_cores,
        device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y);

    // Parallelization scheme
    // We will choose parallelization factors for batch, num_heads, and q_seq_len in that order
    uint32_t batch_parallel_factor = std::min(B, num_cores);
    uint32_t nh_parallel_factor = std::min(num_cores / batch_parallel_factor, NH);
    uint32_t q_parallel_factor = std::min(num_cores / (batch_parallel_factor * nh_parallel_factor), q_num_chunks);

    TT_FATAL(
        batch_parallel_factor * nh_parallel_factor * q_parallel_factor <= num_cores,
        "Parallelism must not exceed number of cores. Got {}, expected at most {}.",
        batch_parallel_factor * nh_parallel_factor * q_parallel_factor,
        num_cores);

    // Ceiling divide to allow for non-perfect divisions
    const uint32_t batch_per_core = tt::div_up(B, batch_parallel_factor);
    const uint32_t nh_per_core = tt::div_up(NH, nh_parallel_factor);
    const uint32_t q_per_core = tt::div_up(q_num_chunks, q_parallel_factor);

    const uint32_t q_buffer_factor = (q_per_core > 1) ? 2 : 1;

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t q_tiles = Sq_chunk_t * DHt * q_buffer_factor;
    uint32_t k_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t v_tiles = Sk_chunk_t * DHt * 2;  // double buffer
    uint32_t mask_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t qk_tiles = Sq_chunk_t * Sk_chunk_t;
    uint32_t out_im_tiles = Sq_chunk_t * DHt;
    uint32_t out0_t = Sq_chunk_t * DHt;
    uint32_t scale_tiles = 1;
    uint32_t statistics_tiles = Sq_chunk_t;  // Single column of values in each iteration

    // Host code is responsible for determining matmul configuration
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    const uint32_t qk_in0_block_w = DHt;
    auto [qk_out_subblock_h, qk_out_subblock_w] =
        detail::determine_largest_subblock_size(Sq_chunk_t, Sk_chunk_t, dst_size);

    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_out_subblock_w;
    const uint32_t qk_num_blocks = DHt / qk_in0_block_w;

    // now for out0
    const uint32_t out_in0_block_w = Sk_chunk_t;

    auto [out_out_subblock_h, out_out_subblock_w] = detail::determine_largest_subblock_size(Sq_chunk_t, DHt, dst_size);

    const uint32_t out_in0_num_subblocks = Sq_chunk_t / out_out_subblock_h;
    const uint32_t out_in1_num_subblocks = DHt / out_out_subblock_w;
    const uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    // Determine granularity for statistics computation
    // Each granularity must evenly divide its tile count to avoid dropping tiles
    const uint32_t stats_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size);
    const uint32_t sub_exp_granularity = detail::find_valid_granularity(Sk_chunk_t, dst_size);
    const uint32_t mul_bcast_granularity = detail::find_valid_granularity(Sq_chunk_t * Sk_chunk_t, dst_size);
    const uint32_t dht_granularity = detail::find_valid_granularity(DHt, dst_size);
    const uint32_t reduce_granularity = detail::find_valid_granularity(Sq_chunk_t, dst_size / 2);

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    class bfloat16 bfloat_identity_scalar(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});

    const uint32_t scale_packed = std::bit_cast<uint32_t>(args.scale);

    // Calculate which K chunks contain the mask boundaries
    // If a tensor does not require masking, set to MAX_UINT32. This avoids a
    // bug in the mask generation code, which would mask a full, valid chunk
    // with -inf.
    const uint32_t mask_chunk_0 =
        (padded_Nk != N) ? (padded_Nkt / Sk_chunk_t) - 1 : UINT32_MAX;  // idx of last chunk in first sequence
    const uint32_t mask_chunk_1 =
        (padded_Lk != L) ? (cat_Skt / Sk_chunk_t) - 1 : UINT32_MAX;  // idx of last chunk in second sequence

    // ---- Data formats / tile sizes ----
    tt::DataFormat q_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_q.dtype());
    tt::DataFormat k_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_k.dtype());
    tt::DataFormat v_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_v.dtype());
    tt::DataFormat mask_df = tt::DataFormat::Bfp4_b;
    tt::DataFormat out_df = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    tt::DataFormat scalar_df =
        (input_tensor_q.dtype() == DataType::FLOAT32) ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;  // need to disable fp32 cbs (Issue #13364)
    tt::DataFormat stats_df = im_df;

    uint32_t q_tile_size = tt::tile_size(q_df);
    uint32_t k_tile_size = tt::tile_size(k_df);
    uint32_t v_tile_size = tt::tile_size(v_df);
    uint32_t mask_tile_size = tt::tile_size(mask_df);
    uint32_t out_tile_size = tt::tile_size(out_df);
    uint32_t scalar_tile_size = tt::tile_size(scalar_df);
    uint32_t im_tile_size = tt::tile_size(im_df);
    uint32_t stats_tile_size = tt::tile_size(stats_df);

    // ---- Dataflow buffers (one per legacy CB) ----
    // q/k/v: reader -> compute. mask/scale/col_identity: writer -> compute. qk_im/out_im/stats:
    // compute INTRA self-loops. out: compute -> writer. None are borrowed (all are
    // interleaved-DRAM-fed via TensorAccessor, or program-local scratch).
    DataflowBufferSpec q_dfb_spec{
        .unique_id = Q_DFB, .entry_size = q_tile_size, .num_entries = q_tiles, .data_format_metadata = q_df};
    DataflowBufferSpec k_dfb_spec{
        .unique_id = K_DFB, .entry_size = k_tile_size, .num_entries = k_tiles, .data_format_metadata = k_df};
    DataflowBufferSpec v_dfb_spec{
        .unique_id = V_DFB, .entry_size = v_tile_size, .num_entries = v_tiles, .data_format_metadata = v_df};
    // The mask DFB is declared unconditionally so the kernels' dfb::mask handle always resolves;
    // when use_joint_mask is false no FIFO op ever touches it (gated by the use_joint_mask CTA).
    DataflowBufferSpec mask_dfb_spec{
        .unique_id = MASK_DFB, .entry_size = mask_tile_size, .num_entries = mask_tiles, .data_format_metadata = mask_df};
    DataflowBufferSpec scale_dfb_spec{
        .unique_id = SCALE_DFB,
        .entry_size = scalar_tile_size,
        .num_entries = scale_tiles,
        .data_format_metadata = scalar_df};
    DataflowBufferSpec col_identity_dfb_spec{
        .unique_id = COL_IDENTITY_DFB,
        .entry_size = scalar_tile_size,
        .num_entries = scale_tiles,
        .data_format_metadata = scalar_df};
    DataflowBufferSpec qk_im_dfb_spec{
        .unique_id = QK_IM_DFB, .entry_size = im_tile_size, .num_entries = qk_tiles, .data_format_metadata = im_df};
    DataflowBufferSpec out_im_a_dfb_spec{
        .unique_id = OUT_IM_A_DFB,
        .entry_size = im_tile_size,
        .num_entries = out_im_tiles,
        .data_format_metadata = im_df};
    DataflowBufferSpec out_im_b_dfb_spec{
        .unique_id = OUT_IM_B_DFB,
        .entry_size = im_tile_size,
        .num_entries = out_im_tiles,
        .data_format_metadata = im_df};
    DataflowBufferSpec max_a_dfb_spec{
        .unique_id = MAX_A_DFB,
        .entry_size = stats_tile_size,
        .num_entries = statistics_tiles,
        .data_format_metadata = stats_df};
    DataflowBufferSpec max_b_dfb_spec{
        .unique_id = MAX_B_DFB,
        .entry_size = stats_tile_size,
        .num_entries = statistics_tiles,
        .data_format_metadata = stats_df};
    DataflowBufferSpec sum_a_dfb_spec{
        .unique_id = SUM_A_DFB,
        .entry_size = stats_tile_size,
        .num_entries = statistics_tiles,
        .data_format_metadata = stats_df};
    DataflowBufferSpec sum_b_dfb_spec{
        .unique_id = SUM_B_DFB,
        .entry_size = stats_tile_size,
        .num_entries = statistics_tiles,
        .data_format_metadata = stats_df};
    DataflowBufferSpec exp_max_diff_dfb_spec{
        .unique_id = EXP_MAX_DIFF_DFB,
        .entry_size = stats_tile_size,
        .num_entries = statistics_tiles,
        .data_format_metadata = stats_df};
    DataflowBufferSpec out_dfb_spec{
        .unique_id = OUT_DFB, .entry_size = out_tile_size, .num_entries = out0_t, .data_format_metadata = out_df};

    // ---- Tensor parameters (6 inputs read by reader, 2 outputs written by writer) ----
    TensorParameter q_in_param{.unique_id = Q_IN_TENSOR, .spec = input_tensor_q.tensor_spec()};
    TensorParameter k_in_param{.unique_id = K_IN_TENSOR, .spec = input_tensor_k.tensor_spec()};
    TensorParameter v_in_param{.unique_id = V_IN_TENSOR, .spec = input_tensor_v.tensor_spec()};
    TensorParameter joint_q_param{.unique_id = JOINT_Q_TENSOR, .spec = joint_tensor_q.tensor_spec()};
    TensorParameter joint_k_param{.unique_id = JOINT_K_TENSOR, .spec = joint_tensor_k.tensor_spec()};
    TensorParameter joint_v_param{.unique_id = JOINT_V_TENSOR, .spec = joint_tensor_v.tensor_spec()};
    TensorParameter out_param{.unique_id = OUT_TENSOR, .spec = output_tensor.tensor_spec()};
    TensorParameter joint_out_param{.unique_id = JOINT_OUT_TENSOR, .spec = joint_output_tensor.tensor_spec()};

    // ---- Kernel defines (granularities + exp approx) applied to all three kernels ----
    Table<std::string, std::string> defines;
    defines.insert({"STATS_GRANULARITY", std::to_string(stats_granularity)});
    defines.insert({"SUB_EXP_GRANULARITY", std::to_string(sub_exp_granularity)});
    defines.insert({"MUL_BCAST_GRANULARITY", std::to_string(mul_bcast_granularity)});
    defines.insert({"DHT_GRANULARITY", std::to_string(dht_granularity)});
    defines.insert({"REDUCE_GRANULARITY", std::to_string(reduce_granularity)});
    defines.insert({"EXP_APPROX_MODE", std::to_string(exp_approx_mode)});

    // ---- Reader: q/k/v + joint q/k/v from DRAM (TensorAccessor) into c_0/c_1/c_2 ----
    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/joint_reader.cpp"},
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = Q_DFB, .accessor_name = "q_in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = K_DFB, .accessor_name = "k_in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = V_DFB, .accessor_name = "v_in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = Q_IN_TENSOR, .accessor_name = "q"},
             TensorBinding{.tensor_parameter_name = K_IN_TENSOR, .accessor_name = "k"},
             TensorBinding{.tensor_parameter_name = V_IN_TENSOR, .accessor_name = "v"},
             TensorBinding{.tensor_parameter_name = JOINT_Q_TENSOR, .accessor_name = "joint_q"},
             TensorBinding{.tensor_parameter_name = JOINT_K_TENSOR, .accessor_name = "joint_k"},
             TensorBinding{.tensor_parameter_name = JOINT_V_TENSOR, .accessor_name = "joint_v"}},
        .compile_time_args =
            {{"B", B},
             {"NH", NH},
             {"DHt", DHt},
             {"Sq_chunk_t", Sq_chunk_t},
             {"Sk_chunk_t", Sk_chunk_t},
             {"k_num_chunks", k_num_chunks},
             {"valid_Nt", valid_Nt},
             {"valid_Lt", valid_Lt},
             {"padded_Nqt", padded_Nqt},
             {"padded_Nkt", padded_Nkt},
             {"padded_Lqt", padded_Lqt},
             {"padded_Lkt", padded_Lkt},
             {"num_cores", num_cores}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"local_batch_start",
                  "local_batch_end",
                  "local_nh_start",
                  "local_nh_end",
                  "local_q_start",
                  "local_q_end"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // ---- Writer: generates mask/scale/col_identity into compute, drains c_16 to DRAM ----
    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/joint_writer.cpp"},
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = MASK_DFB, .accessor_name = "mask", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SCALE_DFB, .accessor_name = "scale", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = COL_IDENTITY_DFB,
                 .accessor_name = "col_identity",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "out"},
             TensorBinding{.tensor_parameter_name = JOINT_OUT_TENSOR, .accessor_name = "joint_out"}},
        .compile_time_args =
            {{"B", B},
             {"NH", NH},
             {"DHt", DHt},
             {"Sq_chunk_t", Sq_chunk_t},
             {"Sk_chunk_t", Sk_chunk_t},
             {"k_num_chunks", k_num_chunks},
             {"valid_Nt", valid_Nt},
             {"valid_Lt", valid_Lt},
             {"padded_Nqt", padded_Nqt},
             {"padded_Nkt", padded_Nkt},
             {"padded_Lqt", padded_Lqt},
             {"padded_Lkt", padded_Lkt},
             {"unpadded_N", N},
             {"unpadded_L", L},
             {"num_cores", num_cores},
             {"identity_scalar_packed", packed_identity_scalar},
             {"scale_val", scale_packed},
             {"use_joint_mask", static_cast<uint32_t>(use_joint_mask)},
             {"mask_chunk_0", mask_chunk_0},
             {"mask_chunk_1", mask_chunk_1}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"local_batch_start",
                  "local_batch_end",
                  "local_nh_start",
                  "local_nh_end",
                  "local_q_start",
                  "local_q_end"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // ---- Compute: flash-attention; consumes q/k/v/mask/scale/col_identity, produces out ----
    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/joint_sdpa.cpp"},
        .compiler_options = {.defines = defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = Q_DFB, .accessor_name = "q_in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = K_DFB, .accessor_name = "k_in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = V_DFB, .accessor_name = "v_in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = MASK_DFB, .accessor_name = "mask", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = SCALE_DFB, .accessor_name = "scale", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = COL_IDENTITY_DFB,
                 .accessor_name = "col_identity",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = QK_IM_DFB, .accessor_name = "qk_im", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = QK_IM_DFB, .accessor_name = "qk_im", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUT_IM_A_DFB, .accessor_name = "out_im_a", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = OUT_IM_A_DFB, .accessor_name = "out_im_a", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUT_IM_B_DFB, .accessor_name = "out_im_b", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = OUT_IM_B_DFB, .accessor_name = "out_im_b", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = MAX_A_DFB, .accessor_name = "max_a", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = MAX_A_DFB, .accessor_name = "max_a", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = MAX_B_DFB, .accessor_name = "max_b", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = MAX_B_DFB, .accessor_name = "max_b", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = SUM_A_DFB, .accessor_name = "sum_a", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = SUM_A_DFB, .accessor_name = "sum_a", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = SUM_B_DFB, .accessor_name = "sum_b", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = SUM_B_DFB, .accessor_name = "sum_b", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = EXP_MAX_DIFF_DFB,
                 .accessor_name = "exp_max_diff",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = EXP_MAX_DIFF_DFB,
                 .accessor_name = "exp_max_diff",
                 .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"B", B},
             {"NH", NH},
             {"Skt", cat_Skt},
             {"DHt", DHt},
             {"Sq_chunk_t", Sq_chunk_t},
             {"Sk_chunk_t", Sk_chunk_t},
             {"k_num_chunks", k_num_chunks},
             {"qk_in0_block_w", qk_in0_block_w},
             {"qk_subblock_w", qk_out_subblock_w},
             {"qk_subblock_h", qk_out_subblock_h},
             {"qk_in0_num_subblocks", qk_in0_num_subblocks},
             {"qk_in1_num_subblocks", qk_in1_num_subblocks},
             {"qk_num_blocks", qk_num_blocks},
             {"out_in0_block_w", out_in0_block_w},
             {"out_subblock_w", out_out_subblock_w},
             {"out_subblock_h", out_out_subblock_h},
             {"out_in0_num_subblocks", out_in0_num_subblocks},
             {"out_in1_num_subblocks", out_in1_num_subblocks},
             {"out_num_blocks", out_num_blocks},
             {"use_joint_mask", static_cast<uint32_t>(use_joint_mask)},
             {"mask_chunk_0", mask_chunk_0},
             {"mask_chunk_1", mask_chunk_1},
             {"scale_fp32", scale_packed}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"local_batch_start",
                  "local_batch_end",
                  "local_nh_start",
                  "local_nh_end",
                  "local_q_start",
                  "local_q_end"}},
        .hw_config =
            ComputeHardwareConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .dst_full_sync_en = dst_full_sync_en,
                .math_approx_mode = math_approx_mode,
            },
    };
    // Flash-attention intermediates + running statistics are produced and consumed within the
    // compute kernel: INTRA self-loops (legal for compute kernels).
    compute_spec.advanced_options.dfb_self_loop_connectivities = {
        {QK_IM_DFB, DFBSelfLoopConnectivity::INTRA},
        {OUT_IM_A_DFB, DFBSelfLoopConnectivity::INTRA},
        {OUT_IM_B_DFB, DFBSelfLoopConnectivity::INTRA},
        {MAX_A_DFB, DFBSelfLoopConnectivity::INTRA},
        {MAX_B_DFB, DFBSelfLoopConnectivity::INTRA},
        {SUM_A_DFB, DFBSelfLoopConnectivity::INTRA},
        {SUM_B_DFB, DFBSelfLoopConnectivity::INTRA},
        {EXP_MAX_DIFF_DFB, DFBSelfLoopConnectivity::INTRA},
    };

    // ---- Per-core runtime args: {local_batch/nh/q start,end} (same on all three kernels) ----
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};
    reader_run.runtime_arg_values.reserve(num_cores);
    writer_run.runtime_arg_values.reserve(num_cores);
    compute_run.runtime_arg_values.reserve(num_cores);

    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};

        uint32_t local_batch_start = (i / (nh_parallel_factor * q_parallel_factor)) * batch_per_core;
        uint32_t local_batch_end = local_batch_start + batch_per_core;
        uint32_t local_nh_start = ((i / q_parallel_factor) % nh_parallel_factor) * nh_per_core;
        uint32_t local_nh_end = local_nh_start + nh_per_core;
        uint32_t local_q_start = (i % q_parallel_factor) * q_per_core;
        uint32_t local_q_end = local_q_start + q_per_core;

        // clamp all to max values for non-even partitioning
        local_batch_start = std::min(local_batch_start, B);
        local_batch_end = std::min(local_batch_end, B);
        local_nh_start = std::min(local_nh_start, NH);
        local_nh_end = std::min(local_nh_end, NH);
        local_q_start = std::min(local_q_start, q_num_chunks);
        local_q_end = std::min(local_q_end, q_num_chunks);

        KernelRunArgs::RuntimeArgValues vals{
            {"local_batch_start", local_batch_start},
            {"local_batch_end", local_batch_end},
            {"local_nh_start", local_nh_start},
            {"local_nh_end", local_nh_end},
            {"local_q_start", local_q_start},
            {"local_q_end", local_q_end}};
        reader_run.runtime_arg_values.push_back({core, vals});
        writer_run.runtime_arg_values.push_back({core, vals});
        compute_run.runtime_arg_values.push_back({core, vals});
    }

    WorkUnitSpec wu{
        .name = "joint_sdpa",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = core_grid,
    };

    ProgramSpec spec{
        .name = "joint_sdpa",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers =
            {q_dfb_spec,
             k_dfb_spec,
             v_dfb_spec,
             mask_dfb_spec,
             scale_dfb_spec,
             col_identity_dfb_spec,
             qk_im_dfb_spec,
             out_im_a_dfb_spec,
             out_im_b_dfb_spec,
             max_a_dfb_spec,
             max_b_dfb_spec,
             sum_a_dfb_spec,
             sum_b_dfb_spec,
             exp_max_diff_dfb_spec,
             out_dfb_spec},
        .tensor_parameters =
            {q_in_param,
             k_in_param,
             v_in_param,
             joint_q_param,
             joint_k_param,
             joint_v_param,
             out_param,
             joint_out_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {reader_run, writer_run, compute_run};
    run_args.tensor_args = {
        {Q_IN_TENSOR, TensorArgument{std::cref(input_tensor_q.mesh_tensor())}},
        {K_IN_TENSOR, TensorArgument{std::cref(input_tensor_k.mesh_tensor())}},
        {V_IN_TENSOR, TensorArgument{std::cref(input_tensor_v.mesh_tensor())}},
        {JOINT_Q_TENSOR, TensorArgument{std::cref(joint_tensor_q.mesh_tensor())}},
        {JOINT_K_TENSOR, TensorArgument{std::cref(joint_tensor_k.mesh_tensor())}},
        {JOINT_V_TENSOR, TensorArgument{std::cref(joint_tensor_v.mesh_tensor())}},
        {OUT_TENSOR, TensorArgument{std::cref(output_tensor.mesh_tensor())}},
        {JOINT_OUT_TENSOR, TensorArgument{std::cref(joint_output_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
