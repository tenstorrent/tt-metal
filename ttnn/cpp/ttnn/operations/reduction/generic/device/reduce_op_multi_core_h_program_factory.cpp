// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core H reduction program factory, ported to the Metal 2.0 host API.
// Reduces along H; work split is by columns (NC * Wt work units). Width-
// sharded inputs and outputs use borrowed-memory DataflowBuffers — the DFB's
// L1 storage is shared with the tensor's shard buffer via
// DataflowBufferSpec::borrowed_from.

#include "reduce_op_device_operation.hpp"
#include "reduce_metal2_factory_helpers.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>

namespace ttnn::prim {

namespace m2 = tt::tt_metal::experimental::metal2_host_api;

namespace {

constexpr const char* H_READER_KERNEL = "reduce_h_reader";
constexpr const char* H_WRITER_KERNEL = "reduce_h_writer";
constexpr const char* H_COMPUTE_KERNEL_G1 = "reduce_h_compute_g1";
constexpr const char* H_COMPUTE_KERNEL_G2 = "reduce_h_compute_g2";
constexpr const char* H_WORK_UNIT_G1 = "reduce_h_wu_g1";
constexpr const char* H_WORK_UNIT_G2 = "reduce_h_wu_g2";

}  // namespace

ttnn::device_operation::ProgramArtifacts ReduceDeviceOperation::ReduceMultiCoreHProgramFactory::create_program_spec(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    const auto& a = tensor_args;
    auto& output = tensor_return_value;
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();
    const uint32_t tile_hw = a.tensor_spec().tile().get_tile_hw();

    uint32_t Wt = W / tile_width;
    uint32_t Ht = H / tile_height;
    uint32_t HtWt = Ht * Wt;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device()->arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    tt_metal::IDevice* device = a.device();

    bool use_width_sharding = a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                              output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    uint32_t chunk_size = use_width_sharding ? 1 : ttnn::get_dest_reg_count(operation_attributes.compute_kernel_config);

    // For min/max with non-unity scalar, the GMPOOL hardware path only respects the scaler's
    // exponent, so the device reduces with scaler=1.0 and the user scalar is applied after the
    // reduction via SFPU mul_unary_tile inside the compute kernel.
    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_cols = NC * Wt;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cols_per_core_group_1, num_cols_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_cols);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_cols);
    }
    TT_FATAL(num_cores > 0, "Reduce H requires at least one worker core");

    // Current sharding only supports width, and that input and output are sharded
    if (use_width_sharding) {
        all_cores = a.shard_spec().value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_cols_per_core_group_1 = NC * (a.shard_spec().value().shape[1] / tile_width);
        num_cols_per_core_group_2 = 0;
    }

    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    // Packed fp32 scalar passed to the compute kernel for mul_unary_tile post-reduction scaling.
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // ---- DFBs ----
    std::vector<m2::DataflowBufferSpec> dataflow_buffers;

    // c_0 — input DFB (private; reader fills from tensor)
    {
        uint32_t num_input_tiles = operation_attributes.negate ? chunk_size : 2;
        if (use_width_sharding) {
            num_input_tiles = 2;  // legacy used 2 tiles in the sharded path
        }
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = IN_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = src0_cb_data_format,
            .tile_format_metadata = a.tensor_spec().tile(),
        });
    }

    // c_1 — only width-sharded: input-shard DFB (borrowed memory)
    if (use_width_sharding) {
        uint32_t num_shard_tiles = a.shard_spec().value().numel() / tile_hw;
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = IN_SHARD_DFB,
            .entry_size = src0_single_tile_size,
            .num_entries = num_shard_tiles,
            .data_format_metadata = src0_cb_data_format,
            .tile_format_metadata = a.tensor_spec().tile(),
            .borrowed_from = INPUT_TENSOR,
        });
    }

    // c_2 — scaler DFB
    dataflow_buffers.push_back(m2::DataflowBufferSpec{
        .unique_id = SCALER_DFB,
        .entry_size = scaler_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scaler_cb_data_format,
        .tile_format_metadata = a.tensor_spec().tile(),
    });

    // c_3 — output DFB (private interleaved; borrowed for width-sharded)
    {
        uint32_t num_output_tiles;
        std::optional<std::string> borrowed_from_output;
        if (use_width_sharding) {
            num_output_tiles = output.shard_spec().value().numel() / tile_hw;
            borrowed_from_output = OUTPUT_TENSOR;
        } else {
            num_output_tiles = operation_attributes.negate ? chunk_size : 2;
        }
        m2::DataflowBufferSpec out_spec{
            .unique_id = OUT_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = num_output_tiles,
            .data_format_metadata = dst_cb_data_format,
            .tile_format_metadata = output.tensor_spec().tile(),
        };
        if (borrowed_from_output.has_value()) {
            out_spec.borrowed_from = *borrowed_from_output;
        }
        dataflow_buffers.push_back(std::move(out_spec));
    }

    if (operation_attributes.negate) {
        // c_4 (acc) and c_5 (ineg): per nc, the kernel advances wr_ptr by
        // Ht * Wt_per_core regardless of how that splits into chunk_size and
        // partial pushes, so sizing the CB at Ht * Wt_per_core makes the
        // trajectory land on fifo_limit exactly. For two core groups, the
        // single-CB option uses Ht * lcm(Wt_g1, Wt_g2) so the same allocation
        // works for both groups.
        const uint32_t compute_Wt_g1 =
            use_width_sharding ? (NC == 0 ? 0 : num_cols_per_core_group_1 / NC) : num_cols_per_core_group_1;
        const uint32_t compute_Wt_g2 = use_width_sharding ? 0 : num_cols_per_core_group_2;
        uint32_t per_nc_advance = 0;
        if (compute_Wt_g2 == 0) {
            per_nc_advance = compute_Wt_g1;
        } else if (compute_Wt_g1 == 0) {
            per_nc_advance = compute_Wt_g2;
        } else {
            per_nc_advance = std::lcm(compute_Wt_g1, compute_Wt_g2);
        }
        TT_FATAL(
            per_nc_advance > 0,
            "Negate H reduce: per-core Wt resolved to 0 (g1={}, g2={}, NC={}, sharded={})",
            compute_Wt_g1,
            compute_Wt_g2,
            NC,
            use_width_sharding);
        // L1 fit check, mirroring the legacy ProgramDescriptor version.
        const uint64_t negate_cb_tiles = static_cast<uint64_t>(Ht) * per_nc_advance;
        const uint64_t per_cb_bytes = negate_cb_tiles * dst_single_tile_size;
        const uint64_t negate_cb_bytes = 2ull * per_cb_bytes;
        const auto lowest_address = device->lowest_occupied_compute_l1_address();
        uint64_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
        const uint64_t base_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        TT_FATAL(
            max_l1_space > base_addr,
            "Negate H reduce: L1 base allocator address {} >= lowest occupied address {}; no room for CBs",
            base_addr,
            max_l1_space);
        max_l1_space -= base_addr;
        TT_FATAL(
            negate_cb_bytes <= max_l1_space,
            "Negate H reduce: cb_acc + cb_ineg ({} B for {} tiles) would not fit in available L1 ({} B). "
            "Caller must use h_reduce_negate_fits_in_l1 to choose the external-negate fallback.",
            negate_cb_bytes,
            negate_cb_tiles,
            max_l1_space);
        TT_FATAL(
            per_cb_bytes <= std::numeric_limits<uint32_t>::max(),
            "Negate H reduce: per-CB size {} B exceeds uint32_t total_size range",
            per_cb_bytes);
        const uint32_t per_cb_num_entries = static_cast<uint32_t>(negate_cb_tiles);

        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = ACC_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = per_cb_num_entries,
            .data_format_metadata = dst_cb_data_format,
            .tile_format_metadata = output.tensor_spec().tile(),
        });
        dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = INEG_DFB,
            .entry_size = dst_single_tile_size,
            .num_entries = per_cb_num_entries,
            .data_format_metadata = dst_cb_data_format,
            .tile_format_metadata = output.tensor_spec().tile(),
        });
    }

    std::map<std::string, std::string> reduce_defines_map =
        reduce_op_utils::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::H);
    if (use_post_mul) {
        reduce_defines_map["REDUCE_POST_MUL"] = "1";
    }
    // Read kernel-side reduce defines (carried through to all kernels).
    auto reduce_defines = DefinesFromMap(reduce_defines_map);

    // ---- Reader kernel ----
    m2::KernelSpec reader;
    reader.unique_id = H_READER_KERNEL;
    if (use_width_sharding) {
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp"};
        // Sharded reader CTAs: scaler_bits (the other three CB-index CTAs in
        // the legacy plumbing are replaced by DFB bindings).
        reader.compile_time_arg_bindings = {{"scaler_bits", scaler_bits}};
        reader.runtime_arguments_schema.named_runtime_args = {
            "num_tiles", "shard_Wt", "Ht", "batch", "row_size_bytes", "batch_size_bytes"};
        auto reader_defines_map = reduce_defines_map;
        reader_defines_map["REDUCE_SCALER"] = "1";
        reader_defines_map["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines_map["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader.compiler_options.defines = DefinesFromMap(reader_defines_map);
        // Reader binds the input-shard DFB as PRODUCER (its writes feed the
        // compute kernel's read path) plus the scaler DFB.
        reader.dfb_bindings = {
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = IN_DFB,
                .local_accessor_name = "in_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            },
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = IN_SHARD_DFB,
                .local_accessor_name = "in_shard_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            },
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = SCALER_DFB,
                .local_accessor_name = "scaler_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            },
        };
        // Sharded reader does NOT use TensorAccessor — it reads from the borrowed-memory
        // DFB. The TensorBinding is still needed at the program level (for the
        // borrowed-memory DFB's runtime address resolution), but not on this kernel.
    } else {
        reader.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/"
            "reader_unary_transpose_wh_universal_input_cols_partitioned.cpp"};
        reader.compile_time_arg_bindings = {
            {"Ht", Ht}, {"Wt", Wt}, {"HtWt", HtWt}, {"scaler_bits", scaler_bits}, {"use_welford", 0u}};
        reader.runtime_arguments_schema.named_runtime_args = {
            "src_addr", "col_start_tile_id", "curr_col_in_batch", "num_cols"};
        auto reader_defines_map = reduce_defines_map;
        reader_defines_map["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
        reader_defines_map["DST_SYNC_FULL"] = dst_full_sync_en ? "1" : "0";
        reader.compiler_options.defines = DefinesFromMap(reader_defines_map);
        reader.dfb_bindings = {
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = IN_DFB,
                .local_accessor_name = "in_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            },
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = SCALER_DFB,
                .local_accessor_name = "scaler_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            },
        };
        reader.tensor_bindings = {
            m2::KernelSpec::TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "input"},
        };
    }
    reader.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
            },
    };

    // ---- Writer kernel ----
    m2::KernelSpec writer;
    writer.unique_id = H_WRITER_KERNEL;
    if (use_width_sharding) {
        writer.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp"};
        writer.runtime_arguments_schema.named_runtime_args = {"num_units"};
    } else {
        writer.source = m2::KernelSpec::SourceFilePath{
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp"};
        writer.runtime_arguments_schema.named_runtime_args = {"num_pages", "start_id"};
    }
    writer.config_spec = m2::DataMovementConfiguration{
        .gen1_data_movement_config =
            m2::DataMovementConfiguration::Gen1DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
            },
    };
    writer.dfb_bindings = {
        m2::KernelSpec::DFBBinding{
            .dfb_spec_name = OUT_DFB,
            .local_accessor_name = "out_dfb",
            .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
        },
    };
    if (!use_width_sharding) {
        writer.tensor_bindings = {
            m2::KernelSpec::TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "output"},
        };
    }

    // ---- Compute kernels ----
    const std::string compute_kernel_path =
        operation_attributes.negate
            ? "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h_neg.cpp"
            : "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce.cpp";

    // For width-sharding, num_cols_per_core_group_1 == NC * shard_Wt. Expose (shard_Wt, NC)
    // to the compute kernel so its (nc, wt_chunk, ht, wt_in_chunk) iteration matches the
    // reader's per-batch tile layout.
    auto make_compute = [&](const char* unique_id, uint32_t cols_per_group) {
        const uint32_t compute_Wt = use_width_sharding ? (cols_per_group / NC) : cols_per_group;
        const uint32_t compute_NC = use_width_sharding ? NC : 1;

        m2::KernelSpec compute;
        compute.unique_id = unique_id;
        compute.source = m2::KernelSpec::SourceFilePath{compute_kernel_path};
        compute.compile_time_arg_bindings = {
            {"Ht", Ht},
            {"Wt", compute_Wt},
            {"NC", compute_NC},
            {"post_mul_scaler_bits", post_mul_scaler_bits},
        };
        compute.compiler_options.defines = reduce_defines;
        compute.config_spec = m2::ComputeConfiguration{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
        };
        compute.dfb_bindings = {
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = IN_DFB,
                .local_accessor_name = "in_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            },
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = SCALER_DFB,
                .local_accessor_name = "scaler_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            },
            m2::KernelSpec::DFBBinding{
                .dfb_spec_name = OUT_DFB,
                .local_accessor_name = "out_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            },
        };
        if (operation_attributes.negate) {
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = ACC_DFB,
                .local_accessor_name = "acc_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            });
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = ACC_DFB,
                .local_accessor_name = "acc_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            });
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = INEG_DFB,
                .local_accessor_name = "ineg_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::PRODUCER,
            });
            compute.dfb_bindings.push_back(m2::KernelSpec::DFBBinding{
                .dfb_spec_name = INEG_DFB,
                .local_accessor_name = "ineg_dfb",
                .endpoint_type = m2::KernelSpec::DFBEndpointType::CONSUMER,
            });
        }
        return compute;
    };

    std::vector<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute(H_COMPUTE_KERNEL_G1, num_cols_per_core_group_1));
    const bool g2_present = !core_group_2.ranges().empty();
    if (g2_present) {
        kernels.push_back(make_compute(H_COMPUTE_KERNEL_G2, num_cols_per_core_group_2));
    }

    // ---- WorkUnits ----
    std::vector<m2::WorkUnitSpec> work_units;
    work_units.push_back(m2::WorkUnitSpec{
        .unique_id = H_WORK_UNIT_G1,
        .kernels = {H_READER_KERNEL, H_WRITER_KERNEL, H_COMPUTE_KERNEL_G1},
        .target_nodes = core_group_1,
    });
    if (g2_present) {
        work_units.push_back(m2::WorkUnitSpec{
            .unique_id = H_WORK_UNIT_G2,
            .kernels = {H_READER_KERNEL, H_WRITER_KERNEL, H_COMPUTE_KERNEL_G2},
            .target_nodes = core_group_2,
        });
    }

    // ---- Assemble spec ----
    m2::ProgramSpec spec;
    spec.program_id = "ttnn::reduce_multi_core_h";
    spec.kernels = std::move(kernels);
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = INPUT_TENSOR, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()},
    };
    spec.work_units = std::move(work_units);

    // ---- Run params: per-core RTAs ----
    std::vector<CoreCoord> cores;
    if (operation_attributes.sub_core_grids.has_value()) {
        for (const auto& range : all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    cores.emplace_back(x, y);
                }
            }
        }
    } else {
        cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }
    TT_FATAL(
        cores.size() == num_cores, "Resolved core list size {} must match split num_cores {}", cores.size(), num_cores);

    m2::ProgramRunParams::KernelRunParams reader_params{.kernel_spec_name = H_READER_KERNEL};
    m2::ProgramRunParams::KernelRunParams writer_params{.kernel_spec_name = H_WRITER_KERNEL};

    if (use_width_sharding) {
        TT_FATAL(NC != 0, "Batch size NC must be non-zero (shape[0]={}, shape[1]={})", shape[0], shape[1]);
        uint32_t shard_Wt = num_cols_per_core_group_1 / NC;
        uint32_t shard_row_size = shard_Wt * src0_single_tile_size;
        uint32_t shard_batch_size = shard_row_size * Ht;
        // Width-sharded path: iterate the actual shard core set (all_cores), not the
        // grid_to_cores sequence — sharded grids may not start at (0,0).
        for (const auto& range : all_cores.ranges()) {
            for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    CoreCoord core{x, y};
                    reader_params.named_runtime_args.push_back({
                        .node = core,
                        .args =
                            {{"num_tiles", num_cols_per_core_group_1 * Ht},
                             {"shard_Wt", shard_Wt},
                             {"Ht", Ht},
                             {"batch", NC},
                             {"row_size_bytes", shard_row_size},
                             {"batch_size_bytes", shard_batch_size}},
                    });
                    writer_params.named_runtime_args.push_back({
                        .node = core,
                        .args = {{"num_units", num_cols_per_core_group_1}},
                    });
                }
            }
        }
    } else {
        TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);
        for (uint32_t i = 0, num_cols_read = 0; i < num_cores; i++) {
            const CoreCoord& core = cores[i];
            uint32_t num_cols_per_core = 0;
            if (core_group_1.contains(core)) {
                num_cols_per_core = num_cols_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_cols_per_core = num_cols_per_core_group_2;
            } else {
                TT_THROW("Core not in specified core ranges");
            }
            reader_params.named_runtime_args.push_back({
                .node = core,
                .args =
                    {{"src_addr", 0u},  // unused: TensorBinding auto-injects the buffer address
                     {"col_start_tile_id", (num_cols_read / Wt * HtWt) + (num_cols_read % Wt)},
                     {"curr_col_in_batch", num_cols_read % Wt},
                     {"num_cols", num_cols_per_core}},
            });
            writer_params.named_runtime_args.push_back({
                .node = core,
                .args = {{"num_pages", num_cols_per_core}, {"start_id", num_cols_read}},
            });
            num_cols_read += num_cols_per_core;
            if (i == num_cores - 1) {
                TT_FATAL(
                    num_cols_read == num_cols,
                    "Reduce H assigned {} columns across cores, expected {}",
                    num_cols_read,
                    num_cols);
            }
        }
    }

    m2::ProgramRunParams run_params;
    run_params.kernel_run_params.push_back(std::move(reader_params));
    run_params.kernel_run_params.push_back(std::move(writer_params));
    run_params.tensor_args = {
        m2::ProgramRunParams::TensorArg{.tensor_parameter_name = INPUT_TENSOR, .tensor = std::cref(a.mesh_tensor())},
        m2::ProgramRunParams::TensorArg{
            .tensor_parameter_name = OUTPUT_TENSOR, .tensor = std::cref(output.mesh_tensor())},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_params)};
}

}  // namespace ttnn::prim
