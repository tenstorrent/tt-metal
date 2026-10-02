// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gate_up_matmul_decode_device_operation.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/shape.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <array>
#include <map>
#include <optional>
#include <vector>

namespace ttnn::operations::matmul_decode {

using namespace tt;
using namespace tt::tt_metal;

// Fused gate+up+GeGLU partial-width-sharded matmul: ONE gather of A, TWO weights, ONE output.
//
// Identical pipeline + geometry to MatmulDecodeDeviceOperation::PartialWidthSharded, but every
// core holds BOTH its gate_b block AND its up_b block (the two weights share the K/N split and the
// core grid), and the compute/writer stages run the partial-matmul + cross-core reduce TWICE (once
// per weight) over the SINGLE gathered A. The reader (A-gather) is shared -- A is gathered exactly
// once -- which halves the per-MLP x-gather and reduce/dispatch vs two separate matmul_decode
// calls. The compute kernel then multiplies the two reduced results (phase 3) so the op emits the
// single GeGLU activation hid = gelu(A @ gate_w) * (A @ up_w), folding the downstream eltwise
// multiply into the op. K_blocks is the standard 2 (pairwise reduce in the compute kernel).
ProgramDescriptor GateUpMatmulDecodeDeviceOperation::GateUpPartialWidthSharded::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& gate_b = tensor_args.gate_b;
    const auto& up_b = tensor_args.up_b;
    auto& hid_out = tensor_return_value;

    const tt::DataFormat in0_data_format = datatype_to_dataformat_converter(input_tensor_a.dtype());
    const tt::DataFormat in1_data_format = datatype_to_dataformat_converter(gate_b.dtype());
    const tt::DataFormat in1b_data_format = datatype_to_dataformat_converter(up_b.dtype());
    const tt::DataFormat out_data_format = datatype_to_dataformat_converter(hid_out.dtype());

    const auto& inputA_tile = input_tensor_a.tensor_spec().tile();
    const auto& inputB_tile = gate_b.tensor_spec().tile();
    const auto& inputBb_tile = up_b.tensor_spec().tile();
    const auto& output_tile = hid_out.tensor_spec().tile();
    const uint32_t in0_tile_size = inputA_tile.get_tile_size(in0_data_format);
    const uint32_t in1_tile_size = inputB_tile.get_tile_size(in1_data_format);
    const uint32_t in1b_tile_size = inputBb_tile.get_tile_size(in1b_data_format);
    const uint32_t out_tile_size = output_tile.get_tile_size(out_data_format);

    const TileDescriptor in0_tile_desc{inputA_tile};
    const TileDescriptor in1_tile_desc{inputB_tile};
    const TileDescriptor in1b_tile_desc{inputBb_tile};
    const TileDescriptor out_tile_desc{output_tile};

    const uint32_t inputA_tile_height = inputA_tile.get_height();
    const uint32_t inputA_tile_width = inputA_tile.get_width();
    const uint32_t inputB_tile_height = inputB_tile.get_height();
    const uint32_t inputB_tile_width = inputB_tile.get_width();
    const uint32_t output_tile_height = output_tile.get_height();
    const uint32_t output_tile_width = output_tile.get_width();

    TT_FATAL(
        inputA_tile_height == output_tile_height,
        "Input tensor A {} and output tile height {} must be equal",
        inputA_tile_height,
        output_tile_height);
    TT_FATAL(inputB_tile_height == tt::constants::TILE_HEIGHT, "Input tensor B tile height must be 32");
    TT_FATAL(inputA_tile_width == tt::constants::TILE_WIDTH, "Input tensor A tile width must be 32");
    TT_FATAL(inputB_tile_width == tt::constants::TILE_WIDTH, "Input tensor B tile width must be 32");
    TT_FATAL(output_tile_width == tt::constants::TILE_WIDTH, "Output tensor tile width must be 32");

    IDevice* device = input_tensor_a.device();

    const uint32_t M_tiles = div_up(operation_attributes.M, inputA_tile_height);
    const uint32_t K_tiles = div_up(operation_attributes.K, tt::constants::TILE_HEIGHT);
    const uint32_t N_tiles = div_up(operation_attributes.N, tt::constants::TILE_WIDTH);

    TT_FATAL(
        M_tiles <= 8,
        "gate_up_matmul_decode requires out_block_h (= M_tiles) <= 8 so it fits in DST, but got M_tiles={}",
        M_tiles);

    // ---- in0 (A) geometry: reshard_input is mandatory for this op. ----
    const uint32_t reshard_cores = operation_attributes.reshard_cores;
    const std::array<uint32_t, 2> inputA_shard_shape = {
        M_tiles * inputA_tile_height, operation_attributes.K / reshard_cores};
    const std::vector<CoreCoord> a_sender_cores = corerange_to_cores(
        tt::tt_metal::num_cores_to_corerangeset(reshard_cores, device->compute_with_storage_grid_size(), true),
        std::nullopt,
        true);
    std::vector<CoreRange> a_sender_ranges;
    a_sender_ranges.reserve(a_sender_cores.size());
    for (const auto& core : a_sender_cores) {
        a_sender_ranges.emplace_back(core, core);
    }
    const CoreRangeSet inputA_core_range_set(a_sender_ranges);
    TT_FATAL(
        inputA_shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Input tensor A shard width must be divisible by the tile width");
    const uint32_t inA_K_tiles_per_core = inputA_shard_shape[1] / tt::constants::TILE_WIDTH;

    // gate_b / up_b share geometry (validated): use gate_b's shard spec for both.
    const std::array<uint32_t, 2> inputB_shard_shape = gate_b.memory_config().shard_spec().value().shape;
    const uint32_t Kc = inputB_shard_shape[0];
    const uint32_t Nc = inputB_shard_shape[1];
    const uint32_t Kc_tiles = Kc / tt::constants::TILE_WIDTH;
    const uint32_t Nc_tiles = Nc / tt::constants::TILE_WIDTH;

    const auto inputB_core_range_set = gate_b.memory_config().shard_spec().value().grid;

    const uint32_t num_B_cores = inputB_core_range_set.num_cores();
    const uint32_t num_B_cores_along_N = N_tiles / Nc_tiles;
    TT_FATAL(num_B_cores % num_B_cores_along_N == 0, "num_B_cores must be divisible by num_B_cores_along_N");
    const uint32_t num_B_cores_along_K = num_B_cores / num_B_cores_along_N;
    const uint32_t K_blocks = K_tiles / Kc_tiles;
    TT_FATAL(num_B_cores_along_K == K_blocks, "num_B_cores_along_K must equal K_blocks");
    const uint32_t N_blocks = num_B_cores / K_blocks;

    const std::vector<CoreCoord> b_cores = corerange_to_cores(inputB_core_range_set, std::nullopt, true);
    std::vector<CoreRange> base_core_ranges;
    base_core_ranges.reserve(N_blocks);
    for (uint32_t i = 0; i < N_blocks; ++i) {
        base_core_ranges.emplace_back(b_cores[i], b_cores[i]);
    }
    const CoreRangeSet base_core_range_set(base_core_ranges);

    // hid_out is width-sharded over the N_blocks base cores; gate_out/up_out are internal scratch
    // on those same cores (phase-2 reduces into them, phase-3 multiplies them into hid_out).
    const auto output_core_range_set = hid_out.memory_config().shard_spec().value().grid;
    TT_FATAL(output_core_range_set.num_cores() == N_blocks, "Output must be sharded across N_blocks cores");

    const auto all_compute_cores = inputA_core_range_set.merge(inputB_core_range_set).merge(output_core_range_set);
    const auto all_compute_cores_with_bbox = tt::tt_metal::CoreRangeSet(all_compute_cores.bounding_box());

    ProgramDescriptor desc;

    // ---- Circular buffers ----
    constexpr uint32_t in0_cb_index = CBIndex::c_0;           // this core's A slice (gather source)
    constexpr uint32_t in1_cb_index = CBIndex::c_1;           // this core's gate_b block (resident)
    constexpr uint32_t gate_out_cb_index = CBIndex::c_2;      // reduced gate (gelu) scratch (base cores)
    constexpr uint32_t full_in0_cb_index = CBIndex::c_3;      // gathered full A
    constexpr uint32_t gate_partial_cb_index = CBIndex::c_4;  // this core's gate partial
    constexpr uint32_t gate_reduce_cb_index = CBIndex::c_5;   // gathered K_blocks gate partials (base)
    constexpr uint32_t in1b_cb_index = CBIndex::c_6;          // this core's up_b block (resident)
    constexpr uint32_t up_partial_cb_index = CBIndex::c_7;    // this core's up partial
    constexpr uint32_t up_reduce_cb_index = CBIndex::c_8;     // gathered K_blocks up partials (base)
    constexpr uint32_t up_out_cb_index = CBIndex::c_9;        // reduced up scratch (base cores)
    constexpr uint32_t hid_out_cb_index = CBIndex::c_10;      // gate_out * up_out -> output shard (base cores)

    const uint32_t block_num_tiles = M_tiles * Nc_tiles;

    // in0: reshard CB (reader NoC-reads A's K-slice into it before the multicast gather).
    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * inA_K_tiles_per_core * in0_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });
    // in1: resident gate_b block (buffer-backed).
    desc.cbs.push_back(CBDescriptor{
        .total_size = Kc_tiles * Nc_tiles * in1_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1_cb_index,
            .data_format = in1_data_format,
            .page_size = in1_tile_size,
            .tile = in1_tile_desc,
        }}},
        .buffer = gate_b.buffer(),
    });
    // in1b: resident up_b block (buffer-backed).
    desc.cbs.push_back(CBDescriptor{
        .total_size = Kc_tiles * Nc_tiles * in1b_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in1b_cb_index,
            .data_format = in1b_data_format,
            .page_size = in1b_tile_size,
            .tile = in1b_tile_desc,
        }}},
        .buffer = up_b.buffer(),
    });
    // gate_out / up_out: reduced-result scratch on the base cores (internal; phase-3 consumes them).
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = gate_out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = up_out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });
    // hid_out: GeGLU output shard = gate_out * up_out (base cores, buffer-backed).
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = output_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = hid_out_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
        .buffer = hid_out.buffer(),
    });
    // full_in0: gathered full A (multicast destination).
    desc.cbs.push_back(CBDescriptor{
        .total_size = M_tiles * K_tiles * in0_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = full_in0_cb_index,
            .data_format = in0_data_format,
            .page_size = in0_tile_size,
            .tile = in0_tile_desc,
        }}},
    });
    // gate_partial / up_partial: each core's matmul partial product (compute -> writer).
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = gate_partial_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = up_partial_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });
    // gate_reduce / up_reduce: gathered K_blocks partials on the base core (writer -> compute).
    // Allocated identically on every core so each sender can use its local write pointer as the
    // (matching) destination L1 address on the base core.
    desc.cbs.push_back(CBDescriptor{
        .total_size = K_blocks * block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = gate_reduce_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = K_blocks * block_num_tiles * out_tile_size,
        .core_ranges = all_compute_cores_with_bbox,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = up_reduce_cb_index,
            .data_format = out_data_format,
            .page_size = out_tile_size,
            .tile = out_tile_desc,
        }}},
    });

    // ---- Semaphores ----
    const uint32_t num_senders = inputA_core_range_set.num_cores();
    constexpr uint32_t gather_sem_id = 0;       // senders -> coordinator (A gather)
    constexpr uint32_t done_sem_id = 1;         // coordinator -> all (A gathered)
    constexpr uint32_t gate_reduce_sem_id = 2;  // partial-producers -> base (gate reduction)
    constexpr uint32_t up_reduce_sem_id = 3;    // partial-producers -> base (up reduction)
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = gather_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = done_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = gate_reduce_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});
    desc.semaphores.push_back(
        SemaphoreDescriptor{.id = up_reduce_sem_id, .core_ranges = all_compute_cores_with_bbox, .initial_value = 0});

    // ---- Reader kernel (A gather + publish BOTH resident weights) ----
    const CoreRange mcast_bbox = all_compute_cores_with_bbox.bounding_box();
    const CoreCoord coordinator_logical = mcast_bbox.start_coord;
    const CoreCoord mcast_start_phys = device->worker_core_from_logical_core(coordinator_logical);
    const CoreCoord mcast_end_phys = device->worker_core_from_logical_core(mcast_bbox.end_coord);
    const uint32_t num_receivers = all_compute_cores_with_bbox.num_cores();
    const uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;

    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
        in0_cb_index,
        full_in0_cb_index,
        shard_num_tiles,
        in0_tile_size,
        num_senders,
        num_receivers,
        static_cast<uint32_t>(mcast_start_phys.x),
        static_cast<uint32_t>(mcast_start_phys.y),
        static_cast<uint32_t>(mcast_end_phys.x),
        static_cast<uint32_t>(mcast_end_phys.y),
        gather_sem_id,
        done_sem_id,
        static_cast<uint32_t>(mcast_start_phys.x),
        static_cast<uint32_t>(mcast_start_phys.y),
        in1_cb_index,
        Kc_tiles * Nc_tiles,
        static_cast<uint32_t>(operation_attributes.reshard_input),
        M_tiles,
        inA_K_tiles_per_core,
        K_tiles,
        in1b_cb_index,  // 20: up_b resident CB (published alongside gate_b)
    };
    tt::tt_metal::TensorAccessorArgs(*input_tensor_a.buffer()).append_to(reader_compile_time_args);

    const std::vector<CoreCoord> sender_cores = corerange_to_cores(inputA_core_range_set, std::nullopt, true);
    std::map<CoreCoord, uint32_t> sender_id_by_core;
    for (uint32_t id = 0; id < sender_cores.size(); id++) {
        sender_id_by_core[sender_cores[id]] = id;
    }
    const std::vector<CoreCoord> all_reader_cores = corerange_to_cores(all_compute_cores_with_bbox, std::nullopt, true);

    {
        KernelDescriptor reader_kernel_desc;
        reader_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/dataflow/reader_gate_up_partial_width_sharded.cpp";
        reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        std::vector<CoreRange> ranges;
        ranges.reserve(all_reader_cores.size());
        for (const auto& core : all_reader_cores) {
            ranges.emplace_back(core, core);
        }
        reader_kernel_desc.core_ranges = CoreRangeSet(ranges);
        reader_kernel_desc.compile_time_args = reader_compile_time_args;
        reader_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::NOC_0,
        };
        reader_kernel_desc.runtime_args.reserve(all_reader_cores.size());
        for (const auto& core : all_reader_cores) {
            const auto it = sender_id_by_core.find(core);
            const bool is_sender = it != sender_id_by_core.end();
            const uint32_t sender_id = is_sender ? it->second : 0;
            const bool is_coordinator = (core == coordinator_logical);
            reader_kernel_desc.emplace_runtime_args(
                core,
                {static_cast<uint32_t>(is_sender),
                 sender_id,
                 static_cast<uint32_t>(is_coordinator),
                 input_tensor_a.buffer()});
        }
        desc.kernels.push_back(std::move(reader_kernel_desc));
    }

    // ---- Writer kernel (cross-core K-reduction for BOTH gate and up) ----
    std::vector<CoreRange> b_core_ranges;
    b_core_ranges.reserve(b_cores.size());
    for (const auto& core : b_cores) {
        b_core_ranges.emplace_back(core, core);
    }

    {
        KernelDescriptor writer_kernel_desc;
        writer_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/dataflow/writer_gate_up_partial_width_sharded.cpp";
        writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_kernel_desc.core_ranges = CoreRangeSet(b_core_ranges);
        writer_kernel_desc.compile_time_args = {
            gate_partial_cb_index,
            gate_reduce_cb_index,
            block_num_tiles,
            out_tile_size,
            K_blocks,
            gate_reduce_sem_id,
            up_partial_cb_index,
            up_reduce_cb_index,
            up_reduce_sem_id,
        };
        writer_kernel_desc.config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_1,
        };
        writer_kernel_desc.runtime_args.reserve(b_cores.size());
        for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
            const uint32_t k_idx = idx / N_blocks;
            const uint32_t n_idx = idx % N_blocks;
            const CoreCoord base_logical = b_cores[n_idx];
            const CoreCoord base_phys = device->worker_core_from_logical_core(base_logical);
            const bool is_base = (k_idx == 0);
            writer_kernel_desc.emplace_runtime_args(
                b_cores[idx],
                {k_idx,
                 static_cast<uint32_t>(base_phys.x),
                 static_cast<uint32_t>(base_phys.y),
                 static_cast<uint32_t>(is_base)});
        }
        desc.kernels.push_back(std::move(writer_kernel_desc));
    }

    // ---- Compute kernel (partial matmul + base-core reduction, for BOTH gate and up) ----
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul_decode/device/kernels/compute/compute_gate_up_partial_width_sharded.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = CoreRangeSet(b_core_ranges);
    compute_kernel_desc.compile_time_args = {
        M_tiles,
        K_tiles,
        Kc_tiles,
        Nc_tiles,
        K_blocks,
        inA_K_tiles_per_core,
        static_cast<uint32_t>(operation_attributes.fused_gelu_approx),  // 0=erf, 1=tanh-approx (gate only)
    };
    const auto ckc = ttnn::init_device_compute_kernel_config(
        input_tensor_a.device()->arch(),
        operation_attributes.compute_kernel_config,
        MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = ckc.math_fidelity,
        .fp32_dest_acc_en = ckc.fp32_dest_acc_en,
        .dst_full_sync_en = ckc.dst_full_sync_en,
        .math_approx_mode = ckc.math_approx_mode,
    };
    compute_kernel_desc.runtime_args.reserve(b_cores.size());
    for (uint32_t idx = 0; idx < b_cores.size(); idx++) {
        const uint32_t k_idx = idx / N_blocks;
        const bool is_base = (k_idx == 0);
        compute_kernel_desc.runtime_args.emplace_back(
            b_cores[idx], KernelDescriptor::CoreRuntimeArgs{k_idx, static_cast<uint32_t>(is_base)});
    }
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::operations::matmul_decode
