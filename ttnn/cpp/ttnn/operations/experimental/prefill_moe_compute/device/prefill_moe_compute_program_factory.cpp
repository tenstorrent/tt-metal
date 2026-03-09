// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_moe_compute_program_factory.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "tt_metal/api/tt-metalium/buffer.hpp"
#include "tt_metal/api/tt-metalium/core_coord.hpp"
#include "tt_metal/api/tt-metalium/device.hpp"
#include "tt_metal/api/tt-metalium/mesh_device.hpp"
#include "tt_metal/api/tt-metalium/program.hpp"
#include <tt-metalium/kernel_types.hpp>

// Fabric includes (used when enable_fabric_reduce or enable_fabric_dispatch is true)
#include <tt-metalium/experimental/fabric/fabric.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::prefill_moe_compute {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr uint32_t TILE_HW = 32;
static constexpr uint32_t BF16_TILE_BYTES = 2 * 1024;  // 32x32 bf16 = 2 KB
static constexpr uint32_t BFP4_TILE_BYTES = 576;       // BFP4_b tile size (512 data + 64 exponent bytes)

// ---------------------------------------------------------------------------
// Kernel paths (relative to tt-metal root)
// ---------------------------------------------------------------------------
static const std::string READER_KERNEL =
    "tests/ttnn/unit_tests/operations/experimental/prefill_moe/kernels/expert_reader_multi.cpp";
static const std::string WRITER_KERNEL =
    "tests/ttnn/unit_tests/operations/experimental/prefill_moe/kernels/expert_writer_multi.cpp";
static const std::string COMPUTE_KERNEL =
    "tests/ttnn/unit_tests/operations/experimental/prefill_moe/kernels/compute_expert_multi.cpp";
static const std::string DISPATCH_KERNEL =
    "tests/ttnn/unit_tests/operations/experimental/prefill_moe/kernels/dispatch_writer_fused.cpp";
static const std::string COMBINE_KERNEL =
    "tests/ttnn/unit_tests/operations/experimental/prefill_moe/kernels/combine_dm_fused.cpp";
static const std::string FABRIC_REDUCE_KERNEL =
    "tests/ttnn/unit_tests/operations/experimental/prefill_moe/kernels/fabric_reduce_dm.cpp";
static const std::string FABRIC_DISPATCH_KERNEL =
    "tests/ttnn/unit_tests/operations/experimental/prefill_moe/kernels/fabric_dispatch.cpp";

// ---------------------------------------------------------------------------
// Helper: compute largest N_BLOCK that divides n_per_core and is <= max_block
// ---------------------------------------------------------------------------
static uint32_t compute_n_block(uint32_t n_per_core, uint32_t max_block = 7) {
    uint32_t limit = std::min(n_per_core, max_block);
    for (uint32_t b = limit; b >= 1; --b) {
        if (n_per_core % b == 0) {
            return b;
        }
    }
    return 1;
}

// ---------------------------------------------------------------------------
// create_at — per-device program creation for MeshWorkloadFactoryConcept
// ---------------------------------------------------------------------------
ttnn::device_operation::CachedProgram<PrefillMoeComputeMeshFactory::shared_variables_t>
PrefillMoeComputeMeshFactory::create_at(
    const operation_attributes_t& attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*output_tensor*/) {
    auto* device = tensor_args.hidden_states.device();
    const uint32_t num_experts = attributes.num_experts;
    const uint32_t num_cores = attributes.num_cores;
    const uint32_t grid_x = attributes.grid_x;
    const uint32_t grid_y = attributes.grid_y;
    const bool enable_fabric_reduce = attributes.enable_fabric_reduce;
    const bool enable_fabric_dispatch = attributes.enable_fabric_dispatch;
    const bool enable_fpu_combine = attributes.enable_fpu_combine;
    const bool need_mesh =
        (attributes.per_device_combine_metadata.size() > 1 || enable_fabric_reduce || enable_fabric_dispatch);

    // ----- Per-device metadata lookup -----
    uint32_t device_index = 0;
    tt::tt_metal::distributed::MeshDevice* mesh_dev = nullptr;
    if (need_mesh) {
        mesh_dev = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device);
        TT_FATAL(mesh_dev != nullptr, "Multi-device metadata or fabric requires a MeshDevice");
        device_index = mesh_coordinate.to_linear_index(mesh_dev->shape());
    }
    TT_FATAL(
        device_index < attributes.per_device_combine_metadata.size(),
        "device_index {} out of range for per_device_combine_metadata of size {}",
        device_index,
        attributes.per_device_combine_metadata.size());
    const auto& combine_metadata = attributes.per_device_combine_metadata[device_index];

    // ----- Derive tile dimensions from tensor shapes (with padding support) -----
    const auto& hs_shape = tensor_args.hidden_states.logical_shape();
    const uint32_t D = hs_shape[3];
    const uint32_t D_tiles = D / TILE_HW;

    const auto& pkt_shape = tensor_args.pkt_buf.logical_shape();
    const uint32_t M_tiles_total = pkt_shape[2] / TILE_HW;  // total tile rows in pkt_buf
    const uint32_t M_tiles = M_tiles_total / num_experts;   // per-expert tile rows

    // Output P_tiles (for combine kernel) — derived from output tensor, not pkt_buf
    const auto& output_shape = tensor_args.output.logical_shape();
    const uint32_t output_M_tiles = output_shape[2] / TILE_HW;

    const auto& gu_shape = tensor_args.gate_up_weights[0].logical_shape();
    const uint32_t D_FF_padded = gu_shape[3];

    const uint32_t k_tiles_gu = D_tiles;
    const uint32_t n_weight_tiles_gu = D_FF_padded / TILE_HW;
    const uint32_t n_weight_per_core_gu = n_weight_tiles_gu / num_cores;
    const uint32_t n_out_per_core = n_weight_per_core_gu / 2;

    const auto& dn_shape = tensor_args.down_weights[0].logical_shape();
    const uint32_t D_FF_HALF_padded = dn_shape[2];
    const uint32_t D_padded = dn_shape[3];

    const uint32_t k_tiles_dn = D_FF_HALF_padded / TILE_HW;
    const uint32_t n_tiles_dn = D_padded / TILE_HW;
    const uint32_t n_per_core_dn = n_tiles_dn / num_cores;

    const uint32_t N_BLOCK_GU = compute_n_block(n_weight_per_core_gu);
    const uint32_t N_BLOCK_DN = compute_n_block(n_per_core_dn);

    // Output tile count (for fabric reduce)
    const uint32_t num_output_tiles = n_tiles_dn;  // [1,1,P,D_padded] with P=32 -> 1 row of tiles

    // ----- Core grids -----
    CoreRange compute_range(CoreCoord(0, 0), CoreCoord(grid_x - 1, grid_y - 1));
    CoreRangeSet compute_cores(std::vector<CoreRange>{compute_range});

    CoreCoord dispatch_core_logical(0, grid_y);
    CoreRange dispatch_range(dispatch_core_logical, dispatch_core_logical);
    CoreRangeSet dispatch_cores(std::vector<CoreRange>{dispatch_range});

    CoreCoord combine_core_logical(1, grid_y);
    CoreRange combine_range(combine_core_logical, combine_core_logical);
    CoreRangeSet combine_cores(std::vector<CoreRange>{combine_range});

    // Base all_cores: compute + dispatch + combine
    std::vector<CoreRange> all_core_ranges = {compute_range, CoreRange(CoreCoord(0, grid_y), CoreCoord(1, grid_y))};

    // Fabric-specific cores
    CoreCoord reduce_core_logical(2, grid_y);
    CoreCoord mux_core_logical(3, grid_y);
    if (enable_fabric_reduce) {
        all_core_ranges.push_back(CoreRange(reduce_core_logical, reduce_core_logical));
        // Note: mux core is NOT in all_cores (its L1 is managed by FabricMuxConfig)
    }
    CoreRangeSet all_cores(all_core_ranges);

    // ----- Physical coordinates -----
    std::vector<std::pair<uint32_t, uint32_t>> phys_coords;
    for (uint32_t y = 0; y < grid_y; ++y) {
        for (uint32_t x = 0; x < grid_x; ++x) {
            auto phys = device->worker_core_from_logical_core(CoreCoord(x, y));
            phys_coords.push_back({phys.x, phys.y});
        }
    }
    uint32_t leader_phys_x = phys_coords[0].first;
    uint32_t leader_phys_y = phys_coords[0].second;

    auto combine_phys = device->worker_core_from_logical_core(combine_core_logical);

    // ----- Compile-time args -----
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(tensor_args.pkt_buf.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(tensor_args.inter_buf.buffer()).append_to(reader_ct_args);
    if (enable_fpu_combine) {
        TensorAccessorArgs(tensor_args.output.buffer()).append_to(reader_ct_args);  // [2] for combine reads
    }

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(tensor_args.gate_up_weights[0].buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(tensor_args.inter_buf.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(tensor_args.down_weights[0].buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(tensor_args.out_bufs[0].buffer()).append_to(writer_ct_args);

    std::vector<uint32_t> compute_ct_args = {
        k_tiles_gu,
        n_weight_per_core_gu,
        n_per_core_dn,
        num_experts,
        N_BLOCK_GU,
        N_BLOCK_DN,
        k_tiles_dn,
        M_tiles,
        output_M_tiles,
        enable_fpu_combine ? 1u : 0u};

    // Dispatch CT args: depends on fabric dispatch mode
    std::vector<uint32_t> dispatch_ct_args;
    if (enable_fabric_dispatch) {
        // Fabric dispatch: [0] TensorAccessorArgs(pkt_buf TILE), [1] TensorAccessorArgs(hs_rm ROW_MAJOR)
        TensorAccessorArgs(tensor_args.pkt_buf.buffer()).append_to(dispatch_ct_args);
        TensorAccessorArgs(tensor_args.hidden_states_rm->buffer()).append_to(dispatch_ct_args);
    } else {
        // Standard dispatch: [0] TensorAccessorArgs(hidden_states TILE), [1] TensorAccessorArgs(pkt_buf TILE)
        TensorAccessorArgs(tensor_args.hidden_states.buffer()).append_to(dispatch_ct_args);
        TensorAccessorArgs(tensor_args.pkt_buf.buffer()).append_to(dispatch_ct_args);
    }

    // Combine CT args: TensorAccessorArgs at [0-3], enable_fabric_reduce at [4]
    // When FPU combine is active, the combine kernel is a no-op — disable its fabric_reduce too.
    std::vector<uint32_t> combine_ct_args;
    TensorAccessorArgs(tensor_args.output.buffer()).append_to(combine_ct_args);
    combine_ct_args.push_back((enable_fabric_reduce && !enable_fpu_combine) ? 1 : 0);

    // ----- Compute per-expert out_buf_tile_row_offsets from combine_metadata -----
    // combine_metadata format: [addr_0, M_0, rw_0...rw_{M0-1}, addr_1, M_1, ...]
    // Each rw packs (row_index, weight_bf16) as: low 16 bits = row, high 16 bits = weight
    std::vector<uint32_t> out_buf_tile_row_offsets(num_experts, 0);
    {
        size_t cm_idx = 0;
        for (uint32_t e = 0; e < num_experts; ++e) {
            // cm_idx+0 = out_buf_addr, cm_idx+1 = M_e
            uint32_t m_e = combine_metadata[cm_idx + 1];
            uint32_t min_row = 0xFFFF;
            for (uint32_t i = 0; i < m_e; ++i) {
                uint32_t rw = combine_metadata[cm_idx + 2 + i];
                uint32_t row = rw & 0xFFFF;
                if (row < min_row) {
                    min_row = row;
                }
            }
            out_buf_tile_row_offsets[e] = (m_e > 0) ? (min_row / TILE_HW) : 0;
            cm_idx += 2 + m_e;
        }
    }

    // ----- Build runtime args -----

    // Reader RT args (per compute core)
    KernelDescriptor::RuntimeArgs reader_rt_args;
    for (uint32_t y = 0; y < grid_y; ++y) {
        for (uint32_t x = 0; x < grid_x; ++x) {
            uint32_t core_idx = y * grid_x + x;
            uint32_t is_leader = (core_idx == 0) ? 1 : 0;

            std::vector<uint32_t> args = {
                static_cast<uint32_t>(tensor_args.pkt_buf.buffer()->address()),
                k_tiles_gu,
                static_cast<uint32_t>(tensor_args.inter_buf.buffer()->address()),
                is_leader,
                num_cores,
                combine_phys.x,
                combine_phys.y,
                num_experts,
                k_tiles_dn,
                M_tiles,
            };
            for (const auto& [px, py] : phys_coords) {
                args.push_back(px);
                args.push_back(py);
            }
            // FPU combine args (appended after physical coords)
            if (enable_fpu_combine) {
                args.push_back(static_cast<uint32_t>(tensor_args.output.buffer()->address()));
                args.push_back(n_tiles_dn);
                args.push_back(core_idx * n_per_core_dn);  // core_d_offset
                args.push_back(n_per_core_dn);
                args.push_back(output_M_tiles);
                for (uint32_t e = 0; e < num_experts; ++e) {
                    args.push_back(static_cast<uint32_t>(tensor_args.out_bufs[e].buffer()->address()));
                }
            }
            reader_rt_args.push_back({CoreCoord(x, y), std::move(args)});
        }
    }

    // Writer RT args (per compute core)
    KernelDescriptor::RuntimeArgs writer_rt_args;
    for (uint32_t y = 0; y < grid_y; ++y) {
        for (uint32_t x = 0; x < grid_x; ++x) {
            uint32_t core_idx = y * grid_x + x;

            std::vector<uint32_t> args = {
                static_cast<uint32_t>(tensor_args.inter_buf.buffer()->address()),
                k_tiles_gu,
                k_tiles_dn,
                n_weight_per_core_gu,
                n_weight_tiles_gu,
                core_idx * n_weight_per_core_gu,
                core_idx * n_out_per_core,
                n_out_per_core,
                n_per_core_dn,
                n_tiles_dn,
                core_idx * n_per_core_dn,
                leader_phys_x,
                leader_phys_y,
                num_experts,
                M_tiles,
            };
            for (uint32_t e = 0; e < num_experts; ++e) {
                args.push_back(static_cast<uint32_t>(tensor_args.gate_up_weights[e].buffer()->address()));
                args.push_back(static_cast<uint32_t>(tensor_args.down_weights[e].buffer()->address()));
                args.push_back(static_cast<uint32_t>(tensor_args.out_bufs[e].buffer()->address()));
                args.push_back(out_buf_tile_row_offsets[e]);
            }
            // FPU combine write-back args
            if (enable_fpu_combine) {
                args.push_back(static_cast<uint32_t>(tensor_args.output.buffer()->address()));
                args.push_back(output_M_tiles);
            }
            writer_rt_args.push_back({CoreCoord(x, y), std::move(args)});
        }
    }

    // Dispatch RT args (single core) — only for standard dispatch
    // For fabric dispatch, RT args are set after program creation (need fabric connection setup)
    KernelDescriptor::RuntimeArgs dispatch_rt_args;
    if (!enable_fabric_dispatch) {
        dispatch_rt_args.push_back(
            {dispatch_core_logical,
             {
                 static_cast<uint32_t>(tensor_args.hidden_states.buffer()->address()),
                 static_cast<uint32_t>(tensor_args.pkt_buf.buffer()->address()),
                 M_tiles_total * D_tiles,  // total tiles to copy (all expert regions)
                 leader_phys_x,
                 leader_phys_y,
             }});
    }

    // Combine RT args (single core) — uses per-device metadata
    // When FPU combine is enabled, the combine kernel is a no-op (num_experts=0).
    std::vector<uint32_t> combine_args = {
        static_cast<uint32_t>(tensor_args.output.buffer()->address()),
        n_tiles_dn,
        output_M_tiles,
        enable_fpu_combine ? 0u : num_experts,
    };
    if (!enable_fpu_combine) {
        combine_args.insert(combine_args.end(), combine_metadata.begin(), combine_metadata.end());
    }
    // When fabric reduce is enabled, append reduce_core NOC address for SEM_COMBINE_DONE signaling
    if (enable_fabric_reduce) {
        auto reduce_phys = device->worker_core_from_logical_core(reduce_core_logical);
        combine_args.push_back(reduce_phys.x);
        combine_args.push_back(reduce_phys.y);
        // SEM_COMBINE_DONE is at semaphore index 4 on the reduce_core.
        // We'll compute the L1 address after creating semaphores.
        // For now, use a placeholder — will be filled after Program construction.
        combine_args.push_back(0);  // placeholder for sem_combine_done_l1
    }
    KernelDescriptor::RuntimeArgs combine_rt_args;
    combine_rt_args.push_back({combine_core_logical, std::move(combine_args)});

    // ----- Semaphore Descriptors -----
    SemaphoreDescriptor sem_barrier{0, tt::CoreType::WORKER, all_cores, 0};
    SemaphoreDescriptor sem_go{1, tt::CoreType::WORKER, all_cores, 0};
    SemaphoreDescriptor sem_pkt_ready{2, tt::CoreType::WORKER, all_cores, 0};
    SemaphoreDescriptor sem_expert_done{3, tt::CoreType::WORKER, all_cores, 0};

    ProgramDescriptor::SemaphoreDescriptors semaphores = {sem_barrier, sem_go, sem_pkt_ready, sem_expert_done};

    // Add fabric reduce semaphores on reduce_core
    if (enable_fabric_reduce) {
        CoreRangeSet reduce_core_set(std::vector<CoreRange>{CoreRange(reduce_core_logical, reduce_core_logical)});
        semaphores.push_back(SemaphoreDescriptor{4, tt::CoreType::WORKER, reduce_core_set, 0});  // SEM_COMBINE_DONE
        semaphores.push_back(SemaphoreDescriptor{5, tt::CoreType::WORKER, reduce_core_set, 0});  // SEM_FABRIC_RECV
    }

    // Add fabric dispatch semaphore on dispatch_core
    if (enable_fabric_dispatch) {
        semaphores.push_back(SemaphoreDescriptor{4, tt::CoreType::WORKER, dispatch_cores, 0});  // SEM_FABRIC_RECV
    }

    // ----- CB Descriptors -----
    // CB0: Activation tile (BF16, 1 tile) on compute cores
    CBDescriptor cb0;
    cb0.total_size = BF16_TILE_BYTES;
    cb0.core_ranges = compute_cores;
    cb0.format_descriptors.push_back({0, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

    // CB1: Weight tiles (BFP4_b, n_weight_per_core_gu tiles) on compute cores
    CBDescriptor cb1;
    cb1.total_size = n_weight_per_core_gu * BFP4_TILE_BYTES;
    cb1.core_ranges = compute_cores;
    cb1.format_descriptors.push_back({1, tt::DataFormat::Bfp4_b, BFP4_TILE_BYTES});

    // CB2: Output tiles (BF16, n_out_per_core tiles) on compute cores
    CBDescriptor cb2;
    cb2.total_size = n_out_per_core * BF16_TILE_BYTES;
    cb2.core_ranges = compute_cores;
    cb2.format_descriptors.push_back({2, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

    // CB3: Dispatch temp buffer on dispatch core
    CBDescriptor cb3;
    if (enable_fabric_dispatch) {
        // Fabric dispatch needs one full token row (D * 2 bytes)
        uint32_t D_bytes = D * 2;
        cb3.total_size = D_bytes;
        cb3.core_ranges = dispatch_cores;
        cb3.format_descriptors.push_back({3, tt::DataFormat::Float16_b, D_bytes});
    } else {
        // Standard dispatch: 1 tile
        cb3.total_size = BF16_TILE_BYTES;
        cb3.core_ranges = dispatch_cores;
        cb3.format_descriptors.push_back({3, tt::DataFormat::Float16_b, BF16_TILE_BYTES});
    }

    ProgramDescriptor::CBDescriptors cbs = {cb0, cb1, cb2, cb3};

    if (enable_fpu_combine) {
        // CB3/4/5 on compute cores for FPU combine
        CBDescriptor cb3_combine;
        cb3_combine.total_size = BF16_TILE_BYTES;
        cb3_combine.core_ranges = compute_cores;
        cb3_combine.format_descriptors.push_back({3, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

        CBDescriptor cb4_combine;
        cb4_combine.total_size = BF16_TILE_BYTES;
        cb4_combine.core_ranges = compute_cores;
        cb4_combine.format_descriptors.push_back({4, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

        CBDescriptor cb5_combine;
        cb5_combine.total_size = BF16_TILE_BYTES;
        cb5_combine.core_ranges = compute_cores;
        cb5_combine.format_descriptors.push_back({5, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

        cbs.push_back(cb3_combine);
        cbs.push_back(cb4_combine);
        cbs.push_back(cb5_combine);
    }

    // CB4/5 on combine core (always needed — combine kernel accesses them before checking num_experts)
    {
        CBDescriptor cb4;
        cb4.total_size = BF16_TILE_BYTES;
        cb4.core_ranges = combine_cores;
        cb4.format_descriptors.push_back({4, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

        CBDescriptor cb5;
        cb5.total_size = BF16_TILE_BYTES;
        cb5.core_ranges = combine_cores;
        cb5.format_descriptors.push_back({5, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

        cbs.push_back(cb4);
        cbs.push_back(cb5);
    }

    // Add fabric dispatch tile assembly CB on dispatch core
    if (enable_fabric_dispatch) {
        // CB6: Tile assembly buffer (D_tiles * 2048 bytes) on dispatch core
        uint32_t tile_assembly_size = D_tiles * BF16_TILE_BYTES;
        CBDescriptor cb6_dispatch;
        cb6_dispatch.total_size = tile_assembly_size;
        cb6_dispatch.core_ranges = dispatch_cores;
        cb6_dispatch.format_descriptors.push_back({6, tt::DataFormat::Float16_b, tile_assembly_size});
        cbs.push_back(cb6_dispatch);
    }

    // Add fabric reduce CBs on reduce_core
    if (enable_fabric_reduce) {
        CoreRangeSet reduce_core_set(std::vector<CoreRange>{CoreRange(reduce_core_logical, reduce_core_logical)});

        // CB6: Send buffer (BF16, 1 tile) on reduce core
        CBDescriptor cb6;
        cb6.total_size = BF16_TILE_BYTES;
        cb6.core_ranges = reduce_core_set;
        cb6.format_descriptors.push_back({6, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

        // CB7: Output tile for accumulation (BF16, 1 tile) on reduce core
        CBDescriptor cb7;
        cb7.total_size = BF16_TILE_BYTES;
        cb7.core_ranges = reduce_core_set;
        cb7.format_descriptors.push_back({7, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

        // CB8: Recv tile for accumulation (BF16, 1 tile) on reduce core
        CBDescriptor cb8;
        cb8.total_size = BF16_TILE_BYTES;
        cb8.core_ranges = reduce_core_set;
        cb8.format_descriptors.push_back({8, tt::DataFormat::Float16_b, BF16_TILE_BYTES});

        cbs.push_back(cb6);
        cbs.push_back(cb7);
        cbs.push_back(cb8);
    }

    // Compute direction flags for fabric dispatch (needed for kernel defines)
    bool dispatch_has_east = false;
    bool dispatch_has_west = false;
    if (enable_fabric_dispatch && mesh_dev != nullptr) {
        auto mesh_shape_early = mesh_dev->shape();
        auto coord_col_early = mesh_coordinate[1];
        dispatch_has_east = (coord_col_early + 1 < mesh_shape_early[1]);
        dispatch_has_west = (coord_col_early > 0);
    }

    // ----- Kernel Descriptors -----
    KernelDescriptor reader_kd;
    reader_kd.kernel_source = READER_KERNEL;
    reader_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kd.core_ranges = compute_cores;
    reader_kd.compile_time_args = reader_ct_args;
    reader_kd.runtime_args = std::move(reader_rt_args);
    reader_kd.config = ReaderConfigDescriptor{};
    if (enable_fpu_combine) {
        reader_kd.defines.emplace_back("ENABLE_FPU_COMBINE", "1");
    }

    KernelDescriptor writer_kd;
    writer_kd.kernel_source = WRITER_KERNEL;
    writer_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kd.core_ranges = compute_cores;
    writer_kd.compile_time_args = writer_ct_args;
    writer_kd.runtime_args = std::move(writer_rt_args);
    writer_kd.config = WriterConfigDescriptor{};
    if (enable_fpu_combine) {
        writer_kd.defines.emplace_back("ENABLE_FPU_COMBINE", "1");
    }

    KernelDescriptor compute_kd;
    compute_kd.kernel_source = COMPUTE_KERNEL;
    compute_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kd.core_ranges = compute_cores;
    compute_kd.compile_time_args = compute_ct_args;
    compute_kd.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::LoFi,
        .fp32_dest_acc_en = false,
        .dst_full_sync_en = true,
        .math_approx_mode = true,
    };

    KernelDescriptor dispatch_kd;
    dispatch_kd.kernel_source = enable_fabric_dispatch ? FABRIC_DISPATCH_KERNEL : DISPATCH_KERNEL;
    dispatch_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    dispatch_kd.core_ranges = dispatch_cores;
    dispatch_kd.compile_time_args = dispatch_ct_args;
    dispatch_kd.runtime_args = std::move(dispatch_rt_args);
    dispatch_kd.config = WriterConfigDescriptor{};
    if (enable_fabric_dispatch) {
        dispatch_kd.defines.emplace_back("EAST_CONNECTION", dispatch_has_east ? "1" : "0");
        dispatch_kd.defines.emplace_back("WEST_CONNECTION", dispatch_has_west ? "1" : "0");
    }

    KernelDescriptor combine_kd;
    combine_kd.kernel_source = COMBINE_KERNEL;
    combine_kd.source_type = KernelDescriptor::SourceType::FILE_PATH;
    combine_kd.core_ranges = combine_cores;
    combine_kd.compile_time_args = combine_ct_args;
    combine_kd.runtime_args = std::move(combine_rt_args);
    combine_kd.config = ReaderConfigDescriptor{};

    // ----- Build ProgramDescriptor and create Program -----
    ProgramDescriptor descriptor;
    descriptor.kernels = {reader_kd, writer_kd, compute_kd, dispatch_kd, combine_kd};
    descriptor.semaphores = semaphores;
    descriptor.cbs = cbs;

    Program program{descriptor};

    // Kernel handles are sequential: 0=reader, 1=writer, 2=compute, 3=dispatch, 4=combine
    shared_variables_t shared{
        0,
        1,
        2,
        3,
        4,
        compute_cores,
        dispatch_core_logical,
        combine_core_logical,
        num_cores,
        grid_x,
        grid_y,
        std::nullopt,
        std::nullopt,
        false};

    // ===================================================================
    // Fabric dispatch setup (after base Program is constructed)
    // ===================================================================
    if (enable_fabric_dispatch) {
        TT_FATAL(mesh_dev != nullptr, "Fabric dispatch requires MeshDevice");
        TT_FATAL(tensor_args.hidden_states_rm.has_value(), "hidden_states_rm required for fabric dispatch");
        TT_FATAL(tensor_args.staging_buf.has_value(), "staging_buf required for fabric dispatch");
        TT_FATAL(
            device_index < attributes.dispatch_metadata.size(),
            "device_index {} out of range for dispatch_metadata of size {}",
            device_index,
            attributes.dispatch_metadata.size());

        auto dispatch_phys = device->worker_core_from_logical_core(dispatch_core_logical);
        const auto& dev_dispatch_meta = attributes.dispatch_metadata[device_index];

        uint32_t D_bytes = D * 2;

        // staging_buf addresses (same on both devices by deterministic allocation)
        auto staging_buf_addr = tensor_args.staging_buf->buffer()->address();
        auto remote_staging_addr = staging_buf_addr;

        // Build dispatch RT args
        std::vector<uint32_t> dispatch_args;
        dispatch_args.push_back(static_cast<uint32_t>(tensor_args.hidden_states_rm->buffer()->address()));  // [0]
        dispatch_args.push_back(static_cast<uint32_t>(staging_buf_addr));                                   // [1]
        dispatch_args.push_back(static_cast<uint32_t>(remote_staging_addr));                                // [2]
        dispatch_args.push_back(static_cast<uint32_t>(tensor_args.pkt_buf.buffer()->address()));            // [3]
        dispatch_args.push_back(leader_phys_x);                                                             // [4]
        dispatch_args.push_back(leader_phys_y);                                                             // [5]
        dispatch_args.push_back(dispatch_phys.x);  // [6] my_phys_x
        dispatch_args.push_back(dispatch_phys.y);  // [7] my_phys_y
        dispatch_args.push_back(D_bytes);          // [8]
        dispatch_args.push_back(D_tiles);          // [9]
        dispatch_args.push_back(M_tiles);          // [10] M_padded_tiles (per-expert tile rows)

        // Transform old-format dispatch_metadata to new kernel format.
        // Old Python format: [local_count, recv_count, send_count, local_indices..., send_indices...]
        // New kernel format: [recv_count, send_count, send_indices..., num_experts, M_0, sources_0..., ...]
        {
            size_t dm_idx = 0;
            uint32_t local_count = dev_dispatch_meta[dm_idx++];
            uint32_t recv_count = dev_dispatch_meta[dm_idx++];
            uint32_t send_count = dev_dispatch_meta[dm_idx++];

            // Extract local_indices and send_indices from old format
            std::vector<uint32_t> local_indices(
                dev_dispatch_meta.begin() + dm_idx, dev_dispatch_meta.begin() + dm_idx + local_count);
            dm_idx += local_count;
            std::vector<uint32_t> send_indices(
                dev_dispatch_meta.begin() + dm_idx, dev_dispatch_meta.begin() + dm_idx + send_count);

            // Append recv_count, send_count, send_indices
            dispatch_args.push_back(recv_count);
            dispatch_args.push_back(send_count);
            dispatch_args.insert(dispatch_args.end(), send_indices.begin(), send_indices.end());

            // Per-expert pkt_buf assembly metadata
            if (attributes.per_expert_dispatch_sources.has_value()) {
                // Per-expert sources provided directly by Python
                const auto& expert_sources = (*attributes.per_expert_dispatch_sources)[device_index];
                dispatch_args.insert(dispatch_args.end(), expert_sources.begin(), expert_sources.end());
            } else {
                // Default: all experts share the same pkt_buf content (replicated)
                uint32_t M_padded = local_count + recv_count;
                dispatch_args.push_back(num_experts);
                for (uint32_t e = 0; e < num_experts; ++e) {
                    dispatch_args.push_back(M_padded);
                    // Local token sources: bit 31 = 0 (local hs_rm)
                    for (uint32_t i = 0; i < local_count; ++i) {
                        dispatch_args.push_back(local_indices[i]);
                    }
                    // Received token sources: bit 31 = 1 (staging_buf)
                    for (uint32_t i = 0; i < recv_count; ++i) {
                        dispatch_args.push_back((1u << 31) | i);
                    }
                }
            }
        }

        // Direction-based fabric connection setup (EAST/WEST for 1xN mesh)
        auto mesh_shape = mesh_dev->shape();
        auto coord_row = mesh_coordinate[0];
        auto coord_col = mesh_coordinate[1];

        // Determine target device for this device's dispatch
        uint32_t target_col;
        if (!attributes.dispatch_target_cols.empty()) {
            target_col = attributes.dispatch_target_cols[device_index];
        } else {
            // Default: forward-preferring for backward compatibility
            target_col = (coord_col + 1 < mesh_shape[1]) ? coord_col + 1 : coord_col - 1;
        }

        // Compute send direction and hop count
        uint32_t dispatch_send_direction;  // 0=EAST, 1=WEST
        uint32_t dispatch_num_hops;
        if (target_col > coord_col) {
            dispatch_send_direction = 0;  // EAST
            dispatch_num_hops = target_col - coord_col;
        } else {
            dispatch_send_direction = 1;  // WEST
            dispatch_num_hops = coord_col - target_col;
        }

        // Which directions have valid immediate neighbors
        bool dispatch_has_east = (coord_col + 1 < mesh_shape[1]);
        bool dispatch_has_west = (coord_col > 0);

        // Append send_direction and num_hops RT args
        dispatch_args.push_back(dispatch_send_direction);
        dispatch_args.push_back(dispatch_num_hops);

        // Append fabric connection RT args for each valid direction
        auto src_node_id = mesh_dev->get_fabric_node_id(mesh_coordinate);
        constexpr uint32_t link = 0;
        if (dispatch_has_east) {
            auto east_coord = ttnn::MeshCoordinate{coord_row, coord_col + 1};
            auto dst_node_id = mesh_dev->get_fabric_node_id(east_coord);
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_node_id, dst_node_id, link, program, dispatch_core_logical, dispatch_args);
        }
        if (dispatch_has_west) {
            auto west_coord = ttnn::MeshCoordinate{coord_row, coord_col - 1};
            auto dst_node_id = mesh_dev->get_fabric_node_id(west_coord);
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_node_id, dst_node_id, link, program, dispatch_core_logical, dispatch_args);
        }

        SetRuntimeArgs(program, 3 /*dispatch_kernel_id*/, dispatch_core_logical, dispatch_args);

        shared.fabric_dispatch_enabled = true;
    }

    // ===================================================================
    // Fabric reduce setup (after base Program is constructed)
    // ===================================================================
    if (enable_fabric_reduce) {
        TT_FATAL(mesh_dev != nullptr, "Fabric reduce requires MeshDevice");
        TT_FATAL(tensor_args.reduce_recv_buf.has_value(), "reduce_recv_buf required for fabric reduce");

        // Virtual coords (used for local NOC and fabric packet addresses)
        auto reduce_phys = device->worker_core_from_logical_core(reduce_core_logical);

        // ---- Compute neighbor ----
        auto mesh_shape = mesh_dev->shape();
        auto coord_row = mesh_coordinate[0];
        auto coord_col = mesh_coordinate[1];

        // For 1xN mesh: forward = EAST (col+1), backward = WEST (col-1)
        std::optional<ttnn::MeshCoordinate> forward_coord;
        std::optional<ttnn::MeshCoordinate> backward_coord;
        if (coord_col + 1 < mesh_shape[1]) {
            forward_coord = ttnn::MeshCoordinate{coord_row, coord_col + 1};
        }
        if (coord_col > 0) {
            backward_coord = ttnn::MeshCoordinate{coord_row, coord_col - 1};
        }
        TT_FATAL(
            forward_coord.has_value() || backward_coord.has_value(), "Fabric reduce requires at least one neighbor");

        auto local_recv_buf_addr = tensor_args.reduce_recv_buf->buffer()->address();
        auto remote_recv_buf_addr = local_recv_buf_addr;

        // ---- Compute reduce hop count from dispatch target ----
        uint32_t reduce_num_hops = 1;  // default: immediate neighbor
        if (!attributes.dispatch_target_cols.empty()) {
            uint32_t reduce_target_col = attributes.dispatch_target_cols[device_index];
            reduce_num_hops =
                (reduce_target_col > coord_col) ? (reduce_target_col - coord_col) : (coord_col - reduce_target_col);
        }

        // ---- Fabric reduce kernel CT args ----
        std::vector<uint32_t> reduce_ct_args;
        TensorAccessorArgs(tensor_args.output.buffer()).append_to(reduce_ct_args);
        reduce_ct_args.push_back(num_output_tiles);
        reduce_ct_args.push_back(reduce_num_hops);

        // ---- Create fabric_reduce kernel ----
        CoreRangeSet reduce_core_set(std::vector<CoreRange>{CoreRange(reduce_core_logical, reduce_core_logical)});

        std::map<std::string, std::string> reduce_defines;
        reduce_defines.emplace("EAST_CONNECTION", (coord_col + 1 < mesh_shape[1]) ? "1" : "0");
        reduce_defines.emplace("WEST_CONNECTION", (coord_col > 0) ? "1" : "0");

        auto reduce_kernel_id = CreateKernel(
            program,
            FABRIC_REDUCE_KERNEL,
            reduce_core_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reduce_ct_args,
                .defines = reduce_defines});

        // ---- Semaphores on reduce_core ----
        auto sem_combine_done_id = CreateSemaphore(program, reduce_core_set, 0);
        auto sem_fabric_recv_id = CreateSemaphore(program, reduce_core_set, 0);

        // ---- Fabric reduce kernel RT args ----
        std::vector<uint32_t> reduce_rt_args;
        reduce_rt_args.push_back(static_cast<uint32_t>(tensor_args.output.buffer()->address()));
        reduce_rt_args.push_back(static_cast<uint32_t>(local_recv_buf_addr));
        reduce_rt_args.push_back(static_cast<uint32_t>(remote_recv_buf_addr));
        reduce_rt_args.push_back(sem_combine_done_id);  // [3] semaphore ID
        reduce_rt_args.push_back(sem_fabric_recv_id);   // [4] semaphore ID
        reduce_rt_args.push_back(reduce_phys.x);        // [5] noc x (virtual)
        reduce_rt_args.push_back(reduce_phys.y);        // [6] noc y (virtual)

        // Direction-based fabric connection for reduce
        // Reduce sends in the OPPOSITE direction of dispatch (return results to partner)
        uint32_t reduce_send_direction;
        if (!attributes.dispatch_target_cols.empty()) {
            uint32_t reduce_target_col = attributes.dispatch_target_cols[device_index];
            reduce_send_direction = (reduce_target_col > coord_col) ? 0 : 1;  // 0=EAST, 1=WEST
        } else {
            // Default: same as dispatch direction (forward-preferring)
            reduce_send_direction = (coord_col + 1 < mesh_shape[1]) ? 0 : 1;
        }
        reduce_rt_args.push_back(reduce_send_direction);

        // Fabric connection RT args for each valid direction
        auto src_node_id = mesh_dev->get_fabric_node_id(mesh_coordinate);
        constexpr uint32_t link = 0;
        bool reduce_has_east = (coord_col + 1 < mesh_shape[1]);
        bool reduce_has_west = (coord_col > 0);
        if (reduce_has_east) {
            auto east_coord = ttnn::MeshCoordinate{coord_row, coord_col + 1};
            auto dst_node_id = mesh_dev->get_fabric_node_id(east_coord);
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_node_id, dst_node_id, link, program, reduce_core_logical, reduce_rt_args);
        }
        if (reduce_has_west) {
            auto west_coord = ttnn::MeshCoordinate{coord_row, coord_col - 1};
            auto dst_node_id = mesh_dev->get_fabric_node_id(west_coord);
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_node_id, dst_node_id, link, program, reduce_core_logical, reduce_rt_args);
        }

        SetRuntimeArgs(program, reduce_kernel_id, reduce_core_logical, reduce_rt_args);

        // ---- Fix up combine kernel's SEM_COMBINE_DONE address ----
        auto& combine_rt = GetRuntimeArgs(program, 4 /*combine_kernel_id*/, combine_core_logical);
        combine_rt[combine_rt.size() - 1] = sem_combine_done_id;

        shared.reduce_kernel_id = reduce_kernel_id;
        shared.reduce_core = reduce_core_logical;
    }

    return {std::move(program), std::move(shared)};
}

// ---------------------------------------------------------------------------
// override_runtime_arguments — updates buffer addresses in cached mesh workload
// ---------------------------------------------------------------------------
void PrefillMoeComputeMeshFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*output_tensor*/) {
    const uint32_t num_experts = attributes.num_experts;

    // Recompute padded tile counts from tensor shapes
    const auto& dn_shape = tensor_args.down_weights[0].logical_shape();
    const uint32_t D_padded = dn_shape[3];
    const uint32_t n_tiles_dn = D_padded / TILE_HW;

    const auto& output_shape = tensor_args.output.logical_shape();
    const uint32_t output_M_tiles = output_shape[2] / TILE_HW;

    uint32_t device_index = 0;
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(range);

        // Determine per-device metadata index
        TT_FATAL(
            device_index < attributes.per_device_combine_metadata.size(),
            "device_index {} out of range for per_device_combine_metadata of size {}",
            device_index,
            attributes.per_device_combine_metadata.size());
        const auto& combine_metadata = attributes.per_device_combine_metadata[device_index];

        // Update reader runtime args (buffer addresses may change)
        for (uint32_t y = 0; y < shared.grid_y; ++y) {
            for (uint32_t x = 0; x < shared.grid_x; ++x) {
                auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id, CoreCoord(x, y));
                reader_args[0] = tensor_args.pkt_buf.buffer()->address();
                reader_args[2] = tensor_args.inter_buf.buffer()->address();
                // Update FPU combine args if enabled
                if (attributes.enable_fpu_combine) {
                    uint32_t combine_base = 10 + 2 * shared.num_cores;
                    reader_args[combine_base + 0] = tensor_args.output.buffer()->address();
                    for (uint32_t e = 0; e < num_experts; ++e) {
                        reader_args[combine_base + 5 + e] = tensor_args.out_bufs[e].buffer()->address();
                    }
                }
            }
        }

        // Update writer runtime args
        for (uint32_t y = 0; y < shared.grid_y; ++y) {
            for (uint32_t x = 0; x < shared.grid_x; ++x) {
                auto& writer_args = GetRuntimeArgs(program, shared.writer_kernel_id, CoreCoord(x, y));
                writer_args[0] = tensor_args.inter_buf.buffer()->address();
                // Per-expert addresses start at index 15 (4 args per expert)
                for (uint32_t e = 0; e < num_experts; ++e) {
                    writer_args[15 + e * 4 + 0] = tensor_args.gate_up_weights[e].buffer()->address();
                    writer_args[15 + e * 4 + 1] = tensor_args.down_weights[e].buffer()->address();
                    writer_args[15 + e * 4 + 2] = tensor_args.out_bufs[e].buffer()->address();
                    // writer_args[15 + e * 4 + 3] = out_buf_tile_row_offset (unchanged)
                }
                // Update FPU combine write-back args if enabled
                if (attributes.enable_fpu_combine) {
                    uint32_t combine_base = 15 + num_experts * 4;
                    writer_args[combine_base + 0] = tensor_args.output.buffer()->address();
                }
            }
        }

        // Update dispatch runtime args
        auto& dispatch_args = GetRuntimeArgs(program, shared.dispatch_kernel_id, shared.dispatch_core);
        if (shared.fabric_dispatch_enabled) {
            dispatch_args[0] = tensor_args.hidden_states_rm->buffer()->address();
            dispatch_args[1] = tensor_args.staging_buf->buffer()->address();
            dispatch_args[2] = tensor_args.staging_buf->buffer()->address();  // remote (same addr assumption)
            dispatch_args[3] = tensor_args.pkt_buf.buffer()->address();
        } else {
            dispatch_args[0] = tensor_args.hidden_states.buffer()->address();
            dispatch_args[1] = tensor_args.pkt_buf.buffer()->address();
        }

        // Rebuild combine args (routing metadata changes each forward pass)
        std::vector<uint32_t> combine_args = {
            static_cast<uint32_t>(tensor_args.output.buffer()->address()),
            n_tiles_dn,
            output_M_tiles,
            num_experts,
        };
        combine_args.insert(combine_args.end(), combine_metadata.begin(), combine_metadata.end());

        if (attributes.enable_fabric_reduce && shared.reduce_core.has_value()) {
            // Re-append reduce_core NOC coords + sem address
            auto& existing_combine = GetRuntimeArgs(program, shared.combine_kernel_id, shared.combine_core);
            size_t base = existing_combine.size() - 3;
            combine_args.push_back(existing_combine[base]);
            combine_args.push_back(existing_combine[base + 1]);
            combine_args.push_back(existing_combine[base + 2]);
        }

        SetRuntimeArgs(program, shared.combine_kernel_id, shared.combine_core, combine_args);

        // Update fabric reduce runtime args (buffer addresses)
        if (attributes.enable_fabric_reduce && shared.reduce_kernel_id.has_value()) {
            auto& reduce_args = GetRuntimeArgs(program, *shared.reduce_kernel_id, *shared.reduce_core);
            reduce_args[0] = tensor_args.output.buffer()->address();
            if (tensor_args.reduce_recv_buf.has_value()) {
                auto recv_addr = tensor_args.reduce_recv_buf->buffer()->address();
                reduce_args[1] = recv_addr;  // local recv_buf
                reduce_args[2] = recv_addr;  // remote recv_buf (same address assumption)
            }
        }

        ++device_index;
    }
}

}  // namespace ttnn::operations::experimental::prefill_moe_compute
