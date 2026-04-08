// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include "gdn_fused_device_operation_types.hpp"
#include "gdn_fused_program_factory.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"
#include <unordered_map>

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

using namespace tt::constants;

// Fixed constants matching the Python kernel
static constexpr uint32_t Kt = 4;                         // key dim tiles (128 / 32)
static constexpr uint32_t Vt = 4;                         // value dim tiles (128 / 32)
static constexpr uint32_t STATE_TILES = Kt * Vt;          // 16
static constexpr uint32_t BF16_TILE_BYTES = 32 * 32 * 2;  // 2048
static constexpr uint32_t PACKED_REDUCE_SCALER = 0x3F803F80;

static const std::string READER_FUSED_PATH =
    "models/demos/qwen35_27b/tt/gdn_kernel/kernels/dataflow/reader_gdn_fused.cpp";
static const std::string WRITER_FUSED_PATH =
    "models/demos/qwen35_27b/tt/gdn_kernel/kernels/dataflow/writer_gdn_fused.cpp";
static const std::string COMPUTE_FUSED_PATH = "models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_fused.cpp";

// Helper: create a single CB with BF16 tile format
static void create_gdn_cb(uint32_t cb_index, uint32_t num_tiles, Program& program, const CoreRangeSet& core_ranges) {
    auto cb_config = CircularBufferConfig(num_tiles * BF16_TILE_BYTES, {{cb_index, tt::DataFormat::Float16_b}});
    cb_config.set_page_size(cb_index, BF16_TILE_BYTES);
    CreateCircularBuffer(program, core_ranges, cb_config);
}

GdnFusedProgramFactory::cached_program_t GdnFusedProgramFactory::create(
    const GdnFusedParams& operation_attributes, const GdnFusedInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    Program program{};

    auto* device = tensor_args.conv_out.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t max_cores = grid.x * grid.y;

    uint32_t num_pairs_total = operation_attributes.num_pairs_total;
    uint32_t num_cores = std::min({operation_attributes.num_cores, num_pairs_total, max_cores});
    uint32_t pairs_per_core = num_pairs_total / num_cores;
    uint32_t remainder = num_pairs_total % num_cores;

    bool state_in_l1 = operation_attributes.state_in_l1;
    bool state_is_sharded = operation_attributes.state_is_sharded;
    uint32_t state_l1_flag = state_in_l1 ? 1 : 0;
    uint32_t sharded_flag = state_is_sharded ? 1 : 0;

    uint32_t Nv_TP = operation_attributes.Nv_TP;
    uint32_t Nk_TP = operation_attributes.Nk_TP;
    uint32_t repeat_factor = operation_attributes.repeat_factor;
    uint32_t key_dim_tp = operation_attributes.key_dim_tp;
    uint32_t key_tile_off = key_dim_tp / 32;
    uint32_t v_tile_off = 2 * key_tile_off;

    // Build core coordinates
    std::vector<CoreCoord> core_coords;
    core_coords.reserve(num_cores);
    for (uint32_t i = 0; i < num_cores; i++) {
        core_coords.emplace_back(i % grid.x, i / grid.x);
    }

    // Build the core range set for all cores (used for CBs)
    std::vector<CoreRange> all_core_ranges;
    all_core_ranges.reserve(num_cores);
    for (const auto& cc : core_coords) {
        all_core_ranges.emplace_back(cc, cc);
    }
    CoreRangeSet core_ranges(all_core_ranges);

    // Create circular buffers (28 CBs, all BF16 tile format)
    create_gdn_cb(0, Kt, program, core_ranges);           // cb_q_raw
    create_gdn_cb(1, Kt, program, core_ranges);           // cb_k_raw
    create_gdn_cb(2, Kt, program, core_ranges);           // cb_k_col
    create_gdn_cb(3, Vt, program, core_ranges);           // cb_v
    create_gdn_cb(4, 1, program, core_ranges);            // cb_g
    create_gdn_cb(5, 1, program, core_ranges);            // cb_beta
    create_gdn_cb(6, STATE_TILES, program, core_ranges);  // cb_state_in
    create_gdn_cb(7, STATE_TILES, program, core_ranges);  // cb_state_b
    create_gdn_cb(8, STATE_TILES, program, core_ranges);  // cb_state_out
    create_gdn_cb(9, 1, program, core_ranges);            // cb_a
    create_gdn_cb(10, 1, program, core_ranges);           // cb_b
    create_gdn_cb(12, 1, program, core_ranges);           // cb_neg_exp_A
    create_gdn_cb(13, 1, program, core_ranges);           // cb_dt_bias
    create_gdn_cb(14, Vt, program, core_ranges);          // cb_norm_w (persistent)
    create_gdn_cb(15, 1, program, core_ranges);           // cb_scale (persistent)
    create_gdn_cb(16, Vt, program, core_ranges);          // cb_out
    create_gdn_cb(17, Kt, program, core_ranges);          // cb_q (normed)
    create_gdn_cb(18, Kt, program, core_ranges);          // cb_k_row (normed)
    create_gdn_cb(21, 1, program, core_ranges);           // cb_scratch
    create_gdn_cb(24, 1, program, core_ranges);           // cb_exp_g
    create_gdn_cb(25, Vt, program, core_ranges);          // cb_kv_mem
    create_gdn_cb(26, Vt, program, core_ranges);          // cb_delta
    create_gdn_cb(27, Vt, program, core_ranges);          // cb_delta_s
    create_gdn_cb(28, Kt, program, core_ranges);          // cb_sq_acc
    create_gdn_cb(29, 1, program, core_ranges);           // cb_tmp
    create_gdn_cb(31, 1, program, core_ranges);           // cb_rms_scale (persistent)
    create_gdn_cb(19, 1, program, core_ranges);           // cb_reduce_scaler (persistent)
    create_gdn_cb(20, 1, program, core_ranges);           // cb_rms_eps (persistent)

    // Per-core pair assignments
    std::vector<uint32_t> core_pair_counts(num_cores);
    uint32_t pair_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        uint32_t n = pairs_per_core + (i < remainder ? 1 : 0);
        core_pair_counts[i] = n;
    }

    // Group cores by pair count (matching Python's groups dict)
    std::unordered_map<uint32_t, std::vector<uint32_t>> groups;  // n_pairs -> core indices
    for (uint32_t i = 0; i < num_cores; i++) {
        groups[core_pair_counts[i]].push_back(i);
    }

    // Shared variables for caching
    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<uint32_t> core_to_reader_group(num_cores);

    uint32_t group_idx = 0;
    for (auto& [n_pairs, core_indices] : groups) {
        if (n_pairs == 0) {
            continue;
        }

        // Build CoreRangeSet for this group
        std::vector<CoreRange> group_ranges;
        group_ranges.reserve(core_indices.size());
        for (uint32_t ci : core_indices) {
            group_ranges.emplace_back(core_coords[ci], core_coords[ci]);
        }
        CoreRangeSet group_core_ranges(group_ranges);

        // Reader compile-time args
        std::vector<uint32_t> reader_ct = {
            Kt,
            Vt,
            BF16_TILE_BYTES,
            state_l1_flag,
            PACKED_REDUCE_SCALER,
            Nv_TP,
            Nk_TP,
            repeat_factor,
            key_tile_off,
            v_tile_off,
            sharded_flag,
        };

        // Writer compile-time args
        std::vector<uint32_t> writer_ct = {
            Kt,
            Vt,
            BF16_TILE_BYTES,
            state_l1_flag,
            0,
            0,
            sharded_flag,
        };

        // Compute compile-time args
        std::vector<uint32_t> compute_ct = {Kt, Vt, n_pairs};

        auto reader_kid =
            CreateKernel(program, READER_FUSED_PATH, group_core_ranges, ReaderDataMovementConfig(reader_ct));

        auto writer_kid =
            CreateKernel(program, WRITER_FUSED_PATH, group_core_ranges, WriterDataMovementConfig(writer_ct));

        CreateKernel(
            program,
            COMPUTE_FUSED_PATH,
            group_core_ranges,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .unpack_to_dest_mode = {},
                .bfp8_pack_precise = false,
                .math_approx_mode = false,
                .compile_args = compute_ct});

        reader_kernel_ids.push_back(reader_kid);
        writer_kernel_ids.push_back(writer_kid);

        for (uint32_t ci : core_indices) {
            core_to_reader_group[ci] = group_idx;
        }
        group_idx++;
    }

    // Set runtime args per core
    pair_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        uint32_t n = core_pair_counts[i];
        uint32_t grp = core_to_reader_group[i];

        SetRuntimeArgs(
            program,
            reader_kernel_ids[grp],
            core_coords[i],
            {
                tensor_args.conv_out.buffer()->address(),      // 0
                tensor_args.a_fused.buffer()->address(),       // 1
                tensor_args.b_fused.buffer()->address(),       // 2
                tensor_args.neg_exp_A.buffer()->address(),     // 3
                tensor_args.dt_bias.buffer()->address(),       // 4
                tensor_args.norm_w.buffer()->address(),        // 5
                tensor_args.scale_tt.buffer()->address(),      // 6
                tensor_args.rms_scale_tt.buffer()->address(),  // 7
                tensor_args.state.buffer()->address(),         // 8
                tensor_args.rms_eps_tt.buffer()->address(),    // 9
                pair_offset,                                   // 10
                n,                                             // 11
            });

        SetRuntimeArgs(
            program,
            writer_kernel_ids[grp],
            core_coords[i],
            {
                tensor_args.output.buffer()->address(),  // 0
                tensor_args.state.buffer()->address(),   // 1
                pair_offset,                             // 2
                n,                                       // 3
            });

        pair_offset += n;
    }

    shared_variables_t shared_variables{
        .reader_kernel_ids = std::move(reader_kernel_ids),
        .writer_kernel_ids = std::move(writer_kernel_ids),
        .cores = std::move(core_coords),
        .core_to_reader_group = std::move(core_to_reader_group),
    };

    return cached_program_t(std::move(program), std::move(shared_variables));
}

void GdnFusedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const GdnFusedParams& /*operation_attributes*/,
    const GdnFusedInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    // Only 3 buffer addresses change per call: conv_out, a_fused, b_fused
    uint32_t conv_out_addr = tensor_args.conv_out.buffer()->address();
    uint32_t a_addr = tensor_args.a_fused.buffer()->address();
    uint32_t b_addr = tensor_args.b_fused.buffer()->address();

    for (uint32_t i = 0; i < shared_vars.cores.size(); i++) {
        const auto& core = shared_vars.cores[i];
        uint32_t grp = shared_vars.core_to_reader_group[i];

        auto& reader_args = GetRuntimeArgs(program, shared_vars.reader_kernel_ids[grp]);
        auto& runtime_args = reader_args.at(core.x).at(core.y);
        runtime_args[0] = conv_out_addr;
        runtime_args[1] = a_addr;
        runtime_args[2] = b_addr;
    }
}

GdnFusedMeshWorkloadFactory::cached_mesh_workload_t GdnFusedMeshWorkloadFactory::create_mesh_workload(
    const GdnFusedParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const GdnFusedInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program =
                GdnFusedProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void GdnFusedMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const GdnFusedParams& operation_attributes,
    const GdnFusedInputs& tensor_args,
    Tensor& tensor_return_value) {
    GdnFusedProgramFactory program_factory;

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

        ttnn::device_operation::mesh_device_operation_utils::apply_override_runtime_arguments(
            program_factory,
            program,
            shared_variables,
            operation_attributes,
            *(coordinate_range.begin()),
            tensor_args,
            tensor_return_value);
    }
}

}  // namespace ttnn::experimental::prim
