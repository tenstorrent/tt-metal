// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_bincount_program_factory.hpp"
#include "masked_bincount_device_operation_types.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

namespace ttnn::experimental::prim::qsr {

ttnn::device_operation::ProgramArtifacts MaskedBincountProgramFactory::create_program_artifacts(
    const MaskedBincountParams& operation_attributes,
    const MaskedBincountInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    const KernelSpecName BRISC{"brisc"};
    const KernelSpecName NCRISC{"ncrisc"};
    const ScratchpadSpecName IN_BRISC{"in_brisc"};
    const ScratchpadSpecName OUT{"out"};
    const ScratchpadSpecName IN_NCRISC{"in_ncrisc"};
    const ScratchpadSpecName GATHER_TMP{"gather_tmp"};
    const ScratchpadSpecName MASK_SCRATCH{"mask_scratch"};
    const ScratchpadSpecName NCRISC_PARTIAL{"ncrisc_partial"};
    const SemaphoreSpecName INIT_SEM{"init_sem"};
    const SemaphoreSpecName DONE_SEM{"done_sem"};
    const SemaphoreSpecName GATHER_SEM{"gather_sem"};
    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};
    const TensorParamName MASK{"mask"};

    const auto& input = tensor_args.input_tensor;
    const auto& expert_mask = tensor_args.expert_mask;

    uint32_t n_routed_experts = operation_attributes.n_routed_experts;

    // Input is TILE + interleaved (no shard spec). Each core gets a contiguous range of the LOGICAL token rows; it
    // reads the TILE pages covering them from interleaved memory and untiles in-kernel (a range may start mid-tile).
    // The split must use the logical count: the padded rows of the last tile hold no picks, and counting them would
    // add phantom tokens to whichever expert the padding value (0) names.
    const uint32_t tile_h = input.tensor_spec().page_config().get_tile().get_height();
    const uint32_t tokens = input.logical_shape()[0];

    // 8 cores (one row): 5120 picks are ~320 per RISC, counted locally; the collector then gathers the 7 other
    // histograms at once. (The previous 64-core binomial tree put 21 sequential merges on the critical path.)
    tt::tt_metal::CoreRangeSet all_cores(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(7, 0)));
    uint32_t num_cores = all_cores.num_cores();
    TT_FATAL(tokens % num_cores == 0, "Token count ({}) must be divisible by the {}-core grid", tokens, num_cores);
    uint32_t shard_height = tokens / num_cores;  // rows per core

    uint32_t h_brisc = shard_height / 2;
    uint32_t h_ncrisc = shard_height - h_brisc;

    // Max TILE pages a RISC's row range can span (misaligned ranges straddle one extra tile).
    uint32_t max_tiles_brisc = (h_brisc + tile_h - 1) / tile_h + 1;
    uint32_t max_tiles_ncrisc = (h_ncrisc + tile_h - 1) / tile_h + 1;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();
    auto* mask_buffer = expert_mask.buffer();
    TT_FATAL(src_buffer != nullptr, "input buffer must be allocated on device");
    TT_FATAL(dst_buffer != nullptr, "output buffer must be allocated on device");
    TT_FATAL(mask_buffer != nullptr, "expert_mask buffer must be allocated on device");

    uint32_t input_page_size = src_buffer->aligned_page_size();  // one TILE (32x32 uint16)
    uint32_t output_page_size = dst_buffer->aligned_page_size();
    uint32_t mask_page_size = mask_buffer->aligned_page_size();

    auto all_cores_vec = tt::tt_metal::corerange_to_cores(all_cores, num_cores, true);
    tt::tt_metal::CoreCoord collector_core = all_cores_vec[0];
    const tt::tt_metal::distributed::MeshDevice* device = input.device();
    auto collector_noc = device->worker_core_from_logical_core(collector_core);

    // --- Scratchpads ---
    // All of this op's L1 buffers are raw scratch: the kernels address them directly and never use FIFO operations.
    // The ones both RISCs of a core touch are multi-bound, synchronized by the kernels through init_sem / done_sem.
    auto shared_scratchpad = [](const ScratchpadSpecName& name, uint32_t size) {
        ScratchpadSpec spec{.unique_id = name, .size_per_node = size};
        spec.advanced_options.allow_instance_multi_binding = true;
        return spec;
    };
    Group<ScratchpadSpec> scratchpads = {
        // BRISC input pages (per-shard)
        ScratchpadSpec{.unique_id = IN_BRISC, .size_per_node = max_tiles_brisc * input_page_size},
        // local output histogram
        shared_scratchpad(OUT, output_page_size),
        // NCRISC input pages (per-shard)
        ScratchpadSpec{.unique_id = IN_NCRISC, .size_per_node = max_tiles_ncrisc * input_page_size},
        // gather temp buffer (collector reads remote histograms here)
        shared_scratchpad(GATHER_TMP, (num_cores - 1) * output_page_size),
        // expert dispatch table (INT32, one value per expert; negative = absent, non-negative = present)
        shared_scratchpad(MASK_SCRATCH, mask_page_size),
        // NCRISC's partial histogram, published for BRISC to add.
        shared_scratchpad(NCRISC_PARTIAL, output_page_size),
    };

    // --- Semaphores ---
    Group<SemaphoreSpec> semaphores = {
        SemaphoreSpec{.unique_id = INIT_SEM, .target_nodes = all_cores},
        SemaphoreSpec{.unique_id = DONE_SEM, .target_nodes = all_cores},
        SemaphoreSpec{.unique_id = GATHER_SEM, .target_nodes = all_cores},
    };

    // Both RISCs run the same source, so their specs bind the same names; only the input pages differ per RISC.
    auto make_kernel_spec = [&](const KernelSpecName& name,
                                const ScratchpadSpecName& in_scratchpad,
                                uint32_t h_count,
                                uint32_t is_initializer,
                                DataMovementProcessor processor,
                                NOC noc) {
        return KernelSpec{
            .unique_id = name,
            .source = std::filesystem::path(
                "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/masked_bincount/device/kernels/"
                "reader_masked_bincount.cpp"),
            .semaphore_bindings =
                {
                    SemaphoreBinding{.semaphore_spec_name = INIT_SEM, .accessor_name = "init"},
                    SemaphoreBinding{.semaphore_spec_name = DONE_SEM, .accessor_name = "done"},
                    SemaphoreBinding{.semaphore_spec_name = GATHER_SEM, .accessor_name = "gather"},
                },
            .scratchpad_bindings =
                {
                    ScratchpadBinding{.scratchpad_spec_name = in_scratchpad, .accessor_name = "in"},
                    ScratchpadBinding{.scratchpad_spec_name = OUT, .accessor_name = "out"},
                    ScratchpadBinding{.scratchpad_spec_name = GATHER_TMP, .accessor_name = "gather_tmp"},
                    ScratchpadBinding{.scratchpad_spec_name = MASK_SCRATCH, .accessor_name = "mask"},
                    ScratchpadBinding{.scratchpad_spec_name = NCRISC_PARTIAL, .accessor_name = "ncrisc_partial"},
                },
            .tensor_bindings =
                {
                    TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"},
                    TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"},
                    TensorBinding{.tensor_parameter_name = MASK, .accessor_name = "mask"},
                },
            .compile_time_args =
                {
                    {"input_page_size", input_page_size},
                    {"output_page_size", output_page_size},
                    {"h_count", h_count},
                    {"num_experts_per_token", operation_attributes.num_experts_per_token},
                    {"n_routed_experts", n_routed_experts},
                    {"is_initializer", is_initializer},
                    {"mask_page_size", mask_page_size},
                    {"tile_h", tile_h},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"h_start", "num_receive", "parent_noc_x", "parent_noc_y"},
                    .runtime_array_args = {{.name = "children_noc", .size = 2 * (num_cores - 1)}},
                },
            .hw_config =
                DataMovementHardwareConfig{
                    .config_1xx =
                        DataMovementHardwareConfig::DataMovement1XXConfig{
                            .processor = processor,
                            .noc = noc,
                        },
                },
        };
    };

    // --- BRISC kernel (RISCV_0, NOC 0) ---
    KernelSpec brisc_spec =
        make_kernel_spec(BRISC, IN_BRISC, h_brisc, 1, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
    // --- NCRISC kernel (RISCV_1, NOC 1) ---
    KernelSpec ncrisc_spec =
        make_kernel_spec(NCRISC, IN_NCRISC, h_ncrisc, 0, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    // --- Per-core runtime args (tree reduction) ---
    KernelRunArgs brisc_run_args{.kernel = BRISC};
    KernelRunArgs ncrisc_run_args{.kernel = NCRISC};
    for (uint32_t i = 0; i < all_cores_vec.size(); i++) {
        uint32_t page_offset = i * shard_height;

        // Flat reduction: core 0 collects from every other core; the others signal core 0.
        std::vector<uint32_t> children_noc(2 * (num_cores - 1), 0);
        uint32_t num_receive = 0;
        if (i == 0) {
            for (uint32_t c = 1; c < num_cores; c++) {
                auto child_noc = device->worker_core_from_logical_core(all_cores_vec[c]);
                children_noc[2 * num_receive] = child_noc.x;
                children_noc[2 * num_receive + 1] = child_noc.y;
                num_receive++;
            }
        }
        uint32_t parent_noc_x = 0xFFFFFFFF;
        uint32_t parent_noc_y = 0xFFFFFFFF;
        if (i > 0) {
            parent_noc_x = collector_noc.x;
            parent_noc_y = collector_noc.y;
        }

        AddRuntimeArgsForNode(
            brisc_run_args.runtime_arg_values,
            all_cores_vec[i],
            {
                {"h_start", page_offset},
                {"num_receive", num_receive},
                {"parent_noc_x", parent_noc_x},
                {"parent_noc_y", parent_noc_y},
            });
        brisc_run_args.runtime_array_arg_values["children_noc"][all_cores_vec[i]] = children_noc;
        // NCRISC returns before the reduction phase, so it never reads the parent / children args.
        AddRuntimeArgsForNode(
            ncrisc_run_args.runtime_arg_values,
            all_cores_vec[i],
            {
                {"h_start", page_offset + h_brisc},
                {"num_receive", 0u},
                {"parent_noc_x", 0xFFFFFFFFu},
                {"parent_noc_y", 0xFFFFFFFFu},
            });
        ncrisc_run_args.runtime_array_arg_values["children_noc"][all_cores_vec[i]] =
            std::vector<uint32_t>(2 * (num_cores - 1), 0);
    }

    ProgramSpec spec{
        .name = "masked_bincount",
        .kernels = {std::move(brisc_spec), std::move(ncrisc_spec)},
        .semaphores = std::move(semaphores),
        .scratchpads = std::move(scratchpads),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = SRC, .spec = input.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = DST, .spec = tensor_return_value.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = MASK, .spec = expert_mask.mesh_tensor().tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {BRISC, NCRISC},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.reserve(2);
    run_args.kernel_run_args.push_back(std::move(brisc_run_args));
    run_args.kernel_run_args.push_back(std::move(ncrisc_run_args));
    run_args.tensor_args.emplace(SRC, input.mesh_tensor());
    run_args.tensor_args.emplace(DST, tensor_return_value.mesh_tensor());
    run_args.tensor_args.emplace(MASK, expert_mask.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::experimental::prim::qsr
