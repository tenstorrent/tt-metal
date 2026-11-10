// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One chip's program assembled from its core roles (ChipProgram::build, chip_values), the head's tensor shapes and
// arena cores. The pieces: head_model.cpp (the model), head_engine_program.cpp, head_gather_program.cpp,
// head_router_program.cpp (a builder a core role), head_row_blocks.cpp, head_workload.cpp (DRAM args, the
// workload, cache-hit patching), head_weights.cpp (weight preparation).
#include "head_chip_program.hpp"

namespace ttnn::operations::experimental::moe_block::moe_head {

ProgramDescriptor ChipProgram::build(HeadPatches* patches) {
    // the engine's semaphores (its kernels' fixed ids), also on the watcher
    const CoreRangeSet semaphore_cores = model.engine_cores().merge(cells({model.watcher}));
    for (uint32_t i = 0; i < kNumOwnSemaphores; ++i) {
        program_descriptor.semaphores.push_back(
            SemaphoreDescriptor{.id = i, .core_ranges = semaphore_cores, .initial_value = 0});
    }
    std::vector<CBDescriptor> gather_cbs, router_cbs, engine_cbs;
    take_pushed();
    auto gather_kernels = gather(gather_cbs);
    auto gather_patches = take_pushed();
    auto router_kernels = router(router_cbs);
    auto router_patches = take_pushed();
    auto y_sender_kernels = y_kernels();
    auto y_sender_patches = take_pushed();
    auto engine_kernels = engine(engine_cbs);
    auto engine_patches = take_pushed();
    TT_FATAL(
        gather_patches.size() == gather_kernels.size() && router_patches.size() == router_kernels.size() &&
            y_sender_patches.size() == y_sender_kernels.size() && engine_patches.size() == engine_kernels.size() &&
            kernel_patches.empty(),
        "moe_head: a kernel's address args not carried to the assembly");
    for (auto* role_cbs : {&gather_cbs, &router_cbs, &engine_cbs}) {
        program_descriptor.cbs.insert(program_descriptor.cbs.end(), role_cbs->begin(), role_cbs->end());
    }
    // (the engine BRISC's kernel_lib mcast families attach after the args go to DRAM: attach_engine_mcasts)
    for (auto [role_kernels, role_patches] :
         {std::pair{&gather_kernels, &gather_patches},
          std::pair{&router_kernels, &router_patches},
          std::pair{&y_sender_kernels, &y_sender_patches},
          std::pair{&engine_kernels, &engine_patches}}) {
        for (size_t i = 0; i < role_kernels->size(); ++i) {
            if (patches) {
                for (auto patch : (*role_patches)[i]) {
                    patch.kernel = u32(program_descriptor.kernels.size());
                    patches->args.push_back(patch);
                }
            }
            program_descriptor.kernels.push_back(std::move((*role_kernels)[i]));
        }
    }
    chip_values();
    if (patches) {
        std::stable_sort(patches->args.begin(), patches->args.end(), [](const auto& left, const auto& right) {
            return left.source < right.source;
        });
        for (uint32_t i = 0; i < patches->args.size();) {
            uint32_t j = i + 1;
            while (j < patches->args.size() && patches->args[j].source == patches->args[i].source) {
                ++j;
            }
            patches->groups.emplace_back(i, j);
            i = j;
        }
        for (uint32_t i = 0; i < program_descriptor.cbs.size(); ++i) {
            if (program_descriptor.cbs[i].buffer != nullptr) {
                patches->arena_cbs.push_back({i, program_descriptor.cbs[i].address_offset});
            }
        }
    }
    return std::move(program_descriptor);
}

HeadTensorShapes head_tensor_shapes(const HeadModel& model) {
    const auto& shape = model.config.shape;
    HeadTensorShapes shapes{};
    const bool width = shape.reduce_scatter_dim() == -1;
    shapes.y_rows = width ? shape.rows : shape.rows / model.tp_size;
    shapes.y_columns = width ? shape.hidden_dim / model.tp_size : shape.hidden_dim;
    // (row blocks: the routing rows, the dispatch buffer and the route words hold every block's tokens, each call its
    // own at block x tokens_per_chip; the shared expert's output is the call's)
    shapes.routing_rows_rows = model.router_row_tiles * model.row_blocks;
    shapes.routing_rows_columns = 32 * 2 * model.top_k;
    shapes.dispatch_buffer_rows = model.num_chips * model.tokens_per_chip_all_blocks;
    shapes.dispatch_buffer_columns = shape.latent.value_or(shape.hidden_dim);
    shapes.route_words = route_words(model.num_chips) * model.tokens_per_chip_all_blocks;
    shapes.slot_words = u32(model.routing_slot_capacity_bytes / 4);
    shapes.slots_rows = 2 * model.num_chips;
    const uint32_t num_experts = model.num_chips * model.experts_per_chip;
    shapes.plan_outputs = {
        {{1, num_experts},
         {1, num_experts},
         {1, model.rows_capacity},
         {model.rows_capacity / 32, 64},
         {1, kPlanWords}}};
    shapes.arena_bytes = u32(model.args_offset + kHeadArgsBytes);  // (the regions, then the kernels' args blocks)
    return shapes;
}

CoreRangeSet head_arena_cores(const HeadModel& model) {
    std::vector<CoreCoord> extra = model.planner_cores;
    for (const auto& port : model.all_ports()) {
        extra.push_back(port);
    }
    extra.push_back(model.watcher);
    return model.engine_cores().merge(cells(extra));
}

void ChipProgram::chip_values() {
    static const std::set<std::string> scalars = {
        "ROUTING_EXCHANGE_MY_CHIP",
        "ROUTING_PLANNER_MY_CHIP",
        "ROUTING_PLANNER_PORTS",
        "ROUTING_PLANNER_NUM_PORTS",
        "ROUTER_MY_CHIP",
        "ROUTER_TP_POSITION",
        "WATCHER_ROUTING_ARGS_OFFSET"};
    for (auto& kernel_descriptor : program_descriptor.kernels) {
        const bool compute = std::holds_alternative<tt::tt_metal::ComputeConfigDescriptor>(kernel_descriptor.config);
        for (auto& [name, value] : kernel_descriptor.defines) {
            if (compute) {
                if (name == "ROUTE_WORDS_MY_CHIP") {
                    name = "ROUTER_CHIP_FROM_READY_PAGE";
                    value = "1";
                }
            } else if (scalars.contains(name)) {
                const uint32_t arg_index = u32(kernel_descriptor.common_runtime_args.size());
                kernel_descriptor.common_runtime_args.push_back(u32(std::stoll(value)));
                value = "(moe_block_args::common(" + integer_literal(arg_index) +
                        "))";  // (this chip's value: its block of the DRAM table)
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::moe_block::moe_head
