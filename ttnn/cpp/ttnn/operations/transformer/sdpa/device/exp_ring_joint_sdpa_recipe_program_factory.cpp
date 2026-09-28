// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_builder.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/exp_ring_recipe_cbs.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa_recipe.hpp"

#include <optional>

#include <tt-metalium/constants.hpp>

namespace ttnn::prim {

namespace {

namespace exp_recipes = ttnn::operations::transformer::sdpa::detail;
namespace exp_ring_cbs = ttnn::operations::transformer::sdpa::exp_ring;

// Named precision recipes B/C/D/E on exp ring joint SDPA: the shared streaming recipe keeps one recurrent
// state per pass resident in L1 across the ring (pass-outer, ring-inner). FAST keeps the exp-ring compute
// with the recipe's fidelity/approximation (set by the entry point).
class RecipeExpRingJointCompute final : public exp_ring_joint_sdpa::ComputeVariant {
public:
    exp_ring_joint_sdpa::KernelSources kernel_sources() const override {
        return {
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_reader_recipe.cpp",
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_ring_joint_writer_recipe.cpp",
            "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/exp_ring_joint_sdpa_recipe.cpp",
        };
    }

    void configure(
        const ExpRingJointSDPAParams& args, const ExpRingJointSDPAInputs& tensor_args, bool fp32_dest_acc_en) override {
        TT_FATAL(
            args.precision && *args.precision != ttnn::transformer::SDPAPrecision::FAST,
            "The exp ring recipe program factory serves the named recipes B-E; FAST uses the exp ring compute");
        policy_ = exp_recipes::resolve_precision_policy(
            exp_recipes::select_recipe(*args.precision, tensor_args.input_k.dtype()));
        TT_FATAL(
            fp32_dest_acc_en == policy_->fp32_destination,
            "Named exp ring recipe expects fp32_dest_acc_en={} from its compute config",
            policy_->fp32_destination);
        k_chunk_tiles_ = args.get_k_chunk_size() / tt::constants::TILE_HEIGHT;
        has_logical_n_tensor_ = tensor_args.has_logical_n_tensor();
    }

    // Recipe matmul subblocks are fixed by the recipe schedule: (FP32 ? 1 : 2) x 4.
    std::optional<uint32_t> fixed_subblock_h() const override { return policy_->fp32_destination ? 1u : 2u; }

    // The dense recipe's host rule (SDPA_RECIPE_QK_W / SDPA_RECIPE_PV_W come from the same helper).
    uint32_t fixed_subblock_w(uint32_t tiles) const override { return exp_recipes::recipe_subblock_width(tiles); }

    bool replace_cbs(
        tt::tt_metal::ProgramDescriptor& desc,
        const tt::tt_metal::CoreRangeSet& sdpa_grid,
        uint32_t Sq_chunk_t,
        uint32_t DHt,
        std::map<std::string, std::string>& defines) override {
        // Adopt the recipe CB layout (fixed indices 0-16) in place of the exp-ring CBs. Only K/V gain second
        // handles for the MUX writer (the recipe's c_14 is exp_max_diff, so the exp aliases move to
        // exp_ring::kRecipe{K,V}WriterAliasCb). Q is single-slot: recipes run pass-outer, so each pass's Q
        // chunk is read once, stays resident across its ring iterations and is popped before the next pass
        // reads its own; the recipe's second Q slot would be dead L1.
        auto recipe_program = exp_recipes::recipe_compute_program(*policy_, sdpa_grid, 1, Sq_chunk_t, k_chunk_tiles_, DHt);
        desc.cbs = std::move(recipe_program.cbs);
        if (has_logical_n_tensor_) {
            // The reader publishes the live logical_n (read from DRAM) to compute here; the exp-ring c_13 is
            // the recipe's second denominator. UInt32, as read_tile_value indexes by the CB format.
            constexpr uint32_t kDerivedPageBytes = 64;
            desc.cbs.push_back(tt::tt_metal::CBDescriptor{
                .total_size = kDerivedPageBytes,
                .core_ranges = sdpa_grid,
                .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(exp_ring_cbs::kRecipeDerivedCb),
                    .data_format = tt::DataFormat::UInt32,
                    .page_size = kDerivedPageBytes,
                }}},
            });
        }
        for (auto& cb : desc.cbs) {
            auto& format = cb.format_descriptors.front();
            if (format.buffer_index == 0) {
                cb.total_size = Sq_chunk_t * DHt * format.page_size;
            } else if (format.buffer_index == 1 || format.buffer_index == 2) {
                auto alias = format;
                alias.buffer_index = static_cast<uint8_t>(
                    format.buffer_index == 1 ? exp_ring_cbs::kRecipeKWriterAliasCb
                                             : exp_ring_cbs::kRecipeVWriterAliasCb);
                cb.format_descriptors.push_back(alias);
            }
        }
        auto& recipe_compute = recipe_program.kernels.front();
        for (const auto& [name, value] : recipe_compute.defines) {
            defines[name] = value;
        }
        compute_config_ = recipe_compute.config;
        return true;
    }

    void check_l1(uint64_t cb_bytes, uint64_t usable_l1, uint32_t q_chunk_size) const override {
        TT_FATAL(
            cb_bytes <= usable_l1,
            "Named exp ring SDPA recipe needs {} B of L1 per core at Q{}/K{} but only {} B are usable; use a "
            "smaller q_chunk_size or k_chunk_size",
            cb_bytes,
            q_chunk_size,
            k_chunk_tiles_ * tt::constants::TILE_HEIGHT,
            usable_l1);
    }

    std::optional<tt::tt_metal::KernelDescriptor::ConfigDescriptor> compute_config() const override {
        return compute_config_;
    }

private:
    std::optional<exp_recipes::PrecisionPolicy> policy_;
    uint32_t k_chunk_tiles_ = 0;
    bool has_logical_n_tensor_ = false;
    std::optional<tt::tt_metal::KernelDescriptor::ConfigDescriptor> compute_config_;
};

}  // namespace

tt::tt_metal::WorkloadDescriptor ExpRingJointSDPARecipeProgramFactory::create_workload_descriptor(
    const ExpRingJointSDPAParams& operation_attributes,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    RecipeExpRingJointCompute variant;
    return exp_ring_joint_sdpa::build_workload_descriptor(
        operation_attributes, tensor_args, tensor_return_value, tensor_coords, variant);
}

ExpRingJointSDPARecipeMeshWorkloadFactory::cached_mesh_workload_t
ExpRingJointSDPARecipeMeshWorkloadFactory::create_mesh_workload(
    const ExpRingJointSDPAParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& tensor_return_value) {
    return descriptor_adapter_t::create_mesh_workload(
        operation_attributes, tensor_coords, tensor_args, tensor_return_value);
}

void ExpRingJointSDPARecipeMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const ExpRingJointSDPAParams& operation_attributes,
    const ExpRingJointSDPAInputs& tensor_args,
    ExpRingJointSDPAResult& tensor_return_value) {
    descriptor_adapter_t::apply_descriptor(cached_workload, operation_attributes, tensor_args, tensor_return_value);
    exp_ring_joint_sdpa::apply_semaphore_runtime_args(cached_workload.workload, operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
