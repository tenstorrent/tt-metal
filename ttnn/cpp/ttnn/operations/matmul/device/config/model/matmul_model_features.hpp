// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Input side of the matmul config model: (matmul problem, candidate program config) -> the analytic base latency
// `base_ns` and the 57 features the GBDT residual consumes. Prediction = base_ns * exp(gbdt(features)).
//
//   base_ns, x_*  an analytic latency equation with constants fitted on the device, its NoC link load (transposed 2D
//                 multicast routed as transposed), its DRAM bank model and 36 mechanism features
//   y_*           the equation's per-resource terms
//   z_*           the L1 per-core compute model
// The features must be exactly the ones the model was trained on: do not change them without retraining.
// Std library only; no ttnn types (an adapter from ttnn tensors / program configs lives elsewhere).
//
// Units: sizes in elements unless stated; the model works in 32x32 tiles internally (ceil(dim / 32)); times in ns
// (base_ns) or device cycles (internal terms, y_* features, z_l1c).

#include <array>
#include <functional>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ttnn/operations/matmul/device/config/model/matmul_model_device_params.hpp"

namespace ttnn::operations::matmul::model {

enum class DataType : std::uint8_t { kBfloat16, kBfloat8B, kBfloat4B, kFloat32 };

// The model covers 32x32 tiles only
inline constexpr std::uint32_t kTileDim = 32;
// Bytes of a 32x32 tile (tt::tile_size): block-float formats add one shared exponent byte per 16 values
std::uint32_t tile_size_bytes(DataType t);

// The model distinguishes DRAM interleaved, L1 interleaved and "sharded" (every L1 shard layout alike).
enum class MemLayout : std::uint8_t {
    kDramInterleaved,
    kL1Interleaved,
    kL1HeightSharded,
    kL1WidthSharded,
    kL1BlockSharded,
};

// kNone and kRelu cost nothing extra (relu is packer-fused); every other fused activation (gelu, silu, ...) is an
// SFPU pass per output tile.
enum class Activation : std::uint8_t { kNone, kRelu, kSfpu };

// Program config family: MatmulMultiCoreReuseMultiCastProgramConfig (k2D), ...MultiCast1DProgramConfig with
// mcast_in0 = true (k1DMcastIn0) / false (k1DMcastIn1), MatmulMultiCoreReuseProgramConfig (kReuse).
// (The plain MultiCore factory is not modeled.)
enum class Family : std::uint8_t { k2D, k1DMcastIn0, k1DMcastIn1, kReuse };

// The matmul problem: C[batch, M, N] = A[batch, M, K] x B[K, N] (+ bias) (+ activation). M, K, N are the logical
// (post-transpose) sizes in elements; batch is the product of A's leading dims.
struct MatmulProblem {
    std::uint32_t M = 0, K = 0, N = 0;
    std::uint32_t batch = 1;
    DataType a_dtype = DataType::kBfloat16;
    DataType b_dtype = DataType::kBfloat16;
    bool has_out_dtype = false;  // false: the output takes A's dtype (ttnn's default)
    DataType out_dtype = DataType::kBfloat16;
    MemLayout a_mem = MemLayout::kDramInterleaved;
    MemLayout b_mem = MemLayout::kDramInterleaved;
    MemLayout out_mem = MemLayout::kDramInterleaved;
    Fidelity fidelity = Fidelity::kHiFi4;
    bool fp32_dest_acc = false;  // compute kernel config fp32_dest_acc_en
    bool packer_l1_acc = false;  // compute kernel config packer_l1_acc
    bool bias = false;           // fused bias add (linear)
    Activation activation = Activation::kNone;
    bool transpose_a = false;  // A stored K x M (changes only the DRAM bank pattern)
    bool transpose_b = false;  // B stored N x K
    // Not used by the model (listed for the adapter's completeness): dst_full_sync, shard specs, l1 reserve.
};

// A candidate program config. Block sizes are in tiles.
struct MatmulCandidate {
    Family family = Family::k2D;
    std::uint32_t grid_x = 0, grid_y = 0;  // compute_with_storage_grid_size
    std::uint32_t in0_block_w = 0;         // K tiles per K block
    std::uint32_t out_subblock_h = 0, out_subblock_w = 0;
    std::uint32_t per_core_M = 0, per_core_N = 0;
    std::uint32_t out_block_h = 0, out_block_w = 0;  // 0 = not set (= per_core_M / per_core_N); ignored for kReuse
    bool fuse_batch = false;                         // 2D / 1D: batch rows stacked into M
    bool transpose_mcast = false;                    // 2D only
};

enum class ScoreStatus : std::uint8_t {
    kOk,
    kInvalidProblem,           // M, K, N or batch is 0
    kGridExceedsDevice,        // grid_x / grid_y larger than the device worker grid
    kNonPositiveBlock,         // per_core_M, per_core_N or in0_block_w is 0
    kInvalidSubblock,          // out_subblock_h / w is 0 (the equation would give a non-finite base)
    kNeedsMoreCores,           // the config's output blocks need more cores than its grid
    kTransposedExceedsDevice,  // 2D transpose_mcast placement does not fit the device grid
    kNonFinite,                // the equation returned a non-finite value
};
std::string_view to_string(ScoreStatus s);

inline constexpr std::size_t kNumFeatures = 57;
// Feature names in the model's order (a blob with other names is refused at load).
extern const std::array<std::string_view, kNumFeatures> kFeatureNames;

// The equation's per-K-step resource terms, cycles: the in0 / in1 sender's fetch and multicast, the
// chip-wide DRAM / L1 bound, the busiest NoC link, compute and the output block write.
struct EquationTerms {
    double fetch0{}, mcast0{}, fetch1{}, mcast1{}, chip{}, link{}, comp{}, write{}, read{};
};

struct ModelInput {
    ScoreStatus status = ScoreStatus::kOk;  // anything but kOk: the row cannot be scored (base_ns and x_/y_ NaN)
    double base_ns = 0.0;
    EquationTerms terms{};
    double l1_cycles = 0.0;  // L1 per-core compute cycles (NaN outside its measured tables; independent of status)
    std::array<double, kNumFeatures> features{};  // NaN = missing (LightGBM's missing value)
};

// Optional memo of the two expensive, problem-independent pieces: the NoC unit link loads of a placement and the
// L1 per-core compute cycles of a (subblock, in0_block_w, K blocks, fidelity, per-core block) shape. Results are
// identical with or without it. Entries are keyed without the device: use one cache per DeviceParams. Not thread safe
// (use one per thread); grows with the number of distinct placements / shapes (bounded by the config space).
class ModelCache {
public:
    using Lanes = std::vector<std::pair<double, double>>;  // (in0, in1) load per byte on each loaded link
    const Lanes* find_link(std::uint64_t key) const;
    const Lanes& insert_link(std::uint64_t key, Lanes lanes);
    const double* find_l1(std::uint64_t k0, std::uint64_t k1) const;
    void insert_l1(std::uint64_t k0, std::uint64_t k1, double cycles);
    void clear();
    std::size_t size() const;

private:
    struct PairHash {
        std::size_t operator()(const std::pair<std::uint64_t, std::uint64_t>& k) const {
            // 2^64 / golden ratio: spreads k.first's bits before combining (boost::hash_combine's constant)
            constexpr std::uint64_t kGoldenRatio64 = 0x9E3779B97F4A7C15ull;
            return std::hash<std::uint64_t>{}(k.first * kGoldenRatio64 ^ k.second);
        }
    };
    std::unordered_map<std::uint64_t, Lanes> link_;
    std::unordered_map<std::pair<std::uint64_t, std::uint64_t>, double, PairHash> l1_;
};

ModelInput compute_model_input(
    const MatmulProblem& problem,
    const MatmulCandidate& candidate,
    const DeviceParams& device,
    ModelCache* cache = nullptr);

}  // namespace ttnn::operations::matmul::model
