// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The device-independent half of the opt-in model selector (ttnn.CONFIG.matmul_model_config): the loaded model,
// candidate scoring and ranking, and the process-wide pick cache. The ttnn side (tensors -> problem, validation,
// live L1 budget, the hook in matmul) is in matmul_model_hook.hpp. Std library only.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "ttnn/operations/matmul/device/config/model/gbdt_model.hpp"
#include "ttnn/operations/matmul/device/config/model/matmul_candidates.hpp"
#include "ttnn/operations/matmul/device/config/model/matmul_model_features.hpp"

namespace ttnn::operations::matmul::model {

// The matmul config model: the analytic base latency and features (matmul_model_features.hpp) and the GBDT residual
// read from a blob (bench/export/export_gbdt.py --format blob in tt-matmul-predictor). Prediction for a candidate:
// base_ns * exp(gbdt(features)).
class ConfigModel {
public:
    // Reads and checks the blob (its feature names must be the ones compute_model_input produces). nullptr with the
    // reason in `error` on any failure; never throws.
    static std::unique_ptr<const ConfigModel> load_file(const std::string& path, std::string& error);
    static std::unique_ptr<const ConfigModel> from_blob(std::span<const std::byte> blob, std::string& error);

    // sha256 of the LightGBM model the blob was exported from (identifies the model in cache keys and logs)
    const std::string& sha256() const { return storage_.source_sha256(); }

    // Predicted device time (ns) of each candidate; NaN where the model can't score it (status != ok).
    std::vector<double> predict_ns(
        const MatmulProblem& problem, std::span<const Candidate> candidates, Arch arch = Arch::kBlackholeP150) const;

private:
    explicit ConfigModel(GbdtModelStorage storage) : storage_(std::move(storage)) {}
    GbdtModelStorage storage_;
};

// Indices of the candidates with a finite prediction, lowest first; ties keep candidate order (so the first of equal
// predictions wins, as a plain argmin would pick).
std::vector<std::size_t> rank_by_prediction(std::span<const double> predicted_ns);

// One decision of the selector, as cached: the chosen candidate (nullopt: fall back to the default selection), its
// circular buffer bytes (re-checked against the live L1 budget on every hit), and a one-line description.
struct Pick {
    std::optional<Candidate> candidate;
    std::uint32_t cb_bytes = 0;
    // get_max_l1_space when the pick was made: the default's K blocking, which a pick keeps, depends on it
    std::uint32_t max_l1_space = 0;
    std::string description;
    // No candidate, for lack of L1: the per-core budget it had (the pick is made again when more is free)
    std::optional<std::uint32_t> failed_budget;
};

// Process-wide map from a matmul's full signature (shapes, dtypes, layouts, memory configs, transposes, bias,
// compute kernel config, device grid and arch) to its pick. Thread safe.
class PickCache {
public:
    std::optional<Pick> find(const std::string& key) const;
    // Keeps the first pick stored under a key (concurrent first calls may both score; either result is the same)
    Pick insert(const std::string& key, const Pick& pick);
    // Stores a pick made again for a key (a pick that failed for lack of L1, retried with more free)
    Pick replace(const std::string& key, Pick pick);
    std::size_t size() const;
    void clear();

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Pick> picks_;
};

}  // namespace ttnn::operations::matmul::model
