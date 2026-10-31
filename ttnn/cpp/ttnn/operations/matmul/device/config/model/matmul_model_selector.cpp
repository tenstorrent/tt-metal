// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/model/matmul_model_selector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <system_error>

#include <tt_stl/assert.hpp>

namespace ttnn::operations::matmul::model {

std::unique_ptr<const ConfigModel> ConfigModel::load_file(const std::string& path, std::string& error) {
    if (path.empty()) {
        error = "no model path (ttnn.CONFIG.matmul_model_path)";
        return nullptr;
    }
    // Read with an explicit size: streaming a directory or special file through istreambuf_iterator can throw
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        error = path + " is not a regular file" + (ec ? " (" + ec.message() + ")" : "");
        return nullptr;
    }
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec) {
        error = "cannot stat " + path + ": " + ec.message();
        return nullptr;
    }
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes(size);
    if (!file || !file.read(bytes.data(), static_cast<std::streamsize>(size))) {
        error = "cannot read " + path;
        return nullptr;
    }
    auto model = from_blob(std::as_bytes(std::span<const char>(bytes.data(), bytes.size())), error);
    if (!model) {
        error = path + ": " + error;
    }
    return model;
}

std::unique_ptr<const ConfigModel> ConfigModel::from_blob(std::span<const std::byte> blob, std::string& error) {
    auto storage = GbdtModelStorage::from_blob(blob, error);
    if (!storage) {
        return nullptr;
    }
    const auto& names = storage->feature_names();
    if (names.size() != kNumFeatures || storage->model().num_features != kNumFeatures) {
        error = "model has " + std::to_string(names.size()) + " features, expected " + std::to_string(kNumFeatures);
        return nullptr;
    }
    for (std::size_t i = 0; i < kNumFeatures; ++i) {
        if (names[i] != kFeatureNames[i]) {
            error = "feature " + std::to_string(i) + " is " + names[i] + ", expected " + std::string(kFeatureNames[i]);
            return nullptr;
        }
    }
    return std::unique_ptr<const ConfigModel>(new ConfigModel(std::move(*storage)));
}

std::vector<double> ConfigModel::predict_ns(
    const MatmulProblem& problem, std::span<const Candidate> candidates, Arch arch) const {
    const DeviceParams& device = device_params(arch);
    // The memo of placement link loads and L1 compute tables is problem independent but device specific; one per
    // thread and arch
    thread_local std::array<ModelCache, 2> caches;
    const auto arch_index = static_cast<std::size_t>(arch);
    TT_FATAL(arch_index < caches.size(), "matmul model: no model cache for arch {}", arch_index);
    ModelCache& cache = caches[arch_index];
    std::vector<double> predicted(candidates.size(), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> rows;
    std::vector<double> base;
    std::vector<std::size_t> scored;
    rows.reserve(candidates.size() * kNumFeatures);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const ModelInput in = compute_model_input(problem, to_model_candidate(candidates[i]), device, &cache);
        if (in.status != ScoreStatus::kOk || !std::isfinite(in.base_ns)) {
            continue;
        }
        rows.insert(rows.end(), in.features.begin(), in.features.end());
        base.push_back(in.base_ns);
        scored.push_back(i);
    }
    std::vector<double> raw(scored.size());
    storage_.model().predict_batch(rows, raw);
    for (std::size_t j = 0; j < scored.size(); ++j) {
        predicted[scored[j]] = base[j] * std::exp(raw[j]);
    }
    return predicted;
}

std::vector<std::size_t> rank_by_prediction(std::span<const double> predicted_ns) {
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < predicted_ns.size(); ++i) {
        if (std::isfinite(predicted_ns[i])) {
            order.push_back(i);
        }
    }
    std::stable_sort(
        order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return predicted_ns[a] < predicted_ns[b]; });
    return order;
}

std::optional<Pick> PickCache::find(const std::string& key) const {
    std::lock_guard lock(mutex_);
    const auto it = picks_.find(key);
    if (it == picks_.end()) {
        return std::nullopt;
    }
    return it->second;
}

Pick PickCache::insert(const std::string& key, const Pick& pick) {
    std::lock_guard lock(mutex_);
    return picks_.try_emplace(key, pick).first->second;
}

Pick PickCache::replace(const std::string& key, Pick pick) {
    std::lock_guard lock(mutex_);
    return picks_.insert_or_assign(key, std::move(pick)).first->second;
}

std::size_t PickCache::size() const {
    std::lock_guard lock(mutex_);
    return picks_.size();
}

void PickCache::clear() {
    std::lock_guard lock(mutex_);
    picks_.clear();
}

}  // namespace ttnn::operations::matmul::model
