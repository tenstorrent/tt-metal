// SPDX-License-Identifier: Apache-2.0

#include <tt/unified/core>
#include "experimental/kernel_args.h"

namespace u = tt::unified;

constexpr uint32_t kKeyChunks = 4;
constexpr auto kRows = u::Axis::Cols;

using Queries = u::Shape<2, 2>;
using KeysTransposed = u::Shape<2, 2>;
using Values = u::Shape<2, 2>;
using Scores = u::Shape<2, 2>;
using Ones = u::Shape<2, 1>;
using Vec = u::reduce_shape<Scores, kRows>;
using Out = u::Shape<2, 2>;

void kernel_main() {
    constexpr uint32_t kDfbQ = get_arg(args::dfb_q);
    constexpr uint32_t kDfbK = get_arg(args::dfb_k);
    constexpr uint32_t kDfbV = get_arg(args::dfb_v);
    constexpr uint32_t kDfbOnes = get_arg(args::dfb_ones);
    constexpr uint32_t kDfbScaler = get_arg(args::dfb_scaler);
    constexpr uint32_t kDfbScores = get_arg(args::dfb_scores);
    constexpr uint32_t kDfbChunkMax = get_arg(args::dfb_chunk_max);
    constexpr uint32_t kDfbProb = get_arg(args::dfb_prob);
    constexpr uint32_t kDfbChunkSum = get_arg(args::dfb_chunk_sum);
    constexpr uint32_t kDfbNewMax = get_arg(args::dfb_new_max);
    constexpr uint32_t kDfbCorrection = get_arg(args::dfb_correction);
    constexpr uint32_t kDfbRescaled = get_arg(args::dfb_rescaled);
    constexpr uint32_t kDfbWeightedV = get_arg(args::dfb_weighted_v);
    constexpr uint32_t kDfbReciprocal = get_arg(args::dfb_reciprocal);
    constexpr uint32_t kDfbOut = get_arg(args::dfb_out);
    constexpr uint32_t kDfbMax = get_arg(args::dfb_max);
    constexpr uint32_t kDfbSum = get_arg(args::dfb_sum);
    constexpr uint32_t kDfbAcc = get_arg(args::dfb_acc);

    const auto q_acc = TensorAccessor(tensor::q);
    const auto k_acc = TensorAccessor(tensor::k);
    const auto v_acc = TensorAccessor(tensor::v);
    const auto ones_acc = TensorAccessor(tensor::ones);
    const auto out = TensorAccessor(tensor::out);

    u::matmul_init<Queries, KeysTransposed>(kDfbQ, kDfbK, kDfbOut);

    u::Storage<Queries> q_storage(kDfbQ);
    u::Storage<KeysTransposed> k_storage(kDfbK);
    u::Storage<Values> v_storage(kDfbV);
    u::Storage<Ones> ones_storage(kDfbOnes);
    u::Storage<u::Shape<1, 1>> scaler_storage(kDfbScaler);
    u::Storage<Scores> scores_storage(kDfbScores);
    u::Storage<Vec> chunk_max_storage(kDfbChunkMax);
    u::Storage<Scores> prob_storage(kDfbProb);
    u::Storage<Vec> chunk_sum_storage(kDfbChunkSum);
    u::Storage<Vec> new_max_storage(kDfbNewMax);
    u::Storage<Vec> correction_storage(kDfbCorrection);
    u::Storage<Out> rescaled_storage(kDfbRescaled);
    u::Storage<Out> weighted_v_storage(kDfbWeightedV);
    u::Storage<Vec> reciprocal_storage(kDfbReciprocal);
    u::Storage<Vec> max_storage(kDfbMax);
    u::Storage<Vec> sum_storage(kDfbSum);
    u::Storage<Out> acc_storage(kDfbAcc);
    u::Storage<Out> out_storage(kDfbOut);

    u::ComputeBlock scaler = u::fill_reduce_scaler<1>(scaler_storage);
    u::ComputeBlock column_of_ones = u::noc_load<0>(ones_storage, ones_acc, 0).wait();
    u::ComputeBlock q = u::noc_load<0>(q_storage, q_acc, 0).wait();

    u::RetainedBlock<Vec> running_max;
    u::RetainedBlock<Vec> running_sum;
    u::RetainedBlock<Out> running_out;

    for (uint32_t j = 0; j < kKeyChunks; ++j) {
        u::ComputeBlock k = u::noc_load<0>(k_storage, k_acc, j).wait();
        u::ComputeBlock v = u::noc_load<0>(v_storage, v_acc, j).wait();

        u::ComputeBlock scores = scores_storage.store(u::matmul(q, k));
        u::ComputeBlock chunk_max = chunk_max_storage.store(u::reduce_max<kRows>(scores, scaler));

        if (j == 0) {
            u::ComputeBlock prob = prob_storage.store((scores - u::bcast<kRows>(chunk_max)).exp());
            running_max = max_storage.store(u::copy(chunk_max));
            running_sum = sum_storage.store(u::matmul(prob, column_of_ones));
            running_out = acc_storage.store(u::matmul(prob, v));
        } else {
            u::ComputeBlock<Vec> previous_max = running_max.release();
            u::ComputeBlock new_max = new_max_storage.store(u::max_(previous_max, chunk_max));
            u::ComputeBlock correction = correction_storage.store((previous_max - new_max).exp());
            u::ComputeBlock prob = prob_storage.store((scores - u::bcast<kRows>(new_max)).exp());
            running_max = max_storage.store(u::copy(new_max));

            u::ComputeBlock chunk_sum = chunk_sum_storage.store(u::matmul(prob, column_of_ones));
            u::ComputeBlock<Vec> previous_sum = running_sum.release();
            running_sum = sum_storage.store(previous_sum * correction + chunk_sum);

            u::ComputeBlock<Out> previous_out = running_out.release();
            u::ComputeBlock rescaled = rescaled_storage.store(previous_out * u::bcast<kRows>(correction));
            u::ComputeBlock weighted_v = weighted_v_storage.store(u::matmul(prob, v));
            running_out = acc_storage.store(rescaled + weighted_v);
        }
    }

    running_max.release();
    u::ComputeBlock<Vec> total_sum = running_sum.release();
    u::ComputeBlock<Out> total_out = running_out.release();

    u::ComputeBlock reciprocal = reciprocal_storage.store(u::recip(total_sum));
    u::noc_store<1>(out_storage.store(total_out * u::bcast<kRows>(reciprocal)), out, 0);
}
