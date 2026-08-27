// SPDX-License-Identifier: Apache-2.0

#include <tt/unified/core>

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
    constexpr uint32_t kCbQ = TT_U_CB(q);
    constexpr uint32_t kCbK = TT_U_CB(k);
    constexpr uint32_t kCbV = TT_U_CB(v);
    constexpr uint32_t kCbOnes = TT_U_CB(ones);
    constexpr uint32_t kCbScaler = TT_U_CB(scaler);
    constexpr uint32_t kCbScores = TT_U_CB(scores);
    constexpr uint32_t kCbChunkMax = TT_U_CB(chunk_max);
    constexpr uint32_t kCbProb = TT_U_CB(prob);
    constexpr uint32_t kCbChunkSum = TT_U_CB(chunk_sum);
    constexpr uint32_t kCbNewMax = TT_U_CB(new_max);
    constexpr uint32_t kCbCorrection = TT_U_CB(correction);
    constexpr uint32_t kCbRescaled = TT_U_CB(rescaled);
    constexpr uint32_t kCbWeightedV = TT_U_CB(weighted_v);
    constexpr uint32_t kCbReciprocal = TT_U_CB(reciprocal);
    constexpr uint32_t kCbOut = TT_U_CB(out);
    constexpr uint32_t kCbMax = TT_U_CB(max);
    constexpr uint32_t kCbSum = TT_U_CB(sum);
    constexpr uint32_t kCbAcc = TT_U_CB(acc);

    const auto q_acc = TensorAccessor(tensor::q));
    const auto k_acc = TensorAccessor(tensor::k));
    const auto v_acc = TensorAccessor(tensor::v));
    const auto ones_acc = TensorAccessor(tensor::ones));
    const auto out = TensorAccessor(tensor::out));

    u::matmul_init<Queries, KeysTransposed>(kCbQ, kCbK, kCbOut);

    u::Storage<Queries> q_storage(kCbQ);
    u::Storage<KeysTransposed> k_storage(kCbK);
    u::Storage<Values> v_storage(kCbV);
    u::Storage<Ones> ones_storage(kCbOnes);
    u::Storage<u::Shape<1, 1>> scaler_storage(kCbScaler);
    u::Storage<Scores> scores_storage(kCbScores);
    u::Storage<Vec> chunk_max_storage(kCbChunkMax);
    u::Storage<Scores> prob_storage(kCbProb);
    u::Storage<Vec> chunk_sum_storage(kCbChunkSum);
    u::Storage<Vec> new_max_storage(kCbNewMax);
    u::Storage<Vec> correction_storage(kCbCorrection);
    u::Storage<Out> rescaled_storage(kCbRescaled);
    u::Storage<Out> weighted_v_storage(kCbWeightedV);
    u::Storage<Vec> reciprocal_storage(kCbReciprocal);
    u::Storage<Vec> max_storage(kCbMax);
    u::Storage<Vec> sum_storage(kCbSum);
    u::Storage<Out> acc_storage(kCbAcc);
    u::Storage<Out> out_storage(kCbOut);

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
