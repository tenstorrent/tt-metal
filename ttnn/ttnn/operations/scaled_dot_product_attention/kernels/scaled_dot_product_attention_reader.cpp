// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
constexpr uint32_t cb_q=tt::CBIndex::c_0, cb_k=tt::CBIndex::c_1, cb_v=tt::CBIndex::c_2, cb_mask=tt::CBIndex::c_3;
constexpr uint32_t cb_scaler_reduce=4, cb_scale_factor=5, cb_o=tt::CBIndex::c_16, cb_max_old=27, cb_sum_old=30;
constexpr uint16_t NEG_INF_BFLOAT16=0xFF80;
inline void fill_bf16_tile_const(uint32_t cb_id, uint16_t val) {
    auto ptr = reinterpret_cast<volatile uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i=0; i<1024; ++i) ptr[i]=val;
}
inline uint16_t fp32_to_bf16(uint32_t bits) {
    uint16_t lsw=static_cast<uint16_t>(bits&0xFFFF);
    return static_cast<uint16_t>((bits+0x7FFFu+(lsw>>15))>>16);
}
void kernel_main() {
    constexpr uint32_t has_mask=get_compile_time_arg_val(0), H_q=get_compile_time_arg_val(1), H_kv=get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes=get_tile_size(cb_q);
    constexpr auto q_args=TensorAccessorArgs<3>();
    constexpr auto k_args=TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args=TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto mask_args=TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    uint32_t rt=0;
    uint32_t num_work_units=get_arg_val<uint32_t>(rt++);
    uint32_t B_q_t=get_arg_val<uint32_t>(rt++), B_kv_t=get_arg_val<uint32_t>(rt++), D_t=get_arg_val<uint32_t>(rt++);
    uint32_t S_q_tiles=get_arg_val<uint32_t>(rt++), S_kv_tiles=get_arg_val<uint32_t>(rt++);
    uint32_t work_b[16], work_h[16];
    for (uint32_t i=0; i<num_work_units; ++i) { work_b[i]=get_arg_val<uint32_t>(rt++); work_h[i]=get_arg_val<uint32_t>(rt++); }
    uint32_t q_addr=get_arg_val<uint32_t>(rt++), k_addr=get_arg_val<uint32_t>(rt++), scale_bits=get_arg_val<uint32_t>(rt++);
    uint32_t v_addr=get_arg_val<uint32_t>(rt++), mask_addr=get_arg_val<uint32_t>(rt++);
    uint32_t num_o_tiles=B_q_t*D_t;
    cb_reserve_back(cb_scale_factor,1);
    { uint16_t b=fp32_to_bf16(scale_bits); auto p=reinterpret_cast<volatile uint16_t*>(get_write_ptr(cb_scale_factor));
      for(uint32_t i=0;i<1024;++i) p[i]=b; }
    cb_push_back(cb_scale_factor,1);
    const auto q_acc=TensorAccessor(q_args,q_addr,tile_bytes), k_acc=TensorAccessor(k_args,k_addr,tile_bytes);
    const auto v_acc=TensorAccessor(v_args,v_addr,tile_bytes);
    [[maybe_unused]] const auto mask_acc=TensorAccessor(mask_args,mask_addr,tile_bytes);
    uint32_t h_q_div_h_kv=H_q/H_kv, num_q_blocks=(S_q_tiles+B_q_t-1)/B_q_t, num_kv_blocks=(S_kv_tiles+B_kv_t-1)/B_kv_t;
    for (uint32_t wu=0; wu<num_work_units; ++wu) {
        DEVICE_PRINT("READER: started wu={}\n", wu);
        uint32_t b=work_b[wu], h_q=work_h[wu], h_kv=h_q/h_q_div_h_kv;
        uint32_t q_base=b*H_q*S_q_tiles*D_t+h_q*S_q_tiles*D_t, k_base=b*H_kv*S_kv_tiles*D_t+h_kv*S_kv_tiles*D_t;
        uint32_t v_base=k_base, mask_base=b*S_q_tiles*S_kv_tiles;
        for (uint32_t qb=0; qb<num_q_blocks; ++qb) {
            uint32_t qrs=qb*B_q_t;
            for(uint32_t t=0;t<B_q_t;++t){cb_reserve_back(cb_max_old,1);fill_bf16_tile_const(cb_max_old,NEG_INF_BFLOAT16);cb_push_back(cb_max_old,1);}
            for(uint32_t t=0;t<B_q_t;++t){cb_reserve_back(cb_sum_old,1);fill_bf16_tile_const(cb_sum_old,0);cb_push_back(cb_sum_old,1);}
            for(uint32_t t=0;t<num_o_tiles;++t){cb_reserve_back(cb_o,1);fill_bf16_tile_const(cb_o,0);cb_push_back(cb_o,1);}
            for(uint32_t qt=0;qt<B_q_t*D_t;++qt){
                cb_reserve_back(cb_q,1);
                noc_async_read_tile(q_base+(qrs+qt/D_t)*D_t+qt%D_t,q_acc,get_write_ptr(cb_q));
                noc_async_read_barrier(); cb_push_back(cb_q,1);
            }
            for (uint32_t kvb=0; kvb<num_kv_blocks; ++kvb) {
                DEVICE_PRINT("READER: kvb={} start\n", kvb);
                uint32_t kvcs=kvb*B_kv_t;
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler_reduce,ckernel::PoolType::MAX,ckernel::ReduceDim::REDUCE_ROW>();
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler_reduce,ckernel::PoolType::SUM,ckernel::ReduceDim::REDUCE_ROW>();
                for(uint32_t k=0;k<D_t;++k) for(uint32_t n=0;n<B_kv_t;++n){
                    cb_reserve_back(cb_k,1); noc_async_read_tile(k_base+(kvcs+n)*D_t+k,k_acc,get_write_ptr(cb_k));
                    noc_async_read_barrier(); cb_push_back(cb_k,1);
                }
                for(uint32_t n=0;n<B_kv_t;++n) for(uint32_t d=0;d<D_t;++d){
                    cb_reserve_back(cb_v,1); noc_async_read_tile(v_base+(kvcs+n)*D_t+d,v_acc,get_write_ptr(cb_v));
                    noc_async_read_barrier(); cb_push_back(cb_v,1);
                }
                if constexpr(has_mask) for(uint32_t qr=0;qr<B_q_t;++qr) for(uint32_t kc=0;kc<B_kv_t;++kc){
                    cb_reserve_back(cb_mask,1); noc_async_read_tile(mask_base+(qrs+qr)*S_kv_tiles+(kvcs+kc),mask_acc,get_write_ptr(cb_mask));
                    noc_async_read_barrier(); cb_push_back(cb_mask,1);
                }
            }
            DEVICE_PRINT("READER: kvb done\n");
        }
    }
    DEVICE_PRINT("READER: done\n");
}
