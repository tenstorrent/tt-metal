// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tuple>

namespace detail{

template <typename T, uint32_t Size, bool ReturnIdx>
inline auto find_if(volatile tt_l1_ptr T* ptr, const uint32_t val) {
    for(uint32_t i=0;i<Size;++i){
        if (ptr[i]==val){
            if constexpr(ReturnIdx){
                return std::make_tuple(true, i);
            }
            else{
                return true;
            }
        }
    }
    if constexpr(ReturnIdx){
        return std::make_tuple(false, 0ul);
    }
    else{
        return false;
    }
}
}  // namespace detail
