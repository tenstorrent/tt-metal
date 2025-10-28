#pragma once

#include <tt-metalium/tensor/tensor_impl_wrapper.hpp>

#include "ttnn/tensor/tensor_impl.hpp"

namespace ttnn {
template <typename... Args>
auto to_string_wrapper(Args&&... args) {
    return tt::tt_metal::tensor_impl::dispatch(
        std::get<0>(std::forward_as_tuple(args...)).dtype(),
        []<typename T>(auto&&... args) { return ttnn::to_string<T>(std::forward<decltype(args)>(args)...); },
        std::forward<Args>(args)...);
}

}  // namespace ttnn
