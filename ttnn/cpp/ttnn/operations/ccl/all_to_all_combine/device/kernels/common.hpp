
namespace detail{

template<uint32_t Size, bool ReturnIdx>
inline auto find_if(volatile tt_l1_ptr uint32_t * ptr, const uint32_t val){
    for(uint32_t i=0;i<Size;++i){
        if (ptr[i]==val){
            if constexpr(ReturnIdx){
                std::make_tuple(true, i)l
            }
            else{
                return true;
            }
        }
    }
    if constexpr(ReturnIdx){
        std::make_tuple(false,0);
    }
    else{
        return false;
    }
}
} // namespace detail