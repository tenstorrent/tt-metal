# Owned by @tenstorrent/metalium-developers-ds-prefill. Add/remove ops for
# this domain here without touching ttnn/CMakeLists.txt.
add_subdirectory(combine)
add_subdirectory(dispatch)
add_subdirectory(extract)
add_subdirectory(inbound_socket_service_sync)
add_subdirectory(insert)
add_subdirectory(masked_bincount)
add_subdirectory(moe_grouped_topk)
add_subdirectory(moe_hash_gate)
add_subdirectory(offset_cumsum)
add_subdirectory(outbound_socket_service_sync)
add_subdirectory(pack_scaled_fp8_kv_cache)
add_subdirectory(per_token_cast_back)
add_subdirectory(per_token_cast_to_fp8)
add_subdirectory(post_combine_reduce)
add_subdirectory(rotary_embedding_indexed)
add_subdirectory(update_padded_kv_cache)
add_subdirectory(zero_padded_kv_cache)

# Domain-level umbrella target: ttnncpp links this one target instead of
# enumerating this team's individual op libraries. Add/remove entries here
# when adding/removing a module — never in ttnn/CMakeLists.txt.
add_library(ttnn_op_deepseek_prefill INTERFACE)
add_library(TTNN::Ops::Experimental::DeepSeekPrefill ALIAS ttnn_op_deepseek_prefill)
target_link_libraries(
    ttnn_op_deepseek_prefill
    INTERFACE
        TTNN::Ops::Experimental::DeepSeekPrefill::Combine
        TTNN::Ops::Experimental::DeepSeekPrefill::Dispatch
        TTNN::Ops::Experimental::DeepSeekPrefill::Extract
        TTNN::Ops::Experimental::DeepSeekPrefill::H2dSocketSync
        TTNN::Ops::Experimental::DeepSeekPrefill::Insert
        TTNN::Ops::Experimental::DeepSeekPrefill::MaskedBincount
        TTNN::Ops::Experimental::DeepSeekPrefill::MoeGroupedTopk
        TTNN::Ops::Experimental::DeepSeekPrefill::MoeHashGate
        TTNN::Ops::Experimental::DeepSeekPrefill::OffsetCumsum
        TTNN::Ops::Experimental::DeepSeekPrefill::D2dSocketSync
        TTNN::Ops::Experimental::DeepSeekPrefill::PackScaledFp8KvCache
        TTNN::Ops::Experimental::DeepSeekPrefill::PerTokenCastBack
        TTNN::Ops::Experimental::DeepSeekPrefill::PerTokenCastToFp8
        TTNN::Ops::Experimental::DeepSeekPrefill::PostCombineReduce
        TTNN::Ops::Experimental::DeepSeekPrefill::RotaryEmbeddingIndexed
        TTNN::Ops::Experimental::DeepSeekPrefill::UpdatePaddedKvCache
        TTNN::Ops::Experimental::DeepSeekPrefill::ZeroPaddedKvCache
)

# target_link_libraries() alone is not enough: per the CMake docs, an OBJECT
# library named in a target's INTERFACE_LINK_LIBRARIES (as these leaves are,
# via the INTERFACE keyword above) is treated as a plain Interface Library by
# further consumers — only usage requirements propagate, not its compiled
# objects. So with the default LIB_TYPE=OBJECT, ttnncpp would silently end up
# without these ops' code (undefined symbols at final link time). Splice the
# object files in directly via generator expression so they still land in
# whatever ultimately links this umbrella target. TARGET_OBJECTS resolves to
# nothing (and is harmless) for leaves that aren't OBJECT libraries — e.g.
# SHARED or STATIC, both of which ttnn can compile these as depending on
# build options — so this is safe unconditionally.
target_sources(
    ttnn_op_deepseek_prefill
    INTERFACE
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::Combine>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::Dispatch>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::Extract>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::H2dSocketSync>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::Insert>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::MaskedBincount>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::MoeGroupedTopk>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::MoeHashGate>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::OffsetCumsum>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::D2dSocketSync>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::PackScaledFp8KvCache>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::PerTokenCastBack>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::PerTokenCastToFp8>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::PostCombineReduce>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::RotaryEmbeddingIndexed>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::UpdatePaddedKvCache>
        $<TARGET_OBJECTS:TTNN::Ops::Experimental::DeepSeekPrefill::ZeroPaddedKvCache>
)
