// File contains enumerations that are common to both kernel and program factories with regards to sharding

namespace ttnn::ccl::common::shard_addr_gen_utils {

enum class Contiguity_types {
    PADDING_BETWEEN_PAGES = 0,
    PADDING_IN_RIGHTMOST_SHARD,
    NO_SHARD_PADDING,
};

enum class ShardingLayout {
    HEIGHT_SHARDED = 0,
    WIDTH_SHARDED,
    BLOCK_SHARDED,
};

}  // namespace ttnn::ccl::common::shard_addr_gen_utils
