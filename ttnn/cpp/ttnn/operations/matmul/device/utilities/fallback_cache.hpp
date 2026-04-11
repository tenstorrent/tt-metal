#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <functional>
#include <mutex>

// Heuristic Fallback Cache for ttnn.auto_matmul
// Stage 1 Prototype for Tenstorrent Bounty

namespace ttnn {
    namespace heuristics {

        struct MatmulConfig {
            int m_tiles;
            int k_tiles;
            int n_tiles;
            int in0_block_w;
            int out_subblock_h;
            int out_subblock_w;
            // Additional architecture-specific parameters (Grayskull/Wormhole)
        };

        struct TensorShape {
            uint32_t m, k, n;
            std::string layout;
            
            bool operator==(const TensorShape& other) const {
                return m == other.m && k == other.k && n == other.n && layout == other.layout;
            }
        };

        struct TensorShapeHash {
            std::size_t operator()(const TensorShape& shape) const {
                std::size_t h1 = std::hash<uint32_t>{}(shape.m);
                std::size_t h2 = std::hash<uint32_t>{}(shape.k);
                std::size_t h3 = std::hash<uint32_t>{}(shape.n);
                std::size_t h4 = std::hash<std::string>{}(shape.layout);
                return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
            }
        };

        class FallbackCache {
        private:
            std::unordered_map<TensorShape, MatmulConfig, TensorShapeHash> cache;
            std::mutex cache_mutex;

        public:
            bool get_config(const TensorShape& shape, MatmulConfig& out_config) {
                std::lock_guard<std::mutex> lock(cache_mutex);
                auto it = cache.find(shape);
                if (it != cache.end()) {
                    out_config = it->second;
                    return true;
                }
                return false;
            }

            void add_config(const TensorShape& shape, const MatmulConfig& config) {
                std::lock_guard<std::mutex> lock(cache_mutex);
                cache[shape] = config;
            }
            
            // Generate heuristic configuration if not found in cache
            MatmulConfig generate_heuristic(const TensorShape& shape) {
                MatmulConfig config;
                // Base tile dimensions (Grayskull/Wormhole operates on 32x32 tiles)
                config.m_tiles = (shape.m + 31) / 32;
                config.k_tiles = (shape.k + 31) / 32;
                config.n_tiles = (shape.n + 31) / 32;
                
                // Heuristic for block dimensions based on M, K, N
                // We want to maximize compute utilization while keeping L1 SRAM usage under control (~1MB per core)
                if (config.m_tiles >= 8 && config.n_tiles >= 8) {
                    config.out_subblock_h = 2;
                    config.out_subblock_w = 4;
                    config.in0_block_w = 2;
                } else if (config.m_tiles >= 4 && config.n_tiles >= 4) {
                    config.out_subblock_h = 2;
                    config.out_subblock_w = 2;
                    config.in0_block_w = 2;
                } else {
                    // Fallback for smaller shapes
                    config.out_subblock_h = 1;
                    config.out_subblock_w = 1;
                    config.in0_block_w = 1;
                }
                
                // Adjust if K is extremely large to prevent OOM
                if (config.k_tiles > 256) {
                    config.in0_block_w = 1; // Reduce block width for large inner dimensions
                }

                return config;
            }
        };

    } // namespace heuristics
} // namespace ttnn
