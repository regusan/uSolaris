#pragma once
#include <TinyReguMath3D.hpp>
#include <usolaris/pixel_format.hpp>
#include <cmath>

namespace usolaris {

// BC1 圧縮テクスチャ（ReadOnly）。4×4 ブロック単位でデコード
struct BC1Texture {
  const BC1Block *data;
  trm3d::vec2i size;

  BC1Texture() : data(nullptr), size{0, 0} {}
  BC1Texture(const BC1Block *d, trm3d::vec2i s) : data(d), size(s) {}

  // ピクセル座標 (x, y) をデコードして返す（ラップ: リピート）
  trm3d::vec3f pixel_at(int x, int y) const {
    uint32_t ux = static_cast<uint32_t>(x) & (size.x - 1);
    uint32_t uy = static_cast<uint32_t>(y) & (size.y - 1);
    int block_idx = (uy >> 2) * (size.x >> 2) + (ux >> 2);
    return data[block_idx].decode_pixel(ux & 3, uy & 3);
  }

  // UV [0,1] 最近傍サンプリング
  trm3d::vec3f sample(trm3d::vec2f uv) const {
    int ix = uv.x >= 0.0f ? static_cast<int>(uv.x * size.x) : static_cast<int>(uv.x * size.x) - 1;
    int iy = uv.y >= 0.0f ? static_cast<int>(uv.y * size.y) : static_cast<int>(uv.y * size.y) - 1;
    return pixel_at(ix, iy);
  }

  // UV [0,1] バイリニアサンプリング（ブロック境界をまたいでも正しくデコード）
  trm3d::vec3f sample_bilinear(trm3d::vec2f uv) const {
    float fx = uv.x * size.x - 0.5f;
    float fy = uv.y * size.y - 0.5f;
    int x0 = fx >= 0.0f ? static_cast<int>(fx) : static_cast<int>(fx) - 1;
    int y0 = fy >= 0.0f ? static_cast<int>(fy) : static_cast<int>(fy) - 1;
    float tx = fx - x0, ty = fy - y0;
    auto c00 = pixel_at(x0,     y0);
    auto c10 = pixel_at(x0 + 1, y0);
    auto c01 = pixel_at(x0,     y0 + 1);
    auto c11 = pixel_at(x0 + 1, y0 + 1);
    auto top = c00 * (1.0f - tx) + c10 * tx;
    auto bot = c01 * (1.0f - tx) + c11 * tx;
    return top * (1.0f - ty) + bot * ty;
  }

  // 固定小数 (16bit) 高速サンプリング。座標範囲 [0, 65535]
  trm3d::vec3f sample_fast(trm3d::vec2u16 uv) const {
    // shift: (16 - log2(size)) でブロック座標に変換
    auto get_shift = [](int v) -> int {
      int n = 0; while (v > 1) { v >>= 1; n++; } return 16 - n;
    };
    int ix = uv.x >> get_shift(size.x);
    int iy = uv.y >> get_shift(size.y);
    return pixel_at(ix, iy);
  }
};

template <int N = 8>
struct BC1Sampler {
    struct CacheEntry {
        int16_t bx = -1;
        int16_t by = -1;
        trm3d::vec3f palette[4];
    };

    CacheEntry cache[N * N];
    const BC1Texture* current_tex = nullptr;

    void bind(const BC1Texture& tex) {
        if (current_tex != &tex) {
            current_tex = &tex;
            clear();
        }
    }

    void clear() {
        for (auto& entry : cache) {
            entry.bx = -1;
        }
    }

    trm3d::vec3f sample(int x, int y) {
        uint32_t ux = static_cast<uint32_t>(x) & (current_tex->size.x - 1);
        uint32_t uy = static_cast<uint32_t>(y) & (current_tex->size.y - 1);

        int bx = ux >> 2;
        int by = uy >> 2;
        
        int cx = bx & (N - 1);
        int cy = by & (N - 1);
        int cache_idx = cy * N + cx;

        if (cache[cache_idx].bx != static_cast<int16_t>(bx) || 
            cache[cache_idx].by != static_cast<int16_t>(by)) {
            
            cache[cache_idx].bx = static_cast<int16_t>(bx);
            cache[cache_idx].by = static_cast<int16_t>(by);
            
            int block_idx = by * (current_tex->size.x / 4) + bx;
            const BC1Block& block = current_tex->data[block_idx];
            
            auto unpack = [](uint16_t c) -> trm3d::vec3f {
                return {((c >> 11) & 0x1F) * 0.03225806f,
                        ((c >> 5) & 0x3F) * 0.01587301f,
                        (c & 0x1F) * 0.03225806f};
            };
            trm3d::vec3f col0 = unpack(block.c0);
            trm3d::vec3f col1 = unpack(block.c1);
            cache[cache_idx].palette[0] = col0;
            cache[cache_idx].palette[1] = col1;
            if (block.c0 > block.c1) {
                cache[cache_idx].palette[2] = col0 * (2.0f / 3.0f) + col1 * (1.0f / 3.0f);
                cache[cache_idx].palette[3] = col0 * (1.0f / 3.0f) + col1 * (2.0f / 3.0f);
            } else {
                cache[cache_idx].palette[2] = col0 * 0.5f + col1 * 0.5f;
                cache[cache_idx].palette[3] = {0.0f, 0.0f, 0.0f};
            }
        }

        int local_x = ux & 3;
        int local_y = uy & 3;
        int block_idx = by * (current_tex->size.x >> 2) + bx;
        int index = current_tex->data[block_idx].extract_index(local_x, local_y);
        
        return cache[cache_idx].palette[index];
    }
};

} // namespace usolaris
