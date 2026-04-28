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
    x = ((x % size.x) + size.x) % size.x;
    y = ((y % size.y) + size.y) % size.y;
    int block_idx = (y / 4) * (size.x / 4) + (x / 4);
    return data[block_idx].decode_pixel(x % 4, y % 4);
  }

  // UV [0,1] 最近傍サンプリング
  trm3d::vec3f sample(trm3d::vec2f uv) const {
    int ix = static_cast<int>(std::floor(uv.x * size.x));
    int iy = static_cast<int>(std::floor(uv.y * size.y));
    return pixel_at(ix, iy);
  }

  // UV [0,1] バイリニアサンプリング（ブロック境界をまたいでも正しくデコード）
  trm3d::vec3f sample_bilinear(trm3d::vec2f uv) const {
    float fx = uv.x * size.x - 0.5f;
    float fy = uv.y * size.y - 0.5f;
    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
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

} // namespace usolaris
