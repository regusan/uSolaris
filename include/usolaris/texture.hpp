#pragma once
#include <TinyReguMath3D.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace usolaris {

// リニアレイアウト（行優先）
struct LinearLayout {
  static int index(int x, int y, trm3d::vec2i size) { return y * size.x + x; }
};

// Z-オーダーレイアウト。バッファサイズは 2^n × 2^n が必要
struct ZOrderLayout {
  static int index(int x, int y, trm3d::vec2i /*size*/) {
    return static_cast<int>(
        morton_encode(static_cast<uint32_t>(x), static_cast<uint32_t>(y)));
  }

private:
  static uint32_t spread_bits(uint32_t v) {
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
  }
  static uint32_t morton_encode(uint32_t x, uint32_t y) {
    return spread_bits(x) | (spread_bits(y) << 1);
  }
};

// 非所有テクスチャ。ピクセルバッファの管理は呼び出し元が行う
template <typename PixelT, typename Layout = LinearLayout>
struct Texture {
  PixelT *data;
  trm3d::vec2i size;

  PixelT &at(int x, int y) { return data[Layout::index(x, y, size)]; }
  const PixelT &at(int x, int y) const {
    return data[Layout::index(x, y, size)];
  }

  // UV座標 [0,1] で最近傍サンプリング
  const PixelT &sample(trm3d::vec2f uv) const {
    int x = static_cast<int>(uv.x * size.x) % size.x;
    int y = static_cast<int>(uv.y * size.y) % size.y;
    return at(x, y);
  }

  // UV座標 [0,1] でバイリニアサンプリング（PixelT のまま返す）
  // PixelT が算術型（float, vec3f 等）の場合は補間済み値を返す
  // PixelT が非算術型（uint32_t 等）の場合は最近傍にフォールバック
  PixelT sample_bilinear(trm3d::vec2f uv) const {
    float fx = uv.x * size.x - 0.5f;
    float fy = uv.y * size.y - 0.5f;
    int x0 = (int)std::floor(fx);
    int y0 = (int)std::floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;

    auto wx = [&](int x) { return ((x % size.x) + size.x) % size.x; };
    auto wy = [&](int y) { return ((y % size.y) + size.y) % size.y; };

    if constexpr (std::is_arithmetic_v<PixelT>) {
      PixelT c00 = at(wx(x0),   wy(y0));
      PixelT c10 = at(wx(x0+1), wy(y0));
      PixelT c01 = at(wx(x0),   wy(y0+1));
      PixelT c11 = at(wx(x0+1), wy(y0+1));
      PixelT top = c00 * (1.0f - tx) + c10 * tx;
      PixelT bot = c01 * (1.0f - tx) + c11 * tx;
      return top * (1.0f - ty) + bot * ty;
    } else {
      return at(wx((int)std::round(fx)), wy((int)std::round(fy)));
    }
  }
};

// Texture<uint32_t> 専用: デコード関数込みのバイリニアサンプリング
// DecodeF: uint32_t → trm3d::vec3f
template <typename Layout, typename DecodeF>
inline trm3d::vec3f sample_bilinear_decode(const Texture<uint32_t, Layout> &tex,
                                            trm3d::vec2f uv, DecodeF decode) {
  float fx = uv.x * tex.size.x - 0.5f;
  float fy = uv.y * tex.size.y - 0.5f;
  int x0 = (int)std::floor(fx);
  int y0 = (int)std::floor(fy);
  float tx = fx - x0;
  float ty = fy - y0;

  auto wx = [&](int x) { return ((x % tex.size.x) + tex.size.x) % tex.size.x; };
  auto wy = [&](int y) { return ((y % tex.size.y) + tex.size.y) % tex.size.y; };

  trm3d::vec3f c00 = decode(tex.at(wx(x0),   wy(y0)));
  trm3d::vec3f c10 = decode(tex.at(wx(x0+1), wy(y0)));
  trm3d::vec3f c01 = decode(tex.at(wx(x0),   wy(y0+1)));
  trm3d::vec3f c11 = decode(tex.at(wx(x0+1), wy(y0+1)));

  trm3d::vec3f top = c00 * (1.0f - tx) + c10 * tx;
  trm3d::vec3f bot = c01 * (1.0f - tx) + c11 * tx;
  return top * (1.0f - ty) + bot * ty;
}

// ミップマップ（複数解像度レベルの非所有コレクション）
template <typename PixelT, typename Layout = LinearLayout>
struct MipTexture {
  const Texture<PixelT, Layout> *levels; // 非所有、レベル0が最高解像度
  int num_levels;
};

// lod に対応する Texture レベルを返す（レベル間は最近傍）
template <typename PixelT, typename Layout>
inline const Texture<PixelT, Layout> &get_mip_level(
    const MipTexture<PixelT, Layout> &mip, float lod) {
  int level = (int)std::round(std::fmax(0.0f, std::fmin((float)(mip.num_levels - 1), lod)));
  return mip.levels[level];
}

} // namespace usolaris
