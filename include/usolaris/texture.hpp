#pragma once
#include <TinyReguMath3D.hpp>
#include <usolaris/pixel_format.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <usolaris/bc1_texture.hpp>

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

// ピクセル型が decode() メソッドを持っているか判定するSFINAE
template <typename, typename = void>
struct has_decode : std::false_type {};

template <typename T>
struct has_decode<T, std::void_t<decltype(std::declval<T>().decode())>> : std::true_type {};

// 構造体が .decode() を持っている場合はそれを呼ぶ
template <typename T>
inline auto decode_pixel(const T& pixel) -> std::enable_if_t<has_decode<T>::value, decltype(pixel.decode())> {
  return pixel.decode();
}

// 持っていない（算術型など）場合はそのまま返す
template <typename T>
inline auto decode_pixel(const T& pixel) -> std::enable_if_t<!has_decode<T>::value, const T&> {
  return pixel;
}

// 非所有テクスチャ。ピクセルバッファの管理は呼び出し元が行う
template <typename PixelT, typename Layout = LinearLayout>
struct Texture {
  PixelT *data;
  trm3d::vec2i size;
  trm3d::vec2i8 shift;

  Texture() : data(nullptr), size{0, 0}, shift{0, 0} {}
  Texture(PixelT *d, trm3d::vec2i s) : data(d), size(s) {
    auto get_shift = [](int v) -> int8_t {
      int n = 0;
      while (v > 1) {
        v >>= 1;
        n++;
      }
      return static_cast<int8_t>(16 - n);
    };
    shift = {get_shift(size.x), get_shift(size.y)};
  }

  PixelT &at(int x, int y) { return data[Layout::index(x, y, size)]; }
  const PixelT &at(int x, int y) const {
    return data[Layout::index(x, y, size)];
  }

  // UV座標 [0,1] で最近傍サンプリング
  const PixelT &sample(trm3d::vec2f uv) const {
    int ix = static_cast<int>(std::floor(uv.x * size.x));
    int iy = static_cast<int>(std::floor(uv.y * size.y));
    ix = ((ix % size.x) + size.x) % size.x;
    iy = ((iy % size.y) + size.y) % size.y;
    return at(ix, iy);
  }

  // 固定小数(16bit)による超高速サンプリング。座標範囲 [0, 65535]
  const PixelT &sample_fast(trm3d::vec2u16 uv) const {
    // 16 - log2(size) 分だけシフトすることで座標を算出
    return at(uv.x >> shift.x, uv.y >> shift.y);
  }

  // UV座標 [0,1] でバイリニアサンプリング
  auto sample_bilinear(trm3d::vec2f uv) const {
    float fx = uv.x * size.x - 0.5f;
    float fy = uv.y * size.y - 0.5f;
    int x0 = (int)std::floor(fx);
    int y0 = (int)std::floor(fy);
    float tx = fx - x0;
    float ty = fy - y0;

    auto wx = [&](int x) { return ((x % size.x) + size.x) % size.x; };
    auto wy = [&](int y) { return ((y % size.y) + size.y) % size.y; };

    auto c00 = decode_pixel(at(wx(x0),   wy(y0)));
    auto c10 = decode_pixel(at(wx(x0+1), wy(y0)));
    auto c01 = decode_pixel(at(wx(x0),   wy(y0+1)));
    auto c11 = decode_pixel(at(wx(x0+1), wy(y0+1)));

    auto top = c00 * (1.0f - tx) + c10 * tx;
    auto bot = c01 * (1.0f - tx) + c11 * tx;
    return top * (1.0f - ty) + bot * ty;
  }

  // 補助: float UV [0, 1] を 16bit固定小数 [0, 65535] に変換
  static trm3d::vec2u16 uv_to_u16(trm3d::vec2f uv) {
    // 65536を掛けることで 1.0 がちょうど 0 (オーバーフロー) になり、ラップ処理がスムーズになる
    return {
        static_cast<uint16_t>(uv.x * 65536.0f),
        static_cast<uint16_t>(uv.y * 65536.0f)};
  }
};



// ミップマップ（汎用: TextureT は Texture<PixelT> でも BC1Texture でも可）
template <typename TextureT>
struct MipTexture {
  const TextureT *levels; // 非所有、レベル0が最高解像度
  int num_levels;
};

// lod に対応するレベルを返す（最近傍）
template <typename TextureT>
inline const TextureT &get_mip_level(const MipTexture<TextureT> &mip, float lod) {
  int level = static_cast<int>(
      std::round(std::fmax(0.0f, std::fmin(static_cast<float>(mip.num_levels - 1), lod))));
  return mip.levels[level];
}

} // namespace usolaris
