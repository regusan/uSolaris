#pragma once
#include "fastmath.hpp"
#include <TinyReguMath3D.hpp>
#include <cmath>
#include <cstdint>
#include <usolaris/texture.hpp>
#include <usolaris/pixel_format.hpp>

namespace usolaris {

// 後方互換性
inline trm3d::vec3f decode_rgb9e5(uint32_t packed) {
  return FormatRGB9E5{packed}.decode();
}

// 方向ベクトル → 球面 UV [0,1]
inline trm3d::vec2f norm_to_uv(trm3d::vec3f dir) {
  constexpr float PI = 3.14159265f;
  dir = trm3d::fast_normalize(dir);
  float u = trm3d::fast_atan2<1>(dir.z, dir.x) / (2.0f * PI) + 0.5f;
  float v =
      trm3d::fast_asin<1>(std::fmax(-1.0f, std::fmin(1.0f, dir.y))) / PI + 0.5f;
  return {u, v};
}

// EnvMapT の各レベルを Texture<FormatRGB9E5> にゼロコピーでラップ
// out は EnvMapT::num_levels 以上の要素を持つバッファを呼び出し元が用意する
template <typename EnvMapT> inline void make_env_mip(Texture<FormatRGB9E5> *out) {
  for (int i = 0; i < EnvMapT::num_levels; i++)
    out[i] = {reinterpret_cast<FormatRGB9E5 *>(const_cast<uint32_t *>(EnvMapT::data + EnvMapT::offsets[i])),
              {EnvMapT::widths[i], EnvMapT::heights[i]}};
}

// BC1 圧縮環境マップ版。EnvMapT は BC1 ヘッダ（uint8_t data[], offsets[]）を持つ
// offsets[] はバイト単位。out は EnvMapT::num_levels 以上の BC1Texture バッファを呼び出し元が用意する
template <typename EnvMapT> inline void make_env_mip_bc1(BC1Texture *out) {
  for (int i = 0; i < EnvMapT::num_levels; i++) {
    const auto *base = reinterpret_cast<const BC1Block *>(EnvMapT::data + EnvMapT::offsets[i]);
    out[i] = BC1Texture{base, {EnvMapT::widths[i], EnvMapT::heights[i]}};
  }
}

} // namespace usolaris

