#pragma once
#include "fastmath.hpp"
#include <TinyReguMath3D.hpp>
#include <cmath>
#include <cstdint>
#include <usolaris/texture.hpp>

namespace usolaris {

// RGB9E5 デコード → float RGB
inline trm3d::vec3f decode_rgb9e5(uint32_t packed) {
  float scale = std::ldexp(1.0f, (int)(packed >> 27) - 24);
  return {
      (packed & 0x1FF) * scale,
      ((packed >> 9) & 0x1FF) * scale,
      ((packed >> 18) & 0x1FF) * scale,
  };
}

// 方向ベクトル → 球面 UV [0,1]
inline trm3d::vec2f norm_to_uv(trm3d::vec3f dir) {
  constexpr float PI = 3.14159265f;
  dir = trm3d::normalize(dir);
  float u = trm3d::fast_atan2<1>(dir.z, dir.x) / (2.0f * PI) + 0.5f;
  float v =
      trm3d::fast_asin<1>(std::fmax(-1.0f, std::fmin(1.0f, dir.y))) / PI + 0.5f;
  return {u, v};
}

// EnvMapT の各レベルを Texture<uint32_t> にゼロコピーでラップ
// out は EnvMapT::num_levels 以上の要素を持つバッファを呼び出し元が用意する
template <typename EnvMapT> inline void make_env_mip(Texture<uint32_t> *out) {
  for (int i = 0; i < EnvMapT::num_levels; i++)
    out[i] = {const_cast<uint32_t *>(EnvMapT::data + EnvMapT::offsets[i]),
              {EnvMapT::widths[i], EnvMapT::heights[i]}};
}

} // namespace usolaris
