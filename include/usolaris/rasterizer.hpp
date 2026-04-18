#pragma once
#include <TinyReguMath3D.hpp>
#include <algorithm>
#include <cstdint>
#include <usolaris/texture.hpp>
#include <usolaris/vertex_transform.hpp>

namespace usolaris {

struct FragmentInput {
  trm3d::vec2f uv;
  trm3d::vec3f normal;
  trm3d::vec3f color;
  trm3d::vec2f reflect_uv;

  FragmentInput operator*(float s) const {
    return {uv * s, normal * s, color * s, reflect_uv * s};
  }
  FragmentInput operator+(const FragmentInput &o) const {
    return {uv + o.uv, normal + o.normal, color + o.color, reflect_uv + o.reflect_uv};
  }
  FragmentInput& operator+=(const FragmentInput &o) {
    uv = uv + o.uv;
    normal = normal + o.normal;
    color = color + o.color;
    reflect_uv = reflect_uv + o.reflect_uv;
    return *this;
  }
};

inline FragmentInput operator*(float s, const FragmentInput &f) {
  return f * s;
}

inline FragmentInput extract_fragment_input(const TransformedVertex& v) {
  return {v.uv, v.normal, v.color, v.reflect_uv};
}

namespace detail {

inline float edge2d(trm3d::vec2f a, trm3d::vec2f b, trm3d::vec2f p) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

} // namespace detail

// Meshlet単位ラスタライザ
template <typename PixelT, typename Layout, typename ShaderFn>
void rasterize_meshlet(Texture<PixelT, Layout> &tex, uint16_t *depth,
                       const TransformedVertex *verts, const uint8_t *prims,
                       int prim_count, const ShaderFn &shader) {
  const float W = static_cast<float>(tex.size.x);
  const float H = static_cast<float>(tex.size.y);

  for (int ii = 0; ii < prim_count; ii++) {
    TransformedVertex v0 = verts[prims[ii * 3 + 0]];
    TransformedVertex v1 = verts[prims[ii * 3 + 1]];
    TransformedVertex v2 = verts[prims[ii * 3 + 2]];

    // 簡易Near-clip: 1つでもカメラ背後(W <= 0)にある頂点を含む場合、描画を破棄
    if (v0.clip_pos.w <= 0.0f || v1.clip_pos.w <= 0.0f || v2.clip_pos.w <= 0.0f) {
      continue;
    }

    // UV wrap-around (シーム) 補正
    float u0 = v0.reflect_uv.x;
    float u1 = v1.reflect_uv.x;
    float u2 = v2.reflect_uv.x;
    if (std::abs(u0 - u1) > 0.5f || std::abs(u1 - u2) > 0.5f || std::abs(u2 - u0) > 0.5f) {
      if (u0 < 0.5f) u0 += 1.0f;
      if (u1 < 0.5f) u1 += 1.0f;
      if (u2 < 0.5f) u2 += 1.0f;
      v0.reflect_uv.x = u0;
      v1.reflect_uv.x = u1;
      v2.reflect_uv.x = u2;
    }

    auto to_screen = [&](const trm3d::vec4f &c) -> trm3d::vec2f {
      float iw = 1.0f / c.w;
      return {(c.x * iw * 0.5f + 0.5f) * W,
              (1.0f - (c.y * iw * 0.5f + 0.5f)) * H};
    };
    auto ndc_z = [](const trm3d::vec4f &c) { return c.z / c.w; };

    trm3d::vec2f s0 = to_screen(v0.clip_pos);
    trm3d::vec2f s1 = to_screen(v1.clip_pos);
    trm3d::vec2f s2 = to_screen(v2.clip_pos);
    float z0 = ndc_z(v0.clip_pos);
    float z1 = ndc_z(v1.clip_pos);
    float z2 = ndc_z(v2.clip_pos);

    float area = detail::edge2d(s0, s1, s2);
    if (area >= 0.0f)
      continue;

    float inv_area = 1.0f / area;

    int x0 = std::max(0, (int)std::min({s0.x, s1.x, s2.x}));
    int y0 = std::max(0, (int)std::min({s0.y, s1.y, s2.y}));
    int x1 = std::min(tex.size.x - 1, (int)std::max({s0.x, s1.x, s2.x}) + 1);
    int y1 = std::min(tex.size.y - 1, (int)std::max({s0.y, s1.y, s2.y}) + 1);

    float de0_dx = s1.y - s2.y, de0_dy = s2.x - s1.x;
    float de1_dx = s2.y - s0.y, de1_dy = s0.x - s2.x;
    float de2_dx = s0.y - s1.y, de2_dy = s1.x - s0.x;

    float dw0_dx = de0_dx * inv_area, dw0_dy = de0_dy * inv_area;
    float dw1_dx = de1_dx * inv_area, dw1_dy = de1_dy * inv_area;
    float dw2_dx = de2_dx * inv_area, dw2_dy = de2_dy * inv_area;

    float dz_dx = z0 * dw0_dx + z1 * dw1_dx + z2 * dw2_dx;
    float dz_dy = z0 * dw0_dy + z1 * dw1_dy + z2 * dw2_dy;

    FragmentInput f0 = extract_fragment_input(v0);
    FragmentInput f1 = extract_fragment_input(v1);
    FragmentInput f2 = extract_fragment_input(v2);

    FragmentInput df_dx = f0 * dw0_dx + f1 * dw1_dx + f2 * dw2_dx;
    FragmentInput df_dy = f0 * dw0_dy + f1 * dw1_dy + f2 * dw2_dy;

    trm3d::vec2f p_start{x0 + 0.5f, y0 + 0.5f};
    float e0_row = detail::edge2d(s1, s2, p_start);
    float e1_row = detail::edge2d(s2, s0, p_start);
    float e2_row = detail::edge2d(s0, s1, p_start);
    float w0s = e0_row * inv_area, w1s = e1_row * inv_area, w2s = e2_row * inv_area;
    
    float z_row = z0 * w0s + z1 * w1s + z2 * w2s;
    FragmentInput f_row = f0 * w0s + f1 * w1s + f2 * w2s;

    for (int y = y0; y <= y1; y++) {
      float e0 = e0_row, e1 = e1_row, e2 = e2_row;
      float z_col = z_row;
      FragmentInput f_col = f_row;

      for (int x = x0; x <= x1; x++) {
        if (!(e0 > 0 || e1 > 0 || e2 > 0)) {
          auto z16 = static_cast<uint16_t>(z_col * 65535.0f);
          int di = y * tex.size.x + x;
          if (z16 < depth[di]) {
            depth[di] = z16;
            tex.at(x, y) = shader(f_col);
          }
        }
        e0 += de0_dx;
        e1 += de1_dx;
        e2 += de2_dx;
        z_col += dz_dx;
        f_col += df_dx;
      }
      e0_row += de0_dy;
      e1_row += de1_dy;
      e2_row += de2_dy;
      z_row += dz_dy;
      f_row += df_dy;
    }
  }
}

} // namespace usolaris
