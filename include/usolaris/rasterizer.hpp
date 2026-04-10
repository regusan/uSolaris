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
};

// マテリアルシェーダ（関数ポインタ + コンテキスト）
template <typename PixelT> struct MaterialShader {
  PixelT (*shade_fn)(const FragmentInput &, const void *ctx);
  const void *ctx;
};

namespace detail {

inline float edge2d(trm3d::vec2f a, trm3d::vec2f b, trm3d::vec2f p) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

} // namespace detail

// Meshlet単位ラスタライザ
// - prims: uint8_tローカルインデックス（prim_count*3個）
// - depth: uint16_t
template <typename PixelT, typename Layout>
void rasterize_meshlet(Texture<PixelT, Layout> &tex, uint16_t *depth,
                       const TransformedVertex *verts, const uint8_t *prims,
                       int prim_count, const MaterialShader<PixelT> &shader) {
  const float W = static_cast<float>(tex.size.x);
  const float H = static_cast<float>(tex.size.y);

  for (int ii = 0; ii < prim_count; ii++) {
    const TransformedVertex &v0 = verts[prims[ii * 3 + 0]];
    const TransformedVertex &v1 = verts[prims[ii * 3 + 1]];
    const TransformedVertex &v2 = verts[prims[ii * 3 + 2]];

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
    trm3d::vec2f duv_dx = v0.uv * dw0_dx + v1.uv * dw1_dx + v2.uv * dw2_dx;
    trm3d::vec2f duv_dy = v0.uv * dw0_dy + v1.uv * dw1_dy + v2.uv * dw2_dy;
    trm3d::vec3f dnrm_dx =
        v0.normal * dw0_dx + v1.normal * dw1_dx + v2.normal * dw2_dx;
    trm3d::vec3f dnrm_dy =
        v0.normal * dw0_dy + v1.normal * dw1_dy + v2.normal * dw2_dy;
    trm3d::vec3f dcol_dx =
        v0.color * dw0_dx + v1.color * dw1_dx + v2.color * dw2_dx;
    trm3d::vec3f dcol_dy =
        v0.color * dw0_dy + v1.color * dw1_dy + v2.color * dw2_dy;

    trm3d::vec2f p_start{x0 + 0.5f, y0 + 0.5f};
    float e0_row = detail::edge2d(s1, s2, p_start);
    float e1_row = detail::edge2d(s2, s0, p_start);
    float e2_row = detail::edge2d(s0, s1, p_start);
    float w0s = e0_row * inv_area, w1s = e1_row * inv_area,
          w2s = e2_row * inv_area;
    float z_row = z0 * w0s + z1 * w1s + z2 * w2s;
    trm3d::vec2f uv_row = v0.uv * w0s + v1.uv * w1s + v2.uv * w2s;
    trm3d::vec3f nrm_row = v0.normal * w0s + v1.normal * w1s + v2.normal * w2s;
    trm3d::vec3f col_row = v0.color * w0s + v1.color * w1s + v2.color * w2s;

    for (int y = y0; y <= y1; y++) {
      float e0 = e0_row, e1 = e1_row, e2 = e2_row;
      float z_col = z_row;
      trm3d::vec2f uv_col = uv_row;
      trm3d::vec3f nrm_col = nrm_row;
      trm3d::vec3f col_col = col_row;

      for (int x = x0; x <= x1; x++) {
        if (!(e0 > 0 || e1 > 0 || e2 > 0)) {
          auto z16 = static_cast<uint16_t>(z_col * 65535.0f);
          int di = y * tex.size.x + x;
          if (z16 < depth[di]) {
            depth[di] = z16;
            tex.at(x, y) = shader.shade_fn(
                FragmentInput{uv_col, nrm_col, col_col}, shader.ctx);
          }
        }
        e0 += de0_dx;
        e1 += de1_dx;
        e2 += de2_dx;
        z_col += dz_dx;
        uv_col += duv_dx;
        nrm_col += dnrm_dx;
        col_col += dcol_dx;
      }
      e0_row += de0_dy;
      e1_row += de1_dy;
      e2_row += de2_dy;
      z_row += dz_dy;
      uv_row += duv_dy;
      nrm_row += dnrm_dy;
      col_row += dcol_dy;
    }
  }
}

} // namespace usolaris
