#pragma once
#include <usolaris/env_map.hpp>
#include <usolaris/mesh.hpp>
#include <usolaris/rasterizer.hpp>
#include <usolaris/vertex_transform.hpp>

namespace usolaris {

// Pass 1: 全Objectの頂点を変換しマテリアルbinに振り分ける
// - xverts: 全Objectの頂点合計数分
// - bins_push:を受け取るコールバック
template <typename PushFn>
void build_bins(const MeshInstance *objects, int object_count, const trm3d::mat4f &vp,
                TransformedVertex *xverts, const trm3d::vec3f& eye, PushFn bins_push) {
  int vert_offset = 0;
  for (int oi = 0; oi < object_count; oi++) {
    const MeshInstance &obj = objects[oi];
    const Mesh &mesh = *obj.mesh;
    auto mvp = vp * obj.model;

    for (int mi = 0; mi < mesh.num_meshlets; mi++) {
      const Meshlet &ml = mesh.meshlets[mi];
      const Vertex *src = mesh.vertices + ml.vert_offset;
      TransformedVertex *dst = xverts + vert_offset;

      for (int vi = 0; vi < ml.vert_count; vi++)
        dst[vi] = DefaultVertexShader::shade(src[vi], mvp, obj.model, eye);

      int mat_id = obj.material_ids[ml.material_slot];
      bins_push(mat_id,
                TransformedMeshlet{dst, mesh.meshlet_prims + ml.prim_offset * 3,
                                   ml.prim_count});

      vert_offset += ml.vert_count;
    }
  }
}

// 指定したMaterial Bin(bins)のメッシュ群を1回のドローコールでラスタライズ
// - bins: material の TransformedMeshlet 配列の先頭ポインタ
// - bin_count: bins の要素数
// - shader: material のシェーダ関数（ラムダ式等）
template <typename PixelT, typename Layout, typename ShaderFn>
void draw_bins(Texture<PixelT, Layout> &tex, uint16_t *depth,
               const TransformedMeshlet *bins, int bin_count,
               const ShaderFn& shader) {
  for (int i = 0; i < bin_count; i++) {
    const TransformedMeshlet &e = bins[i];
    rasterize_meshlet(tex, depth, e.verts, e.prims, e.prim_count, shader);
  }
}

// Post-pass: depth == 0xFFFF のピクセルに背景（sky）を書き込む
// X軸 SPANごとに正確な球面UVを計算し、ほかは補完
template <typename PixelT, typename Layout, typename SkyFn>
void draw_sky(Texture<PixelT, Layout> &tex, const uint16_t *depth,
              const trm3d::mat4f &inv_vp, const trm3d::vec3f &eye,
              SkyFn sky_fn, int span = 16) {
  const float W        = static_cast<float>(tex.size.x);
  const float H        = static_cast<float>(tex.size.y);
  const float inv_span = 1.0f / static_cast<float>(span);

  // スクリーン座標 → 球面UV（正確計算、スパン端点でのみ呼ぶ）
  auto exact_uv = [&](int x, int y) -> trm3d::vec2f {
    float nx = (x + 0.5f) / W * 2.0f - 1.0f;
    float ny = 1.0f - (y + 0.5f) / H * 2.0f;
    trm3d::vec4f world = inv_vp * trm3d::vec4f{nx, ny, 0.0f, 1.0f};
    float iw = 1.0f / world.w;
    trm3d::vec3f dir = trm3d::fast_normalize(trm3d::vec3f{
        world.x * iw - eye.x,
        world.y * iw - eye.y,
        world.z * iw - eye.z});
    return norm_to_uv(dir);
  };

  for (int y = 0; y < tex.size.y; y++) {
    trm3d::vec2f uv_next = exact_uv(0, y);          // 行の先頭だけ計算
    for (int sx = 0; sx < tex.size.x; sx += span) {
      const int    sx_end = sx + span;
      trm3d::vec2f uv_l   = uv_next;
      uv_next             = exact_uv(sx_end, y);     // 右端を計算（次スパンで uv_l に再利用）
      const int    x_end  = std::min(sx_end, tex.size.x);
      for (int x = sx; x < x_end; x++) {
        if (depth[y * tex.size.x + x] != 0xFFFF) continue;
        float        t  = (x - sx) * inv_span;
        trm3d::vec2f uv = uv_l * (1.0f - t) + uv_r * t;
        tex.at(x, y)    = sky_fn(uv);
      }
    }
  }
}

} // namespace usolaris
