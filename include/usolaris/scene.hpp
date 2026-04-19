#pragma once
#include <usolaris/env_map.hpp>
#include <usolaris/mesh.hpp>
#include <usolaris/rasterizer.hpp>
#include <usolaris/vertex_transform.hpp>

namespace usolaris {

// Pass 1: 全Objectを走査し、マテリアルbinに参照(インデックス)を振り分ける
// 変換そのものは行いません
template <typename PushFn>
void build_bins(const MeshInstance *objects, int object_count, PushFn bins_push) {
  for (int oi = 0; oi < object_count; oi++) {
    const MeshInstance &obj = objects[oi];
    const Mesh &mesh = *obj.mesh;

    for (int mi = 0; mi < mesh.num_meshlets; mi++) {
      const Meshlet &ml = mesh.meshlets[mi];
      int mat_id = obj.material_ids[ml.material_slot];
      
      bins_push(mat_id, PendingMeshlet{
                            static_cast<uint16_t>(oi),
                            static_cast<uint16_t>(mi)});
    }
  }
}

// 指定したMaterial Bin(bins)のメッシュ群を1回のドローコールでラスタライズ
// 即時描画(On-Demand Transformation)を行います
template <typename PixelT, typename Layout, typename ShaderFn>
void draw_bins(Texture<PixelT, Layout> &tex, uint16_t *depth,
               const PendingMeshlet *bins, int bin_count,
               const MeshInstance *objects, const trm3d::mat4f &vp, const trm3d::vec3f &eye,
               const ShaderFn& shader) {
  
  TransformedVertex local_verts[64]; // L1キャッシュに収まる小スタック

  for (int i = 0; i < bin_count; i++) {
    const PendingMeshlet &e = bins[i];
    const MeshInstance &obj = objects[e.object_index];
    const Mesh &mesh = *obj.mesh;
    const Meshlet &ml = mesh.meshlets[e.meshlet_index];

    trm3d::mat4f mvp = vp * obj.model;
    const Vertex *src = mesh.vertices + ml.vert_offset;

    // 即時変換
    for (int vi = 0; vi < ml.vert_count; vi++) {
      local_verts[vi] = DefaultVertexShader::shade(src[vi], mvp, obj.model, eye);
    }

    // 即時ラスタライズ
    rasterize_meshlet(tex, depth, local_verts, mesh.meshlet_prims + ml.prim_offset * 3, ml.prim_count, shader);
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
      trm3d::vec2f uv_r   = uv_next;

      // UV wrap-around (シーム) 補正
      if (std::abs(uv_l.x - uv_r.x) > 0.5f) {
        if (uv_l.x < uv_r.x) uv_l.x += 1.0f;
        else uv_r.x += 1.0f;
      }

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
