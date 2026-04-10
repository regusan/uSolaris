#pragma once
#include <usolaris/mesh.hpp>
#include <usolaris/rasterizer.hpp>
#include <usolaris/vertex_transform.hpp>

namespace usolaris {

// Pass 1: 全Objectの頂点を変換しマテリアルbinに振り分ける
// - xverts: 全Objectの頂点合計数分
// - bins_push:を受け取るコールバック
template <typename PushFn>
void build_bins(const Object *objects, int object_count, const trm3d::mat4f &vp,
                TransformedVertex *xverts, PushFn bins_push) {
  int vert_offset = 0;
  for (int oi = 0; oi < object_count; oi++) {
    const Object &obj = objects[oi];
    const Mesh &mesh = *obj.mesh;
    auto mvp = vp * obj.model;

    for (int mi = 0; mi < mesh.num_meshlets; mi++) {
      const Meshlet &ml = mesh.meshlets[mi];
      const Vertex *src = mesh.vertices + ml.vert_offset;
      TransformedVertex *dst = xverts + vert_offset;

      for (int vi = 0; vi < ml.vert_count; vi++)
        dst[vi] = DefaultVertexShader::shade(src[vi], mvp);

      int mat_id = obj.material_ids[ml.material_slot];
      bins_push(mat_id,
                TransformedMeshlet{dst, mesh.meshlet_prims + ml.prim_offset * 3,
                                   ml.prim_count});

      vert_offset += ml.vert_count;
    }
  }
}

// Pass 2: 全マテリアルのbinを1回のドローコールでラスタライズ
// - bins[i]: material i の TransformedMeshlet 配列の先頭ポインタ
// - bin_counts[i]: bins[i] の要素数
// - shaders[i]: material i のシェーダ（関数ポインタ + ctx）
template <typename PixelT, typename Layout>
void draw(Texture<PixelT, Layout> &tex, uint16_t *depth,
          const TransformedMeshlet *const *bins, const int *bin_counts,
          int num_mats, const MaterialShader<PixelT> *shaders) {
  for (int m = 0; m < num_mats; m++) {
    for (int i = 0; i < bin_counts[m]; i++) {
      const TransformedMeshlet &e = bins[m][i];
      rasterize_meshlet(tex, depth, e.verts, e.prims, e.prim_count, shaders[m]);
    }
  }
}

} // namespace usolaris
