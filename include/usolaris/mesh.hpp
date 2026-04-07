#pragma once
#include <TinyReguMath3D.hpp>
#include <usolaris/rasterizer.hpp>
#include <usolaris/vertex.hpp>
#include <usolaris/vertex_transform.hpp>

namespace usolaris {

struct Mesh {
  const Vertex *vertices;
  int num_vertices;
  const uint16_t *indices;
  int num_indices;
  trm3d::mat4f model;
  trm3d::aabb3f aabb; // ローカル空間AABB（カリング用）
};

// Mesh 配列を描画する
template <typename PixelT, typename Layout, typename FragShaderT>
void draw(Texture<PixelT, Layout> &tex, uint16_t *depth, const Mesh *meshes,
          int mesh_count, const trm3d::mat4f &vp, TransformedVertex *scratch,
          const FragShaderT &shader) {
  for (int i = 0; i < mesh_count; i++) {
    const auto &m = meshes[i];
    auto mvp = vp * m.model;
    VAO<Vertex> vao{{m.vertices, m.num_vertices}};
    transform_vertices<Vertex, DefaultVertexShader>(vao, mvp, scratch);
    rasterize<PixelT, Layout, FragShaderT>(tex, depth, scratch, m.indices,
                                           m.num_indices, shader);
  }
}

} // namespace usolaris