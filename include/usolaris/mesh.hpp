#pragma once
#include <TinyReguMath3D.hpp>
#include <cstdint>
#include <usolaris/vertex.hpp>
#include <usolaris/vertex_transform.hpp>

namespace usolaris {

// Mesh内の三角形グループ（頂点・プリムはMeshlet単位でソート済み）
struct Meshlet {
  uint16_t vert_offset;  // mesh.vertices[]へのオフセット
  uint8_t vert_count;    // ≤64
  uint32_t prim_offset;  // mesh.meshlet_prims[]へのオフセット（三角形単位）
  uint8_t prim_count;    // ≤85
  uint8_t material_slot; // object.material_ids[slot]でグローバルIDに変換
  trm3d::aabb3f aabb;    // Object空間AABB（カリング用）
};

// ジオメトリのみ（インスタンス情報なし）
// vertices は Meshlet単位で事前ソート済み
// meshlet_prims は uint8_t のローカルインデックス（0〜vert_count-1）
struct Mesh {
  const Vertex *vertices;
  int num_vertices;
  const uint8_t *meshlet_prims; // 3個で1三角形
  int num_prims;                // 全Meshletのprim_count合計
  const Meshlet *meshlets;
  int num_meshlets;
  trm3d::aabb3f aabb;
};

// Meshのインスタンス
struct MeshInstance {
  const Mesh *mesh;
  trm3d::mat4f model;
  const int *material_ids; // material_ids[slot] = グローバルmaterial index
  int num_material_slots;
};

// 即時描画のためのメッシュレット参照（頂点変換前）
struct PendingMeshlet {
  uint16_t object_index;
  uint16_t meshlet_index;
};

} // namespace usolaris
