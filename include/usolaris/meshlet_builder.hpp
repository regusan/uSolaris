#pragma once
#include <cstring>
#include <usolaris/mesh.hpp>

namespace usolaris {

// マテリアル毎に独立したインデックスバッファを持つセクション
struct MeshSection {
  const uint16_t *indices;
  int             num_indices;   // 3の倍数
  uint8_t         material_slot;
};

// Meshletを構築する
// - in_vertices/indices: 入力メッシュ（標準インデックス付き）
// - sections: マテリアル毎のインデックスバッファ
// - out_vertices: Meshlet単位ソート済み頂点（in_vertices と同サイズ以上）
// - out_prims: uint8_tローカルインデックス（全indices数以上）
// - out_meshlets: 生成Meshlet（num_indices/3以上）
// - scratch: uint8_t[65536]（local_map用、呼び出し元確保）
// 戻り値: 生成されたMeshlet数
inline int build_meshlets(const Vertex *in_vertices, int num_vertices,
                          const MeshSection *sections, int num_sections,
                          Vertex *out_vertices, uint8_t *out_prims,
                          Meshlet *out_meshlets, uint8_t *scratch,
                          uint8_t max_verts = 64, uint8_t max_prims = 85) {
  (void)num_vertices;
  std::memset(scratch, 0xFF, 65536);

  int vert_write   = 0; // out_vertices への書き込み位置
  int prim_write   = 0; // out_prims への書き込み位置（バイト単位）
  int meshlet_count = 0;

  // 現在構築中のMeshlet
  uint8_t  cur_vert_count = 0;
  uint8_t  cur_prim_count = 0;
  uint8_t  cur_slot       = 0;
  int      cur_vert_start = 0;
  int      cur_prim_start = 0; // out_prims のバイトオフセット
  uint16_t used[256];          // global index の reset 用リスト（最大max_verts≤255）

  auto flush = [&]() {
    if (cur_prim_count == 0)
      return;

    Meshlet &ml    = out_meshlets[meshlet_count++];
    ml.vert_offset = static_cast<uint16_t>(cur_vert_start);
    ml.vert_count  = cur_vert_count;
    ml.prim_offset = static_cast<uint32_t>(cur_prim_start / 3);
    ml.prim_count  = cur_prim_count;
    ml.material_slot = cur_slot;

    // AABB をMeshlet内頂点から計算
    trm3d::vec3f mn{ 1e9f,  1e9f,  1e9f};
    trm3d::vec3f mx{-1e9f, -1e9f, -1e9f};
    for (int i = 0; i < cur_vert_count; i++) {
      const Vertex &v = out_vertices[cur_vert_start + i];
      float px = v.px, py = v.py, pz = v.pz;
      if (px < mn.x) mn.x = px; if (px > mx.x) mx.x = px;
      if (py < mn.y) mn.y = py; if (py > mx.y) mx.y = py;
      if (pz < mn.z) mn.z = pz; if (pz > mx.z) mx.z = pz;
    }
    ml.aabb = {mn, mx - mn};
  };

  auto reset_local = [&]() {
    for (int i = 0; i < cur_vert_count; i++)
      scratch[used[i]] = 0xFF;
    cur_vert_start = vert_write;
    cur_prim_start = prim_write;
    cur_vert_count = 0;
    cur_prim_count = 0;
  };

  for (int si = 0; si < num_sections; si++) {
    const MeshSection &sec = sections[si];
    int num_tris = sec.num_indices / 3;

    // セクション切り替え時にflush
    if (cur_prim_count > 0 && cur_slot != sec.material_slot) {
      flush();
      reset_local();
    }
    cur_slot = sec.material_slot;

    for (int ti = 0; ti < num_tris; ti++) {
      uint16_t gi[3] = {
          sec.indices[ti * 3 + 0],
          sec.indices[ti * 3 + 1],
          sec.indices[ti * 3 + 2],
      };

      uint8_t new_verts = 0;
      for (int k = 0; k < 3; k++)
        if (scratch[gi[k]] == 0xFF)
          new_verts++;

      if (cur_vert_count + new_verts > max_verts || cur_prim_count >= max_prims) {
        flush();
        reset_local();
        cur_slot = sec.material_slot;
      }

      for (int k = 0; k < 3; k++) {
        if (scratch[gi[k]] == 0xFF) {
          scratch[gi[k]] = cur_vert_count;
          used[cur_vert_count] = gi[k];
          out_vertices[vert_write++] = in_vertices[gi[k]];
          cur_vert_count++;
        }
        out_prims[prim_write++] = scratch[gi[k]];
      }
      cur_prim_count++;
    }
  }

  flush();
  return meshlet_count;
}

// テスト用：各Meshletの頂点に識別色を付ける（build_meshlets後に呼ぶ）
// out_vertices の cr/cg/cb を上書きする
inline void apply_meshlet_debug_colors(Vertex *out_vertices,
                                       const Meshlet *meshlets,
                                       int num_meshlets) {
  // 識別しやすい色パレット（int8_t: -127〜127 が [-1,1]）
  static const int8_t kPalette[][3] = {
      { 127,   0,   0}, // 赤
      {   0, 127,   0}, // 緑
      {   0,   0, 127}, // 青
      { 127, 127,   0}, // 黄
      { 127,   0, 127}, // マゼンタ
      {   0, 127, 127}, // シアン
      { 127,  64,   0}, // オレンジ
      {  64,   0, 127}, // 紫
  };
  constexpr int kPaletteSize = sizeof(kPalette) / sizeof(kPalette[0]);

  for (int mi = 0; mi < num_meshlets; mi++) {
    const Meshlet &ml = meshlets[mi];
    const auto   *col = kPalette[mi % kPaletteSize];
    for (int vi = 0; vi < ml.vert_count; vi++) {
      Vertex &v = out_vertices[ml.vert_offset + vi];
      v.cr = col[0];
      v.cg = col[1];
      v.cb = col[2];
    }
  }
}

} // namespace usolaris
