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
// NOTE: 最適化のため、span は必ず 2のべき乗（2, 4, 8, 16...）である必要があります
template <typename PixelT, typename Layout, typename SkyFn>
void draw_sky(Texture<PixelT, Layout> &tex, const uint16_t *depth,
              const trm3d::mat4f &inv_vp, const trm3d::vec3f &eye,
              SkyFn sky_fn, int span = 16) {
  // 2のべき乗チェック（2, 4, 8, 16... のみ許可）
  if (!(span > 0 && (span & (span - 1)) == 0)) {
    std::fprintf(stderr, "Error: draw_sky span must be a power of two (given: %d)\n", span);
    std::abort();
  }

  const float W = static_cast<float>(tex.size.x);
  const float H = static_cast<float>(tex.size.y);
  
  // shift量の計算 (span=16なら4)
  int shift = 0;
  for (int t = span; t > 1; t >>= 1) shift++;

  // スクリーン座標 → 球面UV（正確計算）
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

  const int num_tiles_x = (tex.size.x + span - 1) / span + 1;
  trm3d::vec2f uv_row0[128]; // 十分なサイズ（128 * 16 = max 2048 width）
  trm3d::vec2f uv_row1[128];
  trm3d::vec2f* uv_top = uv_row0;
  trm3d::vec2f* uv_bot = uv_row1;

  // 最初の行の正確なUVを計算
  for (int tx = 0; tx < num_tiles_x; ++tx) {
    uv_top[tx] = exact_uv(tx * span, 0);
  }

  // タイルベース（ブロック）描画
  // これにより空間的局所性が劇的に高まり、BC1Samplerのキャッシュヒット率が90%以上になります
  for (int sy = 0; sy < tex.size.y; sy += span) {
    int sy_full = sy + span;
    int sy_end = std::min(sy_full, tex.size.y);
    
    // 次の行のUVを計算
    for (int tx = 0; tx < num_tiles_x; ++tx) {
      uv_bot[tx] = exact_uv(tx * span, sy_full);
    }

    for (int sx = 0, tx = 0; sx < tex.size.x; sx += span, tx++) {
      int sx_full = sx + span;
      int sx_end = std::min(sx_full, tex.size.x);

      trm3d::vec2f uv_tl = uv_top[tx];
      trm3d::vec2f uv_tr = uv_top[tx + 1];
      trm3d::vec2f uv_bl = uv_bot[tx];
      trm3d::vec2f uv_br = uv_bot[tx + 1];

      // UV wrap-around: 左上を基準にしてシームまたぎを補正
      auto fix_wrap = [&](trm3d::vec2f& uv) {
          if (uv.x - uv_tl.x > 0.5f) uv.x -= 1.0f;
          else if (uv_tl.x - uv.x > 0.5f) uv.x += 1.0f;
      };
      fix_wrap(uv_tr);
      fix_wrap(uv_bl);
      fix_wrap(uv_br);

      trm3d::vec2u16 u16_tl = Texture<PixelT, Layout>::uv_to_u16(uv_tl);
      trm3d::vec2u16 u16_tr = Texture<PixelT, Layout>::uv_to_u16(uv_tr);
      trm3d::vec2u16 u16_bl = Texture<PixelT, Layout>::uv_to_u16(uv_bl);
      trm3d::vec2u16 u16_br = Texture<PixelT, Layout>::uv_to_u16(uv_br);

      int32_t du_l = static_cast<int32_t>(u16_bl.x - u16_tl.x);
      int32_t dv_l = static_cast<int32_t>(u16_bl.y - u16_tl.y);
      int32_t du_r = static_cast<int32_t>(u16_br.x - u16_tr.x);
      int32_t dv_r = static_cast<int32_t>(u16_br.y - u16_tr.y);

      for (int y = sy; y < sy_end; y++) {
        int dy = y - sy;
        // 行の両端のUVを補間
        uint16_t row_u_l = u16_tl.x + static_cast<uint16_t>((du_l * dy) >> shift);
        uint16_t row_v_l = u16_tl.y + static_cast<uint16_t>((dv_l * dy) >> shift);
        uint16_t row_u_r = u16_tr.x + static_cast<uint16_t>((du_r * dy) >> shift);
        uint16_t row_v_r = u16_tr.y + static_cast<uint16_t>((dv_r * dy) >> shift);

        // X方向への1ピクセルあたりの増分 (DDA)
        int32_t step_u = static_cast<int32_t>(row_u_r - row_u_l) >> shift;
        int32_t step_v = static_cast<int32_t>(row_v_r - row_v_l) >> shift;

        uint16_t u = row_u_l;
        uint16_t v = row_v_l;

        int depth_idx = y * tex.size.x + sx;
        for (int x = sx; x < sx_end; x++) {
          if (depth[depth_idx++] == 0xFFFF) {
            tex.at(x, y) = sky_fn({u, v});
          }
          u += step_u; // 加算のみ！
          v += step_v;
        }
      }
    }
    // バッファをスワップして次の行へ
    std::swap(uv_top, uv_bot);
  }
}

} // namespace usolaris
