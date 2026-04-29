#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include <random>

#include <usolaris/env_map.hpp>
#include <usolaris/suburban_garden_1k_bc1.hpp>
#include <usolaris/mesh.hpp>
#include <usolaris/meshlet_builder.hpp>
#include <usolaris/primitives.hpp>
#include <usolaris/scene.hpp>
#include <usolaris/texture.hpp>

#pragma pack(push, 1)
struct BGR {
  uint8_t b, g, r;
};
struct BMPFileHeader {
  uint16_t signature;
  uint32_t file_size;
  uint16_t reserved1, reserved2;
  uint32_t pixel_offset;
};
struct BMPDIBHeader {
  uint32_t header_size;
  int32_t width, height;
  uint16_t planes, bits_per_pixel;
  uint32_t compression, image_size;
  int32_t x_resolution, y_resolution;
  uint32_t colors_used, colors_important;
};
#pragma pack(pop)

using ENVMAP = usolaris::SUBURBAN_GARDEN_1K_BC1;



int main() {
  const trm3d::vec2i SIZE = {240, 240};

  BGR *pixels = new BGR[SIZE.x * SIZE.y];
  usolaris::Texture<BGR> tex{pixels, SIZE};
  std::fill(pixels, pixels + SIZE.x * SIZE.y, BGR{0x20, 0x20, 0x20});

  uint16_t *depth = new uint16_t[SIZE.x * SIZE.y];
  std::fill(depth, depth + SIZE.x * SIZE.y, uint16_t{0xFFFF});

  usolaris::BC1Texture env_levels[ENVMAP::num_levels];
  usolaris::make_env_mip_bc1<ENVMAP>(env_levels);
  usolaris::MipTexture<usolaris::BC1Texture> env_mip{env_levels, ENVMAP::num_levels};

  // ICO球生成（フラット頂点）
  constexpr int SUBDIV     = 2;
  constexpr int VERT_COUNT = usolaris::icosphere_vertex_count(SUBDIV);
  constexpr int TRI_COUNT  = VERT_COUNT / 3;

  usolaris::Vertex raw_verts[VERT_COUNT];
  usolaris::make_icosphere(raw_verts, SUBDIV, 0.7f);

  // フラット頂点なのでインデックスは0,1,2,3,...
  uint16_t raw_indices[VERT_COUNT];
  for (int i = 0; i < VERT_COUNT; i++)
    raw_indices[i] = static_cast<uint16_t>(i);

  // 前半をmaterial_slot 0、後半をmaterial_slot 1に分割
  const int half_tris = TRI_COUNT / 2;
  usolaris::MeshSection sections[] = {
      {raw_indices,                  half_tris * 3,               0},
      {raw_indices + half_tris * 3, (TRI_COUNT - half_tris) * 3, 1},
  };

  // Meshlet構築
  usolaris::Vertex  out_verts[VERT_COUNT];
  uint8_t           out_prims[VERT_COUNT]; // フラットなので最大VERT_COUNT
  usolaris::Meshlet out_meshlets[TRI_COUNT];
  uint8_t           scratch[65536];

  int num_meshlets = usolaris::build_meshlets(
      raw_verts, VERT_COUNT,
      sections, 2,
      out_verts, out_prims, out_meshlets,
      scratch);

  std::printf("meshlets: %d\n", num_meshlets);

  // デバッグ色を適用
  usolaris::apply_meshlet_debug_colors(out_verts, out_meshlets, num_meshlets);

  usolaris::Mesh mesh{
      out_verts,    VERT_COUNT,
      out_prims,    TRI_COUNT,
      out_meshlets, num_meshlets,
      {}};

  // material_ids: slot0=metal, slot1=debug_color
  const int mat_ids[] = {0, 1};
  
  constexpr int NUM_OBJECTS = 40;
  std::vector<usolaris::MeshInstance> objects;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist_s(1.8f, 9.0f);     // サイズをさらに3倍 (1.8 ~ 9.0)
  std::uniform_real_distribution<float> dist_pos(-12.0f, 12.0f); // X, Y の散らばり
  std::uniform_real_distribution<float> dist_z(-5.0f, 20.0f);    // 奥行き

  for (int i = 0; i < NUM_OBJECTS; i++) {
    float s = dist_s(rng);
    float x = dist_pos(rng);
    float y = dist_pos(rng);
    float z = dist_z(rng);

    trm3d::mat4f m;
    m[0][0] = s; m[1][1] = s; m[2][2] = s;
    m[3][0] = x; m[3][1] = y; m[3][2] = z;
    
    objects.push_back({&mesh, m, mat_ids, 2});
  }

  const trm3d::vec3f eye{0, 0, -20.0f};
  auto P  = trm3d::perspective(1.5708f, 1.0f, 0.1f, 100.0f);
  auto V  = trm3d::lookAt(eye, trm3d::vec3f{0, 0, 0}, trm3d::vec3f{0, 1, 0});
  auto vp = P * V;

  constexpr int NUM_MATS = 2;
  std::vector<usolaris::PendingMeshlet> bins[NUM_MATS];



  {
    auto t0 = std::chrono::high_resolution_clock::now();
    usolaris::build_bins(objects.data(), NUM_OBJECTS,
        [&](int mat_id, usolaris::PendingMeshlet e) {
          bins[mat_id].push_back(e);
        });
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("build_bins: %.3f ms\n",
                std::chrono::duration<float, std::milli>(t1 - t0).count());
  }

  const usolaris::PendingMeshlet *bin_ptrs[NUM_MATS] = {
      bins[0].data(), bins[1].data()};
  int bin_counts[NUM_MATS] = {(int)bins[0].size(), (int)bins[1].size()};

  // draw + sky: 100回ループして平均
  {
    constexpr int N = 100;
    const auto inv_vp = trm3d::inverse(vp);
    const auto &sky_lev = usolaris::get_mip_level(env_mip, 0.0f);

    float t_draw_sum = 0.f, t_sky_sum = 0.f;
    for (int iter = 0; iter < N; iter++) {
      std::fill(depth, depth + SIZE.x * SIZE.y, uint16_t{0xFFFF});

      auto t0 = std::chrono::high_resolution_clock::now();

      auto &spec_tex = usolaris::get_mip_level(env_mip, 0.0f);
      usolaris::draw_bins(tex, depth, bins[0].data(), bins[0].size(), objects.data(), vp, eye,
                          [&](const usolaris::FragmentInput &f) -> BGR {
                            float u = f.reflect_uv.x - std::floor(f.reflect_uv.x);
                            trm3d::vec3f col = spec_tex.sample_bilinear({u, f.reflect_uv.y}) * 0.5f;
                            auto c = [](float v) -> uint8_t { return (uint8_t)(std::min(v * 255.0f, 255.0f)); };
                            return {c(col.z), c(col.y), c(col.x)};
                          });

      // マテリアル1 (debug_color)
      usolaris::draw_bins(tex, depth, bins[1].data(), bins[1].size(), objects.data(), vp, eye,
                          [&](const usolaris::FragmentInput &f) -> BGR {
                            auto c = [](float v) -> uint8_t {
                              return (uint8_t)(std::min(std::max(v, 0.0f), 1.0f) * 255.0f);
                            };
                            return {c(f.color.z), c(f.color.y), c(f.color.x)};
                          });
      auto t1 = std::chrono::high_resolution_clock::now();

      usolaris::BC1Sampler<8> s_sky_sampler;
      s_sky_sampler.bind(sky_lev);
      // sky_lev (Level 0) は 256x128 なので、2^8 と 2^7。固定小数(16bit)からのシフト量は:
      int sky_shift_x = 16 - 8;
      int sky_shift_y = 16 - 7;

      usolaris::draw_sky(tex, depth, inv_vp, eye, [&](trm3d::vec2u16 uv) -> BGR {
        trm3d::vec3f col_f = s_sky_sampler.sample(uv.x >> sky_shift_x, uv.y >> sky_shift_y);
        auto c = [](float v) -> uint8_t { return (uint8_t)(std::min(v * 255.0f, 255.0f)); };
        return {c(col_f.z), c(col_f.y), c(col_f.x)};
      });
      auto t2 = std::chrono::high_resolution_clock::now();

      t_draw_sum += std::chrono::duration<float, std::milli>(t1 - t0).count();
      t_sky_sum  += std::chrono::duration<float, std::milli>(t2 - t1).count();
    }
    std::printf("draw: %.3f ms avg (%d runs)\n", t_draw_sum / N, N);
    std::printf("sky:  %.3f ms avg (%d runs)\n", t_sky_sum  / N, N);
  }

  delete[] depth;
  BMPFileHeader fh{};
  fh.signature    = 0x4D42;
  fh.pixel_offset = sizeof(BMPFileHeader) + sizeof(BMPDIBHeader);
  fh.file_size    = fh.pixel_offset + SIZE.x * SIZE.y * sizeof(BGR);
  BMPDIBHeader dh{};
  dh.header_size    = sizeof(BMPDIBHeader);
  dh.width          = tex.size.x;
  dh.height         = tex.size.y;
  dh.planes         = 1;
  dh.bits_per_pixel = 24;
  dh.image_size     = SIZE.x * SIZE.y * sizeof(BGR);

  std::ofstream out("output.bmp", std::ios::binary);
  out.write(reinterpret_cast<char *>(&fh), sizeof(fh));
  out.write(reinterpret_cast<char *>(&dh), sizeof(dh));
  out.write(reinterpret_cast<const char *>(tex.data), SIZE.x * SIZE.y * sizeof(BGR));

  delete[] pixels;
  return 0;
}
