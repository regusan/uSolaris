#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include <usolaris/env_map.hpp>
#include <usolaris/ferndale_studio_04_1k.hpp>
#include <usolaris/mesh.hpp>
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

using ENVMAP = usolaris::FERNDALE_STUDIO_04_1K;

// ---- シェーダ関数 ----

static BGR metal_shade(const usolaris::FragmentInput &f, const void *ctx) {
  auto *mip = static_cast<const usolaris::MipTexture<uint32_t> *>(ctx);
  trm3d::vec3f N = trm3d::normalize(f.normal);
  trm3d::vec3f R = trm3d::reflect(trm3d::vec3f{0, 0, 1}, N);
  auto decode    = [](uint32_t p) { return usolaris::decode_rgb9e5(p); };
  auto &spec_tex = usolaris::get_mip_level(*mip, 0.0f);
  trm3d::vec3f col =
      usolaris::sample_bilinear_decode(spec_tex, usolaris::norm_to_uv(R), decode) * 0.5f;
  auto tone = [](float v) -> uint8_t {
    return (uint8_t)(std::min(v / (v + 1.0f) * 255.0f, 255.0f));
  };
  return {tone(col.z), tone(col.y), tone(col.x)};
}

static BGR diffuse_shade(const usolaris::FragmentInput &f, const void *ctx) {
  auto *mip = static_cast<const usolaris::MipTexture<uint32_t> *>(ctx);
  trm3d::vec3f N  = trm3d::normalize(f.normal);
  auto decode     = [](uint32_t p) { return usolaris::decode_rgb9e5(p); };
  auto &diff_tex  = usolaris::get_mip_level(*mip, ENVMAP::num_levels - 1.0f);
  trm3d::vec3f col =
      usolaris::sample_bilinear_decode(diff_tex, usolaris::norm_to_uv(N), decode);
  auto tone = [](float v) -> uint8_t {
    return (uint8_t)(std::min(v / (v + 1.0f) * 255.0f, 255.0f));
  };
  return {tone(col.z), tone(col.y), tone(col.x)};
}

// ---- Meshletビルダ ----

static void build_icosphere_meshlets(int total_tris, int tris_per_meshlet,
                                     int material_slot_fn(int meshlet_idx),
                                     std::vector<usolaris::Meshlet> &out_meshlets,
                                     std::vector<uint8_t> &out_prims) {
  int tri = 0, meshlet_idx = 0;
  while (tri < total_tris) {
    int count = std::min(tris_per_meshlet, total_tris - tri);
    usolaris::Meshlet ml{};
    ml.vert_offset   = static_cast<uint16_t>(tri * 3);
    ml.vert_count    = static_cast<uint8_t>(count * 3);
    ml.prim_offset   = static_cast<uint32_t>(out_prims.size() / 3);
    ml.prim_count    = static_cast<uint8_t>(count);
    ml.material_slot = static_cast<uint8_t>(material_slot_fn(meshlet_idx));
    ml.aabb          = {};
    for (int i = 0; i < count * 3; i++)
      out_prims.push_back(static_cast<uint8_t>(i));
    out_meshlets.push_back(ml);
    tri += count;
    meshlet_idx++;
  }
}

int main() {
  const trm3d::vec2i SIZE = {240, 240};

  BGR *pixels = new BGR[SIZE.x * SIZE.y];
  usolaris::Texture<BGR> tex{pixels, SIZE};
  std::fill(pixels, pixels + SIZE.x * SIZE.y, BGR{0x20, 0x20, 0x20});

  uint16_t *depth = new uint16_t[SIZE.x * SIZE.y];
  std::fill(depth, depth + SIZE.x * SIZE.y, uint16_t{0xFFFF});

  usolaris::Texture<uint32_t> env_levels[ENVMAP::num_levels];
  usolaris::make_env_mip<ENVMAP>(env_levels);
  usolaris::MipTexture<uint32_t> env_mip{env_levels, ENVMAP::num_levels};

  constexpr int SUBDIV          = 2;
  constexpr int VERT_COUNT      = usolaris::icosphere_vertex_count(SUBDIV);
  constexpr int TRI_COUNT       = VERT_COUNT / 3;
  constexpr int TRIS_PER_MESHLET = 21;

  usolaris::Vertex verts_s[VERT_COUNT], verts_m[VERT_COUNT], verts_l[VERT_COUNT];
  usolaris::make_icosphere(verts_s, SUBDIV, 0.4f);
  usolaris::make_icosphere(verts_m, SUBDIV, 0.7f);
  usolaris::make_icosphere(verts_l, SUBDIV, 1.0f);

  std::vector<usolaris::Meshlet> meshlets_s, meshlets_m, meshlets_l;
  std::vector<uint8_t>           prims_s,    prims_m,    prims_l;
  auto mat_slot = [](int mi) { return mi % 2; };
  build_icosphere_meshlets(TRI_COUNT, TRIS_PER_MESHLET, mat_slot, meshlets_s, prims_s);
  build_icosphere_meshlets(TRI_COUNT, TRIS_PER_MESHLET, mat_slot, meshlets_m, prims_m);
  build_icosphere_meshlets(TRI_COUNT, TRIS_PER_MESHLET, mat_slot, meshlets_l, prims_l);

  usolaris::Mesh mesh_s{verts_s, VERT_COUNT, prims_s.data(), (int)prims_s.size() / 3,
                        meshlets_s.data(), (int)meshlets_s.size(), {}};
  usolaris::Mesh mesh_m{verts_m, VERT_COUNT, prims_m.data(), (int)prims_m.size() / 3,
                        meshlets_m.data(), (int)meshlets_m.size(), {}};
  usolaris::Mesh mesh_l{verts_l, VERT_COUNT, prims_l.data(), (int)prims_l.size() / 3,
                        meshlets_l.data(), (int)meshlets_l.size(), {}};

  const int mat_ids[] = {0, 1};
  auto ml = trm3d::translate(trm3d::mat4f{}, trm3d::vec3f{-1.8f, 0.0f, 0.0f});
  auto mm = trm3d::mat4f{};
  auto mr = trm3d::translate(trm3d::mat4f{}, trm3d::vec3f{ 1.8f, 0.0f, 0.0f});
  usolaris::Object objects[] = {
      {&mesh_s, ml, mat_ids, 2},
      {&mesh_m, mm, mat_ids, 2},
      {&mesh_l, mr, mat_ids, 2},
  };

  auto P  = trm3d::perspective(1.5708f, 1.0f, 0.1f, 100.0f);
  auto V  = trm3d::lookAt(trm3d::vec3f{0, 0, -3}, trm3d::vec3f{0, 0, 0},
                          trm3d::vec3f{0, 1, 0});
  auto vp = P * V;

  constexpr int TOTAL_VERTS = 3 * VERT_COUNT;
  usolaris::TransformedVertex xverts[TOTAL_VERTS];

  constexpr int NUM_MATS = 2;
  std::vector<usolaris::TransformedMeshlet> bins[NUM_MATS];

  // シェーダ登録
  usolaris::MaterialShader<BGR> shaders[NUM_MATS] = {
      {metal_shade,   &env_mip},
      {diffuse_shade, &env_mip},
  };

  {
    auto t0 = std::chrono::high_resolution_clock::now();

    usolaris::build_bins(objects, 3, vp, xverts,
        [&](int mat_id, usolaris::TransformedMeshlet e) {
          bins[mat_id].push_back(e);
        });

    const usolaris::TransformedMeshlet *bin_ptrs[NUM_MATS] = {
        bins[0].data(), bins[1].data()};
    int bin_counts[NUM_MATS] = {(int)bins[0].size(), (int)bins[1].size()};

    usolaris::draw(tex, depth, bin_ptrs, bin_counts, NUM_MATS, shaders);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("draw (3 objects, 2 materials): %.3f ms\n",
                std::chrono::duration<float, std::milli>(t1 - t0).count());
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
