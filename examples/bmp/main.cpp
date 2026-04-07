#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>

#include <usolaris/env_map.hpp>
#include <usolaris/ferndale_studio_04_1k.hpp>
#include <usolaris/primitives.hpp>
#include <usolaris/rasterizer.hpp>
#include <usolaris/suburban_garden_1k.h>

#include <usolaris/texture.hpp>
#include <usolaris/vertex_transform.hpp>

// BMP 24bit のピクセル型（BGR順）
#pragma pack(push, 1)
struct BGR {
  uint8_t b, g, r;
};

struct BMPFileHeader {
  uint16_t signature;
  uint32_t file_size;
  uint16_t reserved1;
  uint16_t reserved2;
  uint32_t pixel_offset;
};

struct BMPDIBHeader {
  uint32_t header_size;
  int32_t width;
  int32_t height;
  uint16_t planes;
  uint16_t bits_per_pixel;
  uint32_t compression;
  uint32_t image_size;
  int32_t x_resolution;
  int32_t y_resolution;
  uint32_t colors_used;
  uint32_t colors_important;
};
#pragma pack(pop)

int main() {
  const trm3d::vec2i SIZE = {1024, 1024};

  BGR *pixels = new BGR[SIZE.x * SIZE.y];
  usolaris::Texture<BGR> tex{pixels, SIZE};

  std::fill(pixels, pixels + SIZE.x * SIZE.y, BGR{0x20, 0x20, 0x20});

  uint16_t *depth = new uint16_t[SIZE.x * SIZE.y];
  std::fill(depth, depth + SIZE.x * SIZE.y, uint16_t{0xFFFF});

  // 環境マップ MipTexture 構築（ゼロコピー）
  using ENVMAP = usolaris::FERNDALE_STUDIO_04_1K;
  usolaris::Texture<uint32_t> env_levels[ENVMAP::num_levels];
  usolaris::make_env_mip<ENVMAP>(env_levels);
  usolaris::MipTexture<uint32_t> env_mip{env_levels, ENVMAP::num_levels};

  // ICO球生成
  constexpr int SUBDIV = 2;
  constexpr int VERT_COUNT = usolaris::icosphere_vertex_count(SUBDIV);
  usolaris::Vertex sphere_verts[VERT_COUNT];
  usolaris::make_icosphere(sphere_verts, SUBDIV, 1.0f);
  usolaris::VAO<usolaris::Vertex> vao{{sphere_verts, VERT_COUNT}};

  // MVP 行列
  auto P = trm3d::perspective(0.785f, 1.0f, 0.1f, 100.0f); // 45deg
  auto V = trm3d::lookAt(trm3d::vec3f{0, 0, -3}, trm3d::vec3f{0, 0, 0},
                         trm3d::vec3f{0, 1, 0});
  trm3d::mat4f mvp = P * V;

  // 頂点変換
  usolaris::TransformedVertex transformed[VERT_COUNT];
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    usolaris::transform_vertices<usolaris::Vertex,
                                 usolaris::DefaultVertexShader>(vao, mvp,
                                                                transformed);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("vertex transform: %.3f ms\n",
                std::chrono::duration<float, std::milli>(t1 - t0).count());
  }

  // 連番インデックス（非インデックス化済み頂点列）
  uint16_t indices[VERT_COUNT];
  for (int i = 0; i < VERT_COUNT; i++)
    indices[i] = static_cast<uint16_t>(i);

  // フラグメントシェーダ（IBL diffuse + specular、バイリニア補間）
  struct EnvMirrorShader {
    const usolaris::MipTexture<uint32_t> *mip;

    BGR shade(const usolaris::FragmentInput &f) const {
      constexpr float ROUGHNESS = 0.0f;
      constexpr float NUM_LEVELS = ENVMAP::num_levels;
      trm3d::vec3f N = trm3d::normalize(f.normal);
      trm3d::vec3f I = trm3d::vec3f{0, 0, 1};
      trm3d::vec3f R = trm3d::reflect(I, N);

      auto decode = [](uint32_t p) { return usolaris::decode_rgb9e5(p); };

      // diffuse IBL: 法線方向、最大LOD（最大ボケ ≈ irradiance）
      auto &diff_tex = usolaris::get_mip_level(*mip, NUM_LEVELS - 1.0f);
      trm3d::vec3f diff = usolaris::sample_bilinear_decode(
          diff_tex, usolaris::norm_to_uv(N), decode);

      // specular IBL: 反射方向、roughness に応じた LOD
      auto &spec_tex =
          usolaris::get_mip_level(*mip, ROUGHNESS * (NUM_LEVELS - 1.0f));
      trm3d::vec3f spec = usolaris::sample_bilinear_decode(
          spec_tex, usolaris::norm_to_uv(R), decode);

      trm3d::vec3f col = diff * 0.0f + spec * 0.5f;

      auto tone = [](float v) -> uint8_t {
        float t = v / (v + 1.0f);
        return (uint8_t)(std::min(t * 255.0f, 255.0f));
      };
      return {tone(col.z), tone(col.y), tone(col.x)};
    }
  };

  EnvMirrorShader shader{&env_mip};
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    usolaris::rasterize<BGR, usolaris::LinearLayout, EnvMirrorShader>(
        tex, depth, transformed, indices, VERT_COUNT, shader);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("rasterize:        %.3f ms\n",
                std::chrono::duration<float, std::milli>(t1 - t0).count());
  }

  delete[] depth;

  BMPFileHeader file_header;
  file_header.signature = 0x4D42;
  file_header.reserved1 = 0;
  file_header.reserved2 = 0;
  file_header.pixel_offset = sizeof(BMPFileHeader) + sizeof(BMPDIBHeader);
  file_header.file_size =
      file_header.pixel_offset + SIZE.x * SIZE.y * sizeof(BGR);

  BMPDIBHeader dib_header;
  dib_header.header_size = sizeof(BMPDIBHeader);
  dib_header.width = tex.size.x;
  dib_header.height = tex.size.y;
  dib_header.planes = 1;
  dib_header.bits_per_pixel = 24;
  dib_header.compression = 0;
  dib_header.image_size = SIZE.x * SIZE.y * sizeof(BGR);
  dib_header.x_resolution = 0;
  dib_header.y_resolution = 0;
  dib_header.colors_used = 0;
  dib_header.colors_important = 0;

  std::ofstream outfile("output.bmp", std::ios::binary);
  outfile.write(reinterpret_cast<char *>(&file_header), sizeof(BMPFileHeader));
  outfile.write(reinterpret_cast<char *>(&dib_header), sizeof(BMPDIBHeader));
  outfile.write(reinterpret_cast<const char *>(tex.data),
                SIZE.x * SIZE.y * sizeof(BGR));
  outfile.close();

  delete[] pixels;

  return 0;
}
