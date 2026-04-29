#include <SDL2/SDL.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
#pragma pack(pop)

using ENVMAP = usolaris::SUBURBAN_GARDEN_1K_BC1;

int main(int argc, char* argv[]) {
  (void)argc; (void)argv;

  // ラスタライズ解像度（ウィンドウは少し大きくする）
  const trm3d::vec2i SIZE = {480, 480};

  // SDL初期化
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::printf("SDL_Init Error: %s\n", SDL_GetError());
    return 1;
  }

  SDL_Window *window = SDL_CreateWindow(
      "uSolaris Realtime Viewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
      SIZE.x * 2, SIZE.y * 2, SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI);
  if (!window) return 1;

  SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  if (!renderer) return 1;

  // 表示時にスケーリング（ nearest等でギザギザを維持するか blend かはお好みで ）
  SDL_RenderSetLogicalSize(renderer, SIZE.x, SIZE.y);

  SDL_Texture *sdl_tex = SDL_CreateTexture(
      renderer, SDL_PIXELFORMAT_BGR24, SDL_TEXTUREACCESS_STREAMING, SIZE.x, SIZE.y);
  if (!sdl_tex) return 1;

  // ----- データの準備 (bmp版と同じ) -----
  BGR *pixels = new BGR[SIZE.x * SIZE.y];
  usolaris::Texture<BGR> tex{pixels, SIZE};
  uint16_t *depth = new uint16_t[SIZE.x * SIZE.y];

  usolaris::BC1Texture env_levels[ENVMAP::num_levels];
  usolaris::make_env_mip_bc1<ENVMAP>(env_levels);
  usolaris::MipTexture<usolaris::BC1Texture> env_mip{env_levels, ENVMAP::num_levels};

  // メッシュとMeshlet生成
  constexpr int SUBDIV     = 2;
  constexpr int VERT_COUNT = usolaris::icosphere_vertex_count(SUBDIV);
  constexpr int TRI_COUNT  = VERT_COUNT / 3;

  usolaris::Vertex raw_verts[VERT_COUNT];
  usolaris::make_icosphere(raw_verts, SUBDIV, 0.7f);

  uint16_t raw_indices[VERT_COUNT];
  for (int i = 0; i < VERT_COUNT; i++) raw_indices[i] = static_cast<uint16_t>(i);

  const int half_tris = TRI_COUNT / 2;
  usolaris::MeshSection sections[] = {
      {raw_indices, half_tris * 3, 0},
      {raw_indices + half_tris * 3, (TRI_COUNT - half_tris) * 3, 1},
  };

  usolaris::Vertex  out_verts[VERT_COUNT];
  uint8_t           out_prims[VERT_COUNT];
  usolaris::Meshlet out_meshlets[TRI_COUNT];
  uint8_t           scratch[65536];

  int num_meshlets = usolaris::build_meshlets(
      raw_verts, VERT_COUNT, sections, 2, out_verts, out_prims, out_meshlets, scratch);

  usolaris::apply_meshlet_debug_colors(out_verts, out_meshlets, num_meshlets);
  usolaris::Mesh mesh{out_verts, VERT_COUNT, out_prims, TRI_COUNT, out_meshlets, num_meshlets, {}};

  const int mat_ids[] = {0, 1};
  
  constexpr int NUM_OBJECTS = 40;
  std::vector<usolaris::MeshInstance> objects;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist_s(1.8f, 9.0f);
  std::uniform_real_distribution<float> dist_pos(-12.0f, 12.0f);
  std::uniform_real_distribution<float> dist_z(-5.0f, 20.0f);

  for (int i = 0; i < NUM_OBJECTS; i++) {
    float s = dist_s(rng);
    trm3d::mat4f m;
    m[0][0] = s; m[1][1] = s; m[2][2] = s;
    m[3][0] = dist_pos(rng); m[3][1] = dist_pos(rng); m[3][2] = dist_z(rng);
    objects.push_back({&mesh, m, mat_ids, 2});
  }

  // ----- カメラとインタラクション用変数 -----
  trm3d::vec3f eye{0, 0, -20.0f};
  float cam_yaw = 0.0f;
  float cam_pitch = 0.0f;

  constexpr int NUM_MATS = 2;
  
  auto &spec_tex = usolaris::get_mip_level(env_mip, 0.0f);
  auto &sky_lev = usolaris::get_mip_level(env_mip, 0.0f);

  bool quit = false;
  SDL_Event e;
  
  auto last_time = std::chrono::high_resolution_clock::now();
  int frame_count = 0;

  // メインループ
  while (!quit) {
    while (SDL_PollEvent(&e) != 0) {
      if (e.type == SDL_QUIT) quit = true;
    }

    // キー入力取得
    const uint8_t *keys = SDL_GetKeyboardState(NULL);
    const float speed = 0.5f;
    const float rot_speed = 0.05f;

    // カメラ操作 (十字キーで回転)
    if (keys[SDL_SCANCODE_LEFT])  cam_yaw   += rot_speed;
    if (keys[SDL_SCANCODE_RIGHT]) cam_yaw   -= rot_speed;
    if (keys[SDL_SCANCODE_UP])    cam_pitch += rot_speed;
    if (keys[SDL_SCANCODE_DOWN])  cam_pitch -= rot_speed;

    // ピッチの制限 (真上や真下を向きすぎないように)
    cam_pitch = std::max(-1.57f, std::min(1.57f, cam_pitch));

    // FPSカメラ: ヨー(ワールドY軸周り)とピッチ(ローカルX軸周り)から各ベクトルを構築
    // !! TinyReguMath3D は【右手系 (RH) ・ -Z が前方】の仕様に変更されました !!
    float cy = std::cos(cam_yaw);
    float sy = std::sin(cam_yaw);
    float cp = std::cos(cam_pitch);
    float sp = std::sin(cam_pitch);

    // 右手系での前方ベクトル (Pitch=0, Yaw=0 のとき -Z方向を向く)
    // Upキー(Pitch+)で上(+Y)、Leftキー(Yaw+)で左(-X)を向くように調整
    trm3d::vec3f forward = {
        -sy * cp,
        sp,
        -cy * cp
    };
    
    // 右手系外積ルール (s = f x up) に従ってRightベクトルを生成
    trm3d::vec3f up_world = {0, 1, 0};
    trm3d::vec3f right = trm3d::normalize(trm3d::cross(forward, up_world));
    trm3d::vec3f up = trm3d::cross(right, forward);

    // カメラ操作 (WASDによるローカル移動)
    if (keys[SDL_SCANCODE_W]) eye = eye + forward * speed;
    if (keys[SDL_SCANCODE_S]) eye = eye - forward * speed;
    if (keys[SDL_SCANCODE_A]) eye = eye - right * speed;
    if (keys[SDL_SCANCODE_D]) eye = eye + right * speed;
    if (keys[SDL_SCANCODE_E]) eye.y += speed;
    if (keys[SDL_SCANCODE_Q]) eye.y -= speed;

    auto P  = trm3d::perspective(1.5708f, (float)SIZE.x / SIZE.y, 0.1f, 100.0f);
    auto V  = trm3d::lookAt(eye, eye + forward, up);
    auto vp = P * V;
    auto inv_vp = trm3d::inverse(vp);

    // バッファのクリア
    std::fill(depth, depth + SIZE.x * SIZE.y, uint16_t{0xFFFF});
    
    // Bin のクリアと構築
    std::vector<usolaris::PendingMeshlet> bins[NUM_MATS];
    usolaris::build_bins(objects.data(), NUM_OBJECTS,
        [&](int mat_id, usolaris::PendingMeshlet m) { bins[mat_id].push_back(m); });

    // フレーム描画
    usolaris::draw_bins(tex, depth, bins[0].data(), bins[0].size(), objects.data(), vp, eye, [&](const usolaris::FragmentInput &f) -> BGR {
      float u = f.reflect_uv.x - std::floor(f.reflect_uv.x);
      trm3d::vec3f col = spec_tex.sample_bilinear({u, f.reflect_uv.y}) * 0.5f;
      auto c = [](float v) -> uint8_t { return (uint8_t)(std::min(v * 255.0f, 255.0f)); };
      return {c(col.z), c(col.y), c(col.x)};
    });

    usolaris::draw_bins(tex, depth, bins[1].data(), bins[1].size(), objects.data(), vp, eye, [&](const usolaris::FragmentInput &f) -> BGR {
      auto c = [](float v) -> uint8_t { return (uint8_t)(std::min(std::max(v, 0.0f), 1.0f) * 255.0f); };
      return {c(f.color.z), c(f.color.y), c(f.color.x)};
    });

    usolaris::BC1Sampler<8> s_sky_sampler;
    s_sky_sampler.bind(sky_lev);
    int sky_shift_x = 16 - 8;
    int sky_shift_y = 16 - 7;

    usolaris::draw_sky(tex, depth, inv_vp, eye, [&](trm3d::vec2u16 uv) -> BGR {
      trm3d::vec3f col_f = s_sky_sampler.sample(uv.x >> sky_shift_x, uv.y >> sky_shift_y);
      auto c = [](float v) -> uint8_t { return (uint8_t)(std::min(v * 255.0f, 255.0f)); };
      return {c(col_f.z), c(col_f.y), c(col_f.x)};
    });

    // SDLテクスチャの更新とウィンドウへの転送
    SDL_UpdateTexture(sdl_tex, NULL, tex.data, SIZE.x * sizeof(BGR));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, sdl_tex, NULL, NULL);
    SDL_RenderPresent(renderer);

    frame_count++;
    auto current_time = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(current_time - last_time).count();
    if (elapsed >= 1.0f) {
      char title[256];
      std::snprintf(title, sizeof(title), "uSolaris Realtime Viewer | %d FPS", frame_count);
      SDL_SetWindowTitle(window, title);
      frame_count = 0;
      last_time = current_time;
    }
  }

  delete[] depth;
  delete[] pixels;
  SDL_DestroyTexture(sdl_tex);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
