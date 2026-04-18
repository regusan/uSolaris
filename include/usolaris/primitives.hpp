#pragma once
#include <TinyReguMath3D.hpp>
#include <cmath>
#include <usolaris/vertex.hpp>

namespace usolaris {

// ICO球の頂点数（subdivisions 回細分化後）
constexpr int icosphere_vertex_count(int subdivisions) {
  int faces = 20;
  for (int i = 0; i < subdivisions; i++)
    faces *= 4;
  return faces * 3;
}

namespace detail {

inline Vertex make_sphere_vertex(const trm3d::vec3f &pos, float radius,
                                 int8_t cr, int8_t cg, int8_t cb) {
  trm3d::vec3f n = trm3d::normalize(pos);
  trm3d::vec3f p = n * radius;
  float u = std::atan2(n.z, n.x) / (2.0f * 3.14159265f) + 0.5f;
  float v = std::asin(n.y) / 3.14159265f + 0.5f;
  return {
      p.x,
      p.y,
      p.z,
      (int8_t)(u * 127.0f),
      (int8_t)(v * 127.0f),
      (int8_t)(n.x * 127.0f),
      (int8_t)(n.y * 127.0f),
      (int8_t)(n.z * 127.0f),
      cr,
      cg,
      cb,
  };
}

inline void subdivide(Vertex *out, int &idx, trm3d::vec3f v0, trm3d::vec3f v1,
                      trm3d::vec3f v2, int level, float radius, int8_t cr,
                      int8_t cg, int8_t cb) {
  if (level == 0) {
    out[idx++] = make_sphere_vertex(v0, radius, cr, cg, cb);
    out[idx++] = make_sphere_vertex(v1, radius, cr, cg, cb);
    out[idx++] = make_sphere_vertex(v2, radius, cr, cg, cb);
    return;
  }
  trm3d::vec3f m01 = trm3d::normalize((v0 + v1) * 0.5f);
  trm3d::vec3f m12 = trm3d::normalize((v1 + v2) * 0.5f);
  trm3d::vec3f m20 = trm3d::normalize((v2 + v0) * 0.5f);
  subdivide(out, idx, v0, m01, m20, level - 1, radius, cr, cg, cb);
  subdivide(out, idx, v1, m12, m01, level - 1, radius, cr, cg, cb);
  subdivide(out, idx, v2, m20, m12, level - 1, radius, cr, cg, cb);
  subdivide(out, idx, m01, m12, m20, level - 1, radius, cr, cg, cb);
}

} // namespace detail

// ICO球を生成して out に書き込む（CCW ワインディング）
// out のサイズは icosphere_vertex_count(subdivisions) 以上必要
inline void make_icosphere(Vertex *out, int subdivisions, float radius = 1.0f,
                           int8_t cr = 127, int8_t cg = 127, int8_t cb = 127) {
  const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;

  // 正二十面体の12頂点
  const trm3d::vec3f v[12] = {
      {-1, t, 0},  {1, t, 0},  {-1, -t, 0}, {1, -t, 0}, {0, -1, t},  {0, 1, t},
      {0, -1, -t}, {0, 1, -t}, {t, 0, -1},  {t, 0, 1},  {-t, 0, -1}, {-t, 0, 1},
  };

  // 20面（CCW）
  const int f[20][3] = {
      {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
      {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
      {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
      {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1},
  };

  int idx = 0;
  for (auto &face : f)
    detail::subdivide(out, idx, v[face[0]], v[face[1]], v[face[2]],
                      subdivisions, radius, cr, cg, cb);
}

} // namespace usolaris
