#pragma once
#include <cstdint>

namespace usolaris {

// 頂点属性（圧縮フォーマット）
struct Vertex {
    float px, py, pz;     // 位置
    int8_t u, v;          // UV     (/ 127.0f で [0,1] に展開)
    int8_t nx, ny, nz;    // 法線   (/ 127.0f で [-1,1] に展開)
    int8_t cr, cg, cb;    // 頂点色 (/ 127.0f で [0,1] に展開)
};

// 頂点バッファ（非所有）
template <typename VertexT>
struct VBO {
    const VertexT* data;
    int count;
};

// 頂点配列オブジェクト
template <typename VertexT>
struct VAO {
    VBO<VertexT> vbo;
};

} // namespace usolaris
