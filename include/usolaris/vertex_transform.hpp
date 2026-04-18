#pragma once
#include <TinyReguMath3D.hpp>
#include <usolaris/vertex.hpp>
#include <usolaris/env_map.hpp>

namespace usolaris {

// MVP 変換後の展開済み頂点
struct TransformedVertex {
    trm3d::vec4f clip_pos;
    trm3d::vec2f uv;
    trm3d::vec3f normal;
    trm3d::vec3f color;
    trm3d::vec2f reflect_uv;
};

// デフォルト頂点シェーダ（Vertex 型用）
struct DefaultVertexShader {
    static TransformedVertex shade(const Vertex& v, const trm3d::mat4f& mvp, const trm3d::mat4f& m, const trm3d::vec3f& eye) {
        trm3d::vec3f local_n = {v.nx / 127.0f, v.ny / 127.0f, v.nz / 127.0f};
        trm3d::vec3f world_n = trm3d::fast_normalize(trm3d::mat3f(m) * local_n);

        trm3d::vec3f world_pos = trm3d::vec3f(m * trm3d::vec4f{v.px, v.py, v.pz, 1.0f});
        trm3d::vec3f I = trm3d::fast_normalize(world_pos - eye);
        trm3d::vec3f R = trm3d::reflect(I, world_n);

        return {
            mvp * trm3d::vec4f{v.px, v.py, v.pz, 1.0f},
            {v.u / 127.0f, v.v / 127.0f},
            world_n,
            {v.cr / 127.0f, v.cg / 127.0f, v.cb / 127.0f},
            norm_to_uv(R)
        };
    }
};

// VAO の全頂点に VertexShaderT::shade を適用
// out は vao.vbo.count 以上の要素を持つバッファを呼び出し元が用意する
template <typename VertexT, typename VertexShaderT>
void transform_vertices(const VAO<VertexT>& vao, const trm3d::mat4f& mvp,
                        TransformedVertex* out) {
    for (int i = 0; i < vao.vbo.count; i++)
        out[i] = VertexShaderT::shade(vao.vbo.data[i], mvp);
}

} // namespace usolaris
