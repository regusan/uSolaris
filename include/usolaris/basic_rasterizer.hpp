#pragma once
#include <algorithm>
#include <usolaris/texture.hpp>

namespace usolaris {

namespace detail {
inline int edge(trm3d::vec2i a, trm3d::vec2i b, trm3d::vec2i p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}
} // namespace detail

// 三角形を color で塗りつぶす（CW ワインディング）
template <typename PixelT, typename Layout>
void fill_triangle(Texture<PixelT, Layout>& tex,
                   trm3d::vec2i v0, trm3d::vec2i v1, trm3d::vec2i v2,
                   const PixelT& color) {
    int x0 = std::max(0, std::min({v0.x, v1.x, v2.x}));
    int y0 = std::max(0, std::min({v0.y, v1.y, v2.y}));
    int x1 = std::min(tex.size.x - 1, std::max({v0.x, v1.x, v2.x}));
    int y1 = std::min(tex.size.y - 1, std::max({v0.y, v1.y, v2.y}));

    for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
            trm3d::vec2i p{x, y};
            if (detail::edge(v0, v1, p) >= 0 &&
                detail::edge(v1, v2, p) >= 0 &&
                detail::edge(v2, v0, p) >= 0)
                tex.at(x, y) = color;
        }
    }
}

} // namespace usolaris
