#pragma once
#include <TinyReguMath3D.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace usolaris {

// RGB9E5 フォーマット構造体 (32-bit: Shared Exponent)
struct FormatRGB9E5 {
  uint32_t packed;

  trm3d::vec3f decode() const {
    float scale = std::ldexp(1.0f, (int)(packed >> 27) - 24);
    return {
        (packed & 0x1FF) * scale,
        ((packed >> 9) & 0x1FF) * scale,
        ((packed >> 18) & 0x1FF) * scale,
    };
  }
};

// RGB565 フォーマット構造体 (16-bit: R5 G6 B5)
struct FormatRGB565 {
  uint16_t packed;

  trm3d::vec3f decode() const {
    float r = ((packed >> 11) & 0x1F) * 0.03225806f; // / 31.0f
    float g = ((packed >> 5) & 0x3F) * 0.01587301f;  // / 63.0f
    float b = (packed & 0x1F) * 0.03225806f;
    return {r, g, b};
  }

  static FormatRGB565 encode(trm3d::vec3f color) {
    uint16_t r = static_cast<uint16_t>(std::max(0.0f, std::min(1.0f, color.x)) * 31.0f + 0.5f);
    uint16_t g = static_cast<uint16_t>(std::max(0.0f, std::min(1.0f, color.y)) * 63.0f + 0.5f);
    uint16_t b = static_cast<uint16_t>(std::max(0.0f, std::min(1.0f, color.z)) * 31.0f + 0.5f);
    return {static_cast<uint16_t>((r << 11) | (g << 5) | b)};
  }
};


// BC1 (DXT1) 圧縮ブロック (4×4 ピクセル = 8 byte)
struct BC1Block {
  uint16_t c0, c1;    // RGB565 エンドポイント
  uint32_t indices;   // 16 pixel × 2bit インデックス

  // サブピクセル (bx,by ∈ [0,3]) をデコードして vec3f [0,1] を返す
  trm3d::vec3f decode_pixel(int bx, int by) const {
    int idx = (indices >> ((by * 4 + bx) * 2)) & 0x3;

    auto unpack = [](uint16_t c) -> trm3d::vec3f {
      return {((c >> 11) & 0x1F) * 0.03225806f,
              ((c >> 5) & 0x3F) * 0.01587301f,
              (c & 0x1F) * 0.03225806f};
    };
    const trm3d::vec3f col0 = unpack(c0);
    const trm3d::vec3f col1 = unpack(c1);

    if (c0 > c1) {
      // 4色モード (透過なし)
      switch (idx) {
        case 0: return col0;
        case 1: return col1;
        case 2: return col0 * (2.0f / 3.0f) + col1 * (1.0f / 3.0f);
        default: return col0 * (1.0f / 3.0f) + col1 * (2.0f / 3.0f);
      }
    } else {
      // 3色 + 透明モード (IBL では通常使わない)
      switch (idx) {
        case 0: return col0;
        case 1: return col1;
        case 2: return col0 * 0.5f + col1 * 0.5f;
        default: return {};
      }
    }
  }

  // 特定のサブピクセルのインデックス(0~3)を抽出する
  int extract_index(int bx, int by) const {
    return (indices >> ((by * 4 + bx) * 2)) & 0x3;
  }
};

} // namespace usolaris
