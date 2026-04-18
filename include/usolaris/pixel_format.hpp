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
    float r = ((packed >> 11) & 0x1F) / 31.0f;
    float g = ((packed >> 5) & 0x3F) / 63.0f;
    float b = (packed & 0x1F) / 31.0f;
    return {r, g, b};
  }

  static FormatRGB565 encode(trm3d::vec3f color) {
    uint16_t r = static_cast<uint16_t>(std::max(0.0f, std::min(1.0f, color.x)) * 31.0f + 0.5f);
    uint16_t g = static_cast<uint16_t>(std::max(0.0f, std::min(1.0f, color.y)) * 63.0f + 0.5f);
    uint16_t b = static_cast<uint16_t>(std::max(0.0f, std::min(1.0f, color.z)) * 31.0f + 0.5f);
    return {static_cast<uint16_t>((r << 11) | (g << 5) | b)};
  }
};

} // namespace usolaris
