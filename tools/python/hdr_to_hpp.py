#!/usr/bin/env python3
"""HDR 画像を Pre-filtered RGB9E5 C++ ヘッダに変換する

使い方:
    python3 hdr_to_hpp.py env.hdr                    # → env.h (4段階プリフィルタ)
    python3 hdr_to_hpp.py env.hdr -o out.h           # 出力先指定
    python3 hdr_to_hpp.py env.hdr --var ENV_MAP      # 変数名指定
    python3 hdr_to_hpp.py env.hdr --resize 256 128   # 最高解像度段を指定
    python3 hdr_to_hpp.py env.hdr --levels 4         # ラフネス段数
    python3 hdr_to_hpp.py env.hdr --samples 512      # GGXサンプル数/ピクセル
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image


def read_hdr(path: str) -> np.ndarray:
    """Radiance HDR (.hdr) を float32 RGB (H, W, 3) で返す"""
    with open(path, "rb") as f:
        data = f.read()

    i = 0
    while i < len(data):
        end = data.index(b"\n", i)
        line = data[i:end]
        i = end + 1
        if line == b"":
            break

    end = data.index(b"\n", i)
    size_line = data[i:end].decode()
    i = end + 1
    parts = size_line.split()
    H, W = int(parts[1]), int(parts[3])

    pixels = np.zeros((H, W, 4), dtype=np.uint8)
    for row in range(H):
        if W >= 8:
            r, g, b, e = data[i], data[i+1], data[i+2], data[i+3]
            if r == 2 and g == 2 and (b & 0x80) == 0:
                scanline_w = (b << 8) | e
                i += 4
                for ch in range(4):
                    col = 0
                    while col < scanline_w:
                        code = data[i]; i += 1
                        if code > 128:
                            count = code - 128
                            val = data[i]; i += 1
                            pixels[row, col:col+count, ch] = val
                            col += count
                        else:
                            pixels[row, col:col+code, ch] = list(data[i:i+code])
                            i += code
                            col += code
                continue
        for col in range(W):
            pixels[row, col] = [data[i], data[i+1], data[i+2], data[i+3]]
            i += 4

    exp = pixels[:, :, 3].astype(np.float32)
    scale = np.where(pixels[:, :, 3] > 0,
                     np.ldexp(1.0 / 256.0, exp.astype(int) - 128), 0.0)
    return pixels[:, :, :3].astype(np.float32) * scale[:, :, np.newaxis]


def resize_hdr(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """HDR 画像を Reinhard を介して Lanczos リサイズ"""
    tone = img / (img + 1.0)
    u8 = (np.clip(tone, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB").resize((w, h), Image.LANCZOS)
    t = np.array(pil).astype(np.float32) / 255.0
    return t / np.maximum(1.0 - t, 1e-6)


def dir_to_uv(d: np.ndarray) -> tuple:
    """方向ベクトル (N, 3) → (u, v) 各 (N,)"""
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    u = np.arctan2(d[:, 2], d[:, 0]) / (2 * np.pi) + 0.5
    v = np.arcsin(np.clip(d[:, 1], -1, 1)) / np.pi + 0.5
    return u, v


def sample_env_batch(img: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """方向ベクトル群 (N, 3) → RGB (N, 3) でサンプリング"""
    H, W = img.shape[:2]
    u, v = dir_to_uv(dirs)
    x = (u * W).astype(int) % W
    y = (v * H).astype(int) % H
    return img[y, x]


def sample_env_bilinear(img: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """方向ベクトル群 (N, 3) → RGB (N, 3) バイリニアサンプリング"""
    H, W = img.shape[:2]
    u, v = dir_to_uv(dirs)
    fx = u * W - 0.5
    fy = v * H - 0.5
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    tx = (fx - x0).astype(np.float32)[:, np.newaxis]
    ty = (fy - y0).astype(np.float32)[:, np.newaxis]
    x0 = x0 % W; x1 = (x0 + 1) % W
    y0 = y0 % H; y1 = (y0 + 1) % H
    c00 = img[y0, x0]; c10 = img[y0, x1]
    c01 = img[y1, x0]; c11 = img[y1, x1]
    return (c00 * (1 - tx) + c10 * tx) * (1 - ty) + (c01 * (1 - tx) + c11 * tx) * ty


def build_source_mips(img: np.ndarray) -> list:
    """ソース画像の MIP チェーンを生成（Reinhard 経由 Lanczos リサイズ）"""
    mips = [img]
    while mips[-1].shape[1] > 1 and mips[-1].shape[0] > 1:
        h, w = mips[-1].shape[:2]
        mips.append(resize_hdr(mips[-1], max(1, w // 2), max(1, h // 2)))
    return mips


def sample_env_lod(mips: list, dirs: np.ndarray, lod: np.ndarray) -> np.ndarray:
    """方向ベクトル群 (N, 3) を LOD ごとの MIP からバイリニアサンプリング"""
    results = np.zeros((len(dirs), 3), dtype=np.float32)
    lod_int = np.clip(np.round(lod).astype(int), 0, len(mips) - 1)
    for level in range(len(mips)):
        mask = lod_int == level
        if np.any(mask):
            results[mask] = sample_env_bilinear(mips[level], dirs[mask])
    return results


def pixel_to_dir(ox: int, oy: int, w: int, h: int) -> np.ndarray:
    """ピクセル座標 → 単位方向ベクトル"""
    u = (ox + 0.5) / w
    v = (oy + 0.5) / h
    theta = (v - 0.5) * np.pi
    phi = (u - 0.5) * 2 * np.pi
    return np.array([
        np.cos(theta) * np.cos(phi),
        np.sin(theta),
        np.cos(theta) * np.sin(phi),
    ], dtype=np.float32)


def prefilter_ggx(img: np.ndarray, roughness: float,
                  out_w: int, out_h: int, num_samples: int,
                  src_mips: list) -> np.ndarray:
    """GGX 重要度サンプリングで Pre-filtered Env Map を生成"""
    if roughness == 0.0:
        return resize_hdr(img, out_w, out_h)

    a = roughness * roughness
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)

    src_H, src_W = img.shape[:2]

    # 乱数を事前生成
    rng = np.random.default_rng(42)
    xi1 = rng.random((num_samples,)).astype(np.float32)
    xi2 = rng.random((num_samples,)).astype(np.float32)

    # GGX 半球サンプル (タンジェント空間)
    cos_theta = np.sqrt((1.0 - xi1) / np.maximum(1.0 + (a*a - 1.0) * xi1, 1e-7))
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
    phi = 2.0 * np.pi * xi2
    Hts = np.stack([sin_theta * np.cos(phi),
                    sin_theta * np.sin(phi),
                    cos_theta], axis=-1)  # (S, 3) タンジェント空間 H

    # PDF から LOD を事前計算（V≈N なので VdotH = NdotH = cos_theta）
    #   D(H) = a² / (π * (cos²θ*(a²-1)+1)²)
    #   pdf_l = D(H) / 4
    denom = np.maximum(cos_theta**2 * (a * a - 1.0) + 1.0, 1e-7)
    D = (a * a) / (np.pi * denom * denom)
    pdf_l = np.maximum(D / 4.0, 1e-7)
    lod_bias = 0.5 * np.log2(float(src_W * src_H) / num_samples)
    lod_per_sample = np.clip(lod_bias - 0.5 * np.log2(pdf_l), 0, len(src_mips) - 1)

    for oy in range(out_h):
        if oy % max(1, out_h // 8) == 0:
            print(f"  {oy}/{out_h}", end="\r", flush=True)
        for ox in range(out_w):
            N = pixel_to_dir(ox, oy, out_w, out_h)

            # TBN フレーム構築
            up = np.array([0, 1, 0], dtype=np.float32)
            if abs(N[1]) > 0.99:
                up = np.array([1, 0, 0], dtype=np.float32)
            T = np.cross(up, N)
            T /= np.linalg.norm(T)
            B = np.cross(N, T)

            # H をワールド空間に変換
            H_world = (Hts[:, 0:1] * T +
                       Hts[:, 1:2] * B +
                       Hts[:, 2:3] * N)  # (S, 3)

            # 反射ベクトル L = 2*dot(V,H)*H - V, V ≈ N
            NdotH = np.clip((H_world * N).sum(axis=-1), 0, 1)  # (S,)
            L = 2.0 * NdotH[:, np.newaxis] * H_world - N       # (S, 3)

            NdotL = np.clip((L * N).sum(axis=-1), 0, 1)        # (S,)
            mask = NdotL > 0
            if not np.any(mask):
                continue

            colors = sample_env_lod(src_mips, L[mask], lod_per_sample[mask])
            w = NdotL[mask]
            output[oy, ox] = (colors * w[:, np.newaxis]).sum(axis=0) / w.sum()

    print()
    return output


def encode_rgb9e5(rgb: np.ndarray) -> np.ndarray:
    """float32 RGB (H, W, 3) → uint32 RGB9E5 (H, W)"""
    r = np.maximum(rgb[:, :, 0], 0.0).astype(np.float32)
    g = np.maximum(rgb[:, :, 1], 0.0).astype(np.float32)
    b = np.maximum(rgb[:, :, 2], 0.0).astype(np.float32)
    max_rgb = np.maximum(np.maximum(r, g), b)
    with np.errstate(divide="ignore"):
        exp = np.floor(np.log2(np.where(max_rgb > 0, max_rgb, 1e-38))).astype(np.int32) + 16
    exp = np.clip(exp, 0, 31)
    scale = np.power(2.0, (exp - 24).astype(np.float32))
    rm = np.clip(np.round(r / scale), 0, 511).astype(np.uint32)
    gm = np.clip(np.round(g / scale), 0, 511).astype(np.uint32)
    bm = np.clip(np.round(b / scale), 0, 511).astype(np.uint32)
    return rm | (gm << 9) | (bm << 18) | (exp.astype(np.uint32) << 27)


def write_array(f, data: np.ndarray, indent: str = "        ", cols: int = 8):
    flat = data.flatten()
    for i in range(0, len(flat), cols):
        chunk = flat[i:i+cols]
        f.write(indent + ", ".join(f"0x{v:08X}" for v in chunk) + ",\n")


def main():
    parser = argparse.ArgumentParser(description="HDR → Pre-filtered RGB9E5 C++ ヘッダ変換")
    parser.add_argument("input", help="入力画像 (.hdr)")
    parser.add_argument("-o", "--output", help="出力ファイル")
    parser.add_argument("--var", help="C++ struct 名")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"),
                        default=[256, 128], help="最高解像度段 (デフォルト: 256 128)")
    parser.add_argument("--levels", type=int, default=4, help="ラフネス段数 (デフォルト: 4)")
    parser.add_argument("--samples", type=int, default=256,
                        help="GGX サンプル数/ピクセル (デフォルト: 256)")
    parser.add_argument("--save-preview", action="store_true",
                        help="各 MIP レベルを PNG で保存")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.output or (stem + ".h")
    var = args.var or stem.upper().replace("-", "_").replace(" ", "_")

    print(f"読み込み中: {args.input}")
    img = read_hdr(args.input)
    print(f"元解像度: {img.shape[1]} x {img.shape[0]}")

    print("ソース MIP チェーン生成中...")
    src_mips = build_source_mips(img)
    print(f"  {len(src_mips)} レベル生成")

    base_w, base_h = args.resize
    num_levels = args.levels
    roughness_levels = [i / (num_levels - 1) for i in range(num_levels)]

    # 各レベルの解像度（1段ごとに半分）
    level_sizes = [(max(1, base_w >> i), max(1, base_h >> i))
                   for i in range(num_levels)]

    # 各レベルをプリフィルタ生成
    level_data = []
    level_offsets = [0]
    for li, (rough, (lw, lh)) in enumerate(zip(roughness_levels, level_sizes)):
        print(f"レベル {li} (roughness={rough:.2f}, {lw}x{lh}) ...")
        filtered = prefilter_ggx(img, rough, lw, lh, args.samples, src_mips)
        encoded = encode_rgb9e5(filtered)
        level_data.append(encoded)
        level_offsets.append(level_offsets[-1] + lw * lh)

        if args.save_preview:
            tone = filtered / (filtered + 1.0)
            u8 = (np.clip(tone, 0, 1) * 255).astype(np.uint8)
            preview_path = f"{stem}_mip{li}_r{rough:.2f}_{lw}x{lh}.png"
            Image.fromarray(u8, mode="RGB").save(preview_path)
            print(f"  プレビュー: {preview_path}")

    total_px = level_offsets[-1]
    print(f"総データ: {total_px * 4 / 1024:.1f} KB")

    # ヘッダ出力
    guard = f"USOLARIS_{var}_H"
    with open(out_path, "w") as f:
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("namespace usolaris {\n\n")
        f.write(f"// Pre-filtered 環境マップ (GGX, {num_levels}段)\n")
        f.write(f"// 元画像: {os.path.basename(args.input)}\n")
        f.write(f"struct {var} {{\n")
        f.write(f"    static constexpr int num_levels = {num_levels};\n")

        ws_str = ", ".join(str(w) for w, _ in level_sizes)
        hs_str = ", ".join(str(h) for _, h in level_sizes)
        ro_str = ", ".join(f"{r:.4f}f" for r in roughness_levels)
        of_str = ", ".join(str(o) for o in level_offsets[:-1])

        f.write(f"    static constexpr int widths[{num_levels}]   = {{{ws_str}}};\n")
        f.write(f"    static constexpr int heights[{num_levels}]  = {{{hs_str}}};\n")
        f.write(f"    static constexpr float roughness[{num_levels}] = {{{ro_str}}};\n")
        f.write(f"    static constexpr int offsets[{num_levels}]  = {{{of_str}}};\n")
        f.write(f"    static constexpr uint32_t data[{total_px}] = {{\n")

        for li, enc in enumerate(level_data):
            lw, lh = level_sizes[li]
            f.write(f"        // level {li}: roughness={roughness_levels[li]:.2f}  {lw}x{lh}\n")
            write_array(f, enc)

        f.write("    };\n};\n\n} // namespace usolaris\n\n")
        f.write(f"#endif // {guard}\n")

    print(f"出力: {out_path}")


if __name__ == "__main__":
    main()
