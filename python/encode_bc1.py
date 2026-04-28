#!/usr/bin/env python3
"""
encode_bc1.py - PNG/HDR → BC1 圧縮 C++ ヘッダ変換ツール

使い方:
  python encode_bc1.py <input.(png|hdr|exr)> <output.hpp> <VarName> [--exposure 1.0]

入力:
  LDR (PNG/BMP/JPG): そのまま BC1 エンコード
  HDR (HDR/EXR):     Reinhard トーンマップ後 BC1 エンコード

制約:
  幅・高さは 4 の倍数であること
"""

import argparse
import struct
import numpy as np
from PIL import Image
import os


# ---------------------------------------------------------------------------
# RGB565 変換
# ---------------------------------------------------------------------------
def to_rgb565(r: float, g: float, b: float) -> int:
    r5 = int(np.clip(r, 0, 1) * 31.0 + 0.5) & 0x1F
    g6 = int(np.clip(g, 0, 1) * 63.0 + 0.5) & 0x3F
    b5 = int(np.clip(b, 0, 1) * 31.0 + 0.5) & 0x1F
    return (r5 << 11) | (g6 << 5) | b5


def from_rgb565(c: int):
    r = ((c >> 11) & 0x1F) / 31.0
    g = ((c >> 5) & 0x3F) / 63.0
    b = (c & 0x1F) / 31.0
    return np.array([r, g, b], dtype=np.float32)


# ---------------------------------------------------------------------------
# BC1 ブロックエンコード（4×4 = 16 ピクセル、float32 [16,3]）
# ---------------------------------------------------------------------------
def encode_bc1_block(pixels: np.ndarray) -> bytes:
    """pixels: shape (16, 3), float32 [0,1]"""
    lo = pixels.min(axis=0)
    hi = pixels.max(axis=0)

    c0_val = to_rgb565(hi[0], hi[1], hi[2])
    c1_val = to_rgb565(lo[0], lo[1], lo[2])

    # c0 > c1 を強制して 4色モードに（透過なし）
    if c0_val <= c1_val:
        # 完全に単色の場合はインデックス 0 で全埋め
        if c0_val == c1_val:
            return struct.pack('<HHI', c0_val, c1_val, 0x00000000)
        c0_val, c1_val = c1_val, c0_val
        hi, lo = lo, hi

    col0 = from_rgb565(c0_val)
    col1 = from_rgb565(c1_val)
    palette = np.array([
        col0,
        col1,
        col0 * (2.0 / 3.0) + col1 * (1.0 / 3.0),
        col0 * (1.0 / 3.0) + col1 * (2.0 / 3.0),
    ], dtype=np.float32)

    # 各ピクセルを最近傍パレットエントリに割り当て
    bits = np.uint32(0)
    for i, px in enumerate(pixels):
        dists = np.sum((px - palette) ** 2, axis=1)
        idx = int(np.argmin(dists))
        bits |= np.uint32(idx) << np.uint32(i * 2)

    return struct.pack('<HHI', c0_val, c1_val, int(bits))


# ---------------------------------------------------------------------------
# 画像読み込み（LDR / HDR 両対応）
# ---------------------------------------------------------------------------
def load_image_ldr(path: str) -> np.ndarray:
    """
    LDR: [0,255] uint8 → [0,1] float32
    HDR: Reinhard トーンマップ → [0,1] float32
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.hdr', '.exr'):
        # Pillow で HDR を float32 で開く
        img = Image.open(path)
        arr = np.array(img, dtype=np.float32)
        # チャンネル確認
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr[:, :, :3]

        # Reinhard 全チャンネルグローバルトーンマップ
        lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        white = np.percentile(lum[lum > 0], 99) if lum.max() > 0 else 1.0
        arr = arr / (white + 1e-6)           # 上位1%を基準に正規化
        arr = arr / (arr + 1.0)              # Reinhard
        arr = np.clip(arr, 0, 1)
    else:
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0

    return arr


# ---------------------------------------------------------------------------
# 画像全体をBC1エンコード
# ---------------------------------------------------------------------------
def encode_image_bc1(arr: np.ndarray) -> bytes:
    """arr: (H, W, 3) float32"""
    H, W = arr.shape[:2]
    assert W % 4 == 0 and H % 4 == 0, f"幅({W})・高さ({H})は4の倍数が必要"

    blocks = bytearray()
    for by in range(H // 4):
        for bx in range(W // 4):
            block = arr[by*4:(by+1)*4, bx*4:(bx+1)*4].reshape(16, 3)
            blocks += encode_bc1_block(block)
    return bytes(blocks)


# ---------------------------------------------------------------------------
# C++ ヘッダ出力
# ---------------------------------------------------------------------------
def write_cpp_header(raw: bytes, width: int, height: int, var_name: str, path: str,
                     roughness: float = 0.0, mip_level: int = 0,
                     source_path: str = ''):
    lines = [
        '#pragma once',
        '#include <cstdint>',
        '',
        f'// BC1 compressed texture: {width}x{height}',
        f'// Source: {os.path.basename(source_path)}',
        f'// Blocks: {len(raw) // 8} ({width // 4}x{height // 4})',
        f'namespace usolaris {{',
        f'struct {var_name} {{',
        f'  static constexpr int width  = {width};',
        f'  static constexpr int height = {height};',
        f'  static constexpr int num_blocks_x = {width // 4};',
        f'  static constexpr int num_blocks_y = {height // 4};',
        f'  static constexpr uint8_t data[] = {{',
    ]

    # 8 バイト（1ブロック）ずつ改行
    for i in range(0, len(raw), 8):
        chunk = raw[i:i+8]
        hex_str = ', '.join(f'0x{b:02X}' for b in chunk)
        lines.append(f'    {hex_str},')

    lines += [
        '  };',
        '};',
        '} // namespace usolaris',
        '',
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Written: {path}  ({len(raw)} bytes, {width}x{height})')


# ---------------------------------------------------------------------------
# ミップマップセット用ヘッダ出力（複数レベルをまとめる）
# ---------------------------------------------------------------------------
def write_mip_cpp_header(levels: list, var_name: str, path: str, source_path: str = ''):
    """
    levels: list of (raw_bytes, width, height, roughness)
    """
    num_levels = len(levels)

    # オフセット計算（ブロック数単位で）
    offsets = []
    acc = 0
    for raw, w, h, _ in levels:
        offsets.append(acc)
        acc += len(raw)

    total_bytes = sum(len(r) for r, _, _, _ in levels)

    lines = [
        '#pragma once',
        '#include <cstdint>',
        '',
        f'// BC1 compressed mip-mapped texture',
        f'// Source: {os.path.basename(source_path)}',
        f'// Total: {total_bytes} bytes ({num_levels} mip levels)',
        f'namespace usolaris {{',
        f'struct {var_name} {{',
        f'  static constexpr int num_levels = {num_levels};',
        f'  static constexpr int widths[{num_levels}]   = {{{", ".join(str(w) for _,w,_,_ in levels)}}};',
        f'  static constexpr int heights[{num_levels}]  = {{{", ".join(str(h) for _,_,h,_ in levels)}}};',
        f'  static constexpr float roughness[{num_levels}] = {{{", ".join(f"{r:.4f}f" for _,_,_,r in levels)}}};',
        f'  static constexpr int offsets[{num_levels}]  = {{{", ".join(str(o) for o in offsets)}}};',
        f'  // Data layout: concatenated BC1 blocks, level 0 first (highest res)',
        f'  static constexpr uint8_t data[] = {{',
    ]

    for level_idx, (raw, w, h, roughness) in enumerate(levels):
        lines.append(f'    // level {level_idx}: roughness={roughness:.2f}  {w}x{h}')
        for i in range(0, len(raw), 8):
            chunk = raw[i:i+8]
            hex_str = ', '.join(f'0x{b:02X}' for b in chunk)
            lines.append(f'    {hex_str},')

    lines += [
        '  };',
        '};',
        '} // namespace usolaris',
        '',
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Written mip header: {path}  ({total_bytes} bytes, {num_levels} levels)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='PNG/HDR → BC1 C++ ヘッダ変換')
    parser.add_argument('inputs', nargs='+',
                        help='入力画像(複数でミップセット。roughnessは自動計算)')
    parser.add_argument('-o', '--output', required=True,
                        help='出力 .hpp ファイルパス')
    parser.add_argument('-n', '--name', required=True,
                        help='C++ struct 名')
    args = parser.parse_args()

    if len(args.inputs) == 1:
        # 単一画像モード
        path = args.inputs[0]
        arr = load_image_ldr(path)
        H, W = arr.shape[:2]
        raw = encode_image_bc1(arr)
        write_cpp_header(raw, W, H, args.name, args.output, source_path=path)
    else:
        # ミップセットモード（入力順が level 0, 1, 2, ... と仮定）
        levels = []
        n = len(args.inputs)
        for i, path in enumerate(args.inputs):
            roughness = i / max(n - 1, 1)
            arr = load_image_ldr(path)
            H, W = arr.shape[:2]
            raw = encode_image_bc1(arr)
            print(f'  Encoding level {i}: {W}x{H} roughness={roughness:.4f}')
            levels.append((raw, W, H, roughness))
        write_mip_cpp_header(levels, args.name, args.output,
                             source_path=args.inputs[0])


if __name__ == '__main__':
    main()
