cmake_minimum_required(VERSION 3.16)
project(uSolaris LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ライブラリ本体（ヘッダオンリー）
add_subdirectory(extern/TinyReguMath3D)

add_library(usolaris INTERFACE)
target_include_directories(usolaris INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(usolaris INTERFACE TinyReguMath3D)

target_compile_options(usolaris INTERFACE
    -Ofast             # -O3に加え、数学的厳密さを無視して高速化（fast-mathを含む）
    -march=native      # 実行マシンのCPU命令セット（AVX2, AVX-512等）をフル活用
    -flto              # リンク時最適化。関数呼び出しをまたいで最適化
    -fomit-frame-pointer # デバッグ用情報を捨ててレジスタを計算に回す
)
# サブディレクトリ
add_subdirectory(examples)
