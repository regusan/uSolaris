find_package(SDL2 REQUIRED)

add_executable(example_sdl2 main.cpp)

target_link_libraries(example_sdl2 PRIVATE usolaris ${SDL2_LIBRARIES})
target_include_directories(example_sdl2 PRIVATE ${SDL2_INCLUDE_DIRS})
