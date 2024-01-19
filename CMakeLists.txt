# CMakeList.txt : CMake project for yolo8-cls_libtorch, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("yolo8-cls_libtorch")

include_directories(${PROJECT_SOURCE_DIR})
# Add source to this project's executable.
add_executable (yolo8-cls_libtorch "yolo8-cls_libtorch.cpp" "yolo8-cls_libtorch.h" "fpng.cpp" "fpng.h")


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET yolo8-cls_libtorch PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
set(CMAKE_PREFIX_PATH ${} CACHE PATH "Path to libtorch root directory")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(yolo8-cls_libtorch "${TORCH_LIBRARIES}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET yolo8-cls_libtorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:yolo8-cls_libtorch>)
endif (MSVC)