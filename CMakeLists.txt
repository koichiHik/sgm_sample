cmake_minimum_required(VERSION 3.5)
project(algo_practice)

# CMake Package Collection
FIND_PACKAGE(OpenCV CONFIG PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)

set(LIB_NAME "sgbm_practice")

set(LIB_SRC
  src/main.cpp
  src/sgbm.cpp
  )

include_directories(${OpenCV_INCLUDE_DIRS})

# Generate Exe Files
add_executable("${LIB_NAME}_test" ${LIB_SRC})
target_link_libraries("${LIB_NAME}_test" ${OpenCV_LIBS})
