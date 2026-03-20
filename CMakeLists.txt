cmake_minimum_required(VERSION 3.10)
project(libfremen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(fremen_core STATIC
  src/fremen_model.cpp
)

target_include_directories(fremen_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)