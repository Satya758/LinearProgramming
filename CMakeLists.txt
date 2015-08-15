cmake_minimum_required(VERSION 2.6)
project(linearprogramming)

add_executable(linearprogramming main.cpp)

install(TARGETS linearprogramming RUNTIME DESTINATION bin)
