cmake_minimum_required(VERSION 3.25)
project(main)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)  
    message(STATUS "====Release====")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)  
    # 设置C/C++编译器的优化选项（可选）  
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")  
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")  

    message(STATUS ${CMAKE_BUILD_TYPE})
    message(STATUS ${CMAKE_CONFIGURATION_TYPES})
    message(STATUS ${CMAKE_CXX_FLAGS_RELEASE})
endif()

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Freetype REQUIRED)



add_executable(main main.cpp cvxFont.cpp decode.cpp)

target_link_libraries(main ${OpenCV_LIBS}  Freetype::Freetype spdlog::spdlog)
