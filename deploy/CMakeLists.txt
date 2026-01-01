cmake_minimum_required(VERSION 3.20)
project(deploy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/Dev/lib/opencv/build/x64/vc16/lib")
set(Torch_DIR "C:/Dev/lib/libtorch/share/cmake/Torch")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
enable_language(CUDA)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(deploy main.cpp)

target_include_directories(deploy PRIVATE ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(deploy PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})

if(MSVC)
    set(ENV{VSLANG} "1033")
    add_compile_options(/utf-8)

    set(DEPLOY_OUTPUT_DIR $<TARGET_FILE_DIR:deploy>)

    file(GLOB TORCH_DLLS "${Torch_INSTALL_PREFIX}/lib/*.dll")
    if(TORCH_DLLS)
        add_custom_command(TARGET deploy POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                ${DEPLOY_OUTPUT_DIR}
                COMMENT "Copying Torch DLLs to output directory...")
    endif()

    file(GLOB OPENCV_DLLS "C:/Dev/lib/opencv/build/x64/vc16/bin/*.dll")
    if(OPENCV_DLLS)
        add_custom_command(TARGET deploy POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OPENCV_DLLS}
                ${DEPLOY_OUTPUT_DIR}
                COMMENT "Copying OpenCV DLLs to output directory...")
    endif()
endif()