cmake_minimum_required(VERSION 3.20)
project(devswSTL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(devswSTL SHARED
    src/Private/devswSTL.cpp
    src/Public/devswSTL.h
    src/Public/Traits.h src/Public/Allocators.h
	src/Public/AVX.h
	src/Public/AlignedVector.h src/Private/AlignedVector.cpp
	src/Public/AvxIntrinsics.h
    src/Public/Memory.h
    src/Public/Stack.h
    src/Public/List.h
    src/Public/Deque.h
    src/Public/Set.h
    src/Public/Iterators.h
    src/Public/Map.h
    src/Public/Queue.h
    src/Public/BlockingQueue.h
)

target_include_directories(devswSTL PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Public
)

set_target_properties(devswSTL PROPERTIES FOLDER "Base")

