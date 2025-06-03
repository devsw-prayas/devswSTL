cmake_minimum_required(VERSION 3.20)
project(devswSTL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Switch: Use bundled Instrumenta or link externally
option(INSTRUMENTA_STANDALONE "Build Instrumenta internally for devswSTL" ON)

if (INSTRUMENTA_STANDALONE)
    # Assume Instrumenta is present at external/Instrumenta
    add_subdirectory(external/Instrumenta)
    include_directories(external/Instrumenta/src/Public)
    set(INSTRUMENTA_LIB Instrumenta)
else()
    set(INSTRUMENTA_LIB Instrumenta)
endif()

include_directories(src/Public)

add_library(devswSTL SHARED
    src/Private/devswSTL.cpp
    src/Public/devswSTL.h
)

target_link_libraries(devswSTL PRIVATE ${INSTRUMENTA_LIB})

set_target_properties(devswSTL PROPERTIES FOLDER "Base")
