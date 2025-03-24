cmake_minimum_required(VERSION 3.19)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

project(
    pyro_ED
)

find_package(xdiag REQUIRED)

# Find HDF5 (both C and C++ versions)
find_package(HDF5 REQUIRED COMPONENTS C CXX)


# Debugging output
message(STATUS "HDF5 Include Dirs: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 Libraries: ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES}")

# Ensure we use the correct libraries in case imported targets are missing
set(HDF5_C_LIBS "")
set(HDF5_CXX_LIBS "")

if(HDF5_C_LIBRARIES)
    set(HDF5_C_LIBS ${HDF5_C_LIBRARIES})
elseif(TARGET HDF5::HDF5_C)
    set(HDF5_C_LIBS HDF5::HDF5_C)
endif()

if(HDF5_CXX_LIBRARIES)
    set(HDF5_CXX_LIBS ${HDF5_CXX_LIBRARIES})
elseif(TARGET HDF5::HDF5_CXX)
    set(HDF5_CXX_LIBS HDF5::HDF5_CXX)
endif()


#if(NOT ${CMAKE\_BUILD\_TYPE} STREQUAL "Debug" OR NOT ${CMAKE\_BUILD\_TYPE} STREQUAL "Release")
#set(CMAKE\_BUILD\_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
#endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Using Release settings")
    set(CMAKE_CXX_FLAGS "-DNDEBUG -Og")
else()
    message(STATUS "Using Debug settings")
    set(CMAKE_CXX_FLAGS "-Wall -Wpedantic -g -O0")
endif()
    

if(CMAKE_ENABLE_TESTS)
# the tests
    add_executable(tfim tests/tfim.cpp)
    target_link_libraries(tfim PRIVATE xdiag::xdiag)

    add_executable(tfim_no_sym tests/tfim_no_sym.cpp)
    target_link_libraries(tfim_no_sym PRIVATE xdiag::xdiag)

    add_executable(heischain tests/heischain.cpp)
    target_link_libraries(heischain PRIVATE xdiag::xdiag)

    add_executable(rationalmath tests/rationalmath.cpp)
endif()

# main files
#add_executable(pyro16 pyro16.cpp)
#target_link_libraries(pyro16 PRIVATE xdiag::xdiag)

add_executable(pyro16_0field pyro16_0field.cpp)
target_include_directories(pyro16_0field PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(pyro16_0field PRIVATE xdiag::xdiag ${HDF5_C_LIBS})

#add_executable(pyro32 pyro32.cpp)
#target_link_libraries(pyro32 PRIVATE xdiag::xdiag)
#
#add_executable(cluster12 cluster12.cpp)
#target_link_libraries(cluster12 PRIVATE xdiag::xdiag)
#
#add_executable(cluster12_full cluster12_full.cpp)
#target_link_libraries(cluster12_full PRIVATE xdiag::xdiag)
#
#add_executable(pyro16_full pyro16_full.cpp)
#target_link_libraries(pyro16_full PRIVATE xdiag::xdiag)
#
#add_executable(ring_ED ring_ED.cpp)
#target_link_libraries(ring_ED PRIVATE xdiag::xdiag)
