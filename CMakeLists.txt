cmake_minimum_required(VERSION 3.19)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

project(
    pyro_ED
)

find_package(xdiag REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -Wpedantic -DNDEBUG")

# the tests
add_executable(tfim tests/tfim.cpp)
target_link_libraries(tfim PRIVATE xdiag::xdiag)

add_executable(tfim_no_sym tests/tfim_no_sym.cpp)
target_link_libraries(tfim_no_sym PRIVATE xdiag::xdiag)

add_executable(heischain tests/heischain.cpp)
target_link_libraries(heischain PRIVATE xdiag::xdiag)

add_executable(rationalmath tests/rationalmath.cpp)

# main files
add_executable(pyro16 pyro16.cpp)
target_link_libraries(pyro16 PRIVATE xdiag::xdiag)

add_executable(pyro32 pyro32.cpp)
target_link_libraries(pyro32 PRIVATE xdiag::xdiag)

add_executable(cluster12 cluster12.cpp)
target_link_libraries(cluster12 PRIVATE xdiag::xdiag)

add_executable(cluster12_full cluster12_full.cpp)
target_link_libraries(cluster12_full PRIVATE xdiag::xdiag)

add_executable(pyro16_full pyro16_full.cpp)
target_link_libraries(pyro16_full PRIVATE xdiag::xdiag)

add_executable(ring_ED ring_ED.cpp)
target_link_libraries(ring_ED PRIVATE xdiag::xdiag)
