# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.2/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/alaricsanders/mygh/ED_DOQSI

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/alaricsanders/mygh/ED_DOQSI/build

# Include any dependencies generated for this target.
include CMakeFiles/pyro16.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pyro16.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pyro16.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pyro16.dir/flags.make

CMakeFiles/pyro16.dir/pyro16.cpp.o: CMakeFiles/pyro16.dir/flags.make
CMakeFiles/pyro16.dir/pyro16.cpp.o: /Users/alaricsanders/mygh/ED_DOQSI/pyro16.cpp
CMakeFiles/pyro16.dir/pyro16.cpp.o: CMakeFiles/pyro16.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/alaricsanders/mygh/ED_DOQSI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pyro16.dir/pyro16.cpp.o"
	/opt/homebrew/bin/g++-14 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pyro16.dir/pyro16.cpp.o -MF CMakeFiles/pyro16.dir/pyro16.cpp.o.d -o CMakeFiles/pyro16.dir/pyro16.cpp.o -c /Users/alaricsanders/mygh/ED_DOQSI/pyro16.cpp

CMakeFiles/pyro16.dir/pyro16.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/pyro16.dir/pyro16.cpp.i"
	/opt/homebrew/bin/g++-14 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/alaricsanders/mygh/ED_DOQSI/pyro16.cpp > CMakeFiles/pyro16.dir/pyro16.cpp.i

CMakeFiles/pyro16.dir/pyro16.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/pyro16.dir/pyro16.cpp.s"
	/opt/homebrew/bin/g++-14 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/alaricsanders/mygh/ED_DOQSI/pyro16.cpp -o CMakeFiles/pyro16.dir/pyro16.cpp.s

# Object files for target pyro16
pyro16_OBJECTS = \
"CMakeFiles/pyro16.dir/pyro16.cpp.o"

# External object files for target pyro16
pyro16_EXTERNAL_OBJECTS =

pyro16: CMakeFiles/pyro16.dir/pyro16.cpp.o
pyro16: CMakeFiles/pyro16.dir/build.make
pyro16: /usr/local/lib/libxdiag.a
pyro16: /opt/homebrew/Cellar/gcc/14.2.0/lib/gcc/current/libgomp.dylib
pyro16: /opt/homebrew/lib/libhdf5_cpp.310.0.4.dylib
pyro16: /opt/homebrew/lib/libhdf5.310.4.0.dylib
pyro16: CMakeFiles/pyro16.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/alaricsanders/mygh/ED_DOQSI/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pyro16"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pyro16.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pyro16.dir/build: pyro16
.PHONY : CMakeFiles/pyro16.dir/build

CMakeFiles/pyro16.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pyro16.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pyro16.dir/clean

CMakeFiles/pyro16.dir/depend:
	cd /Users/alaricsanders/mygh/ED_DOQSI/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/alaricsanders/mygh/ED_DOQSI /Users/alaricsanders/mygh/ED_DOQSI /Users/alaricsanders/mygh/ED_DOQSI/build /Users/alaricsanders/mygh/ED_DOQSI/build /Users/alaricsanders/mygh/ED_DOQSI/build/CMakeFiles/pyro16.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/pyro16.dir/depend

