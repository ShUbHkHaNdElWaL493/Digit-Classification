# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build"

# Include any dependencies generated for this target.
include include/CMakeFiles/LAYERS.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include include/CMakeFiles/LAYERS.dir/compiler_depend.make

# Include the progress variables for this target.
include include/CMakeFiles/LAYERS.dir/progress.make

# Include the compile flags for this target's objects.
include include/CMakeFiles/LAYERS.dir/flags.make

include/CMakeFiles/LAYERS.dir/layers.cpp.o: include/CMakeFiles/LAYERS.dir/flags.make
include/CMakeFiles/LAYERS.dir/layers.cpp.o: /home/shubh_khandelwal/Documents/Projects/Digit\ Classification/C++/include/layers.cpp
include/CMakeFiles/LAYERS.dir/layers.cpp.o: include/CMakeFiles/LAYERS.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object include/CMakeFiles/LAYERS.dir/layers.cpp.o"
	cd "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/include" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT include/CMakeFiles/LAYERS.dir/layers.cpp.o -MF CMakeFiles/LAYERS.dir/layers.cpp.o.d -o CMakeFiles/LAYERS.dir/layers.cpp.o -c "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/include/layers.cpp"

include/CMakeFiles/LAYERS.dir/layers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/LAYERS.dir/layers.cpp.i"
	cd "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/include" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/include/layers.cpp" > CMakeFiles/LAYERS.dir/layers.cpp.i

include/CMakeFiles/LAYERS.dir/layers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/LAYERS.dir/layers.cpp.s"
	cd "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/include" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/include/layers.cpp" -o CMakeFiles/LAYERS.dir/layers.cpp.s

# Object files for target LAYERS
LAYERS_OBJECTS = \
"CMakeFiles/LAYERS.dir/layers.cpp.o"

# External object files for target LAYERS
LAYERS_EXTERNAL_OBJECTS =

include/libLAYERS.a: include/CMakeFiles/LAYERS.dir/layers.cpp.o
include/libLAYERS.a: include/CMakeFiles/LAYERS.dir/build.make
include/libLAYERS.a: include/CMakeFiles/LAYERS.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libLAYERS.a"
	cd "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/include" && $(CMAKE_COMMAND) -P CMakeFiles/LAYERS.dir/cmake_clean_target.cmake
	cd "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/include" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LAYERS.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
include/CMakeFiles/LAYERS.dir/build: include/libLAYERS.a
.PHONY : include/CMakeFiles/LAYERS.dir/build

include/CMakeFiles/LAYERS.dir/clean:
	cd "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/include" && $(CMAKE_COMMAND) -P CMakeFiles/LAYERS.dir/cmake_clean.cmake
.PHONY : include/CMakeFiles/LAYERS.dir/clean

include/CMakeFiles/LAYERS.dir/depend:
	cd "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++" "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/include" "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build" "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/include" "/home/shubh_khandelwal/Documents/Projects/Digit Classification/C++/build/include/CMakeFiles/LAYERS.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : include/CMakeFiles/LAYERS.dir/depend

