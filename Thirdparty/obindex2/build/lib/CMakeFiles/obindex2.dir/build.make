# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build

# Include any dependencies generated for this target.
include lib/CMakeFiles/obindex2.dir/depend.make

# Include the progress variables for this target.
include lib/CMakeFiles/obindex2.dir/progress.make

# Include the compile flags for this target's objects.
include lib/CMakeFiles/obindex2.dir/flags.make

lib/CMakeFiles/obindex2.dir/src/binary_descriptor.cc.o: lib/CMakeFiles/obindex2.dir/flags.make
lib/CMakeFiles/obindex2.dir/src/binary_descriptor.cc.o: ../lib/src/binary_descriptor.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/CMakeFiles/obindex2.dir/src/binary_descriptor.cc.o"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/obindex2.dir/src/binary_descriptor.cc.o -c /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_descriptor.cc

lib/CMakeFiles/obindex2.dir/src/binary_descriptor.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/obindex2.dir/src/binary_descriptor.cc.i"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_descriptor.cc > CMakeFiles/obindex2.dir/src/binary_descriptor.cc.i

lib/CMakeFiles/obindex2.dir/src/binary_descriptor.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/obindex2.dir/src/binary_descriptor.cc.s"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_descriptor.cc -o CMakeFiles/obindex2.dir/src/binary_descriptor.cc.s

lib/CMakeFiles/obindex2.dir/src/binary_tree_node.cc.o: lib/CMakeFiles/obindex2.dir/flags.make
lib/CMakeFiles/obindex2.dir/src/binary_tree_node.cc.o: ../lib/src/binary_tree_node.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object lib/CMakeFiles/obindex2.dir/src/binary_tree_node.cc.o"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/obindex2.dir/src/binary_tree_node.cc.o -c /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_tree_node.cc

lib/CMakeFiles/obindex2.dir/src/binary_tree_node.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/obindex2.dir/src/binary_tree_node.cc.i"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_tree_node.cc > CMakeFiles/obindex2.dir/src/binary_tree_node.cc.i

lib/CMakeFiles/obindex2.dir/src/binary_tree_node.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/obindex2.dir/src/binary_tree_node.cc.s"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_tree_node.cc -o CMakeFiles/obindex2.dir/src/binary_tree_node.cc.s

lib/CMakeFiles/obindex2.dir/src/binary_tree.cc.o: lib/CMakeFiles/obindex2.dir/flags.make
lib/CMakeFiles/obindex2.dir/src/binary_tree.cc.o: ../lib/src/binary_tree.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object lib/CMakeFiles/obindex2.dir/src/binary_tree.cc.o"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/obindex2.dir/src/binary_tree.cc.o -c /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_tree.cc

lib/CMakeFiles/obindex2.dir/src/binary_tree.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/obindex2.dir/src/binary_tree.cc.i"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_tree.cc > CMakeFiles/obindex2.dir/src/binary_tree.cc.i

lib/CMakeFiles/obindex2.dir/src/binary_tree.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/obindex2.dir/src/binary_tree.cc.s"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_tree.cc -o CMakeFiles/obindex2.dir/src/binary_tree.cc.s

lib/CMakeFiles/obindex2.dir/src/binary_index.cc.o: lib/CMakeFiles/obindex2.dir/flags.make
lib/CMakeFiles/obindex2.dir/src/binary_index.cc.o: ../lib/src/binary_index.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object lib/CMakeFiles/obindex2.dir/src/binary_index.cc.o"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/obindex2.dir/src/binary_index.cc.o -c /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_index.cc

lib/CMakeFiles/obindex2.dir/src/binary_index.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/obindex2.dir/src/binary_index.cc.i"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_index.cc > CMakeFiles/obindex2.dir/src/binary_index.cc.i

lib/CMakeFiles/obindex2.dir/src/binary_index.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/obindex2.dir/src/binary_index.cc.s"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib/src/binary_index.cc -o CMakeFiles/obindex2.dir/src/binary_index.cc.s

# Object files for target obindex2
obindex2_OBJECTS = \
"CMakeFiles/obindex2.dir/src/binary_descriptor.cc.o" \
"CMakeFiles/obindex2.dir/src/binary_tree_node.cc.o" \
"CMakeFiles/obindex2.dir/src/binary_tree.cc.o" \
"CMakeFiles/obindex2.dir/src/binary_index.cc.o"

# External object files for target obindex2
obindex2_EXTERNAL_OBJECTS =

lib/libobindex2.so: lib/CMakeFiles/obindex2.dir/src/binary_descriptor.cc.o
lib/libobindex2.so: lib/CMakeFiles/obindex2.dir/src/binary_tree_node.cc.o
lib/libobindex2.so: lib/CMakeFiles/obindex2.dir/src/binary_tree.cc.o
lib/libobindex2.so: lib/CMakeFiles/obindex2.dir/src/binary_index.cc.o
lib/libobindex2.so: lib/CMakeFiles/obindex2.dir/build.make
lib/libobindex2.so: /usr/local/lib/libopencv_stitching.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_superres.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_videostab.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_aruco.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_bgsegm.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_bioinspired.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_ccalib.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_dpm.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_face.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_freetype.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_fuzzy.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_hdf.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_hfs.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_img_hash.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_line_descriptor.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_optflow.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_reg.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_rgbd.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_saliency.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_sfm.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_stereo.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_structured_light.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_surface_matching.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_tracking.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_xfeatures2d.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_ximgproc.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_xobjdetect.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_xphoto.so.3.4.20
lib/libobindex2.so: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
lib/libobindex2.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
lib/libobindex2.so: /usr/local/lib/libopencv_highgui.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_videoio.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_shape.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_viz.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_video.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_datasets.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_plot.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_text.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_dnn.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_ml.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_objdetect.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_calib3d.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_features2d.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_flann.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_photo.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_imgproc.so.3.4.20
lib/libobindex2.so: /usr/local/lib/libopencv_core.so.3.4.20
lib/libobindex2.so: lib/CMakeFiles/obindex2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared library libobindex2.so"
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/obindex2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/CMakeFiles/obindex2.dir/build: lib/libobindex2.so

.PHONY : lib/CMakeFiles/obindex2.dir/build

lib/CMakeFiles/obindex2.dir/clean:
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib && $(CMAKE_COMMAND) -P CMakeFiles/obindex2.dir/cmake_clean.cmake
.PHONY : lib/CMakeFiles/obindex2.dir/clean

lib/CMakeFiles/obindex2.dir/depend:
	cd /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2 /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/lib /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib /home/rushmian/catkin_ws/src/ov2slam/Thirdparty/obindex2/build/lib/CMakeFiles/obindex2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/CMakeFiles/obindex2.dir/depend
