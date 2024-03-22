
SET(SRC_LIST_DATA data_reading/data_reading.cpp data_reading/capture.cpp)

SET(SRC_LIST_EXTR plane_edge_extraction/plane_feature_matching.cpp plane_edge_extraction/plane_extraction.cpp plane_edge_extraction/edge_point_extraction.cpp)

SET(SRC_LIST_FIT plane_fitting/plane_param_estimation.cpp)

SET(SRC_LIST_MOTION motion_estimation/pose_estimation.cpp motion_estimation/motion_estimation.cpp motion_estimation/display.cpp motion_estimation/test.cpp)

SET(SRC_LIST_BACKEND backend/traj_puzzle.cpp backend/loop_closing.cpp backend/pose_graph_optimization.cpp backend/bundle_adjustment.cpp)

#SET(SRC_LIST_FUSE plane_fusing/plane_fusing.cpp)


set( CMAKE_BUILD_TYPE Release )
#set(CMAKE_CXX_FLAGS "-std=c++11")
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
#set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package( G2O )
include_directories( ${G2O_INCLUDE_DIRS} ) 
find_package(Eigen3 REQUIRED) 
find_package(CSparse REQUIRED) 
find_package(Cholmod REQUIRED) 
include_directories(${CSPARSE_INCLUDE_DIR}) 
include_directories(${EIGEN3_INCLUDE_DIR}) 
include_directories(${CHOLMOD_INCLUDE_DIR})
#set( G2O_LIBS g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension)
SET(G2O_LIBS  GL GLU cholmod g2o_cli g2o_core g2o_incremental g2o_interactive g2o_interface  g2o_parser g2o_solver_cholmod g2o_solver_csparse g2o_solver_dense g2o_solver_pcg g2o_stuff g2o_types_icp g2o_types_sba g2o_types_sim3 g2o_types_slam2d g2o_types_slam3d ${QT_LIBRARIES} ${QT_QTOPENGL_LIBRARY} ${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${OpenCV_LIBS})

#Openni2
#FIND_PATH(OpenNI2_INCLUDE_DIRS OpenNI.h HINTS $ENV{OPENNI2_INCLUDE64} $ENV{OPENNI2_INCLUDE} PATH_SUFFIXES openni2)
#FIND_LIBRARY(OpenNI2_LIBRARY NAMES OpenNI2 HINTS $ENV{OPENNI2_LIB64} $ENV{OPENNI2_LIB} $ENV{OPENNI2_REDIST})
#
#IF (OpenNI2_INCLUDE_DIRS AND OpenNI2_LIBRARY)
#   SET(OpenNI2_FOUND TRUE)
#ENDIF (OpenNI2_INCLUDE_DIRS AND OpenNI2_LIBRARY)
#
#IF (OpenNI2_FOUND)
#   # show which OpenNI2 was found only if not quiet
#   SET(OpenNI2_LIBRARIES ${OpenNI2_LIBRARY})
#   MESSAGE(STATUS "Found OpenNI2: ${OpenNI2_LIBRARIES}")
#ELSE (OpenNI2_FOUND)
#   # fatal error if OpenNI2 is required but not found
#   IF (OpenNI2_FIND_REQUIRED)
#      MESSAGE(FATAL_ERROR "Could not find OpenNI2. Environment variables OPENNI2_INCLUDE (directory containing OpenNI.h) and OPENNI2_LIB (directory containing OpenNI2 library) could bet set.")
#   ENDIF (OpenNI2_FIND_REQUIRED)
#ENDIF (OpenNI2_FOUND)
#include_directories(${OPENNI2_INCLUDE_DIRS})


################################################################
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)
# Set cmake prefix path to enable cmake to find freenect2
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} $ENV{HOME}/freenect2/lib/cmake/freenect2)
#find_package(freenect2 REQUIRED)
#include_directories("/usr/include/libusb-1.0/" ${freenect2_INCLUDE_DIR})


INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/ann_1.1.2/include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

link_directories(${PROJECT_SOURCE_DIR}/ann_1.1.2/lib)

ADD_LIBRARY(data_reading SHARED ${SRC_LIST_DATA})
target_link_libraries(data_reading ${PCL_LIBRARIES} ${OpenCV_LIBS})# ${freenect2_LIBRARIES}) #${OpenNI2_LIBRARIES} 

ADD_LIBRARY(plane_edge_extration SHARED ${SRC_LIST_EXTR})
target_link_libraries(plane_edge_extration ${PCL_LIBRARIES} ${OpenCV_LIBS} ANN)

ADD_LIBRARY(plane_fitting SHARED ${SRC_LIST_FIT})
target_link_libraries(plane_fitting ${PCL_LIBRARIES} ${OpenCV_LIBS} ANN)

ADD_LIBRARY(motion_estimation SHARED ${SRC_LIST_MOTION})
target_link_libraries(motion_estimation ${PCL_LIBRARIES} ${OpenCV_LIBS} ANN ${G2O_LIBS})

ADD_LIBRARY(backend SHARED ${SRC_LIST_BACKEND})
target_link_libraries(backend ${PCL_LIBRARIES} ${OpenCV_LIBS} ANN ${G2O_LIBS})

#ADD_LIBRARY(plane_fusing SHARED ${SRC_LIST_FUSE})
#target_link_libraries(plane_fusing ${PCL_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS})
