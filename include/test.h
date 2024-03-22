/*==============================================
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2019-01-24 09:43
#
# Filename:		test.h
#
# Description: 
#
===============================================*/
#pragma once
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/solver.h"
#include "g2o/types/icp/types_icp.h"
#include <eigen3/Eigen/Eigenvalues>



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>
#include <pcl-1.8/pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl-1.8/pcl/segmentation/planar_region.h>
#include <pcl-1.8/pcl/features/organized_edge_detection.h>
#include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/visualization/pcl_painter2D.h>
#include <pcl-1.8/pcl/registration/transforms.h>
#include <pcl-1.8/pcl/filters/voxel_grid.h>
#include "ANN/ANN.h"
#include "types.h"
#include "pose_estimation.h"

namespace ulysses
{
	class Test
	{
	public:

		Test() {}

		~Test() {}
		
		void test();

	private:
		
		
	};

}
