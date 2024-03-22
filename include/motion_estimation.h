/*==============================================
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2018-11-27 09:55
#
# Filename:		pose_estimation.h
#
# Description: 
#
===============================================*/
#pragma once
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>
#include <pcl-1.8/pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl-1.8/pcl/segmentation/planar_region.h>
#include <pcl-1.8/pcl/features/organized_edge_detection.h>
#include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/visualization/pcl_painter2D.h>
#include <pcl-1.8/pcl/registration/transforms.h>
#include <pcl-1.8/pcl/filters/voxel_grid.h>
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
#include "ANN/ANN.h"
#include "types.h"
#include "pose_estimation.h"

namespace ulysses
{

	class MeasurementProjectiveRay
	{
	public:
		MeasurementProjectiveRay() {}

		MeasurementProjectiveRay(ProjectiveRay *c, ProjectiveRay *r)
		{ cur=c; ref=r; }

		ProjectiveRay *cur,*ref;

//		Eigen::Vector3d transform_pt_ref(g2o::SE3Quat Tcr)
//		{
//			return cur->plane->projected_point(Tcr*ref->xyz);
//		}
	};

	class Edge_ProjectiveRay: public g2o::BaseBinaryEdge<3, MeasurementProjectiveRay, g2o::VertexSE3Expmap, g2o::VertexSE3Expmap>
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		Edge_ProjectiveRay() 
		{
			debug=false;
		}
		
		virtual void computeError();

//		virtual void linearizeOplus();

		virtual bool read ( std::istream& in ) {}
		virtual bool write ( std::ostream& out ) const {}
		
		void setDebug(bool d) {debug=d;}


	private:

		bool debug;
	};

	class Edge_ProjectiveRay_2: public g2o::BaseBinaryEdge<3, MeasurementProjectiveRay, g2o::VertexSE3Expmap, g2o::VertexSE3Expmap>
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		Edge_ProjectiveRay_2() 
		{
			debug=false;
		}
		
		virtual void computeError();

//		virtual void linearizeOplus();

		virtual bool read ( std::istream& in ) {}
		virtual bool write ( std::ostream& out ) const {}
		
		void setDebug(bool d) {debug=d;}


	private:

		bool debug;
	};

	class MotionEstimation
	{
	public:

		MotionEstimation() 
		{
			max_iter_icp=10;
			max_iter_g2o=10;
			remove("alignScans_motion.txt");
//			remove("alpha.txt");
//			remove("gradient.txt");
			debug=false;
			usePln=true;
			usePt=true;
			thres_association=0.01;
		}

		~MotionEstimation() {}
		
		void setDebug(bool d) {debug=d;}
		void setVisual(bool v) {visual=v;}
		void usePlnPt(bool pln, bool pt) {usePln=pln; usePt=pt;}

		// alignScans
		// - align two scans using planes and edges;
		// - initial guess is scan->Tcr;
		double alignScans(Scan *scan, Scan *scan_ref, Transform &Tcr,
					boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

		void setMaxIterationICP(int i) {max_iter_icp=i;}
		void setMaxIterationLM(int i) {max_iter_g2o=i;}

		void setThresAssociation(double t) {thres_association=t;}

		Eigen::Matrix<double,10,1> computeResidual(Scan *scan, Transform Tcr);

		// buildCorrespondence
		// - build correspondences between scan_cur and scan_ref;
		// - and store them in the scan_cur.plane_matches;
		void buildCorrespondence(Scan *scan_cur, Scan *scan_ref, Transform Tcr, Transform Tcr_pre, int iterations);

        void test_loop();

		void test();

	private:
		
		bool debug;
		bool visual;
		std::ofstream fp;

		bool usePln, usePt;

		int max_iter_icp;
		int max_iter_g2o;
		double thres_association;

		// largest sqrt eigenvalue of Psi_pi;
		double lambda_pi_1;
		// corresponding eigenvector;
		Eigen::Matrix<double,6,1> q_pi_1;
		// index of the largest eigenvalue;
		int idx_1;
		// all the eigenvalues and eigenvectors of Psi_pi;
		Eigen::Matrix<double,6,1> Lambda_p;
		Eigen::Matrix<double,6,6> Q_p;
		Eigen::Matrix<double,6,1> Lambda_pi;
		Eigen::Matrix<double,6,6> Q_pi;
		Eigen::Matrix<double,6,1> Lambda;
		Eigen::Matrix<double,6,6> Q;

		double alignEdges(Scan *scan_cur, Scan *scan_ref, Transform &Tcr, 
						boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);
		double optimizeEdges(Scan *scan, Transform &Tcr);


		double alignShadows(Scan *scan_cur, Scan *scan_ref, Transform &Tcr, 
						boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);
		void fuseEdges(Scan *scan_cur, Transform &Tcr);
		double optimizeShadows(Scan *scan_cur, Transform &Tcr);


		// planeDist
		// - distance between two planes;
		// - return [pln_cur-Tcr(pln_ref)]^T*Cov_cur_inv*[pln_cur-Tcr(pln_ref)];

		void compute_Psi(Scan *scan, Transform Tcr);
		
	};
}

