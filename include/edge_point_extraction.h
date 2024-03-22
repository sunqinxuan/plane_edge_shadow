/*==============================================
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2019-01-08 10:36
#
# Filename:		edge_point_extraction.h
#
# Description: 
#
===============================================*/

#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>
#include <pcl-1.8/pcl/features/organized_edge_detection.h>
#include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/segmentation/rgb_plane_coefficient_comparator.h>
#include <pcl-1.8/pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl-1.8/pcl/segmentation/planar_region.h>
#include <pcl-1.8/pcl/sample_consensus/sac_model_line.h>
#include <pcl-1.8/pcl/sample_consensus/ransac.h>
#include <pcl-1.8/pcl/common/centroid.h>
#include <pcl-1.8/pcl/common/eigen.h>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/LU>
#include <limits>
#include "ANN/ANN.h"
#include "types.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include "plane_extraction.h"

namespace ulysses
{
	class EdgePointExtraction : public pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>, public pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>
	{
	public:
		using OrganizedEdgeBase<pcl::PointXYZRGBA, pcl::Label>::EDGELABEL_NAN_BOUNDARY;
		using OrganizedEdgeBase<pcl::PointXYZRGBA, pcl::Label>::EDGELABEL_OCCLUDING;
		using OrganizedEdgeBase<pcl::PointXYZRGBA, pcl::Label>::EDGELABEL_OCCLUDED;
		using OrganizedEdgeBase<pcl::PointXYZRGBA, pcl::Label>::EDGELABEL_HIGH_CURVATURE;
		using OrganizedEdgeBase<pcl::PointXYZRGBA, pcl::Label>::EDGELABEL_RGB_CANNY;	

		EdgePointExtraction()
		{
			// edge detection;
			setDepthDisconThreshold (0.05);
			setMaxSearchNeighbors (50);
			setEdgeType (EDGELABEL_OCCLUDING | EDGELABEL_OCCLUDED); // EDGELABEL_HIGH_CURVATURE | EDGELABEL_OCCLUDING | EDGELABEL_OCCLUDED
			remove("extract_EdgePoints.txt");
			sqRad_ANN=0.01;
			K_ANN=20;
			edge_meas=20;
			thres_ratio=0.5;
			thres_angle=0.5;
		}

		~EdgePointExtraction() {}

		void setDebug(bool d) {debug=d;}

		void setThres(double th, int max, double pxl, double dist, double rad, int k, double meas, double ratio, double angle) 
		{
			setDepthDisconThreshold (th);
			setMaxSearchNeighbors (max);
			thres_pxl_sq=pxl*pxl;
			thres_occluded_dist=dist;
			sqRad_ANN=rad*rad;
			K_ANN=k;
			edge_meas=meas;
			thres_ratio=ratio;
			thres_angle=angle;
		}

		void setThresLine(double r, double t, int th, double min, double max, double dir, double dist, double split, int min_points)
		{
			rho=r; 
			theta=t; 
			threshold=th; 
			minLineLength=min; 
			maxLineGap=max;
			thres_sim_dir=dir;
			thres_sim_dist=dist;
			thres_split=split;
			min_points_on_line=min_points;
		}

		void setThresPln(double min_inliers, double ang_thres, double dist_thres, double max_curv)
		{
			//double min_inliers=5000, ang_thres=0.017453*5.0, dist_thres=0.05, max_curv=0.01;
			// plane segmentation;
			setMinInliers(min_inliers);
			setAngularThreshold(ang_thres*0.017453); // 5deg
			setDistanceThreshold(dist_thres); // 5cm 
			setMaximumCurvature(max_curv);
			
		}

		void extractEdgePoints(Scan *scan);

		void fitLines(Scan *scan);
//		void fitLinesHough(Scan *scan);
		void fitLinesHough(std::vector<EdgePoint*>& edge_points, std::list<Line*>& lines_occluding, unsigned int EDGE_LABEL);
		void fitLinesLS(Line *line);

		bool fitSphere(EdgePoint *edge_point);

	private:
		
		std::ofstream fp;
		bool debug;
		double thres_pxl_sq;
		double thres_occluded_dist;
		double thres_ratio, thres_angle;

		// radius=0.1m;
//		static constexpr double sqRad_ANN=0.01;
		double sqRad_ANN;
		// K=20;
		int K_ANN;

		double edge_meas;

		pcl::PointCloud<pcl::Label>::Ptr labels_plane;
		pcl::PointCloud<pcl::Label>::Ptr labels_edge;
//		std::vector<pcl::PointIndices> inlier_indices_plane;

		int num_plane;
		void segmentPlanes(Scan *scan);

		double rho; 
		double theta; 
		int threshold; 
		double minLineLength; 
		double maxLineGap;

		double thres_sim_dir, thres_sim_dist;
		double thres_split;
		int min_points_on_line;
	};

//	class PlaneFeatureExtraction : public pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>
//	{
//	public:
//
//		PlaneFeatureExtraction() {}
//		~PlaneFeatureExtraction() {}
//
////		void loadScan(Scan *s) {scan=s;}
//
//		// initialize
//		// - some default arguments set for the plane segmentation and the edge detection;
////		void initialize();
//
//		void setDebug(bool d) {debug=d;}
//		// generatePlaneFeature
//		// - scan->observed_planes;
//		// - after pose estimation (scan->Tcg required);
//		// - all the plane coefficients and point coordinates are in local frame;
////		void generatePlaneFeature(Scan *scan);
//
//	private:
//		
//		bool debug;
//		// allocated before load in;
//		//Scan *scan;
//	
//		// segmentPlanes
//		// - plane segmentation;
//		// - save to scan
//		//   + planes
//		//     - planar_regions
//		//     - plane_indices
//		void segmentPlanes(Scan *scan);
//
//	};
}

