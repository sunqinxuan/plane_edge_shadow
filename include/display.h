/*==============================================
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2019-01-10 16:49
#
# Filename:		display.h
#
# Description: 
#
===============================================*/
//#include "pose_estimation.h"
//
#ifndef _DISPLAY_H_
#define _DISPLAY_H_

#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl-1.8/pcl/filters/voxel_grid.h>
#include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/visualization/pcl_painter2D.h>
#include <pcl-1.8/pcl/registration/transforms.h>
#include "types.h"


//using namespace ulysses;
typedef pcl::PointXYZRGBA PointT;

namespace ulysses
{

extern void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void);

// displayPlanes
// - display planes from one scan in different colors;
extern void displayPlanes(Scan *scan, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

extern void displayLines(Scan *scan, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

// displayPlanes_2scans
// - display planes from 2 scans;
// - different color for each scan;
// - scan_ref is transformed by Tcr;
extern void displayPlanes_2scans(Scan *scan, Scan *scan_ref, Transform Tcr, 
							    boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

// displayScans_2scans
// - display raw scan data from 2 scans;
// - different color for each scan;
// - scan_ref is transformed by Tcr;
extern void displayScans_2scans(Scan *scan, Scan *scan_ref, Transform Tcr, 
						 boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

// display_addScan
// - add one scan to vis after calling this function;
// - the scan is transformed to global frame by scan->Tcg;
extern void display_addScan(Scan *scan, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

// displayEdgePoints
// - display edge points in one scan;
extern void displayEdgePoints(Scan *scan, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

extern void displayMatchedEdgePoints(Scan *scan, Transform Tcr, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

extern void displayMatchedProjRays(Scan *scan, Transform Tcr, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);


extern void displayMatchedShadowPoints(Scan *scan, Transform Tcr, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

extern void displayTraj(Map *map, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

extern void display_addCameraPose(Scan *scan, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

extern void display_LoopClosure(LoopClosure loop_closure, boost::shared_ptr<pcl::visualization::PCLVisualizer> vis);

}

#endif
