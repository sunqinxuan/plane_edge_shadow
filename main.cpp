/*==============================================
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2019-01-07 09:52
#
# Filename:		main.cpp
#
# Description: 
#
===============================================*/

#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl-1.8/pcl/visualization/cloud_viewer.h>
#include <pcl-1.8/pcl/features/integral_image_normal.h>
#include <pcl-1.8/pcl/registration/transforms.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <eigen3/Eigen/StdVector>
#include "data_reading.h"
//#include "plane_segmentation.h"
#include "plane_feature_matching.h"
#include "pose_estimation.h"
#include "motion_estimation.h"
//#include "plane_map_update.h"
#include "plane_param_estimation.h"
#include "plane_extraction.h"
#include "edge_point_extraction.h"
#include "loop_closing.h"
#include "pose_graph_optimization.h"
#include <pcl-1.8/pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl-1.8/pcl/segmentation/planar_region.h>
#include <pcl-1.8/pcl/features/organized_edge_detection.h>
#include <pcl-1.8/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.8/pcl/visualization/pcl_painter2D.h>
#include <pcl-1.8/pcl/visualization/cloud_viewer.h>
#include <pcl-1.8/pcl/filters/voxel_grid.h>
#include <limits>
//#include "display.h"
#include "traj_puzzle.h"
#include "capture.h"
#include "bundle_adjustment.h"
//#include "plane_fusing.h"


#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"

#include "test.h"

using namespace Eigen;
using namespace std;
using namespace g2o;
using namespace ulysses;


// sampling distributions
  class Sample
  {

    static default_random_engine gen_real;
    static default_random_engine gen_int;
  public:
    static int uniform(int from, int to);

    static double uniform();

    static double gaussian(double sigma);
  };


  default_random_engine Sample::gen_real;
  default_random_engine Sample::gen_int;

  int Sample::uniform(int from, int to)
  {
    uniform_int_distribution<int> unif(from, to);
    int sam = unif(gen_int);
    return  sam;
  }

  double Sample::uniform()
  {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double sam = unif(gen_real);
    return  sam;
  }

  double Sample::gaussian(double sigma)
  {
    std::normal_distribution<double> gauss(0.0, sigma);
    double sam = gauss(gen_real);
    return  sam;
  }

class motion
{
public:
    motion() {}
    ~motion() {}

    void test();
};


void motion::test()
{
  double euc_noise = 0.01;       // noise in position, m
  //  double outlier_ratio = 0.1;


  SparseOptimizer optimizer;
  optimizer.setVerbose(false);

  // variable-size block solver
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<BlockSolverX>(g2o::make_unique<LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));

  optimizer.setAlgorithm(solver);

  vector<Vector3d> true_points;
  for (size_t i=0;i<1000; ++i)
  {
    true_points.push_back(Vector3d((Sample::uniform()-0.5)*3,
                                   Sample::uniform()-0.5,
                                   Sample::uniform()+10));
  }


  // set up two poses
  int vertex_id = 0;
  for (size_t i=0; i<2; ++i)
  {
    // set up rotation and translation for this node
    Vector3d t(0,0,i);
    Quaterniond q;
    q.setIdentity();

    Eigen::Isometry3d cam; // camera pose
    cam = q;
    cam.translation() = t;

    // set up node
    VertexSE3 *vc = new VertexSE3();
    vc->setEstimate(cam);

    vc->setId(vertex_id);      // vertex id

    cerr << t.transpose() << " | " << q.coeffs().transpose() << endl;

    // set first cam pose fixed
    if (i==0)
      vc->setFixed(true);

    // add to optimizer
    optimizer.addVertex(vc);

    vertex_id++;                
  }

  // set up point matches
  for (size_t i=0; i<true_points.size(); ++i)
  {
    // get two poses
    VertexSE3* vp0 = 
      dynamic_cast<VertexSE3*>(optimizer.vertices().find(0)->second);
    VertexSE3* vp1 = 
      dynamic_cast<VertexSE3*>(optimizer.vertices().find(1)->second);

    // calculate the relative 3D position of the point
    Vector3d pt0,pt1;
    pt0 = vp0->estimate().inverse() * true_points[i];
    pt1 = vp1->estimate().inverse() * true_points[i];

    // add in noise
    pt0 += Vector3d(Sample::gaussian(euc_noise ),
                    Sample::gaussian(euc_noise ),
                    Sample::gaussian(euc_noise ));

    pt1 += Vector3d(Sample::gaussian(euc_noise ),
                    Sample::gaussian(euc_noise ),
                    Sample::gaussian(euc_noise ));

    // form edge, with normals in varioius positions
    Vector3d nm0, nm1;
    nm0 << 0, i, 1;
    nm1 << 0, i, 1;
    nm0.normalize();
    nm1.normalize();

    Edge_V_V_GICP * e           // new edge with correct cohort for caching
        = new Edge_V_V_GICP(); 

    e->setVertex(0, vp0);      // first viewpoint

    e->setVertex(1, vp1);      // second viewpoint

    EdgeGICP meas;
    meas.pos0 = pt0;
    meas.pos1 = pt1;
    meas.normal0 = nm0;
    meas.normal1 = nm1;

    e->setMeasurement(meas);
    //        e->inverseMeasurement().pos() = -kp;
    
    meas = e->measurement();
    // use this for point-plane
    e->information() = meas.prec0(0.01);

    // use this for point-point 
    //    e->information().setIdentity();

    //    e->setRobustKernel(true);
    //e->setHuberWidth(0.01);

    optimizer.addEdge(e);
  }

  // move second cam off of its true position
  VertexSE3* vc = 
    dynamic_cast<VertexSE3*>(optimizer.vertices().find(1)->second);
  Eigen::Isometry3d cam = vc->estimate();
  cam.translation() = Vector3d(0,0,0.2);
  vc->setEstimate(cam);
  cout<<"before initialization "<<endl;

  optimizer.initializeOptimization();
  optimizer.computeActiveErrors();
  cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << endl;

  optimizer.setVerbose(true);

  optimizer.optimize(5);

  cout << endl << "Second vertex should be near 0,0,1" << endl;
  cout <<  dynamic_cast<VertexSE3*>(optimizer.vertices().find(0)->second)
    ->estimate().translation().transpose() << endl;
  cout <<  dynamic_cast<VertexSE3*>(optimizer.vertices().find(1)->second)
    ->estimate().translation().transpose() << endl;
}


//
// set up simulated system with noise, optimize it
//

//int main()
//{
//    MotionEstimation me;
//    for(size_t i=0;i<10;i++)
//    {
//        cout<<endl<<i<<" ##########################"<<endl;
//        me.alignScans();
//    }
//    return 0;
//}

int main (int argc, char *argv[])
{
	vtkObject::GlobalWarningDisplayOff();
	// some default settings;
	std::string sequence_name="";
//	std::string sequence_name="/home/sun/dataset/rgbd_dataset_freiburg3_structure_notexture_far/";
	double time_start=1341836538.6;
	double time_interval=0.1;
	int min_inliers=1000;
	bool usePln=true, usePt=false;
	int pln_fitting_method=0;
	double m_fp=2.85e-3, sigma_u=0.5, sigma_v=0.5;
	int vis_every_n_frames=10;
	int total_frames=1000;
	double alpha=1, beta=1;
	int max_icp=10, max_lm=10;
	int occluding=2, occluded=4, curvature=0, canny=0;
	bool useWeight=true;
	double delta_time=2, delta_angle=6.0, delta_dist=0.1;
	std::string traj_path="/home/sun/traj/";
	enum Mode {DEBUG, VIS_SCAN, CLOSE_LOOP, CLOSE_LOOP_FILE, TRAJ_PUZZLE, RELEASE, ONLINE, COLLECT, VIEW} mode;
	int key_frame=1;
	int kinect2=0;
	double thres_association=0.01;
	bool visual;
	double pln_angle=0.2, pln_dist=0.05, pln_color=0.5;
	double edge_th=0.05, edge_pxl=10, edge_dist=0.1, edge_rad=0.1, edge_meas=20;
	int edge_max=50, edge_k=20;
	double edge_ratio, edge_angle;
	int pln_cell1=10,pln_cell2=20,pln_cell3=1;
	bool save_forDisplay=false;
	double fitline_rho, fitline_theta, fitline_minLineLength, fitline_maxLineGap;
	int fitline_threshold;
	double fitline_sim_dir=10.0, fitline_sim_dist=0.1;
	double fitline_split=0.05;
	int fitline_min_points_on_line=50;
	double plane_ang_thres=5.0, plane_dist_thres=0.05, plane_max_curv=0.01;

	for(int i=1;i<argc;i++)
	{
		if(strcmp(argv[i],"-mode")==0)
		{
			if(strcmp(argv[i+1],"debug")==0)
				mode=DEBUG;
			if(strcmp(argv[i+1],"vis_scan")==0)
				mode=VIS_SCAN;
			if(strcmp(argv[i+1],"close_loop")==0)
				mode=CLOSE_LOOP;
			if(strcmp(argv[i+1],"traj_puzzle")==0)
				mode=TRAJ_PUZZLE;
			if(strcmp(argv[i+1],"close_loop_file")==0)
				mode=CLOSE_LOOP_FILE;
			if(strcmp(argv[i+1],"release")==0)
				mode=RELEASE;
			if(strcmp(argv[i+1],"online")==0)
				mode=ONLINE;
			if(strcmp(argv[i+1],"collect")==0)
				mode=COLLECT;
			if(strcmp(argv[i+1],"view")==0)
				mode=VIEW;
		}
		if(strcmp(argv[i],"-ds")==0) {sequence_name=argv[i+1];}
		if(strcmp(argv[i],"-st")==0) {time_start=atof(argv[i+1]);}
		if(strcmp(argv[i],"-ti")==0) {time_interval=atof(argv[i+1]);}
		if(strcmp(argv[i],"-mi")==0) {min_inliers=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-pln_angle")==0) {pln_angle=atof(argv[i+1]);}
		if(strcmp(argv[i],"-pln_dist")==0) {pln_dist=atof(argv[i+1]);}
		if(strcmp(argv[i],"-pln_color")==0) {pln_color=atof(argv[i+1]);}
		if(strcmp(argv[i],"-pln_cell1")==0) {pln_cell1=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-pln_cell2")==0) {pln_cell2=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-pln_cell3")==0) {pln_cell3=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-edge_th")==0) {edge_th=atof(argv[i+1]);}
		if(strcmp(argv[i],"-edge_pxl")==0) {edge_pxl=atof(argv[i+1]);}
		if(strcmp(argv[i],"-edge_dist")==0) {edge_dist=atof(argv[i+1]);}
		if(strcmp(argv[i],"-edge_max")==0) {edge_max=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-edge_rad")==0) {edge_rad=atof(argv[i+1]);}
		if(strcmp(argv[i],"-edge_k")==0) {edge_k=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-edge_meas")==0) {edge_meas=atof(argv[i+1]);}
		if(strcmp(argv[i],"-edge_ratio")==0) {edge_ratio=atof(argv[i+1]);}
		if(strcmp(argv[i],"-edge_angle")==0) {edge_angle=atof(argv[i+1]);}
		if(strcmp(argv[i],"-fitline_rho")==0) {fitline_rho=atof(argv[i+1]);}
		if(strcmp(argv[i],"-fitline_theta")==0) {fitline_theta=atof(argv[i+1]);fitline_theta=fitline_theta*M_PI/180.0;}
		if(strcmp(argv[i],"-fitline_minLineLength")==0) {fitline_minLineLength=atof(argv[i+1]);}
		if(strcmp(argv[i],"-fitline_maxLineGap")==0) {fitline_maxLineGap=atof(argv[i+1]);}
		if(strcmp(argv[i],"-fitline_threshold")==0) {fitline_threshold=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-fitline_sim_dir")==0) {fitline_sim_dir=atof(argv[i+1]);}
		if(strcmp(argv[i],"-fitline_sim_dist")==0) {fitline_sim_dist=atof(argv[i+1]);}
		if(strcmp(argv[i],"-fitline_split")==0) {fitline_split=atof(argv[i+1]);}
		if(strcmp(argv[i],"-fitline_min_points_on_line")==0) {fitline_min_points_on_line=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-icp")==0) {max_icp=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-lm")==0) {max_lm=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-plane_ang_thres")==0) {plane_ang_thres=atof(argv[i+1]);}
		if(strcmp(argv[i],"-plane_dist_thres")==0) {plane_dist_thres=atof(argv[i+1]);}
		if(strcmp(argv[i],"-plane_max_curv")==0) {plane_max_curv=atof(argv[i+1]);}
		if(strcmp(argv[i],"-pln")==0)
		{
			if(strcmp(argv[i+1],"1")==0)
				usePln=true;
			if(strcmp(argv[i+1],"0")==0)
				usePln=false;
		}
		if(strcmp(argv[i],"-pt")==0)
		{
			if(strcmp(argv[i+1],"1")==0)
				usePt=true;
			if(strcmp(argv[i+1],"0")==0)
				usePt=false;
		}
		if(strcmp(argv[i],"-plnfit")==0) {pln_fitting_method=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-vis")==0) {vis_every_n_frames=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-frames")==0) {total_frames=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-alpha")==0) {alpha=atof(argv[i+1]);}
		if(strcmp(argv[i],"-beta")==0) {beta=atof(argv[i+1]);}
		if(strcmp(argv[i],"-associate")==0) {thres_association=atof(argv[i+1]);}
		if(strcmp(argv[i],"-occluding")==0)
		{
			if(strcmp(argv[i+1],"1")==0)
				occluding=2;
			if(strcmp(argv[i+1],"0")==0)
				occluding=0;
		}
		if(strcmp(argv[i],"-occluded")==0)
		{
			if(strcmp(argv[i+1],"1")==0)
				occluded=4;
			if(strcmp(argv[i+1],"0")==0)
				occluded=0;
		}
		if(strcmp(argv[i],"-curvature")==0)
		{
			if(strcmp(argv[i+1],"1")==0)
				curvature=8;
			if(strcmp(argv[i+1],"0")==0)
				curvature=0;
		}
		if(strcmp(argv[i],"-canny")==0)
		{
			if(strcmp(argv[i+1],"1")==0)
				canny=16;
			if(strcmp(argv[i+1],"0")==0)
				canny=0;
		}
		if(strcmp(argv[i],"-useWeight")==0)
		{
			if(strcmp(argv[i+1],"1")==0)
				useWeight=true;
			if(strcmp(argv[i+1],"0")==0)
				useWeight=false;
		}
		if(strcmp(argv[i],"-traj_path")==0) {traj_path=argv[i+1];}
		if(strcmp(argv[i],"-delta_time")==0) {delta_time=atof(argv[i+1]);}
		if(strcmp(argv[i],"-delta_angle")==0) {delta_angle=atof(argv[i+1]);}
		if(strcmp(argv[i],"-delta_dist")==0) {delta_dist=atof(argv[i+1]);}
		if(strcmp(argv[i],"-key_frame")==0) {key_frame=atoi(argv[i+1]);}
		if(strcmp(argv[i],"-kinect2")==0)
		{
			if(strcmp(argv[i+1],"2")==0)
				kinect2=2;
			if(strcmp(argv[i+1],"1")==0)
				kinect2=1;
			if(strcmp(argv[i+1],"0")==0)
				kinect2=0;
		}
		if(strcmp(argv[i],"-visual")==0)
		{
			if(strcmp(argv[i+1],"1")==0)
				visual=true;
			if(strcmp(argv[i+1],"0")==0)
				visual=false;
		}
		if(strcmp(argv[i],"-display")==0)
		{
			if(strcmp(argv[i+1],"1")==0)
				save_forDisplay=true;
			if(strcmp(argv[i+1],"0")==0)
				save_forDisplay=false;
		}
	}

	if(mode==TRAJ_PUZZLE)
	{
		std::cout<<traj_path<<std::endl;
		TrajPuzzle traj_puzzle(traj_path);
		traj_puzzle.readTrajFiles();
		return 0;
	}

	ulysses::Scan *scan_ref;
	ulysses::Scan *scan_cur;
	ulysses::IntrinsicParam cam;
	cam.m_fp=m_fp;
	cam.sigma_u=sigma_u;
	cam.sigma_v=sigma_v;
	if(kinect2==2)
	{
		cam.fx=540.686;
		cam.fy=540.686;
		cam.cx=479.75;
		cam.cy=269.75;
		cam.width=960;
		cam.height=540;
	}
	if(kinect2==1)
	{
		cam.fx=367.933;//params.fx;
		cam.fy=367.933;//params.fy;
		cam.cx=254.169;//params.cx;
		cam.cy=204.267;//params.cy;
		cam.width=512;
		cam.height=424;
		cam.factor=1000.0;
	}
	if(kinect2==0)
	{
		cam.fx=567.6;
		cam.fy=570.2;
		cam.cx=324.7;
		cam.cy=250.1;
		cam.width=640;
		cam.height=480;
	
	}
	std::vector<ulysses::PlanePair> matched_planes;
    ulysses::Transform Tcr_align_planes, Tgc, Tcr, Tcr_gt;
	Eigen::Matrix<double,6,1> xi_cr, xi_cr_gt, delta_xi;
	bool debug=(mode==DEBUG);

	DataReading *data_reading=new ulysses::DataReading();
	data_reading->setPath(sequence_name);
	data_reading->setDebug(true);
	data_reading->setSampleInterval(time_interval);
	
	ulysses::PlaneParamEstimation plane_fitting;
	plane_fitting.setDebug(debug);
	plane_fitting.setPlnFittingMethod(pln_fitting_method);

	ulysses::PlaneFeatureMatching pfm;
	pfm.setDebug(debug);

	ulysses::PoseEstimation pe;
	pe.setDebug(debug);
	pe.setVisual(visual);
	pe.usePlnPt(usePln,usePt);
	pe.useEdgeWeight(useWeight);
	pe.setAlpha(alpha);
	pe.setBeta(beta);
	pe.setMaxIterationICP(max_icp);
	pe.setMaxIterationLM(max_lm);
	pe.setThresAssociation(thres_association);
	ulysses::PoseEstimation::ConstraintCase case_cur, case_pre;

	ulysses::MotionEstimation me;
	me.setDebug(debug);
	me.setVisual(visual);
	me.usePlnPt(usePln,usePt);
	me.setMaxIterationICP(max_icp);
	me.setMaxIterationLM(max_lm);
	me.setThresAssociation(thres_association);

	ulysses::PlaneExtraction extract;
	extract.setDebug(debug);
	extract.setMinPlaneSize(min_inliers);
	extract.setThres(pln_angle,pln_dist,pln_color);
	extract.allocBottomGrid(pln_cell1,pln_cell2,pln_cell3);

	ulysses::EdgePointExtraction edge_ext;
	edge_ext.setDebug(debug);
	edge_ext.setThres(edge_th,edge_max,edge_pxl,edge_dist,edge_rad,edge_k,edge_meas,edge_ratio,edge_angle);
	edge_ext.setThresLine(fitline_rho, fitline_theta, fitline_threshold, fitline_minLineLength, fitline_maxLineGap, fitline_sim_dir, fitline_sim_dist,fitline_split,fitline_min_points_on_line);
	edge_ext.setThresPln(min_inliers,plane_ang_thres,plane_dist_thres,plane_max_curv);
	int edge_type=occluding|occluded|canny;
	edge_ext.setEdgeType(edge_type);
//	std::cout<<"edge type - "<<edge_type<<std::endl;

	ulysses::Map *map=new ulysses::Map;

	ulysses::BundleAdjustment ba;
	ba.setDebug(debug);
	ba.setVisual(visual);


//	ulysses::PlaneFusing pf;
//	pf.setDebug(debug);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> vis (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	vis->setBackgroundColor (0, 0, 0);
//	vis->setBackgroundColor (0.78, 0.78, 0.78);
//	vis->addCoordinateSystem (0.5);
	vis->initCameraParameters ();
	vis->registerKeyboardCallback (keyboardEventOccurred, (void*)vis.get ());

//	boost::shared_ptr<pcl::visualization::PCLPainter2D> fig (new pcl::visualization::PCLPainter2D ("Figure"));
//	fig->setWindowSize (600, 400);

	if(mode==VIEW)
	{
//		pcl::visualization::CloudViewer view("point_cloud_map");
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud_map(new pcl::PointCloud<pcl::PointXYZRGBA>);
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
//		pcl::VoxelGrid<pcl::PointXYZRGBA> filter;
//		filter.setLeafSize(0.005,0.005,0.005);
		std::ifstream fp_traj;
		fp_traj.open("traj.txt",std::ios::in);
		double tx,ty,tz,qx,qy,qz,qw,time;
		char ch;
		int count=0;
        ulysses::Transform T0;
		if(time_start!=0)
		{
			while(true)
			{
				fp_traj>>time>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
				ulysses::Transform Tgc(Eigen::Quaterniond(qw,qx,qy,qz), Eigen::Vector3d(tx,ty,tz));
				if(fabs(time-time_start)<0.01)
				{
					T0=Tgc;
					break;
				}
			}
		}
		fp_traj.close();

		vis->removeAllPointClouds();
		vis->removeAllShapes();
//		vis->spin();
		fp_traj.open("traj.txt",std::ios::in);
		while(true)
		{
			fp_traj>>time>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
			if(time<time_start)
			{
				if(count%10!=0) 
				{
					count++;
					continue;
				}
			}
			else
			{
				if(count%key_frame!=0) 
				{
					count++;
					continue;
				}
			}

			scan_cur=new Scan;
			data_reading->loadScan_once(scan_cur,cam,time);
			ulysses::Transform Tgc(Eigen::Quaterniond(qw,qx,qy,qz), Eigen::Vector3d(tx,ty,tz));
//			scan_cur->Tcg=Tgc.inv();
			scan_cur->id=count;
//			if(count==0)
//				T0=Tgc;
			if(fabs(time-time_start)<0.01)
			{
				vis->spin();
			}
			scan_cur->Tcg=Tgc.inv()*T0;
			Tgc=scan_cur->Tcg.inv();
			std::cout<<std::fixed<<time<<" "<<Tgc.t.transpose()<<" "<<Tgc.Quaternion().transpose()<<std::endl;
//			pcl::transformPointCloud(*scan_cur->point_cloud,*point_cloud,Tgc.getMatrix4f());
//			filter.setInputCloud(point_cloud);
//			filter.filter(*point_cloud);
//			*point_cloud_map=*point_cloud_map+*point_cloud;
//			for(size_t i=0;i<scan_cur->point_cloud->size();i++)
//			{
//				if(scan_cur->point_cloud->at(i).y<-0.3 || scan_cur->point_cloud->at(i).y>0.3)
//				{
//					scan_cur->point_cloud->at(i).x=0;
//					scan_cur->point_cloud->at(i).y=0;
//					scan_cur->point_cloud->at(i).z=0;
//				}
//			}
			display_addScan(scan_cur,vis);
			display_addCameraPose(scan_cur,vis);
			vis->spinOnce(200);
//			vis->removeAllPointClouds();
//			view.showCloud(point_cloud_map);
			std::cout<<std::fixed<<time<<std::endl;
			delete scan_cur;
			if(count==0)
			{
				vis->spin();
//				std::cout<<"Press enter to continue...\n"<<std::endl;
//				ch=std::cin.get();
			}
			count++;
			if(fp_traj.eof())
			{
				vis->spin();
//				std::cout<<"Press enter to continue...\n"<<std::endl;
//				ch=std::cin.get();
				break;
			}
		}
		fp_traj.close();
		return 0;
	}

	ofstream fp;
	fp.open("traj.txt",std::ios::out);
	ofstream fp_edge;
	fp_edge.open("traj_edge.txt",std::ios::out);
	ofstream fp_shadow;
	fp_shadow.open("traj_shadow.txt",std::ios::out);
	ofstream fp_forDisplay;
	

	int first=0;
	int filenum = first;
	timeval start, end;
	double timeused;

	ofstream fp_error;
	fp_error.open("error.txt",std::ios::out);
	ofstream fp_residual;
	fp_residual.open("residual.txt",std::ios::out);
	ofstream fp_notes;
	fp_notes.open("notes.txt",std::ios::out);

	boost::shared_ptr<pcl::visualization::PCLPainter2D> fig_error (new pcl::visualization::PCLPainter2D ("Error"));
	fig_error->setWindowSize (600, 400);
	fig_error->setPenColor(0,0,0,200);
	fig_error->addLine(0,200,600,200);

	data_reading->Initialize(time_start);
	double err_pos_pre=0,err_ang_pre=0;
	double err_pos_pre_edge=0,err_ang_pre_edge=0;
	double err_pos_pre_shadow=0,err_ang_pre_shadow=0;
	double err_pos=0, err_ang=0;
	double err_pos_edge=0, err_ang_edge=0;
	double err_pos_shadow=0, err_ang_shadow=0;
	double res_occluding_Tcr=0, res_occluding_Tcr_gt=0;
	double res_occluding_Tcr_pre=0, res_occluding_Tcr_gt_pre=0;
	double res_occluded_Tcr=0, res_occluded_Tcr_gt=0;
	double res_occluded_Tcr_pre=0, res_occluded_Tcr_gt_pre=0;
	while(!data_reading->isEOF() && filenum<total_frames)
	{
		std::cout<<std::endl<<"***************** frame "<<filenum<<" ******************"<<std::endl;
		fp_notes<<std::endl<<"***************** frame\t"<<filenum<<"\t"<<std::fixed<<data_reading->getTime()<<" ******************"<<std::endl;

		// load point cloud from the image sequence;
		gettimeofday(&start,NULL);
		scan_cur=new Scan;
		scan_cur->cam=cam;
//		data_reading->read(scan_cur,cam,rgb,dep);
		if(!data_reading->loadScan(scan_cur,cam)) break;
		gettimeofday(&end,NULL);
		timeused=(1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec)/1000;
		std::cout<<"time reading data from freiburg:"<<timeused<<std::endl;
		scan_cur->id=filenum;

//		for(size_t i=0;i<scan_cur->point_cloud->height;i++)
//		{
//			for(size_t j=0;j<scan_cur->point_cloud->width;j++)
//			{
//				fp<<scan_cur->point_cloud->points[i*640+j].x<<"\t"
//				  <<scan_cur->point_cloud->points[i*640+j].y<<"\t"
//				  <<scan_cur->point_cloud->points[i*640+j].z<<"\t"<<std::endl;
//			}
//		}
//		return 0;

//		if (!vis->updatePointCloud (scan_cur->point_cloud, "scan"))
//			vis->addPointCloud (scan_cur->point_cloud, "scan");
//		vis->spin();

		// load points to the grid structure;
		gettimeofday(&start,NULL);
		if(!extract.loadPoints(scan_cur))
		{
			cout<<"no points for PlaneExtraction structure"<<endl;
			return 0;
		}
		gettimeofday(&end,NULL);
		timeused=(1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec)/1000;
		cout<<"time loading points:"<<timeused<<"ms"<<endl;

		// extract planes;
		gettimeofday(&start,NULL);
		extract.extractPlanes(scan_cur);
		gettimeofday(&end,NULL);
		timeused=(1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec)/1000;
		cout<<"extracted planes: "<<scan_cur->observed_planes.size()<<std::endl;
		cout<<"time extracting planes:"<<timeused<<"ms"<<endl;
		fp_notes<<"extracted "<<scan_cur->observed_planes.size()<<" planes using "<<timeused<<"ms"<<endl;
//		fp_error<<std::endl<<std::fixed<<data_reading->getTime()<<"\t";
//		fp_error<<timeused<<"\t";

//		std::cout<<"after extraction"<<std::endl;
//		for(size_t i=0;i<scan_cur->observed_planes.size();i++)
//		{
//			std::cout<<"plane - "<<i<<std::endl;
//			std::cout<<scan_cur->observed_planes[i]->normal.transpose()<<", "
//					 <<scan_cur->observed_planes[i]->d<<std::endl;
//			std::cout<<scan_cur->observed_planes[i]->points.size()<<std::endl;
//		}

		// estimate plane parameters;
		gettimeofday(&start,NULL);
		for(size_t i=0;i<scan_cur->observed_planes.size();i++)
		{ plane_fitting.estimatePlaneParams(scan_cur->observed_planes[i],cam); }
		gettimeofday(&end,NULL);
		timeused=(1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec)/1000;
		std::cout<<"time estimating plane parameters:"<<timeused<<std::endl;
//		fp_error<<timeused<<"\t";

		// cout extracted planes;
		for(size_t i=0;i<scan_cur->observed_planes.size();i++)
		{
			std::cout<<"plane - "<<i<<std::endl;
			std::cout<<scan_cur->observed_planes[i]->normal.transpose()<<", "
					 <<scan_cur->observed_planes[i]->d<<std::endl;
			std::cout<<scan_cur->observed_planes[i]->points.size()<<std::endl;
//			std::cout<<"determinant - "<<scan_cur->observed_planes[i]->cov_inv.determinant()<<std::endl;
		}

		fp_notes<<"extracted planes\t(id\tnormal\td\tplane_size)"<<std::endl;
		for(size_t i=0;i<scan_cur->observed_planes.size();i++)
		{
			fp_notes<<"\t"<<i<<"\t";
			fp_notes<<scan_cur->observed_planes[i]->normal.transpose()<<"\t"
					 <<scan_cur->observed_planes[i]->d<<"\t";
			fp_notes<<scan_cur->observed_planes[i]->points.size()<<std::endl;
		}

		// extract edge points;
		gettimeofday(&start,NULL);
		edge_ext.extractEdgePoints(scan_cur);
//		edge_ext.fitLinesHough(scan_cur);
//		for(size_t i=0;i<scan_cur->edge_points.size();i++)
//		{
//			edge_ext.fitSphere(scan_cur->edge_points[i]);
//			fp_notes<<scan_cur->edge_points[i]->meas_Edge<<"\t";
//		}
//		fp_notes<<std::endl;
		gettimeofday(&end,NULL);
		timeused=(1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec)/1000;
		cout<<"extracted edge points: "<<scan_cur->edge_points.size()<<std::endl;
		cout<<"time extracting edge points:"<<timeused<<"ms"<<endl;
		fp_notes<<"extracted "<<scan_cur->edge_points.size()<<" edge points and "<<scan_cur->edge_points_occluded.size()<<" shadow points using "<<timeused<<"ms"<<endl;
//		fp_error<<timeused<<"\t";

//		displayLines(scan_cur,vis);
//		vis->spin();

//		if(mode==DEBUG)
		if(mode==DEBUG && visual)
		{
			displayPlanes(scan_cur,vis);
			vis->spin();
			displayEdgePoints(scan_cur,vis);
			vis->spin();
		}

		if(filenum>first)
		{
			scan_cur->scan_ref=scan_ref;

			// plane feature matching;
			gettimeofday(&start,NULL);
			pfm.match(scan_ref->observed_planes, scan_cur->observed_planes, scan_cur->plane_matches);
			gettimeofday(&end,NULL);
			timeused=(1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec)/1000;
			std::cout<<"time matching plane feature:"<<timeused<<std::endl;
			fp_notes<<"associate "<<scan_cur->plane_matches.size()<<" pairs of planes using "<<timeused<<"ms"<<std::endl;

			// cout matched planes;
			std::cout<<"matched planes --- "<<std::endl;
			for(size_t i=0;i<scan_cur->plane_matches.size();i++)
			{
				std::cout<<i<<" --- "<<std::endl;
				std::cout<<"\t<"<<scan_cur->plane_matches[i].cur->id<<","<<scan_cur->plane_matches[i].ref->id<<">"<<std::endl;
				std::cout<<"\t"<<scan_cur->plane_matches[i].cur->normal.transpose()<<", "<<scan_cur->plane_matches[i].cur->d<<std::endl;
				std::cout<<"\t"<<scan_cur->plane_matches[i].ref->normal.transpose()<<", "<<scan_cur->plane_matches[i].ref->d<<std::endl;
			}

			fp_notes<<"matched planes\t(id\t<cur,ref>\tnormal_cur\td_cur\tnormal_ref\td_ref)"<<std::endl;
			for(size_t i=0;i<scan_cur->plane_matches.size();i++)
			{
				fp_notes<<i<<"\t<"<<scan_cur->plane_matches[i].cur->id<<","<<scan_cur->plane_matches[i].ref->id<<">"<<std::endl;
				fp_notes<<"\t"<<scan_cur->plane_matches[i].cur->normal.transpose()<<"\t"<<scan_cur->plane_matches[i].cur->d<<std::endl;
				fp_notes<<"\t"<<scan_cur->plane_matches[i].ref->normal.transpose()<<"\t"<<scan_cur->plane_matches[i].ref->d<<std::endl;
			}


			int *size=new int[2];
			size=fig_error->getWindowSize();

			Tcr_gt=scan_cur->Tcg_gt*scan_ref->Tcg_gt.inv();
			xi_cr_gt=Tcr_gt.getMotionVector();

//				displayMatchedEdgePoints(scan_cur,Tcr_gt,vis);
//				std::cout<<"measured"<<std::endl;
//				vis->spin();

			////////////////////////////////////////////////////////////////////////
			for(size_t i=0;i<scan_cur->projective_rays.size();i++)
			{
				scan_cur->projective_rays[i]->occluded->xyz
					=scan_cur->projective_rays[i]->occluded_proj->xyz;
				scan_cur->projective_rays[i]->occluded->xyz
					=scan_cur->projective_rays[i]->occluded_proj->xyz;
			}
			for(size_t i=0;i<scan_ref->projective_rays.size();i++)
			{
				scan_ref->projective_rays[i]->occluded->xyz
					=scan_ref->projective_rays[i]->occluded_proj->xyz;
				scan_ref->projective_rays[i]->occluded->xyz
					=scan_ref->projective_rays[i]->occluded_proj->xyz;
			}

//				displayMatchedEdgePoints(scan_cur,Tcr_gt,vis);
//				std::cout<<"calculated"<<std::endl;
//				vis->spin();

			////////////////////////////////////////////
			// align scans;
			// use only edges;
			gettimeofday(&start,NULL);
			Tcr.setIdentity();
//			me.usePlnPt(true,false);
			me.usePlnPt(usePln,usePt);
            //me.test();
			me.alignScans(scan_cur,scan_ref,Tcr,vis);
			gettimeofday(&end,NULL);
			timeused=(1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec)/1000;
			std::cout<<"time aligning edges:"<<timeused<<std::endl;
			fp_notes<<"aligning edges in "<<timeused<<" ms"<<std::endl;

			scan_cur->Tcr_edge=Tcr;
			scan_cur->Tcg_edge=scan_cur->Tcr_edge*scan_ref->Tcg_edge; // Tcg=Tcr*Trg;
			/* xi_cr=Tcr.getMotionVector(); */
			/* delta_xi=xi_cr_gt-xi_cr; */
			/* err_pos_edge=delta_xi.block<3,1>(0,0).norm(); */
			/* err_ang_edge=delta_xi.block<3,1>(3,0).norm(); */
			/* std::cout<<"using only edge\t"<<err_pos_edge<<"\t"<<err_ang_edge<<std::endl; */
			/* fp_notes<<"using only edge"<<std::endl; */
			/* fp_notes<<"\ttranslation error\t"<<err_pos_edge<<std::endl; */
			/* fp_notes<<"\trotation error\t\t"<<err_ang_edge<<std::endl; */
			/* fig_error->setPenColor(0,0,255,200); */
			/* fig_error->addLine(filenum*2,err_pos_pre_edge*size[0]+0.5*size[1],(filenum+1)*2,err_pos_edge*size[0]+0.5*size[1]); */
			/* fig_error->setPenColor(0,0,255,200); */
			/* fig_error->addLine(filenum*2,err_ang_pre_edge*size[0],(filenum+1)*2,err_ang_edge*size[0]); */
			/* err_pos_pre_edge=err_pos_edge; */
			/* err_ang_pre_edge=err_ang_edge; */

			/* Eigen::Matrix<double,10,1> res=me.computeResidual(scan_cur,Tcr); */
			/* fp_notes<<"residual at Tcr_opt_edge is\t"<<res.transpose()<<std::endl; */

			/* fp_error<<std::fixed<<data_reading->getTime()<<"\t"<<err_pos_edge<<"\t"<<err_ang_edge<<"\t"; */
			/* fp_residual<<std::fixed<<data_reading->getTime()<<"\t"<<res.transpose()<<"\t"; */

			std::cout<<"Tcr edge"<<std::endl<<Tcr.getMatrix4f()<<std::endl;

//			if(debug)
//			{
//				displayMatchedEdgePoints(scan_cur,Tcr,vis);
//				std::cout<<"residuals at Tcr"<<std::endl;
//				vis->spin();
//				displayMatchedEdgePoints(scan_cur,Tcr_gt,vis);
//				std::cout<<"residuals at Tcr_gt"<<std::endl;
//				vis->spin();
//			}

			if(mode==VIS_SCAN && err_pos_edge>0.1)
			{
				displayMatchedEdgePoints(scan_cur,Tcr,vis);
				vis->spin();
			}

			/* //////////////////////////////////////////// */
			/* // align scans; */
			/* // use only shadows; */
			/* gettimeofday(&start,NULL); */
/* //			Tcr.setIdentity(); */
			/* me.usePlnPt(false,true); */
			/* me.alignScans(scan_cur,scan_ref,Tcr,vis); */
			/* gettimeofday(&end,NULL); */
			/* timeused=(1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec)/1000; */
			/* std::cout<<"time aligning shadows:"<<timeused<<std::endl; */
			/* fp_notes<<"aligning shadows in "<<timeused<<" ms"<<std::endl; */
/* //			fp_error<<timeused<<"\t"; */
/* //			fp_error<<std::endl; */

			/* scan_cur->Tcr=Tcr; */
			/* scan_cur->Tcg=scan_cur->Tcr*scan_ref->Tcg; // Tcg=Tcr*Trg; */
			/* xi_cr=Tcr.getMotionVector(); */
			/* //std::cout<<"xi_cr\t\t"<<xi_cr.transpose()<<std::endl; */
/* //			Tcr_gt=scan_cur->Tcg_gt*scan_ref->Tcg_gt.inv(); */
/* //			std::cout<<"Tcg\t"<<scan_cur->Tcg_gt.t.transpose()<<" "<<scan_cur->Tcg_gt.Quaternion().transpose()<<std::endl; */
/* //			std::cout<<"Trg\t"<<scan_ref->Tcg_gt.t.transpose()<<" "<<scan_ref->Tcg_gt.Quaternion().transpose()<<std::endl; */
/* //			std::cout<<"Tcr_gt\t"<<Tcr_gt.t.transpose()<<" "<<Tcr_gt.Quaternion().transpose()<<std::endl; */
/* //			xi_cr_gt=Tcr_gt.getMotionVector(); */
			/* //std::cout<<"xi_cr_gt\t"<<xi_cr_gt.transpose()<<std::endl; */
			/* delta_xi=xi_cr_gt-xi_cr; */
			/* //std::cout<<"delta_xi\t"<<delta_xi.transpose()<<std::endl; */
			/* err_pos_shadow=delta_xi.block<3,1>(0,0).norm(); */
			/* err_ang_shadow=delta_xi.block<3,1>(3,0).norm(); */
			/* std::cout<<"using only shadow\t"<<err_pos_shadow<<"\t"<<err_ang_shadow<<std::endl; */
			/* fp_notes<<"using only shadow"<<std::endl; */
			/* fp_notes<<"\ttranslation error\t"<<err_pos_shadow<<std::endl; */
			/* fp_notes<<"\trotation error\t\t"<<err_ang_shadow<<std::endl; */
			/* fig_error->setPenColor(255,0,0,200); */
			/* fig_error->addLine(filenum*2,err_pos_pre_shadow*size[0]+0.5*size[1],(filenum+1)*2,err_pos_shadow*size[0]+0.5*size[1]); */
			/* fig_error->setPenColor(255,0,0,200); */
			/* fig_error->addLine(filenum*2,err_ang_pre_shadow*size[0],(filenum+1)*2,err_ang_shadow*size[0]); */
			/* err_pos_pre_shadow=err_pos_shadow; */
			/* err_ang_pre_shadow=err_ang_shadow; */
			/* //fig_error->spinOnce(); */

			/* std::cout<<"Tcr shadow"<<std::endl<<Tcr.getMatrix4f()<<std::endl; */

			/* res=me.computeResidual(scan_cur,Tcr); */
			/* fp_notes<<"residual at Tcr_opt_shadow is\t"<<res.transpose()<<std::endl; */

			/* fp_error<<err_pos_shadow<<"\t"<<err_ang_shadow<<std::endl; */
			/* fp_residual<<res.transpose()<<"\t"; */

			/* res=me.computeResidual(scan_cur,Tcr_gt); */
			/* fp_residual<<res.transpose()<<std::endl;; */

			/* fig_error->spinOnce(); */

//			if(mode==DEBUG)
			if(mode==DEBUG && visual)
			{
				displayScans_2scans(scan_cur,scan_ref,ulysses::Transform(),vis);
//				displayMatchedEdgePoints(scan_cur,ulysses::Transform(),vis);
				std::cout<<"initial pose"<<std::endl;
				vis->spin();
				displayScans_2scans(scan_cur,scan_ref,scan_cur->Tcr_edge,vis);
//				displayMatchedEdgePoints(scan_cur,scan_cur->Tcr_edge,vis);
				std::cout<<"edge-only"<<std::endl;
				vis->spin();
//				displayScans_2scans(scan_cur,scan_ref,scan_cur->Tcr,vis);
//				displayMatchedEdgePoints(scan_cur,scan_cur->Tcr,vis);
//				std::cout<<"shadow"<<std::endl;
//				vis->spin();
//				displayScans_2scans(scan_cur,scan_ref,Tcr_gt,vis);
//				displayMatchedEdgePoints(scan_cur,Tcr_gt,vis);
//				std::cout<<"groundtruth"<<std::endl;
//				vis->spin();
			}



			if(save_forDisplay)
			{
				fp_forDisplay.open("forDisplay.txt",std::ios::out);
				me.buildCorrespondence(scan_cur,scan_ref,Tcr_gt,Tcr_gt,0);
				fp_forDisplay<<scan_cur->Tcr_edge.getMatrix4f()<<std::endl;
				fp_forDisplay<<scan_cur->Tcr.getMatrix4f()<<std::endl;
				fp_forDisplay<<Tcr_gt.getMatrix4f()<<std::endl;
				fp_forDisplay<<scan_cur->plane_matches.size()<<"\t"
						     <<scan_cur->observed_planes.size()<<"\t"
							 <<scan_ref->observed_planes.size()<<std::endl;
				for(size_t i=0;i<scan_cur->plane_matches.size();i++)
				{
					fp_forDisplay<<scan_cur->plane_matches[i].cur->id<<"\t"
								 <<scan_cur->plane_matches[i].ref->id<<"\t"
					<<scan_cur->plane_matches[i].cur->normal.transpose()<<"\t"<<scan_cur->plane_matches[i].cur->d<<"\t"
					<<scan_cur->plane_matches[i].ref->normal.transpose()<<"\t"<<scan_cur->plane_matches[i].ref->d<<std::endl;
				}
				fp_forDisplay<<scan_cur->projective_ray_matches.size()<<std::endl;
				for(size_t i=0;i<scan_cur->projective_ray_matches.size();i++)
				{
					fp_forDisplay<<scan_cur->projective_ray_matches[i]->cur->plane->id<<"\t"
								 <<scan_cur->projective_ray_matches[i]->ref->plane->id<<"\t"
								 <<scan_cur->projective_ray_matches[i]->cur->occluding->xyz.transpose()<<"\t"
								 <<scan_cur->projective_ray_matches[i]->cur->occluded->xyz.transpose()<<"\t"
								 <<scan_cur->projective_ray_matches[i]->ref->occluding->xyz.transpose()<<"\t"
								 <<scan_cur->projective_ray_matches[i]->ref->occluded->xyz.transpose()<<std::endl;
				}
				fp_forDisplay.close();
				std::cout<<"forDisplay.txt saved !"<<std::endl;
				vis->spin();
			}


//			if(mode==DEBUG)
			delete scan_ref;
		}
		else
		{
			scan_cur->Tcg.setIdentity();
			scan_cur->Tcg_edge.setIdentity();
		}

		if(mode==VIS_SCAN)
		{
			display_addScan(scan_cur,vis);
			display_addCameraPose(scan_cur,vis);
			if(filenum%vis_every_n_frames==0)
			{
				vis->spin();
				vis->removeAllPointClouds();
			}
		}

		Tgc=scan_cur->Tcg.inv();
		fp<<std::fixed<<data_reading->getTime()<<" "<<Tgc.t.transpose()<<" "<<Tgc.Quaternion().transpose()<<std::endl;
		Tgc=scan_cur->Tcg_edge.inv();
		fp_edge<<std::fixed<<data_reading->getTime()<<" "<<Tgc.t.transpose()<<" "<<Tgc.Quaternion().transpose()<<std::endl;

		scan_ref=scan_cur;
		filenum++;
	}

//	fp<<std::endl<<"plane landmarks in the map"<<std::endl;
//	for(size_t i=0;i<map->planes.size();i++)
//	{
//		fp<<map->planes[i]->id<<" "<<map->planes[i]->n.transpose()<<" "<<map->planes[i]->d<<std::endl;
//	}
	fp.close();
	fp_edge.close();
	fp_shadow.close();
	fp_error.close();
	fp_residual.close();
	return 0;
}
