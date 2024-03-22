/*==============================================
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@gmail.com
#
# Last modified:	2018-12-03 09:22
#
# Filename:		motion_estimation.cpp
#
# Description: 
#
===============================================*/

#include "motion_estimation.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace g2o;
using namespace Eigen;
//std::ofstream fp_debug;

namespace ulysses
{
	void Edge_ProjectiveRay::computeError()
	{
		const g2o::VertexSE3Expmap* v0 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
		const g2o::VertexSE3Expmap* v1 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );
		g2o::SE3Quat Tcr=v0->estimate().inverse()*v1->estimate();
		// point-line distance
//		Eigen::Matrix3d R=Tcr.rotation().toRotationMatrix();
		Eigen::Vector3d t=Tcr.translation();
		Eigen::Vector3d dir=Tcr*_measurement.ref->occluding->xyz-t;
		Eigen::Vector3d p=_measurement.cur->occluding->xyz;

		Eigen::Matrix<double,1,1> mu_mat=dir.transpose()*(p-t);
		double mu=mu_mat(0,0)/(dir.norm()*dir.norm());
		_error=p-mu*dir-t;

//		Eigen::Vector3d tmp=_measurement.cur->plane->projected_point(p,t);
//		_error=tmp-

//		g2o::SE3Quat Trc=Tcr.inverse();
//		Eigen::Matrix3d R=Trc.rotation().toRotationMatrix();
//		Eigen::Vector3d t=Trc.translation();
//		Eigen::Vector3d nr=_measurement.ref->plane->normal;
//		double dr=_measurement.ref->plane->d;
//		Eigen::Vector3d pr=_measurement.ref->occluding->xyz;
//		double ntd=nr.transpose()*t+dr;
//		double ntp=nr.transpose()*(t-pr);
//		double npd=nr.transpose()*pr+dr;
//		Eigen::Vector3d cp_pic=pr*ntd/ntp-t*npd/ntp;
//		cp_pic=Tcr*cp_pic;
//		_error=cp_pic-_measurement.cur->occluded->xyz;
	}

	void Edge_ProjectiveRay_2::computeError()
	{
		const g2o::VertexSE3Expmap* v0 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
		const g2o::VertexSE3Expmap* v1 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );
		g2o::SE3Quat Tcr=v0->estimate().inverse()*v1->estimate();
		// point-line distance
//		Eigen::Matrix3d R=Tcr.rotation().toRotationMatrix();
//		Eigen::Vector3d t=Tcr.translation();
		Eigen::Vector3d p=Tcr*_measurement.ref->occluded->xyz;
		Eigen::Vector3d n=_measurement.cur->plane->normal;
		Eigen::Matrix<double,1,1> tmp=n.transpose()*p;
		double tmp2=tmp(0,0)+_measurement.cur->plane->d;
		_error=tmp2*n;
//		g2o::SE3Quat Trc=Tcr.inverse();
//		Eigen::Matrix3d R=Trc.rotation().toRotationMatrix();
//		Eigen::Vector3d t=Trc.translation();
//		Eigen::Vector3d nr=_measurement.ref->plane->normal;
//		double dr=_measurement.ref->plane->d;
//		Eigen::Vector3d pr=_measurement.ref->occluding->xyz;
//		double ntd=nr.transpose()*t+dr;
//		double ntp=nr.transpose()*(t-pr);
//		double npd=nr.transpose()*pr+dr;
//		Eigen::Vector3d cp_pic=pr*ntd/ntp-t*npd/ntp;
//		cp_pic=Tcr*cp_pic;
//		_error=cp_pic-_measurement.cur->occluded->xyz;
	}

	Eigen::Matrix<double,10,1> MotionEstimation::computeResidual(Scan *scan_cur, Transform Tcr)
	{
		Eigen::Matrix3d R=Tcr.R;
		Eigen::Vector3d t=Tcr.t;
		Eigen::Vector3d res_line,res_pln;
		double error_line=0,error_pln=0;
		double error_line1=0,error_pln1=0;
		double error_n=0,error_d=0;
		double error_edge=0;

		buildCorrespondence(scan_cur,scan_cur->scan_ref,Tcr,Tcr,0);

		for(size_t i=0;i<scan_cur->projective_ray_matches.size();i++)
		{
//			scan_cur->projective_ray_matches[i]->cur->occluded->xyz
//				=scan_cur->projective_ray_matches[i]->cur->occluded_proj->xyz;
//			scan_cur->projective_ray_matches[i]->ref->occluded->xyz
//				=scan_cur->projective_ray_matches[i]->ref->occluded_proj->xyz;

			Eigen::Vector3d dir=Tcr.transformPoint(scan_cur->projective_ray_matches[i]->ref->occluded->xyz)-t;
			Eigen::Vector3d p=scan_cur->projective_ray_matches[i]->cur->occluded->xyz;
			Eigen::Matrix<double,1,1> mu_mat=dir.transpose()*(p-t);
			double mu=mu_mat(0,0)/(dir.norm()*dir.norm());
			res_line=p-mu*dir-t;
			error_line+=res_line.transpose()*res_line;

			dir=scan_cur->projective_ray_matches[i]->cur->occluded->xyz;
			p=Tcr.transformPoint(scan_cur->projective_ray_matches[i]->ref->occluding->xyz);
			mu_mat=dir.transpose()*p;
			mu=mu_mat(0,0)/(dir.norm()*dir.norm());
			res_line=p-mu*dir;
			error_line1+=res_line.transpose()*res_line;

			p=Tcr.transformPoint(scan_cur->projective_ray_matches[i]->ref->occluded->xyz);
			Eigen::Vector3d n=scan_cur->projective_ray_matches[i]->cur->plane->normal;
			Eigen::Matrix<double,1,1> tmp=n.transpose()*p;
			double tmp2=tmp(0,0)+scan_cur->projective_ray_matches[i]->cur->plane->d;
			res_pln=tmp2*n;
			error_pln+=res_pln.transpose()*res_pln;

			p=scan_cur->projective_ray_matches[i]->cur->occluded->xyz;
			n=R*scan_cur->projective_ray_matches[i]->ref->plane->normal;
			tmp=n.transpose()*p;
			double d=scan_cur->projective_ray_matches[i]->ref->plane->d-t.transpose()*n;
			tmp2=tmp(0,0)+d;
			res_pln=tmp2*n;
			error_pln1+=res_pln.transpose()*res_pln;

			p=Tcr.transformPoint(scan_cur->projective_ray_matches[i]->ref->occluding->xyz);
			p=scan_cur->projective_ray_matches[i]->cur->occluding->xyz-p;
			error_edge+=p.transpose()*p;
		}
		for(size_t i=0;i<scan_cur->plane_matches.size();i++)
		{
			Eigen::Vector3d n=scan_cur->plane_matches[i].cur->normal-R*scan_cur->plane_matches[i].ref->normal;
			error_n+=n.norm();
			Eigen::Matrix<double,1,1> tmp=t.transpose()*R*scan_cur->plane_matches[i].ref->normal;
			double d=fabs(scan_cur->plane_matches[i].ref->d-tmp(0,0)-scan_cur->plane_matches[i].cur->d);
			error_d+=d;
		}
//		error_line/=scan_cur->projective_ray_matches.size();
//		error_pln/=scan_cur->projective_ray_matches.size();
		Eigen::Matrix<double,10,1> ret;
		ret.setZero();
		ret(0)=error_line;
		ret(1)=error_line1;
		ret(2)=error_pln;
		ret(3)=error_pln1;
		ret(4)=error_n;
		ret(5)=error_d;
		ret(6)=error_edge;
		return ret;
	}


	double MotionEstimation::alignEdges(Scan *scan_cur, Scan *scan_ref, Transform &Tcr, 
									boost::shared_ptr<pcl::visualization::PCLVisualizer> vis)
	{
		fp.open("alignScans_motion.txt",std::ios::app);

//		if(debug && visual)
//		{
//			displayMatchedEdgePoints(scan_cur,Tcr,vis);
//			vis->spin();
//		}

		double error;
//		error=alignEdges(scan_cur,scan_ref,Tcr,vis);

//		fuseEdges(scan_cur,Tcr);

//		if(debug)
//		{
//			displayMatchedProjRays(scan_cur,Tcr,vis);
//			vis->spin();
//		}

		fp.close();

		return error;
	}

	void MotionEstimation::fuseEdges(Scan *scan_cur, Transform &Tcr)
	{
		for(size_t i=0;i<scan_cur->projective_ray_matches.size();i++)
		{
			scan_cur->projective_ray_matches[i]->fused_edge_point_cur
				=0.5*(scan_cur->projective_ray_matches[i]->cur->occluding->xyz
				+Tcr.transformPoint(scan_cur->projective_ray_matches[i]->ref->occluding->xyz));
//			Transform Trc=Tcr.inv();
//			scan_cur->projective_ray_matches[i]->fused_edge_point_ref
//				=Trc.transformPoint(scan_cur->projective_ray_matches[i]->fused_edge_point_cur);
		}
	}

	double MotionEstimation::alignScans(Scan *scan_cur, Scan *scan_ref, Transform &Tcr, 
									boost::shared_ptr<pcl::visualization::PCLVisualizer> vis)
	{
		char id[20];

		int iterations = 0;
		bool converged = false;
		double chi2_pre=DBL_MAX, chi2=DBL_MAX;
		Transform Tcr_pre=Tcr;
		double error;

		fp.open("alignScans_motion.txt",std::ios::app);
		if(debug) fp<<std::endl<<"alignEdges *******************************************************"<<std::endl;
		
		while(iterations<max_iter_icp && !converged)
		{
			if(debug)
			{
				fp<<std::endl<<"icp iteration ========================================"<<std::endl;
				std::cout<<std::endl<<"icp iteration ========================================"<<std::endl;
			}

			buildCorrespondence(scan_cur,scan_ref,Tcr,Tcr_pre, iterations);

//			Transform Tcr_gt=scan_cur->Tcg_gt*scan_ref->Tcg_gt.inv();
//			buildCorrespondence(scan_cur,scan_ref,Tcr,Tcr_pre, iterations);

//			if(debug && visual)
////			if(debug && visual && usePln==false && usePt==true)
//			{
//				displayMatchedEdgePoints(scan_cur,Tcr,vis);
//				vis->spin();
//			}

			chi2_pre=chi2;
            //test();
			chi2=optimizeEdges(scan_cur,Tcr);
			std::cout<<iterations<<" - "<<chi2<<std::endl;
			
//			if(debug && visual)
////			if(debug && visual && usePln==false && usePt==true)
//			{
//				displayMatchedEdgePoints(scan_cur,Tcr,vis);
//				vis->spin();
//			}

			iterations++;
			if(chi2>chi2_pre)
				converged=true;
		}

		fp.close();

		return error;
	}

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

void MotionEstimation::test_loop()
{
	for(size_t i=0;i<30;i++)
	{
		cout<<endl<<i<<" ************************************************"<<endl;
		test();
	}
}

void MotionEstimation::test()
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

	double MotionEstimation::optimizeEdges(Scan *scan_cur, Transform &Tcr)
	{
//		typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> Block;
		g2o::SparseOptimizer optimizer;
		optimizer.setVerbose(debug);
//		std::unique_ptr<Block::LinearSolverType> linearSolver;
		//if (DENSE) {
//		linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
		//} else {
		//linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<Block::PoseMatrixType>>();
		//}
//		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( g2o::make_unique<Block>(std::move(linearSolver)) );
//		g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( g2o::make_unique<Block>(std::move(linearSolver)) );
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));
		optimizer.setAlgorithm(solver);

//		g2o::VertexSE3Expmap* pose0 = new g2o::VertexSE3Expmap();
//		g2o::SE3Quat Tcg(Tcr.R,Tcr.t);
//		pose0->setEstimate(Tcg.inverse());
//		pose0->setId(0);
//		pose0->setFixed(false);
//		optimizer.addVertex(pose0);

		Eigen::Isometry3d Tcg;
		Tcg=Tcr.Quat();
		Tcg.translation()=Tcr.t;
		g2o::VertexSE3 *pose0 = new g2o::VertexSE3();
		pose0->setEstimate(Tcg.inverse());
		pose0->setId(0);
		pose0->setFixed(false);
		optimizer.addVertex(pose0);
		

//		g2o::VertexSE3Expmap* pose1 = new g2o::VertexSE3Expmap();
//		g2o::SE3Quat Trg(Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero());
//		pose1->setEstimate(Trg.inverse());
//		pose1->setId(1);
//		pose1->setFixed(true);
//		optimizer.addVertex(pose1);

		Eigen::Isometry3d Trg;
		Trg.setIdentity();
		g2o::VertexSE3 *pose1 = new g2o::VertexSE3();
		pose1->setEstimate(Trg.inverse());
		pose1->setId(1);
		pose1->setFixed(true);
		optimizer.addVertex(pose1);

		int count_edge=0;
//		if(false)
		if(usePln)
		for(size_t i=0;i<scan_cur->projective_ray_matches.size();i++)
		{
			MeasurementPoint measure_pt;
			measure_pt.pt_cur=scan_cur->projective_ray_matches[i]->cur->occluding->xyz;
			measure_pt.pt_ref=scan_cur->projective_ray_matches[i]->ref->occluding->xyz;
//			EdgeGICP measure_pt;
//			measure_pt.pos0=scan_cur->projective_ray_matches[i]->cur->occluding->xyz;
//			measure_pt.pos1=scan_cur->projective_ray_matches[i]->ref->occluding->xyz;

			Edge_EdgePoint *edge=new Edge_EdgePoint();
//			Edge_V_V_GICP *edge=new g2o::Edge_V_V_GICP();
			edge->setVertex(0,pose0);
			edge->setVertex(1,pose1);
			edge->setMeasurement(measure_pt);
			measure_pt=edge->measurement();
//			edge->information()=measure_pt.prec0(0.01);
//			edge->setInformation(scan_cur->projective_ray_matches[i]->cur->occluding->cov);
			edge->setInformation(Eigen::Matrix3d::Identity());
			edge->setId(i);
			optimizer.addEdge(edge);
			count_edge++;
		}
		std::cout<<"\toptimizing edge-points "<<count_edge<<std::endl;
		if(debug)
			fp<<"optimizing edge-points "<<count_edge<<std::endl;

		int count_shadow=0;
//		if(false)
		if(usePt)
		for(size_t i=0;i<scan_cur->projective_ray_matches.size();i++)
		{
			MeasurementProjectiveRay measure_pt(scan_cur->projective_ray_matches[i]->cur, 
												scan_cur->projective_ray_matches[i]->ref);
			Edge_ProjectiveRay *edge=new Edge_ProjectiveRay();
			edge->setVertex(0,pose0);
			edge->setVertex(1,pose1);
			edge->setMeasurement(measure_pt);
//			edge->setInformation(scan_cur->projective_ray_matches[i]->cur->occluded->cov.inverse());
			edge->setInformation(Eigen::Matrix3d::Identity()*0.5);
			edge->setId(count_edge+i*2);
			optimizer.addEdge(edge);

			Edge_ProjectiveRay_2 *edge2=new Edge_ProjectiveRay_2();
			edge2->setVertex(0,pose0);
			edge2->setVertex(1,pose1);
			edge2->setMeasurement(measure_pt);
//			edge->setInformation(scan_cur->projective_ray_matches[i]->cur->occluded->cov.inverse());
			edge2->setInformation(Eigen::Matrix3d::Identity());
			edge2->setId(count_edge+i*2+1);
			optimizer.addEdge(edge2);
			count_shadow++;
		}
		std::cout<<"\toptimizing occluded edge-points "<<count_shadow<<std::endl;
		if(debug)
			fp<<"optimizing occluded edge-points "<<count_shadow<<std::endl;

		optimizer.initializeOptimization();
		std::cout<<"\toptimizer.initializeOptimization() "<<std::endl;
		int result=optimizer.optimize ( max_iter_g2o );

		Tcg = pose0->estimate().inverse();
		Tcr.R = Tcg.rotation();
		Tcr.t = Tcg.translation();

		double chi2=optimizer.activeChi2();
		optimizer.clear(); // delete vertices and edges;

		return chi2;
	}


	void MotionEstimation::buildCorrespondence(Scan *scan_cur, Scan *scan_ref, Transform Tcr, Transform Tcr_pre, int iterations)
	{
		double thresh=thres_association*thres_association/(double)(iterations+1);

		ANNkd_tree *kdtree;
		ANNpoint cur_point=annAllocPt(3);
		ANNpointArray ref_points=annAllocPts(scan_ref->projective_rays.size(),3);
		ANNidxArray index=new ANNidx[1];
		ANNdistArray distance=new ANNdist[1];

		// build kdtree with scan_ref->projective_rays[i]->occluding;
		for(size_t i=0;i<scan_ref->projective_rays.size();i++)
		{
			Eigen::Vector3d pr=Tcr.transformPoint(scan_ref->projective_rays[i]->occluding->xyz);
			ref_points[i][0]=pr(0);
			ref_points[i][1]=pr(1);
			ref_points[i][2]=pr(2);
		}
		kdtree=new ANNkd_tree(ref_points,scan_ref->projective_rays.size(),3);

		// fill scan_cur->projective_ray_matches;
		scan_cur->projective_ray_matches.clear();
		for(size_t i=0;i<scan_cur->projective_rays.size();i++)
		{
//			if(scan_cur->projective_rays[i]->occluding->rgb.norm()==0)
//				continue;
			cur_point[0]=scan_cur->projective_rays[i]->occluding->xyz(0);
			cur_point[1]=scan_cur->projective_rays[i]->occluding->xyz(1);
			cur_point[2]=scan_cur->projective_rays[i]->occluding->xyz(2);
			kdtree->annkSearch(cur_point,1,index,distance);
			if(distance[0]<thresh)
			{
				for(size_t j=0;j<scan_cur->plane_matches.size();j++)
				{
					if(scan_cur->projective_rays[i]->plane==scan_cur->plane_matches[j].cur 
					   && scan_ref->projective_rays[index[0]]->plane==scan_cur->plane_matches[j].ref
					   && scan_ref->projective_rays[index[0]]->occluding->dir.transpose()*scan_cur->projective_rays[i]->occluding->dir>0.8)
					{
						ProjectiveRayPair *match=new ProjectiveRayPair(scan_cur->projective_rays[i],scan_ref->projective_rays[index[0]]);
						scan_cur->projective_ray_matches.push_back(match);
					}
				}
//				Eigen::Vector3d tmp=scan_cur->projective_rays[i]->occluded->xyz-scan_ref->projective_rays[index[0]]->occluded->xyz;
//				if(tmp.norm()<thres_association*5)
//				{
//					ProjectiveRayPair *match=new ProjectiveRayPair(scan_cur->projective_rays[i],scan_ref->projective_rays[index[0]]);
//					scan_cur->projective_ray_matches.push_back(match);
//				}
			}
		}
//	
//		// fill scan_cur->point_matches_occluded;
//		scan_cur->point_matches_occluded.clear();
//		for(size_t i=0;i<scan_cur->edge_points_occluded.size();i++)
//		{
//			if(!scan_cur->edge_points_occluded[i]->isEdge || scan_cur->edge_points_occluded[i]->occluding==0 || scan_cur->edge_points_occluded[i]->plane==0)
//				continue;
//			cur_point[0]=scan_cur->edge_points_occluded[i]->occluding->xyz(0);
//			cur_point[1]=scan_cur->edge_points_occluded[i]->occluding->xyz(1);
//			cur_point[2]=scan_cur->edge_points_occluded[i]->occluding->xyz(2);
//			kdtree->annkSearch(cur_point,1,index,distance);
//			if(distance[0]<thresh*2.0 && scan_ref->edge_points[index[0]]->isEdge)
//			{
//				EdgePointPair match(scan_cur->edge_points_occluded[i],scan_ref->edge_points[index[0]]);
//				scan_cur->point_matches_occluded.push_back(match);
//			}
//		}

		delete kdtree;
//		delete kdtree_occluded;
		delete index;
		delete distance;
		annDeallocPt(cur_point);
//		annDeallocPt(cur_point_occluded);
		annDeallocPts(ref_points);
//		annDeallocPts(ref_points_occluded);
	}

	void MotionEstimation::compute_Psi(Scan *scan, Transform Tcr)
	{
		std::ofstream fp_op;
		fp_op.open("occluded_points.txt",std::ios::out);
		if(debug)
			fp<<std::endl<<"compute_Psi_pi"<<std::endl;

		scan->Psi_p.setZero();
		scan->Psi_pi.setZero();
		scan->Psi.setZero();
		Eigen::Matrix<double,3,6> J_p;
		Eigen::Matrix<double,3,6> J_pi;
		Transform Trc=Tcr.inv();
		Eigen::Vector4d pln;
//		std::ofstream fp_delta;
//		fp_delta.open("delta.txt",std::ios::app);
//		fp<<"size "<<scan->plane_matches.size()<<std::endl;
		for(size_t i=0;i<scan->point_matches.size();i++)
		{
			J_p.block<3,3>(0,0)=-Trc.R.transpose();
			J_p.block<3,3>(0,3)=Trc.R.transpose()*g2o::skew(scan->point_matches[i].ref->xyz-Trc.t);
			scan->Psi_p+=J_p.transpose()*scan->point_matches[i].cur->cov.inverse()*J_p;
//			scan->Psi_p+=J_p.transpose()*J_p;
		}
		double v1_pre=0,v2_pre=0;
		for(size_t i=0;i<scan->point_matches_occluded.size();i++)
		{
			if(scan->point_matches_occluded[i].cur->plane==0)
				continue;
			pln.block<3,1>(0,0)=scan->point_matches_occluded[i].cur->plane->normal;
			pln(3)=scan->point_matches_occluded[i].cur->plane->d;
			pln=Trc.transformPlane(pln);
			Eigen::Vector3d n=pln.block<3,1>(0,0);
			double d=pln(3);
			Eigen::Vector3d p=scan->point_matches_occluded[i].ref->xyz;
			double npd=n.transpose()*p+d;
			double ntd=n.transpose()*Trc.t+d;
			double npt=n.transpose()*(p-Trc.t);
			J_pi.block<3,3>(0,0)=Trc.R.transpose()*(Trc.t-p)*n.transpose()*(npd/(npt*npt))+Trc.R.transpose()*(ntd/npt);
			J_pi.block<3,3>(0,3)=Trc.R*g2o::skew(Trc.t-p)*(ntd/npt);
			scan->Psi_pi+=J_pi.transpose()*scan->point_matches_occluded[i].cur->cov.inverse()*J_pi;
//			scan->Psi_pi+=J_pi.transpose()*J_pi;
			fp_op<<i<<"\t"<<npd/(npt*npt)<<"\t"<<ntd/npt<<std::endl;
//			int *size=new int[2];
//			size=fig->getWindowSize();
//			double v1=npd/(npt*npt)*0.1*size[1]+0.75*size[1];
//			double v2=ntd/npt*0.1*size[1]+0.25*size[1];
//			double delta=double(size[0])/scan->point_matches_occluded.size();
//			fig->setPenColor(255,0,0,200);
//			fig->addLine(i*delta,v1_pre,(i+1)*delta,v1);
//			fig->setPenColor(0,0,255,200);
//			fig->addLine(i*delta,v2_pre,(i+1)*delta,v2);
////			fig->spinOnce();
//			v1_pre=v1;
//			v2_pre=v2;
		}
//		fig->spinOnce();

//		fp_delta.close();
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6> > es;
		es.compute(scan->Psi_p);
		Lambda_p=es.eigenvalues();
		Q_p=es.eigenvectors();
		es.compute(scan->Psi_pi);
		Lambda_pi=es.eigenvalues();
		Q_pi=es.eigenvectors();

		for(size_t i=0;i<6;i++)
		{
			for(size_t j=0;j<6;j++)
			{
				scan->Psi(i,j)=scan->Psi_p(i,j)+scan->Psi_pi(i,j);
			}
		}
//		scan->Psi=scan->Psi_p+scan->Psi_pi;
		es.compute(scan->Psi);
		Lambda=es.eigenvalues();
		Q=es.eigenvectors();

//		if(debug)
//		{
//			fp<<std::endl;
//			fp<<"Psi_p"<<std::endl<<scan->Psi_p<<std::endl;
//			fp<<"Psi_p eigenvalues: "<<Lambda_p.transpose()<<std::endl;
//			fp<<"Psi_p eigenvectors"<<std::endl<<Q_p<<std::endl;
//			fp<<std::endl;
//			fp<<"Psi_pi"<<std::endl<<scan->Psi_pi<<std::endl;
//			fp<<"Psi_pi eigenvalues: "<<Lambda_pi.transpose()<<std::endl;
//			fp<<"Psi_pi eigenvectors"<<std::endl<<Q_pi<<std::endl;
//			fp<<std::endl;
//			fp<<"Psi"<<std::endl<<scan->Psi<<std::endl;
//			fp<<"Psi eigenvalues: "<<Lambda.transpose()<<std::endl;
//			fp<<"Psi eigenvectors"<<std::endl<<Q<<std::endl;
//			fp<<std::endl;
//		}

//		lambda_pi_1=DBL_MIN;
//		idx_1=-1;
//		for(size_t i=0;i<6;i++)
//		{
//			if(Lambda_pi(i)>lambda_pi_1)
//			{
//				lambda_pi_1=Lambda_pi(i);
//				idx_1=i;
//			}
//		}
////		sq_lambda_pi_1=sqrt(lambda_pi_1);
//		for(size_t l=0;l<6;l++)
//			q_pi_1(l)=Q_pi(l,idx_1);
		fp_op.close();
	}

}

