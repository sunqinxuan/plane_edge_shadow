/*==============================================
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2018-11-28 09:23
#
# Filename:		pose_estimation.cpp
#
# Description: 
#
===============================================*/

#include "pose_estimation.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//std::ofstream fp_debug;

namespace ulysses
{

	void Edge_Plane::computeError()
	{
		const g2o::VertexSE3Expmap* v0 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
		const g2o::VertexSE3Expmap* v1 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );
		// Tcr=Tcg*Trg.inv();
		g2o::SE3Quat Tcr=v0->estimate().inverse()*v1->estimate();
		Eigen::Vector4d plane_ref_trans=_measurement.transform_pln_ref(Tcr);
		_error=_measurement.pln_cur-plane_ref_trans;
	}

	void Edge_EdgePoint::computeError()
	{
		const g2o::VertexSE3 *v0 =static_cast<const g2o::VertexSE3*> ( _vertices[0] );
		const g2o::VertexSE3 *v1 =static_cast<const g2o::VertexSE3*> ( _vertices[1] );
//		const g2o::VertexSE3Expmap* v0 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
//		const g2o::VertexSE3Expmap* v1 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );
		// Tcr=Tcg*Trg.inv();
		Eigen::Isometry3d Tcr=v0->estimate().inverse()*v1->estimate();
		Eigen::Vector3d point_ref_trans=Tcr*_measurement.pt_ref;
//		g2o::SE3Quat Tcr=v0->estimate().inverse()*v1->estimate();
//		Eigen::Vector3d point_ref_trans=_measurement.transform_pt_ref(Tcr);
		_error=_measurement.pt_cur-point_ref_trans;
	}

	void Edge_EdgePoint_occluded::computeError()
	{
		const g2o::VertexSE3Expmap* v0 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
		const g2o::VertexSE3Expmap* v1 =static_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );
		// Tcr=Tcg*Trg.inv();
		g2o::SE3Quat Tcr=v0->estimate().inverse()*v1->estimate();
		Eigen::Vector3d point_ref_trans=_measurement.transform_pt_ref(Tcr);
		_error=_measurement.cur->xyz-point_ref_trans;
//		fp_debug<<std::endl<<"computeError()"<<std::endl;
//		fp_debug<<"\tref_occluding\t"<<_measurement.ref->xyz.transpose()<<"\t"<<_measurement.ref_occluding_trans.transpose()<<std::endl;
//		fp_debug<<"\tcur_occluding\t"<<_measurement.cur->xyz.transpose()<<std::endl;
//		fp_debug<<"\tcur_occluded \t"<<_measurement.cur->occluded->xyz.transpose()<<std::endl;
//		fp_debug<<"\tref_occluded_trans\t"<<_measurement.ref_occluded_trans.transpose()<<std::endl;
//		fp_debug<<"\tcur_plane\t"<<_measurement.cur->occluded->plane->normal.transpose()<<std::endl;
//		fp_debug<<"\terror \t"<<_error.transpose()<<std::endl;
	}


	Transform PoseEstimation::alignPlanes(std::vector<PlanePair> matched_planes)
	{
		std::ofstream fp;
		fp.open("alignPlanes.txt",std::ios::app);
		if(debug)
			fp<<std::endl<<"*********************************************************"<<std::endl;
		Transform Tcr_align_planes;
		_case=constraint_case(matched_planes);
		if(debug)
		{
			fp<<"constrain case - "<<_case<<std::endl;
			fp<<"svd of H - "<<H_singularValues.transpose()<<std::endl;
			fp<<H_svd_U<<std::endl;
			fp<<H_svd_V<<std::endl;
		}
		Eigen::Matrix3d tmp_mat3d,tmp_inverse;
		bool invertible;
		Tcr_align_planes.R=H_svd_V*H_svd_U.transpose(); // R_cr
		if(debug)
			fp<<"Rotation - "<<std::endl<<Tcr_align_planes.R<<std::endl;
		Eigen::MatrixXd A=Eigen::MatrixXd::Zero(MAX_PLNS,3);
		Eigen::VectorXd d=Eigen::VectorXd::Zero(MAX_PLNS);
		if(_case==DoF_6)
		{
			//Tcr_align_planes.R=H_svd_U*H_svd_V.transpose();
			for(int i=0;i<matched_planes.size();i++)
			{
				A.block<1,3>(i,0)=matched_planes[i].cur->normal.transpose();
				d(i)=matched_planes[i].ref->d-matched_planes[i].cur->d;
			}
			if(debug)
			{
				fp<<"A - "<<std::endl<<A<<std::endl;
				fp<<"d - "<<d.transpose()<<std::endl;
			}
		}
		else if(_case==DoF_5)
		{
			//Tcr_align_planes.R=H_svd_U*H_svd_V.transpose();
			if(abs(Tcr_align_planes.R.determinant()+1.0f)<1.0e-4)
			{
				H_svd_U.block<3,1>(0,2)=-H_svd_U.block<3,1>(0,2);
				Tcr_align_planes.R=H_svd_U*H_svd_V.transpose();
				if(debug)
				{
					fp<<"U':"<<std::endl<<H_svd_U<<std::endl;
					fp<<"det(R'):"<<Tcr_align_planes.R.determinant()<<std::endl;
				}
			}
			else
			{
				if(debug)
				{
					fp<<"U:"<<std::endl<<H_svd_U<<std::endl;
					fp<<"det(R):"<<Tcr_align_planes.R.determinant()<<std::endl;
				}
			}
			for(int i=0;i<matched_planes.size();i++)
			{
				A.block<1,3>(i,0)=matched_planes[i].cur->normal.transpose();
				d(i)=matched_planes[i].ref->d-matched_planes[i].cur->d;
			}
			A.block<1,3>(matched_planes.size(),0)=H_svd_V.block<3,1>(0,2).transpose();
		}
		else if(_case==DoF_3)
		{
			Eigen::Matrix3d H1, H_svd_U1, H_svd_V1;
			H1=H+H_svd_U.block<3,1>(0,2)*H_svd_V.block<3,1>(0,2).transpose();
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(H1, Eigen::ComputeFullU | Eigen::ComputeFullV);
			H_svd_U1=svd.matrixU();
			H_svd_V1=svd.matrixV();
			Tcr_align_planes.R=H_svd_U1*H_svd_V1.transpose();
			if(abs(Tcr_align_planes.R.determinant()+1.0f)<1.0e-4)
			{
				H_svd_U1.block<3,1>(0,2)=-H_svd_U1.block<3,1>(0,2);
				Tcr_align_planes.R=H_svd_U1*H_svd_V1.transpose();
				if(debug)
				{
					fp<<"U1':"<<std::endl<<H_svd_U1<<std::endl;
					fp<<"det(R'):"<<Tcr_align_planes.R.determinant()<<std::endl;
				}
			}
			else
			{
				if(debug)
				{
					fp<<"U1:"<<std::endl<<H_svd_U1<<std::endl;
					fp<<"det(R):"<<Tcr_align_planes.R.determinant()<<std::endl;
				}
			}
			for(int i=0;i<matched_planes.size();i++)
			{
				A.block<1,3>(i,0)=matched_planes[i].cur->normal.transpose();
				d(i)=matched_planes[i].ref->d-matched_planes[i].cur->d;
			}
			A.block<1,3>(matched_planes.size(),0)=H_svd_V.block<3,1>(0,1).transpose();
			A.block<1,3>(matched_planes.size()+1,0)=H_svd_V.block<3,1>(0,2).transpose();
		}
		tmp_mat3d=A.transpose()*A;
		tmp_mat3d.computeInverseWithCheck(tmp_inverse,invertible);
		if(invertible)
		{
			Tcr_align_planes.t=tmp_inverse*A.transpose()*d;
			if(debug)
			{
				fp<<"inv(ATA) - "<<std::endl<<tmp_inverse<<std::endl;
				fp<<"translation - "<<Tcr_align_planes.t.transpose()<<std::endl;
			}
		}
		else
		{
			std::cerr<<"matrix A uninvertible!"<<std::endl;
			return Transform();
		}
		fp.close();
		return Tcr_align_planes;
	}


	PoseEstimation::ConstraintCase PoseEstimation::constraint_case(std::vector<PlanePair> matched_planes)
	{
		ofstream fp;
		fp.open("constraint_case.txt",std::ios::app);
		H.setZero();
		for(int i=0;i<matched_planes.size();i++)
		{
			H=H+matched_planes[i].ref->normal*matched_planes[i].cur->normal.transpose();
		}
		if(debug)
			fp<<"H - "<<std::endl<<H<<std::endl;

		Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
		H_svd_U=svd.matrixU();
		H_svd_V=svd.matrixV();
		H_singularValues=svd.singularValues();
		if(debug)
		{
			fp<<"singular - "<<H_singularValues.transpose()<<std::endl;
			fp<<"H_svd_U - "<<std::endl<<H_svd_U<<std::endl;
			fp<<"H_svd_V - "<<std::endl<<H_svd_V<<std::endl;
		}

		if(H_singularValues(0)>100*H_singularValues(1))
			return DoF_3;
		else if(H_singularValues(1)>100*H_singularValues(2))
			return DoF_5;
		else
			return DoF_6;
	}

	double PoseEstimation::alignScans(Scan *scan_cur, Scan *scan_ref, Transform &Tcr, 
									boost::shared_ptr<pcl::visualization::PCLVisualizer> vis)
	{
		char id[20];
		fp.open("alignScans.txt",std::ios::app);
//		fp_debug.open("debug.txt",std::ios::app);
		if(debug) fp<<std::endl<<"******************************************************************"<<std::endl;
//		if(debug) fp_debug<<std::endl<<"******************************************************************"<<std::endl;

		int iterations = 0;
		bool converged = false;
		double chi2_pre=DBL_MAX, chi2=DBL_MAX;
		Transform Tcr_pre=Tcr;
		double error;
		
//		confirmPlaneCorrespondence(scan_cur,Tcr);

//		fp<<"ref planes "<<std::endl;
		for(size_t i=0;i<scan_ref->observed_planes.size();i++)
		{
//			fp<<i<<"\t"<<scan_ref->observed_planes[i]->normal.transpose()<<std::endl;
			scan_ref->observed_planes[i]->pln_pointer=0;
		}
//		fp<<"pln matches "<<std::endl;
		for(size_t i=0;i<scan_cur->plane_matches.size();i++)
		{
			scan_cur->plane_matches[i].ref->pln_pointer=scan_cur->plane_matches[i].cur;
//			fp<<i<<std::endl<<scan_cur->plane_matches[i].ref->normal.transpose()<<std::endl<<scan_cur->plane_matches[i].ref->pln_pointer->normal.transpose()<<std::endl;
		}

		while(iterations<max_iter_icp && !converged)
		{
			if(debug)
			{
				fp<<std::endl<<"========================================"<<std::endl;
				fp<<"icp iteration - "<<iterations<<std::endl;
			}

			buildCorrespondence(scan_cur,scan_ref,Tcr,Tcr_pre, iterations);

			if(visual)
			{
				displayMatchedShadowPoints(scan_cur,Tcr,vis);
				vis->spin();
			}

			//if(scan_cur->plane_matches.size()==0 || scan_cur->point_matches.size()<10)
			//	return -1;
//			confirmPlaneCorrespondence(scan_cur,Tcr);

//			compute_EdgePoint_weight(scan_cur,Tcr);

//			if(debug)
//			{
//				fp<<"Weight_p - "<<Weight_p<<std::endl;
//				fp<<Lambda_pi.transpose()<<std::endl;
////				fp<<scan_cur->point_matches[0].weight<<"\t"<<scan_cur->point_matches[0].v_pk.transpose()<<std::endl;
//				for(size_t i=0;i<scan_cur->point_matches.size();i++)
//				{
//					fp<<scan_cur->point_matches[i].weight<<"\t"<<scan_cur->point_matches[i].v_pk(0)<<"\t"<<scan_cur->point_matches[i].v_pk(1)<<"\t"<<scan_cur->point_matches[i].v_pk(2)<<"\t"<<scan_cur->point_matches[i].v_pk(3)<<"\t"<<scan_cur->point_matches[i].v_pk(4)<<"\t"<<scan_cur->point_matches[i].v_pk(5)<<std::endl;
//				}
//			}

			chi2_pre=chi2;
			chi2=optimize(scan_cur,Tcr);
			std::cout<<iterations<<" - "<<chi2<<std::endl;
			
			Transform Tcr_gt=scan_cur->Tcg_gt*scan_ref->Tcg_gt.inv();

			error=0;
//			fp<<"error_occluding"<<std::endl;
			for(size_t i=0;i<scan_cur->point_matches.size();i++)
			{
				Eigen::Vector3d vec3d=scan_cur->point_matches[i].cur->xyz;
				vec3d=vec3d-Tcr.transformPoint(scan_cur->point_matches[i].ref->xyz);
//				vec3d=vec3d-Tcr.transformPoint(scan_cur->point_matches[i].ref->xyz);
//				vec3d=vec3d-scan_cur->point_matches[i].ref->xyz;
				error+=vec3d.norm();
//				fp<<i<<"\t"<<vec3d.norm()<<std::endl;
			}
			error/=scan_cur->point_matches.size();
			std::cout<<"error_occluding\t"<<error<<std::endl;

			error=0;
//			fp<<"error_occluding"<<std::endl;
			for(size_t i=0;i<scan_cur->point_matches.size();i++)
			{
				Eigen::Vector3d vec3d=scan_cur->point_matches[i].cur->xyz;
				vec3d=vec3d-Tcr_gt.transformPoint(scan_cur->point_matches[i].ref->xyz);
//				vec3d=vec3d-Tcr.transformPoint(scan_cur->point_matches[i].ref->xyz);
//				vec3d=vec3d-scan_cur->point_matches[i].ref->xyz;
				error+=vec3d.norm();
//				fp<<i<<"\t"<<vec3d.norm()<<std::endl;
			}
			error/=scan_cur->point_matches.size();
			std::cout<<"error_occluding_gt\t"<<error<<std::endl;


			error=0;
//			fp<<"error_occluded"<<std::endl;
			for(size_t i=0;i<scan_cur->point_matches_occluded.size();i++)
			{
				if(scan_cur->point_matches_occluded[i].cur->plane==0)
					continue;
				Eigen::Vector3d vec3d=scan_cur->point_matches_occluded[i].cur->plane->projected_point(Tcr.transformPoint(scan_cur->point_matches_occluded[i].ref->xyz));
//				Eigen::Vector3d vec3d=scan_cur->point_matches_occluded[i].ref->xyz;
				vec3d=vec3d-scan_cur->point_matches_occluded[i].cur->xyz;
				error+=vec3d.norm();
//				fp<<i<<"\t"<<vec3d.norm()<<std::endl;
			}
			error/=scan_cur->point_matches_occluded.size();
			std::cout<<"error_occluded\t"<<error<<std::endl;

			error=0;
//			fp<<"error_occluded"<<std::endl;
			for(size_t i=0;i<scan_cur->point_matches_occluded.size();i++)
			{
				if(scan_cur->point_matches_occluded[i].cur->plane==0)
					continue;
				Eigen::Vector3d vec3d=scan_cur->point_matches_occluded[i].cur->plane->projected_point(Tcr_gt.transformPoint(scan_cur->point_matches_occluded[i].ref->xyz));
//				Eigen::Vector3d vec3d=scan_cur->point_matches_occluded[i].ref->xyz;
				vec3d=vec3d-scan_cur->point_matches_occluded[i].cur->xyz;
				error+=vec3d.norm();
//				fp<<i<<"\t"<<vec3d.norm()<<std::endl;
			}
			error/=scan_cur->point_matches_occluded.size();
			std::cout<<"error_occluded_gt\t"<<error<<std::endl;

//			displayEdgePoints(scan_cur,vis);
//			vis->spin();
			if(visual)
			{
				displayMatchedShadowPoints(scan_cur,Tcr,vis);
				vis->spin();
			}

			iterations++;
			if(chi2>chi2_pre)
				converged=true;
		}

//		fig->clearFigures();
//		fig->setPenColor(0,0,0,200);
//		fig->addLine(0,200,600,200);
		compute_Psi(scan_cur,Tcr);

//		if(debug)
		{
			Transform Trc=Tcr.inv();
			Eigen::Matrix<double,6,1> xi_rc=Trc.getMotionVector();
			Transform Trc_gt=scan_ref->Tcg_gt*scan_cur->Tcg_gt.inv();
			Eigen::Matrix<double,6,1> xi_rc_gt=Trc_gt.getMotionVector();
			Eigen::Matrix<double,6,1> delta_xi=xi_rc_gt-xi_rc;
			Eigen::Matrix<double,6,1> error_distr;
			Eigen::Matrix<double,6,1> error_distr_p;
			Eigen::Matrix<double,6,1> error_distr_pi;
			for(size_t i=0;i<6;i++)
			{
	//			Lambda(i)
				Eigen::Matrix<double,6,1> q=Q.block<6,1>(0,i);
				Eigen::Matrix<double,6,1> q_p=Q_p.block<6,1>(0,i);
				Eigen::Matrix<double,6,1> q_pi=Q_pi.block<6,1>(0,i);
				error_distr(i)=fabs(delta_xi.dot(q));
				error_distr_p(i)=fabs(delta_xi.dot(q_p));
				error_distr_pi(i)=fabs(delta_xi.dot(q_pi));
			}
			fp<<"Lambda\t"<<Lambda.transpose()<<std::endl;
			fp<<"error \t"<<error_distr.transpose()<<std::endl;
			fp<<"Lambda_p\t"<<Lambda_p.transpose()<<std::endl;
			fp<<"error_p \t"<<error_distr_p.transpose()<<std::endl;
			fp<<"Lambda_pi\t"<<Lambda_pi.transpose()<<std::endl;
			fp<<"error_pi \t"<<error_distr_pi.transpose()<<std::endl;
			fp<<"Q_pi"<<std::endl<<Q_pi<<std::endl;
			fp<<std::endl;
		}
//		scan_cur->Tcr=Tcr;

		fp.close();
//		fp_debug.close();



		return error;
	}



	double PoseEstimation::optimize(Scan *scan_cur, Transform &Tcr)
	{
		typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> Block;
		g2o::SparseOptimizer optimizer;
		optimizer.setVerbose(debug);
		std::unique_ptr<Block::LinearSolverType> linearSolver;
		//if (DENSE) {
		linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
		//} else {
		//linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<Block::PoseMatrixType>>();
		//}
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( g2o::make_unique<Block>(std::move(linearSolver)) );
//		g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( g2o::make_unique<Block>(std::move(linearSolver)) );
		optimizer.setAlgorithm(solver);

		g2o::VertexSE3Expmap* pose0 = new g2o::VertexSE3Expmap();
		g2o::SE3Quat Tcg(Tcr.R,Tcr.t);
		pose0->setEstimate(Tcg.inverse());
		pose0->setId(0);
		pose0->setFixed(false);
		optimizer.addVertex(pose0);

		g2o::VertexSE3Expmap* pose1 = new g2o::VertexSE3Expmap();
		g2o::SE3Quat Trg(Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero());
		pose1->setEstimate(Trg.inverse());
		pose1->setId(1);
		pose1->setFixed(true);
		optimizer.addVertex(pose1);

//		if(usePln)
//		for(size_t i=0;i<scan_cur->plane_matches.size();i++)
//		{
//			if(scan_cur->plane_matches[i].cur==0)
//				continue;
//			MeasurementPlane measure_pln;
//			measure_pln.pln_cur.block<3,1>(0,0)=scan_cur->plane_matches[i].cur->normal;
//			measure_pln.pln_cur(3)=scan_cur->plane_matches[i].cur->d;
//			measure_pln.pln_ref.block<3,1>(0,0)=scan_cur->plane_matches[i].ref->normal;
//			measure_pln.pln_ref(3)=scan_cur->plane_matches[i].ref->d;
//
//			Edge_Plane *edge=new Edge_Plane();
//			edge->setVertex(0,pose0);
//			edge->setVertex(1,pose1);
//			edge->setMeasurement(measure_pln);
//			edge->setInformation(scan_cur->plane_matches[i].computeCov(Tcr).inverse());
//			edge->setId(i);
//			optimizer.addEdge(edge);
//		}

		int count=0;
		if(usePln)
		for(size_t i=0;i<scan_cur->point_matches.size();i++)
		{
//			if(scan_cur->point_matches[i].weight<thres_weight)
//				continue;
			MeasurementPoint measure_pt;
			measure_pt.pt_cur=scan_cur->point_matches[i].cur->xyz;
			measure_pt.pt_ref=scan_cur->point_matches[i].ref->xyz;

			Edge_EdgePoint *edge=new Edge_EdgePoint();
			edge->setVertex(0,pose0);
			edge->setVertex(1,pose1);
			edge->setMeasurement(measure_pt);
			edge->setInformation(scan_cur->point_matches[i].cur->cov.inverse());
//			edge->setInformation(Eigen::Matrix3d::Identity());
			edge->setId(i);
			optimizer.addEdge(edge);
			count++;
//			edge->computeError();
//			Eigen::Vector3d tmp = edge->error();
//			fp<<i<<"\t"<<edge->chi2()<<"\t"<<tmp.transpose()<<std::endl;
		}
		std::cout<<"\toptimizing edge-points "<<count<<std::endl;
		if(debug)
			fp<<"optimizing edge-points "<<count<<std::endl;

		count=0;
		if(usePt)
		for(size_t i=0;i<scan_cur->point_matches_occluded.size();i++)
		{
			if(scan_cur->point_matches_occluded[i].cur->plane==0)
				continue;
//			if(scan_cur->point_matches[i].occluded_match==false)
//				continue;

//			std::cout<<scan_cur->point_matches[i].cur->xyz.transpose()<<std::endl;
//			std::cout<<scan_cur->point_matches[i].ref->xyz.transpose()<<std::endl;
//			std::cout<<scan_cur->point_matches[i].cur->occluded->xyz.transpose()<<std::endl;
//			std::cout<<scan_cur->point_matches[i].ref->occluded->xyz.transpose()<<std::endl;
//			std::cout<<scan_cur->point_matches[i].cur->occluded->plane->normal.transpose()<<std::endl;

			MeasurementPoint_occluded measure_pt(scan_cur->point_matches_occluded[i].cur, scan_cur->point_matches_occluded[i].ref);
//			std::cout<<measure_pt.point_pair.occluded_match<<"\t"<<measure_pt.point_pair.cur<<"\t"<<measure_pt.point_pair.ref<<"\t"<<measure_pt.point_pair.weight<<std::endl;
//			measure_pt.point_pair=scan_cur->point_matches[i];
//			measure_pt.cur=scan_cur->point_matches[i].cur;
//			measure_pt.ref=scan_cur->point_matches[i].ref;

//			std::cout<<"measurement"<<std::endl;
//			std::cout<<measure_pt.cur->xyz.transpose()<<std::endl;
//			std::cout<<measure_pt.ref->xyz.transpose()<<std::endl;
//			std::cout<<measure_pt.cur->occluded->xyz.transpose()<<std::endl;
//			std::cout<<measure_pt.ref->occluded->xyz.transpose()<<std::endl;
//			std::cout<<measure_pt.cur->occluded->plane->normal.transpose()<<std::endl;

			Edge_EdgePoint_occluded *edge=new Edge_EdgePoint_occluded();
			edge->setVertex(0,pose0);
			edge->setVertex(1,pose1);
			edge->setMeasurement(measure_pt);
			edge->setInformation(scan_cur->point_matches_occluded[i].cur->cov.inverse());
//			edge->setInformation(Eigen::Matrix3d::Identity());
			edge->setId(scan_cur->point_matches.size()+i);
			optimizer.addEdge(edge);
			count++;

//			edge->computeError();
//			Eigen::Vector3d tmp = edge->error();
//			fp<<"\t"<<scan_cur->point_matches.size()+i<<"\t"<<edge->chi2()<<"\t"<<tmp.transpose()<<std::endl;
//			measure_pt=edge->Measurement();
//			fp<<std::endl<<i<<std::endl;
////			tmp=Tcr.transformPoint(measure_pt.ref->xyz);
//			fp<<"ref_occluding\t"<<measure_pt.ref->xyz.transpose()<<"\t"<<measure_pt.ref_occluding_trans.transpose()<<std::endl;
//			fp<<"cur_occluding\t"<<measure_pt.cur->xyz.transpose()<<std::endl;
//			fp<<"cur_occluded \t"<<measure_pt.cur->occluded->xyz.transpose()<<std::endl;
//			fp<<"ref_occluded_trans\t"<<measure_pt.ref_occluded_trans.transpose()<<std::endl;
//			fp<<"cur_plane\t"<<measure_pt.cur->occluded->plane->normal.transpose()<<std::endl;
//			Eigen::Vector3d tmp = edge->error();
//			Eigen::Vector3d tmp1 = measure_pt.ref_occluded_trans-measure_pt.cur->occluded->xyz;
//			fp<<"error \t"<<tmp1.transpose()<<"\t"<<tmp.transpose()<<std::endl;
//			fp<<measure_pt.Tcr_<<std::endl;
//			fp<<measure_pt.Tcr_*measure_pt.ref->xyz<<std::endl;
		}
		std::cout<<"\toptimizing occluded edge-points "<<count<<std::endl;
		if(debug)
			fp<<"optimizing occluded edge-points "<<count<<std::endl;

		optimizer.initializeOptimization();
		int result=optimizer.optimize ( max_iter_g2o );

		Tcg = pose0->estimate().inverse();
		Tcr.R = Tcg.rotation().toRotationMatrix();
		Tcr.t = Tcg.translation();

		double chi2=optimizer.activeChi2();
//		for(optimizer.activeEdges()
		optimizer.clear(); // delete vertices and edges;

		return chi2;
	}


	void PoseEstimation::buildCorrespondence(Scan *scan_cur, Scan *scan_ref, Transform Tcr, Transform Tcr_pre, int iterations)
	{
		double thresh=thres_association*thres_association/(double)(iterations+1);

		ANNkd_tree *kdtree;
		ANNpoint cur_point=annAllocPt(3);
		ANNpointArray ref_points=annAllocPts(scan_ref->edge_points.size(),3);
		ANNidxArray index=new ANNidx[1];
		ANNdistArray distance=new ANNdist[1];

		// build kdtree with scan_ref->edge_points;
		for(size_t i=0;i<scan_ref->edge_points.size();i++)
		{
			Eigen::Vector3d pr=Tcr.transformPoint(scan_ref->edge_points[i]->xyz);
			ref_points[i][0]=pr(0);
			ref_points[i][1]=pr(1);
			ref_points[i][2]=pr(2);
		}
		kdtree=new ANNkd_tree(ref_points,scan_ref->edge_points.size(),3);

		// fill scan_cur->point_matches;
		scan_cur->point_matches.clear();
		for(size_t i=0;i<scan_cur->edge_points.size();i++)
		{
			if(!scan_cur->edge_points[i]->isEdge)
				continue;
			cur_point[0]=scan_cur->edge_points[i]->xyz(0);
			cur_point[1]=scan_cur->edge_points[i]->xyz(1);
			cur_point[2]=scan_cur->edge_points[i]->xyz(2);
			kdtree->annkSearch(cur_point,1,index,distance);
			if(distance[0]<thresh && scan_ref->edge_points[index[0]]->isEdge)
			{
				EdgePointPair match(scan_cur->edge_points[i],scan_ref->edge_points[index[0]]);
				scan_cur->point_matches.push_back(match);
			}
		}
		
//		ANNkd_tree *kdtree_occluded;
//		ANNpoint cur_point_occluded=annAllocPt(3);
//		ANNpointArray ref_points_occluded=annAllocPts(scan_ref->edge_points_occluded.size(),3);
//		for(size_t i=0;i<scan_ref->edge_points_occluded.size();i++)
//		{
//			if(scan_ref->edge_points_occluded[i]->occluding==0 || scan_ref->edge_points_occluded[i]->plane==0)
//			{
//				ref_points_occluded[i][0]=0;
//				ref_points_occluded[i][1]=0;
//				ref_points_occluded[i][2]=0;
//				continue;
//			}
//			if(scan_ref->edge_points_occluded[i]->plane->pln_pointer==0)
//			{
//				ref_points_occluded[i][0]=0;
//				ref_points_occluded[i][1]=0;
//				ref_points_occluded[i][2]=0;
//				continue;
//			}
//			Eigen::Vector3d tmp=scan_ref->edge_points_occluded[i]->plane->pln_pointer->projected_point(Tcr.transformPoint(scan_ref->edge_points_occluded[i]->occluding->xyz));
//			ref_points_occluded[i][0]=tmp(0);
//			ref_points_occluded[i][1]=tmp(1);
//			ref_points_occluded[i][2]=tmp(2);
//		}
//		kdtree_occluded=new ANNkd_tree(ref_points_occluded,scan_ref->edge_points_occluded.size(),3);

		// fill scan_cur->point_matches_occluded;
		scan_cur->point_matches_occluded.clear();
		for(size_t i=0;i<scan_cur->edge_points_occluded.size();i++)
		{
			if(!scan_cur->edge_points_occluded[i]->isEdge || scan_cur->edge_points_occluded[i]->occluding==0 || scan_cur->edge_points_occluded[i]->plane==0)
				continue;
			cur_point[0]=scan_cur->edge_points_occluded[i]->occluding->xyz(0);
			cur_point[1]=scan_cur->edge_points_occluded[i]->occluding->xyz(1);
			cur_point[2]=scan_cur->edge_points_occluded[i]->occluding->xyz(2);
			kdtree->annkSearch(cur_point,1,index,distance);
			if(distance[0]<thresh*2.0 && scan_ref->edge_points[index[0]]->isEdge)
			{
				EdgePointPair match(scan_cur->edge_points_occluded[i],scan_ref->edge_points[index[0]]);
				scan_cur->point_matches_occluded.push_back(match);
			}
		}

		delete kdtree;
//		delete kdtree_occluded;
		delete index;
		delete distance;
		annDeallocPt(cur_point);
//		annDeallocPt(cur_point_occluded);
		annDeallocPts(ref_points);
//		annDeallocPts(ref_points_occluded);
	}

	void PoseEstimation::confirmPlaneCorrespondence(Scan *scan_cur, Transform &Tcr)
	{
		typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> Block;
		g2o::SparseOptimizer optimizer;
		optimizer.setVerbose(true);
		std::unique_ptr<Block::LinearSolverType> linearSolver;
		linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( g2o::make_unique<Block>(std::move(linearSolver)) );
		optimizer.setAlgorithm(solver);

		g2o::VertexSE3Expmap* pose0 = new g2o::VertexSE3Expmap();
		g2o::SE3Quat Tcg(Tcr.R,Tcr.t);
		pose0->setEstimate (Tcg.inverse());
		pose0->setId ( 0 );
		pose0->setFixed(false);
		optimizer.addVertex ( pose0 );

		g2o::VertexSE3Expmap* pose1 = new g2o::VertexSE3Expmap();
		g2o::SE3Quat Trg(Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero());
		pose1->setEstimate (Trg.inverse());
		pose1->setId ( 1 );
		pose1->setFixed(true);
		optimizer.addVertex ( pose1 );

		for(size_t i=0;i<scan_cur->plane_matches.size();i++)
		{
			if(scan_cur->plane_matches[i].cur==0)
				continue;
			MeasurementPlane measure_pln;
			measure_pln.pln_cur.block<3,1>(0,0)=scan_cur->plane_matches[i].cur->normal;
			measure_pln.pln_cur(3)=scan_cur->plane_matches[i].cur->d;
			measure_pln.pln_ref.block<3,1>(0,0)=scan_cur->plane_matches[i].ref->normal;
			measure_pln.pln_ref(3)=scan_cur->plane_matches[i].ref->d;

			Edge_Plane *edge=new Edge_Plane();
			edge->setVertex(0,pose0);
			edge->setVertex(1,pose1);
			edge->setMeasurement(measure_pln);
			edge->setInformation(scan_cur->plane_matches[i].cur->cov_inv);
			edge->setId(i);
			optimizer.addEdge(edge);
		}

		optimizer.initializeOptimization();
		optimizer.optimize ( max_iter_g2o );

		std::vector<double> pln_errors;
		double pln_error_mean=0;
		for(size_t i=0;i<scan_cur->plane_matches.size();i++)
		{
			double tmp=optimizer.activeEdges()[i]->chi2();
			pln_errors.push_back(tmp);
			pln_error_mean+=tmp;
		}
		pln_error_mean/=pln_errors.size();
		std::cout<<"pln_error_mean "<<pln_error_mean<<std::endl;
		double pln_error_sigma=0;
		for(size_t i=0;i<pln_errors.size();i++)
		{
			pln_error_sigma+=(pln_errors[i]-pln_error_mean)*(pln_errors[i]-pln_error_mean);
		}
		pln_error_sigma/=(pln_errors.size());
		pln_error_sigma=sqrt(pln_error_sigma);
		for(size_t i=0;i<pln_errors.size();i++)
		{
			if(pln_errors[i]-pln_error_mean>3*pln_error_sigma)
			{
				scan_cur->plane_matches[i].cur=0;
			}
		}

		optimizer.clear(); 
	}


	void PoseEstimation::compute_Psi(Scan *scan, Transform Tcr)
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

	void PoseEstimation::compute_EdgePoint_weight(Scan *scan, Transform Tcr)
	{
//		if(debug)
//			fp<<"compute_EdgePoint_weight"<<std::endl;
		Eigen::Matrix<double,3,6> J_pk;
		Eigen::Matrix<double,6,1> v_pk_sum=Eigen::Matrix<double,6,1>::Zero();
//		double sq_lambda_pk;
		for(size_t k=0;k<scan->point_matches.size();k++)
		{
			Eigen::Vector3d pc=scan->point_matches[k].cur->xyz;
			Eigen::Vector3d pr=Tcr.transformPoint(scan->point_matches[k].ref->xyz);
			Eigen::Matrix3d cov_inv=scan->point_matches[k].cur->cov.inverse();

			J_pk.block<3,3>(0,0)=Eigen::Matrix3d::Identity();
			J_pk.block<3,3>(0,3)=g2o::skew(Tcr.R*pr);
			scan->point_matches[k].Psi_pk=J_pk.transpose()*cov_inv*J_pk;

			for(size_t j=0;j<6;j++)
			{
				scan->point_matches[k].v_pk(j)=Q_pi.block<6,1>(0,j).transpose()*scan->point_matches[k].Psi_pk*Q_pi.block<6,1>(0,j);
			}
//			std::cout<<scan->point_matches[k].v_pk.transpose()<<std::endl;
			v_pk_sum+=scan->point_matches[k].v_pk;

		}
		for(size_t k=0;k<scan->point_matches.size();k++)
		{
			if(!useWeight)
			{
				scan->point_matches[k].weight=1;
				continue;
			}

			scan->point_matches[k].weight=0;
			for(size_t l=0;l<6;l++)
			{
				double nu_pkl=scan->point_matches[k].v_pk(l)/v_pk_sum(l);
				double nu_pil=exp(alpha*Lambda_pi(l)/lambda_pi_1);
				scan->point_matches[k].weight+=nu_pkl/nu_pil;
			}
//			scan->point_matches[k].weight/=6;
		}
		double num=0,dem=0;
		for(size_t l=0;l<6;l++)
		{
			num+=Lambda_pi(l);
			for(size_t k=0;k<scan->point_matches.size();k++)
			{
				dem+=scan->point_matches[k].weight*scan->point_matches[k].v_pk(l);
			}
		}
		Weight_p=num/dem;
		
	}
}

