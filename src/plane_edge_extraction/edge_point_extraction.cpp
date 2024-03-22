/*==============================================
#
# Author: Sun Qinxuan
#
# Email: sunqinxuan@outlook.com
#
# Last modified:	2019-01-19 21:46
#
# Filename:		edge_point_extraction.cpp
#
# Description: 
#
===============================================*/

#include "edge_point_extraction.h"
#include <pcl-1.8/pcl/filters/extract_indices.h>

namespace ulysses
{
	void EdgePointExtraction::extractEdgePoints(Scan *scan)
	{
		fp.open("extract_EdgePoints.txt",std::ios::app);
		if(debug)
			fp<<std::endl<<"******************************************************************"<<std::endl;

//		segmentPlanes(scan);
		// 							0    1    2    3    4    5    6    7    8    9   10   11   12   13
//		unsigned char red [14] = {255,   0,   0, 255, 255,   0, 130,   0,   0, 130, 130,   0, 130, 255};
//		unsigned char grn [14] = {  0, 255,   0, 255,   0, 255,   0, 130,   0, 130,   0, 130, 130, 255};
//		unsigned char blu [14] = {  0,   0, 255,   0, 255, 255,   0,   0, 130,   0, 130, 130, 130, 255};
		// 							0    1    2    3    4    5    6    7    8    9   10   11   12   13
		unsigned char red [11] = {255, 255,   0, 130,   0,   0, 130, 130,   0, 130, 255};
		unsigned char grn [11] = {255,   0, 255,   0, 130,   0, 130,   0, 130, 130, 255};
		unsigned char blu [11] = {  0, 255, 255,   0,   0, 130,   0, 130, 130, 130, 255};

		// edge
		labels_edge=pcl::PointCloud<pcl::Label>::Ptr (new pcl::PointCloud<pcl::Label>);
		std::vector<pcl::PointIndices> edge_indices;

		// for edge detection;
		// change the invalid depth in scan->point_cloud from zero to infinite;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGBA>);
		*cloud_tmp=*scan->point_cloud;
		for(size_t i=0;i<cloud_tmp->height;i++)
		{
			for(size_t j=0;j<cloud_tmp->width;j++)
			{
				double dep=cloud_tmp->points[cloud_tmp->width*i+j].z;
				if(std::abs(dep)<1e-4)
				{
					cloud_tmp->points[cloud_tmp->width*i+j].z=std::numeric_limits<double>::max();
				}
			}
		}

		// edge detection;
		if (getEdgeType () & EDGELABEL_HIGH_CURVATURE)
		{
			pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>::setInputNormals(scan->normal_cloud);
		}
		pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>::setInputCloud(cloud_tmp);
		pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>::compute(*labels_edge, edge_indices);


			std:cout<<"plane num "<<num_plane<<std::endl;
//			for (size_t i=0;i<labels_plane->height;i++)
//			{
//				for (size_t j=0;j<labels_plane->width;j++)
//				{
//					fp<<labels_plane->at(j,i).label<<"\t";
//				}
//				fp<<std::endl;
//			}

		if(debug)
		{
			fp<<"organized edge detection "<<std::endl;
			fp<<"\tEDGELABEL_NAN_BOUNDARY - "<<edge_indices[0].indices.size()<<std::endl;
			fp<<"\tEDGELABEL_OCCLUDING - "<<edge_indices[1].indices.size()<<std::endl;
			fp<<"\tEDGELABEL_OCCLUDED - "<<edge_indices[2].indices.size()<<std::endl;
			fp<<"\tEDGELABEL_HIGH_CURVATURE - "<<edge_indices[3].indices.size()<<std::endl;
			fp<<"\tEDGELABEL_RGB_CANNY - "<<edge_indices[4].indices.size()<<std::endl;
		}

		// scan->edge_points;
		// fitting local line segment of each edge point;
		ANNkd_tree *kdtree; // kdtree built by occluding points;
		ANNpoint query_point=annAllocPt(3);
		ANNidxArray index=new ANNidx[K_ANN];
		ANNdistArray distance=new ANNdist[K_ANN];
		// edge_indices[1] - occluding points;
		// edge_indices[2] - occluded points;
		ANNpointArray edge_points=annAllocPts(edge_indices[1].indices.size(),3);
		scan->edge_points.resize(edge_indices[1].indices.size());
		// kdtree build by the pixel coordinates of the occluding points;
		// used for the association of occluding and occluded points;
		ANNkd_tree *kdtree_pixel;
		ANNidxArray index_pixel=new ANNidx[K_ANN];
		ANNdistArray distance_pixel=new ANNdist[K_ANN];
		ANNpointArray edge_points_pixel=annAllocPts(edge_indices[1].indices.size(),2);
		ANNpoint query_point_pixel=annAllocPt(2);
		if(debug)
		{
			fp<<"occluding points"<<std::endl;
		}
		for(size_t i=0;i<edge_indices[1].indices.size();i++)
		{
			int idx=edge_indices[1].indices[i];
			edge_points[i][0]=scan->point_cloud->at(idx).x;
			edge_points[i][1]=scan->point_cloud->at(idx).y;
			edge_points[i][2]=scan->point_cloud->at(idx).z;
			edge_points_pixel[i][0]=scan->pixel_cloud->at(idx).x;
			edge_points_pixel[i][1]=scan->pixel_cloud->at(idx).y;
			// fill scan->edge_points;
			scan->edge_points[i]=new EdgePoint;
			scan->edge_points[i]->xyz(0)=scan->point_cloud->at(idx).x;
			scan->edge_points[i]->xyz(1)=scan->point_cloud->at(idx).y;
			scan->edge_points[i]->xyz(2)=scan->point_cloud->at(idx).z;
			scan->edge_points[i]->pixel(0)=scan->pixel_cloud->at(idx).x;
			scan->edge_points[i]->pixel(1)=scan->pixel_cloud->at(idx).y;
			scan->edge_points[i]->rgb(0)=0;
			scan->edge_points[i]->rgb(1)=0;
			scan->edge_points[i]->rgb(2)=0;
			scan->edge_points[i]->index=idx;
			if(debug)
			{
//				fp<<"\t"<<i<<" - "<<scan->edge_points[i]->xyz.transpose()<<std::endl;
			}
		}
		// build the kd-tree using the occluding edge points;
		kdtree=new ANNkd_tree(edge_points,scan->edge_points.size(),3);
		// build the kd-tree using the pixel coordinates of the occluding edge points;
		kdtree_pixel=new ANNkd_tree(edge_points_pixel,scan->edge_points.size(),2);
		// for each occluding edge point;
		// search for the nearest neighbor in the occluding edge points;
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es;
		Eigen::Vector3d Lambda;
		Eigen::Matrix3d U;
		Eigen::Vector3d e1;
		e1.setOnes();
		for(size_t i=0;i<scan->edge_points.size();i++)
		{
			query_point[0]=scan->edge_points[i]->xyz(0);
			query_point[1]=scan->edge_points[i]->xyz(1);
			query_point[2]=scan->edge_points[i]->xyz(2);
			int points_in_radius=kdtree->annkFRSearch(query_point,sqRad_ANN,K_ANN,index,distance);
			//	ANNpoint q, // query point
			//	ANNdist sqRad, // squared radius
			//	int k = 0, // number of near neighbors to return
			//	ANNidxArray nn_idx = NULL, // nearest neighbor array (modified)
			//	ANNdistArray dd = NULL, // dist to near neighbors (modified)
			//	double eps = 0.0); // error bound
			if(points_in_radius>7)
			{
				scan->edge_points[i]->isEdge=true;
				for(size_t j=0;j<K_ANN;j++)
				{
					if(index[j]==ANN_NULL_IDX)
						continue;
					scan->edge_points[i]->neighbors.push_back(scan->edge_points[index[j]]);
				}
				Eigen::Vector3d mean=Eigen::Vector3d::Zero();
				for(size_t j=0;j<scan->edge_points[i]->neighbors.size();j++)
				{
					mean+=scan->edge_points[i]->neighbors[j]->xyz;
				}
				mean=mean/scan->edge_points[i]->neighbors.size();
				scan->edge_points[i]->cov.setZero();
				for(size_t j=0;j<scan->edge_points[i]->neighbors.size();j++)
				{
					Eigen::Vector3d vec3d=scan->edge_points[i]->neighbors[j]->xyz-mean;
					scan->edge_points[i]->cov+=vec3d*vec3d.transpose();
				}
				scan->edge_points[i]->cov=scan->edge_points[i]->cov/(scan->edge_points[i]->neighbors.size()-1);
				if(scan->edge_points[i]->cov.determinant()<1e-20)
				{
					scan->edge_points[i]->isEdge=false;
				}
				else
				{
					es.compute(scan->edge_points[i]->cov);
					Lambda=es.eigenvalues();
					U=es.eigenvectors();
//					fp<<Lambda.transpose()<<"\t"<<Lambda(2)/Lambda(0)<<"\t"<<Lambda(2)/Lambda(1)<<"\t"<<Lambda(1)/Lambda(2)/0.01+(Lambda(1)/Lambda(0)-1)/5.0<<std::endl;
					scan->edge_points[i]->meas_Edge=Lambda(1)/Lambda(2)/0.01+(Lambda(1)/Lambda(0)-1)/5.0;
					if(scan->edge_points[i]->meas_Edge>edge_meas)
					{
						scan->edge_points[i]->isEdge=false;
					}
					else
					{
						Eigen::Vector3d u1=U.block<3,1>(0,2);
						Eigen::Vector3d u2=U.block<3,1>(0,1);
						Eigen::Vector3d u3=U.block<3,1>(0,0);
						scan->edge_points[i]->cov=u2*u2.transpose()+u3*u3.transpose();
						if(u1.transpose()*e1<0)
							scan->edge_points[i]->dir=-u1;
						else
							scan->edge_points[i]->dir=u1;
					}
				}
			}
		}
		// occluded points;
		// shadow on the plane;
//		kdtree_pixel=new ANNkd_tree(edge_points_pixel,edge_indices[1].indices.size(),2);
		if(debug)
		{
			fp<<"occluded points"<<std::endl;
		}
		ANNkd_tree *kdtree_occluded;
		ANNidxArray index_occluded=new ANNidx[K_ANN];
		ANNdistArray distance_occluded=new ANNdist[K_ANN];
		ANNpointArray edge_points_occluded=annAllocPts(edge_indices[2].indices.size(),3);
//		kdtree_occluded=new ANNkd_tree(edge_points_pixel,edge_indices[2].indices.size(),3);
		scan->edge_points_occluded.resize(edge_indices[2].indices.size());
		for(size_t i=0;i<edge_indices[2].indices.size();i++)
		{
			int idx=edge_indices[2].indices[i];
			edge_points_occluded[i][0]=scan->point_cloud->at(idx).x;
			edge_points_occluded[i][1]=scan->point_cloud->at(idx).y;
			edge_points_occluded[i][2]=scan->point_cloud->at(idx).z;
			// fill scan->edge_points_occluded;
			scan->edge_points_occluded[i]=new EdgePoint;
			scan->edge_points_occluded[i]->xyz(0)=scan->point_cloud->at(idx).x;
			scan->edge_points_occluded[i]->xyz(1)=scan->point_cloud->at(idx).y;
			scan->edge_points_occluded[i]->xyz(2)=scan->point_cloud->at(idx).z;
			scan->edge_points_occluded[i]->pixel(0)=scan->pixel_cloud->at(idx).x;
			scan->edge_points_occluded[i]->pixel(1)=scan->pixel_cloud->at(idx).y;
			scan->edge_points_occluded[i]->index=idx;
//			scan->edge_points_occluded[i]->occluding=0;
//			scan->edge_points_occluded[i]->plane=0;
			if(debug)
			{
//				fp<<"\t"<<i<<" - "<<scan->edge_points_occluded[i]->xyz.transpose()<<std::endl;
			}
		}
		// build the kd-tree using the occluded edge points;
		kdtree_occluded=new ANNkd_tree(edge_points_occluded,scan->edge_points_occluded.size(),3);
		// for each occluded edge point;
		// search for the nearest 3D neighbor in the occluded edge points;
		// and search for the nearest 3D neighbor in the image plane for the occluding point;
		for(size_t i=0;i<scan->edge_points_occluded.size();i++)
		{
//			// search for nearest occluded point in the 3D space;
//			query_point[0]=scan->edge_points_occluded[i]->xyz(0);
//			query_point[1]=scan->edge_points_occluded[i]->xyz(1);
//			query_point[2]=scan->edge_points_occluded[i]->xyz(2);
//			int points_in_radius=kdtree_occluded->annkFRSearch(query_point,sqRad_ANN,K_ANN,index_occluded,distance_occluded);
//			if(points_in_radius>7)
//			{
//				scan->edge_points_occluded[i]->isEdge=true;
//				for(size_t j=0;j<K_ANN;j++)
//				{
//					if(index_occluded[j]==ANN_NULL_IDX)
//						continue;
//					scan->edge_points_occluded[i]->neighbors.push_back(scan->edge_points_occluded[index_occluded[j]]);
//				}
//				Eigen::Vector3d mean=Eigen::Vector3d::Zero();
//				for(size_t j=0;j<scan->edge_points_occluded[i]->neighbors.size();j++)
//				{
//					mean+=scan->edge_points_occluded[i]->neighbors[j]->xyz;
//				}
//				mean=mean/scan->edge_points_occluded[i]->neighbors.size();
//				scan->edge_points_occluded[i]->cov.setZero();
//				for(size_t j=0;j<scan->edge_points_occluded[i]->neighbors.size();j++)
//				{
//					Eigen::Vector3d vec3d=scan->edge_points_occluded[i]->neighbors[j]->xyz-mean;
//					scan->edge_points_occluded[i]->cov+=vec3d*vec3d.transpose();
//				}
//				scan->edge_points_occluded[i]->cov=scan->edge_points_occluded[i]->cov/(scan->edge_points_occluded[i]->neighbors.size()-1);
//				if(scan->edge_points_occluded[i]->cov.determinant()<1e-20)
//					scan->edge_points_occluded[i]->isEdge=false;
//			}
			// search for nearest occluding point in the image plane;
			query_point_pixel[0]=scan->edge_points_occluded[i]->pixel(0);
			query_point_pixel[1]=scan->edge_points_occluded[i]->pixel(1);
			kdtree_pixel->annkSearch(query_point_pixel,1,index_pixel,distance_pixel,0);
			EdgePoint *occluding=scan->edge_points[index_pixel[0]];
			EdgePoint *occluded=scan->edge_points_occluded[i];
			if(distance_pixel[0]<thres_pxl_sq && occluding->isEdge)// && occluded->isEdge)
			{
				scan->edge_points_occluded[i]->occluding=scan->edge_points[index_pixel[0]];
				scan->edge_points_occluded[i]->occluding->occluded=scan->edge_points_occluded[i];

				double dist_min=DBL_MAX;
				int idx_min=-1;
				for(size_t j=0;j<scan->observed_planes.size();j++)
				{
					Eigen::Vector3d tmp=occluded->xyz-scan->observed_planes[j]->projected_point(occluding->xyz);
					double dist=tmp.norm();
					if(dist<dist_min) {dist_min=dist;idx_min=j;}
				}
				if(dist_min<thres_occluded_dist)
				{
					double ratio=occluding->xyz.norm()/occluded->xyz.norm();
					Eigen::Matrix<double,1,1> tmp=scan->observed_planes[idx_min]->normal.transpose()*occluding->xyz;
					double angle=-tmp(0,0)/occluding->xyz.norm();
//					fp<<ratio<<"\t"<<angle<<std::endl;
					if(ratio>thres_ratio && angle>thres_angle)
					{
						ProjectiveRay *proj_ray=new ProjectiveRay(occluding,occluded);
						proj_ray->plane=scan->observed_planes[idx_min];
						proj_ray->occluded_proj=new EdgePoint;
						proj_ray->occluded_proj->xyz=proj_ray->plane->projected_point(occluding->xyz);
						scan->projective_rays.push_back(proj_ray);
					}
//					fp<<dist_min<<std::endl;
				}

//				if(debug)
//				{
//					fp<<i<<"\t"<<index_pixel[0]<<"\t"<<distance_pixel[0]
//						 <<"\n\t"<<scan->edge_points_occluded[i]->xyz.transpose()
//						 <<"\n\t"<<scan->edge_points_occluded[i]->occluding->xyz.transpose()<<std::endl;
//				}
			}
//			else
//			{
//				scan->edge_points_occluded[i]->occluding=0;
////				if(debug)
////				{
////					fp<<i<<"\t"<<index_pixel[0]<<"\t"<<distance_pixel[0]
////						 <<"\n\t"<<scan->edge_points_occluded[i]->xyz.transpose()
////						 <<"\n\t"<<scan->edge_points_occluded[i]->occluding<<std::endl;
////				}
//			}
			for(size_t j=0;j<scan->observed_planes.size();j++)
			{
//				fp<<scan->observed_planes[j]->point_plane_dist(scan->edge_points_occluded[i]->xyz)<<"\t";
				if(scan->observed_planes[j]->point_plane_dist(scan->edge_points_occluded[i]->xyz)<=0.1)
				{
					if(scan->edge_points_occluded[i]->plane==0)
					{
						scan->edge_points_occluded[i]->plane=scan->observed_planes[j];
					}
					else if(scan->edge_points_occluded[i]->plane->point_plane_dist(scan->edge_points_occluded[i]->xyz)>scan->observed_planes[j]->point_plane_dist(scan->edge_points_occluded[i]->xyz))
					{
						scan->edge_points_occluded[i]->plane=scan->observed_planes[j];
					}
				}
			}
//			if(scan->edge_points_occluded[i]->plane==0)
//				fp<<"---\t"<<scan->edge_points_occluded[i]->plane;
//			else
//				fp<<"---\t"<<scan->edge_points_occluded[i]->plane->point_plane_dist(scan->edge_points_occluded[i]->xyz);
//			fp<<std::endl;
		}



		/*
		fitLinesHough(scan->edge_points,scan->lines_occluding,EDGELABEL_OCCLUDING);
		std::cout<<"fitted lines in occluding points: "<<scan->lines_occluding.size()<<std::endl;
		for(std::list<Line*>::iterator it_line=scan->lines_occluding.begin();it_line!=scan->lines_occluding.end();it_line++)
		{
			fitLinesLS(*it_line);
			std::cout<<"\t"<<(*it_line)->m<<"\t"<<(*it_line)->n<<"\t"<<(*it_line)->x0<<"\t"<<(*it_line)->y0<<std::endl;

			double z=(*it_line)->end_point_1(2);
			(*it_line)->end_point_1(0)=(*it_line)->m*z+(*it_line)->x0;
			(*it_line)->end_point_1(1)=(*it_line)->n*z+(*it_line)->y0;
			z=(*it_line)->end_point_2(2);
			(*it_line)->end_point_2(0)=(*it_line)->m*z+(*it_line)->x0;
			(*it_line)->end_point_2(1)=(*it_line)->n*z+(*it_line)->y0;

			(*it_line)->end_point_1_img=scan->cam.project((*it_line)->end_point_1);
			(*it_line)->end_point_2_img=scan->cam.project((*it_line)->end_point_2);
			double x1=(*it_line)->end_point_1_img(0);
			double y1=(*it_line)->end_point_1_img(1);
			double x2=(*it_line)->end_point_2_img(0);
			double y2=(*it_line)->end_point_2_img(1);
			(*it_line)->a=(y1-y2)/(x1*y2-x2*y1);
			(*it_line)->b=(x2-x1)/(x1*y2-x2*y1);
		}

		fitLinesHough(scan->edge_points_occluded,scan->lines_occluded,EDGELABEL_OCCLUDED);
		std::cout<<"fitted lines in occluded points: "<<scan->lines_occluded.size()<<std::endl;
		for(std::list<Line*>::iterator it_line=scan->lines_occluded.begin();it_line!=scan->lines_occluded.end();it_line++)
		{
			fitLinesLS(*it_line);
			std::cout<<"\t"<<(*it_line)->m<<"\t"<<(*it_line)->n<<"\t"<<(*it_line)->x0<<"\t"<<(*it_line)->y0<<std::endl;

			double z=(*it_line)->end_point_1(2);
			(*it_line)->end_point_1(0)=(*it_line)->m*z+(*it_line)->x0;
			(*it_line)->end_point_1(1)=(*it_line)->n*z+(*it_line)->y0;
			z=(*it_line)->end_point_2(2);
			(*it_line)->end_point_2(0)=(*it_line)->m*z+(*it_line)->x0;
			(*it_line)->end_point_2(1)=(*it_line)->n*z+(*it_line)->y0;

			(*it_line)->end_point_1_img=scan->cam.project((*it_line)->end_point_1);
			(*it_line)->end_point_2_img=scan->cam.project((*it_line)->end_point_2);
			double x1=(*it_line)->end_point_1_img(0);
			double y1=(*it_line)->end_point_1_img(1);
			double x2=(*it_line)->end_point_2_img(0);
			double y2=(*it_line)->end_point_2_img(1);
			(*it_line)->a=(y1-y2)/(x1*y2-x2*y1);
			(*it_line)->b=(x2-x1)/(x1*y2-x2*y1);
		}

		for(std::list<Line*>::iterator it_occluding=scan->lines_occluding.begin();it_occluding!=scan->lines_occluding.end();it_occluding++)
		{
			double x1=(*it_occluding)->end_point_1_img(0);
			double y1=(*it_occluding)->end_point_1_img(1);
			double x2=(*it_occluding)->end_point_2_img(0);
			double y2=(*it_occluding)->end_point_2_img(1);
			for(std::list<Line*>::iterator it_occluded=scan->lines_occluded.begin();it_occluded!=scan->lines_occluded.end();it_occluded++)
			{
				double a=(*it_occluded)->a;
				double b=(*it_occluded)->b;
				double d1=fabs(a*x1+b*y1+1)/(sqrt(a*a+b*b));
				double d2=fabs(a*x2+b*y2+1)/(sqrt(a*a+b*b));
				std::cout<<d1<<","<<d2<<"\t";
				if(d1<10 && d2<10)
				{
					(*it_occluding)->occluded.push_back(*it_occluded);
				}
			}
			std::cout<<std::endl;
		}

		for(std::list<Line*>::iterator it_occluding=scan->lines_occluding.begin();it_occluding!=scan->lines_occluding.end();it_occluding++)
		{
			if((*it_occluding)->occluded.size()==0)
				continue;
			for(std::list<EdgePoint*>::iterator it=(*it_occluding)->points.begin();it!=(*it_occluding)->points.end();it++)
			{
				(*it)->xyz(0)=(*it_occluding)->m*(*it)->xyz(2)+(*it_occluding)->x0;
				(*it)->xyz(1)=(*it_occluding)->n*(*it)->xyz(2)+(*it_occluding)->y0;
			}
			for(size_t i=0;i<(*it_occluding)->occluded.size();i++)
			{
				for(std::list<EdgePoint*>::iterator it =(*it_occluding)->occluded[i]->points.begin();
													it!=(*it_occluding)->occluded[i]->points.end();it++)
				{
					(*it)->xyz(0)=(*it_occluding)->m*(*it)->xyz(2)+(*it_occluding)->x0;
					(*it)->xyz(1)=(*it_occluding)->n*(*it)->xyz(2)+(*it_occluding)->y0;
				}
			}
		}
		*/
		

		if(debug)
		{
//			fp<<"extracted edge Points - "<<scan->edge_points.size()<<std::endl;
//			for(size_t i=0;i<scan->edge_points.size();i++)
//			{
//				fp<<i<<" - "<<scan->edge_points[i]->xyz.transpose();
//				if(scan->edge_points[i]->isEdge)
//				{
//					fp<<"\tneighbors - "<<scan->edge_points[i]->neighbors.size()<<std::endl;
//					fp<<"\tcov - "<<std::endl<<scan->edge_points[i]->cov<<std::endl;
//				}
//			}
		}
		annDeallocPt(query_point);
		annDeallocPt(query_point_pixel);
		annDeallocPts(edge_points);
		annDeallocPts(edge_points_pixel);
		annDeallocPts(edge_points_occluded);
		delete kdtree;
		delete kdtree_pixel;
		delete kdtree_occluded;
		delete index;
		delete index_pixel;
		delete index_occluded;
		delete distance;
		delete distance_pixel;
		delete distance_occluded;
		fp.close();
	}

	void EdgePointExtraction::fitLinesLS(Line *line)
	{
		int N=line->points.size();
		Eigen::Matrix2d A,B;
		A.setZero();
		B.setZero();
		B(1,1)=N;
		for(std::list<EdgePoint*>::iterator it=line->points.begin();it!=line->points.end();it++)
		{
			A(0,0)+=(*it)->xyz(0)*(*it)->xyz(2);
			A(0,1)+=(*it)->xyz(0);
			A(1,0)+=(*it)->xyz(1)*(*it)->xyz(2);
			A(1,1)+=(*it)->xyz(1);
			B(0,0)+=(*it)->xyz(2)*(*it)->xyz(2);
			B(0,1)+=(*it)->xyz(2);
			B(1,0)+=(*it)->xyz(2);
		}
		Eigen::Matrix2d param=A*B.inverse();
		line->m=param(0,0);
		line->n=param(1,0);
		line->x0=param(0,1);
		line->y0=param(1,1);
	}

	void EdgePointExtraction::fitLinesHough(std::vector<EdgePoint*>& edge_points, std::list<Line*>& lines_occluding, unsigned int EDGE_LABEL)
	{
//		fp.open("extract_EdgePoints.txt",std::ios::app);
		fp<<"in the fitLinesHough function ###############################"<<std::endl;
//		cv::Mat img_occluding=cv::Mat::zeros(480,640,CV_8UC3);
		cv::Mat img_occluding=cv::Mat::zeros(540,960,CV_8UC3);
		for (size_t i=0;i<labels_edge->height;i++)
		{
			for (size_t j=0;j<labels_edge->width;j++)
			{
//					for(size_t k=0;k<num_plane;k++)
//					{
//						if(labels_plane->at(j,i).label==k)
//						{
//							img_occluding.at<cv::Vec3b>(i,j)[0]=blu[k];
//							img_occluding.at<cv::Vec3b>(i,j)[1]=grn[k];
//							img_occluding.at<cv::Vec3b>(i,j)[2]=red[k];
//						}
//					}
				if(labels_edge->at(j,i).label==EDGE_LABEL)//occluding
				{
					img_occluding.at<cv::Vec3b>(i,j)[0]=255;//blue
					img_occluding.at<cv::Vec3b>(i,j)[1]=0;
					img_occluding.at<cv::Vec3b>(i,j)[2]=0;
				}
//					if(labels_edge->at(j,i).label==4)//occluded
//					{
//						img_occluding.at<cv::Vec3b>(i,j)[0]=0;
//						img_occluding.at<cv::Vec3b>(i,j)[1]=0;
//						img_occluding.at<cv::Vec3b>(i,j)[2]=255;//red
//						for(size_t k=0;k<num_plane;k++)
//						{
//							if(labels_plane->at(j,i).label==k)
//							{
//								img_occluding.at<cv::Vec3b>(i,j)[0]=0;
//								img_occluding.at<cv::Vec3b>(i,j)[1]=255;
//								img_occluding.at<cv::Vec3b>(i,j)[2]=0;
//							}
//						}
//					}
			}
		}
//		cv::imshow("img_occluding",img_occluding);
//		cv::waitKey(0);
		cv::Mat contours=cv::Mat::zeros(480,640,CV_8UC1);
		for(size_t i=0;i<edge_points.size();i++)
		{
			int x=edge_points[i]->pixel(1);
			int y=edge_points[i]->pixel(0);
			contours.at<unsigned char>(x,y)=255;
		}

		// extract 2D lines in image using Hough transform (OpenCV);
		// void HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0 )
		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP(contours, lines, rho, theta, threshold, minLineLength, maxLineGap);
		std::vector<Line*> lines_tmp;
		lines_tmp.resize(lines.size());
//		for(std::vector<cv::Vec4i>::iterator it=lines.begin();it!=lines.end();it++)
		for(size_t i=0;i<lines.size();i++)
		{
			cv::Point pt1(lines[i][0],lines[i][1]);
			cv::Point pt2(lines[i][2],lines[i][3]);
			cv::line(img_occluding, pt1, pt2, CV_RGB(0,255,0));
			double x1=lines[i][0];
			double y1=lines[i][1];
			double x2=lines[i][2];
			double y2=lines[i][3];
			double tmp=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
			double a=(y1-y2)/tmp;
			double b=(x2-x1)/tmp;
			double d=(x1*y2-x2*y1)/tmp;
			Line *line=new Line;
			for(size_t j=0;j<edge_points.size();j++)
			{
				int x=edge_points[j]->pixel(0);
				int y=edge_points[j]->pixel(1);
				if(fabs(a*x+y*b+d)<5.0 && edge_points[j]->onLine==false)
				{
					edge_points[j]->onLine=true;
					line->points_tmp.push_back(edge_points[j]);
					img_occluding.at<cv::Vec3b>(int(y),int(x))[0]=0;
					img_occluding.at<cv::Vec3b>(int(y),int(x))[1]=0;
					img_occluding.at<cv::Vec3b>(int(y),int(x))[2]=255;//red
				}
			}
			// lines_occluding.push_back(line);
			lines_tmp[i]=line;
		}
		//cv::imshow("img_occluding",img_occluding);
		//cv::waitKey(0);

		// 3D RANSAC (PCL);
		// merge similar lines;
		// same them in ordered list;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
		Eigen::VectorXf line_coeff;
		std::vector<int> line_inliers;
		Eigen::Matrix<double,1,1> tmp;
		bool merge=false;
		for(size_t i=0;i<lines_tmp.size();i++)
		{
			fp<<i<<"-th line in lines_tmp "<<lines_tmp[i]->points_tmp.size()<<std::endl;
			if(lines_tmp[i]->points_tmp.size()<=10)
			{
				delete lines_tmp[i];
				continue;
			}
			merge=false;
			cloud->points.resize(lines_tmp[i]->points_tmp.size());
			for(size_t j=0;j<lines_tmp[i]->points_tmp.size();j++)
			{
				cloud->points[j].x=lines_tmp[i]->points_tmp[j]->xyz(0);
				cloud->points[j].y=lines_tmp[i]->points_tmp[j]->xyz(1);
				cloud->points[j].z=lines_tmp[i]->points_tmp[j]->xyz(2);
//				fp<<lines_tmp[i]->points[j]->xyz.transpose()<<std::endl;
			}
			line_inliers.clear();
			pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr model_line
					(new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud));
			pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_line);
			ransac.setDistanceThreshold (.01);
//			model_line->setInputCloud(cloud);
//			ransac.setSampleConsensusModel(model_line);
			ransac.computeModel();
			ransac.getInliers(line_inliers);
			fp<<"inliers "<<line_inliers.size()<<std::endl;
			ransac.getModelCoefficients(line_coeff);
			fp<<"coefficients "<<line_coeff.transpose()<<std::endl;
			lines_tmp[i]->end_point_1(0)=line_coeff(0);
			lines_tmp[i]->end_point_1(1)=line_coeff(1);
			lines_tmp[i]->end_point_1(2)=line_coeff(2);
			lines_tmp[i]->end_point_2(0)=line_coeff(0);
			lines_tmp[i]->end_point_2(1)=line_coeff(1);
			lines_tmp[i]->end_point_2(2)=line_coeff(2);
			lines_tmp[i]->dir(0)=line_coeff(3);
			lines_tmp[i]->dir(1)=line_coeff(4);
			lines_tmp[i]->dir(2)=line_coeff(5);
			if(debug)
			{
				fp<<"========================================="<<std::endl;
				fp<<i<<"-th line "<<line_coeff.transpose()<<std::endl;
			}
			for(std::list<Line*>::iterator it_line=lines_occluding.begin();
										   it_line!=lines_occluding.end();it_line++)
			{
				double sim_dir =lines_tmp[i]->similarity_dir (*it_line);
				double sim_dist=lines_tmp[i]->similarity_dist(*it_line);
				if(sim_dir<thres_sim_dir && sim_dist<thres_sim_dist)
				{
					if(debug)
					{
						fp<<"similar with a line in lines_occluding"<<std::endl;
						fp<<"sim_dir ="<<sim_dir<<std::endl;
						fp<<"sim_dist="<<sim_dist<<std::endl;
					}
					// merge [i] and [j];
//					for(size_t k=0;k<lines_tmp[i]->points.size();k++)
					for(size_t k=0;k<line_inliers.size();k++)
					{
						tmp=(*it_line)->dir.transpose()
							*(lines_tmp[i]->points_tmp[line_inliers[k]]->xyz-(*it_line)->end_point_2);
						if(tmp(0,0)>0)
						{
							(*it_line)->points.push_back(lines_tmp[i]->points_tmp[line_inliers[k]]);
						}
						else
						for(std::list<EdgePoint*>::iterator it=(*it_line)->points.begin();
															it!=(*it_line)->points.end();it++)
						{
							tmp=(*it_line)->dir.transpose()
								*(lines_tmp[i]->points_tmp[line_inliers[k]]->xyz-(*it)->xyz);
							if(tmp(0,0)<0)
							{
								(*it_line)->points.insert(it,lines_tmp[i]->points_tmp[line_inliers[k]]);
								break;
							}
						}
						std::list<EdgePoint*>::iterator it=(*it_line)->points.begin();
						(*it_line)->end_point_1=(*it)->xyz;
//						(*it_line)->end_point_1_img=(*it)->pixel;
						it=(*it_line)->points.end(); it--;
						(*it_line)->end_point_2=(*it)->xyz;
//						(*it_line)->end_point_2_img=(*it)->pixel;
					}
					Eigen::Vector3d vec3d=(*it_line)->end_point_1-(*it_line)->end_point_2;
					(*it_line)->length=vec3d.norm();
					// update lines_occluding[j]->dir here;
					delete lines_tmp[i];
					merge=true;
					break;
				}
			}
			if(merge) continue;
			// else add a new line in lines_occluding
			if(debug)
				fp<<"add a new line in lines_occluding"<<std::endl;
			Line *line=new Line;
//			line->points.resize(line_inliers.size());
//			line->end_point_1=lines_tmp[i]->end_point_1;
//			line->end_point_2=lines_tmp[i]->end_point_2;
			line->dir=lines_tmp[i]->dir;
//			fp<<"inlier size "<<line_inliers.size()<<std::endl; // more than 10;
			for(size_t k=0;k<line_inliers.size();k++)
			{
				if(line->points.size()==0)
				{
					line->points.push_back(lines_tmp[i]->points_tmp[line_inliers[k]]);
					line->end_point_1=lines_tmp[i]->points_tmp[line_inliers[k]]->xyz;
					line->end_point_2=lines_tmp[i]->points_tmp[line_inliers[k]]->xyz;
//					fp<<"first pushed point "<<lines_tmp[i]->points_tmp[line_inliers[k]]->xyz.transpose()<<std::endl;
				}
				else
				{
					tmp=line->dir.transpose()*(lines_tmp[i]->points_tmp[line_inliers[k]]->xyz-line->end_point_2);
					if(tmp(0,0)>0)
					{
						line->points.push_back(lines_tmp[i]->points_tmp[line_inliers[k]]);
//						fp<<"at the end "<<lines_tmp[i]->points_tmp[line_inliers[k]]->xyz.transpose()<<"\t"<<tmp<<std::endl;
					}
					else
					for(std::list<EdgePoint*>::iterator it=line->points.begin();it!=line->points.end();it++)
					{
						tmp=line->dir.transpose()*(lines_tmp[i]->points_tmp[line_inliers[k]]->xyz-(*it)->xyz);
						if(tmp(0,0)<0)
						{
							line->points.insert(it,lines_tmp[i]->points_tmp[line_inliers[k]]);
//							fp<<"in the middle "<<lines_tmp[i]->points_tmp[line_inliers[k]]->xyz.transpose()<<"\t"<<(*it)->xyz.transpose()<<std::endl;
							break;
						}
					}
					std::list<EdgePoint*>::iterator it=line->points.begin();
					line->end_point_1=(*it)->xyz;
//					line->end_point_1_img=(*it)->pixel;
					it=line->points.end(); it--;
					line->end_point_2=(*it)->xyz;
//					line->end_point_2_img=(*it)->pixel;
//					Eigen::Vector3d pt=lines_tmp[i]->points_tmp[line_inliers[k]]->xyz;
//					Eigen::Vector3d ep1=line->end_point_1;
//					Eigen::Vector3d ep2=line->end_point_2;
//					Eigen::Vector3d dir=line->dir;
//					tmp=dir.transpose()*(pt-ep1);
//					if(tmp(0,0)<0) line->end_point_1=pt;
//					tmp=dir.transpose()*(pt-ep2);
//					if(tmp(0,0)>0) line->end_point_2=pt;
				}
			}
			Eigen::Vector3d vec3d=line->end_point_1-line->end_point_2;
			line->length=vec3d.norm();
			lines_occluding.push_back(line);
			delete lines_tmp[i];

			if(debug)
			{
				std::list<Line*>::iterator itt_line=std::prev(lines_occluding.end());
//				size_t ii=lines_occluding.size()-1;
				fp<<(*itt_line)->end_point_1.transpose()<<std::endl;
				for(std::list<EdgePoint*>::iterator it=(*itt_line)->points.begin();it!=(*itt_line)->points.end();it++)
				{
					fp<<"\t"<<(*it)->xyz.transpose()<<"\t"<<(*itt_line)->dir.transpose()*((*it)->xyz-(*itt_line)->end_point_1)<<std::endl;
				}
				fp<<(*itt_line)->end_point_2.transpose()<<std::endl;
			}
		}

//		for(size_t i=0;i<lines_occluding.size();i++)
		fp<<std::endl<<"splitting lines "<<std::endl;
		for(std::list<Line*>::iterator it_line=lines_occluding.begin();
									   it_line!=lines_occluding.end();it_line++)
		{
			fp<<"\t"<<lines_occluding.size()<<std::endl;
			for(std::list<EdgePoint*>::iterator it=(*it_line)->points.begin();it!=(*it_line)->points.end();it++)
			{
				if(it==(*it_line)->points.begin()) continue;
				std::list<EdgePoint*>::iterator it_pre=it;
				it_pre--;
				Eigen::Vector3d vec3d=(*it)->xyz-(*it_pre)->xyz;
				if(vec3d.norm()>thres_split)
				{
//					(*it_line)->end_point_2=(*it_pre)->xyz;
					Line *line=new Line;
					line->points.splice(line->points.begin(),(*it_line)->points,it,(*it_line)->points.end());
//					std::list<EdgePoint*>::iterator itt=line->points.begin();
//					line->end_point_1=(*itt)->xyz;
//					line->end_point_2=(*std::prev(line->points.end()))->xyz;
//					vec3d=line->end_point_1-line->end_point_2;
//					line->length=vec3d.norm();
//					line->dir=(*it_line)->dir;
					lines_occluding.push_back(line);
					break;
				}
			}
		}

		fp<<std::endl<<"cunning the lines "<<std::endl;
		for(std::list<Line*>::iterator it_line=lines_occluding.begin();
									   it_line!=lines_occluding.end();)
		{
			fp<<"\t"<<(*it_line)->points.size()<<std::endl;
			// cunning the lines that does not contain enough points;
			if((*it_line)->points.size()<min_points_on_line)
			{
				delete *it_line;
				it_line=lines_occluding.erase(it_line);
			}
			else
			{
				std::list<EdgePoint*>::iterator itt=(*it_line)->points.begin();
				(*it_line)->end_point_1=(*itt)->xyz;
				(*it_line)->end_point_2=(*std::prev((*it_line)->points.end()))->xyz;
				Eigen::Vector3d vec3d=(*it_line)->end_point_1-(*it_line)->end_point_2;
				(*it_line)->length=vec3d.norm();
				(*it_line)->dir=(*it_line)->end_point_2-(*it_line)->end_point_1;
				(*it_line)->dir.normalize();
				it_line++;
			}
		//	compute the dir
		}
//		fp.close();


//		cv::Mat img_occluded=cv::Mat::zeros(480,640,CV_8UC3);
//		for (size_t i=0;i<labels_edge->height;i++)
//		{
//			for (size_t j=0;j<labels_edge->width;j++)
//			{
//				if(labels_edge->at(j,i).label==4)//occluded
//				{
//					img_occluded.at<cv::Vec3b>(i,j)[0]=255;
//					img_occluded.at<cv::Vec3b>(i,j)[1]=0;
//					img_occluded.at<cv::Vec3b>(i,j)[2]=0;//red
//				}
//			}
//		}
//		contours=cv::Mat::zeros(480,640,CV_8UC1);
//		for (size_t i=0;i<labels_edge->height;i++)
//		{
//			for (size_t j=0;j<labels_edge->width;j++)
//			{
//				if(labels_edge->at(j,i).label==4)
//				{
//					contours.at<unsigned char>(i,j)=255;
//				}
//			}
//		}
//		//void HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0 )
//		lines.clear();
//		cv::HoughLinesP(contours, lines, rho, theta, threshold, minLineLength, maxLineGap);

	}

	void EdgePointExtraction::fitLines(Scan *scan)
	{
		fp.open("extract_EdgePoints.txt",std::ios::app);
		int bins_theta=10,bins_phy=20;
//		std::vector<Eigen::Vector2d> dirs_sphere;
//		dirs_sphere.resize(scan->edge_points.size());
		Eigen::Vector2d dir;
		std::vector<std::vector<EdgePoint*> > cells;
		cells.resize(bins_theta*bins_phy);
		std::vector<std::vector<EdgePoint*> > lines;
		for(size_t i=0;i<scan->edge_points.size();i++)
		{
			if(scan->edge_points[i]->isEdge==false)
				continue;
			dir(0)=acos(scan->edge_points[i]->dir(2));
			dir(1)=atan2(scan->edge_points[i]->dir(1),scan->edge_points[i]->dir(0));
			int row=dir(0)/(M_PI/bins_theta);
			int col=(dir(1)+M_PI)/(M_PI*2.0/bins_phy);
			int index=bins_phy*row+col;
			cells[index].push_back(scan->edge_points[i]);
		}
//		for(size_t i=0;i<cells.size();i++)
//		{
//			int row=i/bins_phy;
//			int col=i%bins_phy;
//			if(col==0) fp<<std::endl;
//			fp<<cells[i].size()<<"\t";
//		}
		std::vector<Sorted_Cell> sorted_cells;
		sorted_cells.resize(cells.size());
		for(size_t i=0;i<cells.size();i++)
		{
			sorted_cells[i].index=i;
			sorted_cells[i].num_point=cells[i].size();
		}
		std::sort(sorted_cells.begin(),sorted_cells.end());
		std::vector<Sorted_Cell>::iterator iter_sorted_cells=sorted_cells.end()-1;
		unsigned char red [14] = {255,   0,   0, 255, 255,   0, 130,   0,   0, 130, 130,   0, 130, 255};
		unsigned char grn [14] = {  0, 255,   0, 255,   0, 255,   0, 130,   0, 130,   0, 130, 130, 255};
		unsigned char blu [14] = {  0,   0, 255,   0, 255, 255,   0,   0, 130,   0, 130, 130, 130, 255};
		int count=0;
		timeval time_seed;

		while(true)
		{
//			fp<<"cell "<<count<<" - "<<iter_sorted_cells->num_point<<std::endl;
			int idx=iter_sorted_cells->index;
//			Eigen::Vector3d n0;
//			n0.setZero();
//			for(size_t i=0;i<cells[idx].size();i++)
//			{
////				cells[idx][i]->rgb(0)=red[count];
////				cells[idx][i]->rgb(1)=grn[count];
////				cells[idx][i]->rgb(2)=blu[count];
//				n0+=cells[idx][i]->xyz;
////				fp<<"\t"<<cells[idx][i]->xyz.transpose()<<"\t"<<cells[idx][i]->rgb.transpose()<<std::endl;
//			}
//			n0/=iter_sorted_cells->num_point;
			int iter=0;
			while(true)
			{
//				srand((int)time(0));
				gettimeofday(&time_seed,NULL);
				srand((int)time_seed.tv_usec);
				int index=(int)(rand()%iter_sorted_cells->num_point);
				Eigen::Vector3d p0=cells[idx][index]->xyz;
				Eigen::Vector3d n0=cells[idx][index]->dir;
				std::vector<EdgePoint*> tmp_line;
				for(size_t i=0;i<scan->edge_points.size();i++)
				{
					if(scan->edge_points[i]->isEdge==false)
						continue;
					if(scan->edge_points[i]->rgb.norm()>0)
						continue;
					Eigen::Vector3d p=scan->edge_points[i]->xyz;
					Eigen::Matrix<double,1,1> mu_tmp=n0.transpose()*(p-p0);
					double mu=mu_tmp(0,0)/(n0.norm()*n0.norm());
					p=p-mu*n0-p0;
					mu=p.norm();
					double ang=n0.transpose()*scan->edge_points[i]->dir;
					if(mu<0.03 && ang>0.9)
						tmp_line.push_back(scan->edge_points[i]);
				}
				int thres=(int(iter_sorted_cells->num_point/500.0)+1)*100;
//				fp<<"\t"<<time_seed.tv_usec<<"\t"<<index<<"\t"<<thres<<"\t"<<tmp_line.size()<<std::endl;
				if(tmp_line.size()>=thres)
				{
					for(size_t i=0;i<tmp_line.size();i++)
					{
						tmp_line[i]->rgb(0)=red[count];
						tmp_line[i]->rgb(1)=grn[count];
						tmp_line[i]->rgb(2)=blu[count];
//						fp<<"\t\t"<<tmp_line[i]->xyz.transpose()<<std::endl;
					}
					count++;
					lines.push_back(tmp_line);
					break;
				}
				if(iter>20)
					break;
				iter++;
			}
			if(iter_sorted_cells==sorted_cells.begin()) break;
			iter_sorted_cells--;
			if(iter_sorted_cells->num_point<50) break;
		}
		fp.close();
	}

	bool EdgePointExtraction::fitSphere(EdgePoint *edge_point)
	{
		fp.open("extract_EdgePoints.txt",std::ios::app);
		Eigen::Vector3d center;
		double radius;

		double x_bar=0,y_bar=0,z_bar=0;
		double xy_bar=0,xz_bar=0,yz_bar=0;
		double x2_bar=0,y2_bar=0,z2_bar=0;
		double x2y_bar=0,x2z_bar=0,xy2_bar=0,y2z_bar=0,xz2_bar=0,yz2_bar=0;
		double x3_bar=0,y3_bar=0,z3_bar=0;
		for(size_t i=0;i<edge_point->neighbors.size();i++)
		{
			double x=edge_point->neighbors[i]->xyz(0);
			double y=edge_point->neighbors[i]->xyz(1);
			double z=edge_point->neighbors[i]->xyz(2);
			x_bar  +=x;
			y_bar  +=y;
			z_bar  +=z;
			xy_bar +=x*y;
			xz_bar +=x*z;
			yz_bar +=y*z;
			x2_bar +=x*x;
			y2_bar +=y*y;
			z2_bar +=z*z;
			x2y_bar+=x*x*y;
			x2z_bar+=x*x*z;
			xy2_bar+=x*y*y;
			y2z_bar+=y*y*z;
			xz2_bar+=x*z*z;
			yz2_bar+=y*y*z;
			x3_bar +=x*x*x;
			y3_bar +=y*y*y;
			z3_bar +=z*z*z;
		}
		x_bar  /=edge_point->neighbors.size();
		y_bar  /=edge_point->neighbors.size();
		z_bar  /=edge_point->neighbors.size();
		xy_bar /=edge_point->neighbors.size();
		xz_bar /=edge_point->neighbors.size();
		yz_bar /=edge_point->neighbors.size();
		x2_bar /=edge_point->neighbors.size();
		y2_bar /=edge_point->neighbors.size();
		z2_bar /=edge_point->neighbors.size();
		x2y_bar/=edge_point->neighbors.size();
		x2z_bar/=edge_point->neighbors.size();
		xy2_bar/=edge_point->neighbors.size();
		y2z_bar/=edge_point->neighbors.size();
		xz2_bar/=edge_point->neighbors.size();
		yz2_bar/=edge_point->neighbors.size();
		x3_bar /=edge_point->neighbors.size();
		y3_bar /=edge_point->neighbors.size();
		z3_bar /=edge_point->neighbors.size();
		
		Eigen::Matrix3d A;
		A(0,0)=x2_bar-x_bar*x_bar;
		A(0,1)=xy_bar-x_bar*y_bar;
		A(0,2)=xz_bar-x_bar*z_bar;
		A(1,0)=xy_bar-x_bar*y_bar;
		A(1,1)=y2_bar-y_bar*y_bar;
		A(1,2)=yz_bar-y_bar*z_bar;
		A(2,0)=xz_bar-x_bar*z_bar;
		A(2,1)=yz_bar-y_bar*z_bar;
		A(2,2)=z2_bar-z_bar*z_bar;

		Eigen::Vector3d b;
		b(0)=(x3_bar -x_bar*x2_bar)+(xy2_bar-x_bar*y2_bar)+(xz2_bar-x_bar*z2_bar);
		b(1)=(x2y_bar-y_bar*x2_bar)+(y3_bar -y_bar*y2_bar)+(yz2_bar-y_bar*z2_bar);
		b(2)=(x2z_bar-z_bar*x2_bar)+(y2z_bar-z_bar*y2_bar)+(z3_bar -z_bar*z2_bar);
		b*=0.5;

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
		Eigen::Vector3d Lambda=es.eigenvalues();
		Eigen::Matrix3d U=es.eigenvectors();
		
//		fp<<std::endl<<edge_point->neighbors.size()<<"\t"<<Lambda.transpose()<<std::endl;
//		fp<<A<<std::endl;

		Eigen::Matrix3d A_inv;
		bool invertible;
		A.computeInverseWithCheck(A_inv,invertible);
		if(invertible)
		{
			center=A_inv*b;
			double x0=center(0);
			double y0=center(1);
			double z0=center(2);
			radius=sqrt(x2_bar-2*x0*x_bar+x0*x0
					   +y2_bar-2*y0*y_bar+y0*y0
					   +z2_bar-2*z0*z_bar+z0*z0);
			edge_point->meas_Edge=1.0/radius;
			fp.close();
			return true;
		}
		else
		{
			edge_point->meas_Edge=-1;
			fp.close();
			return false;
		}

//		edge_point->meas_Edge
	}



	void EdgePointExtraction::segmentPlanes(Scan *scan)
	{
		std::vector<pcl::PlanarRegion<pcl::PointXYZRGBA>, Eigen::aligned_allocator<pcl::PlanarRegion<pcl::PointXYZRGBA> > > regions;
		std::vector<pcl::PointIndices> inlier_indices;
		std::vector<pcl::ModelCoefficients> model_coefficients;
		labels_plane=pcl::PointCloud<pcl::Label>::Ptr (new pcl::PointCloud<pcl::Label>);
		std::vector<pcl::PointIndices> label_indices;
		std::vector<pcl::PointIndices> boundary_indices;

//		// segment
//		scan->segment_label_cloud=pcl::PointCloud<pcl::Label>::Ptr (new pcl::PointCloud<pcl::Label>);
//		scan->segment_indices.clear();
//		scan->segment_boundary_indices.clear();

		// plane
//		scan->planar_regions.clear();
//		scan->plane_indices.clear();

		// rgb_comparator
		//pcl::RGBPlaneCoefficientComparator<pcl::PointXYZRGBA, pcl::Normal>::Ptr rgb_comparator;
		//rgb_comparator.reset (new pcl::RGBPlaneCoefficientComparator<pcl::PointXYZRGBA, pcl::Normal> ());
		//rgb_comparator->setColorThreshold(20);
		//pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>::setComparator(rgb_comparator);

		// plane segmentation;
		pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>::setInputNormals(scan->normal_cloud);
		pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>::setInputCloud(scan->point_cloud);
		pcl::OrganizedMultiPlaneSegmentation<pcl::PointXYZRGBA, pcl::Normal, pcl::Label>::segmentAndRefine(regions, model_coefficients, inlier_indices, labels_plane, label_indices, boundary_indices);
		num_plane=regions.size();


//		std::cout<<"planar regions:"<<std::endl;
//		for(size_t i=0;i<scan->planar_regions.size();i++)
//		{
//			Eigen::Vector4f co=scan->planar_regions[i].getCoefficients();
//			std::cout<<co.transpose()<<std::endl;
//		}
//		std::cout<<"similarity between"<<std::endl;
//		for(size_t i=0;i<scan->planar_regions.size();i++)
//		{
//			for(size_t j=0;j<scan->planar_regions.size();j++)
//			{
//				Eigen::Vector4f coi=scan->planar_regions[i].getCoefficients();
//				Eigen::Vector4f coj=scan->planar_regions[j].getCoefficients();
//				std::cout<<coi.block<3,1>(0,0).transpose()*coj.block<3,1>(0,0)<<","<<fabs(fabs(coi(3))-fabs(coj(3)))<<"\t";
//			}
//			std::cout<<std::endl;
//		}

		// generate the plane labels;
//		scan->plane_label_cloud=pcl::PointCloud<pcl::Label>::Ptr (new pcl::PointCloud<pcl::Label>);
		pcl::Label invalid_pt;
		invalid_pt.label = unsigned (0);
		labels_plane->points.resize (scan->point_cloud->size(), invalid_pt);
		labels_plane->width = scan->point_cloud->width;
		labels_plane->height = scan->point_cloud->height;
		for(size_t i=0;i<labels_plane->points.size();i++)
		{
			labels_plane->points[i].label=0;
		}
		for(size_t i=0;i<inlier_indices.size();i++)
		{
			for(size_t j=0;j<inlier_indices[i].indices.size();j++)
			{
				labels_plane->at(inlier_indices[i].indices[j]).label=i+1;
			}
		}

		// save the planar regions to scan->observed_planes;
		scan->observed_planes.clear();
		for(size_t i=0;i<regions.size();i++)
		{
			Eigen::Vector3f centroid=regions[i].getCentroid();
			Eigen::Vector4f coefficients=regions[i].getCoefficients();
			Plane *plane=new Plane;
			plane->centroid(0)=centroid(0);
			plane->centroid(1)=centroid(1);
			plane->centroid(2)=centroid(2);
			if(coefficients(3)<0) coefficients=-coefficients;
			coefficients=coefficients/coefficients.block<3,1>(0,0).norm();
			if(debug)
				fp<<i<<"-th plane "<<coefficients.transpose()<<std::endl;
			plane->normal(0)=coefficients(0);
			plane->normal(1)=coefficients(1);
			plane->normal(2)=coefficients(2);
			plane->d=coefficients(3);
			plane->id=i;
			plane->inlier_indices=inlier_indices[i];
			scan->observed_planes.push_back(plane);
		}

	}

}

