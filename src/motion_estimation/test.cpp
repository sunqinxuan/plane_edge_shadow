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

#include "test.h"

using namespace std;
using namespace g2o;
using namespace Eigen;
//std::ofstream fp_debug;

namespace ulysses
{
// sampling distributions
  class Sample_test
  {

    static default_random_engine gen_real;
    static default_random_engine gen_int;
  public:
    static int uniform(int from, int to);

    static double uniform();

    static double gaussian(double sigma);
  };


  default_random_engine Sample_test::gen_real;
  default_random_engine Sample_test::gen_int;

  int Sample_test::uniform(int from, int to)
  {
    uniform_int_distribution<int> unif(from, to);
    int sam = unif(gen_int);
    return  sam;
  }

  double Sample_test::uniform()
  {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double sam = unif(gen_real);
    return  sam;
  }

  double Sample_test::gaussian(double sigma)
  {
    std::normal_distribution<double> gauss(0.0, sigma);
    double sam = gauss(gen_real);
    return  sam;
  }


void Test::test()
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
    true_points.push_back(Vector3d((Sample_test::uniform()-0.5)*3,
                                   Sample_test::uniform()-0.5,
                                   Sample_test::uniform()+10));
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
    pt0 += Vector3d(Sample_test::gaussian(euc_noise ),
                    Sample_test::gaussian(euc_noise ),
                    Sample_test::gaussian(euc_noise ));

    pt1 += Vector3d(Sample_test::gaussian(euc_noise ),
                    Sample_test::gaussian(euc_noise ),
                    Sample_test::gaussian(euc_noise ));

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

}

