// #include <boost.h>
#include <bits/stdc++.h>
// #include <open3d/Open3D.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <ros/ros.h>

using namespace std;

/*
 * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is
 * time stamp). Taken from LeGO LOAM.
 */
struct PointXYZIRPYT {
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRPYT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity,
                                            intensity)(float, roll,
                                                       roll)(float, pitch,
                                                             pitch)(float, yaw,
                                                                    yaw)(double,
                                                                         time,
                                                                         time))

typedef PointXYZIRPYT PointTypePose;

class GetSurroundingCloud {
 private:
  // constants
  std::string mapDir =
      "/home/shrinivas/KGP/SLAM/catkin_ws_indy/Outputs/StereoMap/";
  float resolution = 2.0;
  float radius = 40.0;
  int num_frames=36;

  // Point Clouds
  pcl::PointCloud<pcl::PointXYZI>::Ptr globalmap, localMap, inputMap;
  pcl::PointCloud<PointTypePose>::Ptr globalPose6D;

  vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> allFrames;

  // Oct Tree
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZI> octree;

  //----------------------------------------------------//
  //-------------- Utility Functions -------------------//
  // ---------------------------------------------------//
  Eigen::Matrix4f ConvertToTransformMatrix(float d[3], float q[3]) {
    Eigen::Matrix3f rotMat;

    rotMat = Eigen::AngleAxisf(q[0], Eigen::Vector3f::UnitX()) *
             Eigen::AngleAxisf(q[1], Eigen::Vector3f::UnitY()) *
             Eigen::AngleAxisf(q[2], Eigen::Vector3f::UnitZ());

    Eigen::Vector3f transMat;
    transMat << d[0], d[1], d[2];

    Eigen::Vector4f lastRow;
    lastRow << 0.0, 0.0, 0.0, 1.0;

    Eigen::Matrix4f transformMat;

    transformMat.block<3, 3>(0, 0) = rotMat;
    transformMat.block<3, 1>(0, 3) = transMat;
    transformMat.block<1, 4>(3, 0) = lastRow;

    return transformMat;
  }

  // Get Point Cloud from local filesystem and transform it
  pcl::PointCloud<pcl::PointXYZI>::Ptr getPointCloud(int index) {
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << index;
    std::string cloudDir = mapDir + ss.str() + "/cloud.pcd";
    pcl::PointCloud<pcl::PointXYZI>::Ptr tempCloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr output(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile<pcl::PointXYZI>(cloudDir, *tempCloud);
    float q[3], d[3];

    q[0] = globalPose6D->points[index].yaw;
    q[1] = globalPose6D->points[index].roll;
    q[2] = globalPose6D->points[index].pitch;

    d[0] = globalPose6D->points[index].z;  // z
    d[1] = globalPose6D->points[index].x;  // x
    d[2] = globalPose6D->points[index].y;  // y

    // cout<<q[0]<<" "<<q[1]<<" "<<q[2]<<"\n";
    // cout<<d[1]<<" "<<d[2]<<" "<<d[0]<<"\n";

    Eigen::Matrix4f transform = ConvertToTransformMatrix(d, q);

    pcl::transformPointCloud(*tempCloud, *output, transform);
    return output;
  }

  // Visualise a Point Cloud
  pcl::visualization::PCLVisualizer::Ptr simpleVis(
      pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud, string name) {
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer(name));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZI>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
  }

  // Get all surrounding transformed frames
  void getAllFrames(){
    for(int i=0;i<=num_frames;i++){
      allFrames.push_back(getPointCloud(i));
    }
  }

  //----------------------------------------------------//
  //-------------- Extractor Function ------------------//
  // ---------------------------------------------------//
  void extract_local_map(float X, float Y, float Z, float radius) {
    pcl::PointXYZI searchPoint;

    localMap.reset(new pcl::PointCloud<pcl::PointXYZI>());

    searchPoint.x = Y;
    searchPoint.y = Z;
    searchPoint.z = X;

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    if (octree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                            pointRadiusSquaredDistance) > 0) {
      cout << pointIdxRadiusSearch.size() << "\n";

      for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
        *localMap += *allFrames[pointIdxRadiusSearch[i]];
        // cout<<pointRadiusSquaredDistance[i]<<"\n";
      }
    }
  }

 public:
  //----------------------------------------------------//
  //-------------- Constructor -------------------------//
  // ---------------------------------------------------//
  GetSurroundingCloud() : octree(resolution) {
    globalmap.reset(new pcl::PointCloud<pcl::PointXYZI>);
    globalPose6D.reset(new pcl::PointCloud<PointTypePose>);
    localMap.reset(new pcl::PointCloud<pcl::PointXYZI>);
    inputMap.reset(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::io::loadPCDFile<pcl::PointXYZI>(mapDir + "cloudKeyPoses3D.pcd",
                                        *globalmap);
    pcl::io::loadPCDFile<PointTypePose>(mapDir + "cloudKeyPoses6D.pcd",
                                        *globalPose6D);

    localMap.reset(new pcl::PointCloud<pcl::PointXYZI>);
    inputMap.reset(new pcl::PointCloud<pcl::PointXYZI>);

    octree.setInputCloud(globalmap);
    octree.addPointsFromInputCloud();

    // getAllFrames();

    // SaveFullCloud();

    cout<<"All Frames Saved. Starting Code\n";
  }

  void SaveFullCloud(){
      pcl::PointCloud<pcl::PointXYZI>::Ptr fullCloud(new pcl::PointCloud<pcl::PointXYZI>);

      for(int i=0;i<=num_frames;i++){
        *fullCloud+=*getPointCloud(i);
      }

      pcl::io::savePCDFileASCII (mapDir +"Complete_Cloud.pcd", *fullCloud);
  }

  //----------------------------------------------------//
  //-------------- Main Function -----------------------//
  // ---------------------------------------------------//
  void run() {
    // this we will get from a subscriber call back
    pcl::io::loadPCDFile<pcl::PointXYZI>(mapDir + "../Map/134.768931732.pcd",
                                        *inputMap);

    /**
     * Transform Input according to odometry input
     * Here we get odometry from Poses Vector. During real
     * run we will get it from odometry topic
     **/
    float q[3], d[3];

    q[0] = globalPose6D->points[0].yaw;
    q[1] = globalPose6D->points[0].roll;
    q[2] = globalPose6D->points[0].pitch;

    d[0] = globalPose6D->points[0].z;
    d[1] = globalPose6D->points[0].x;
    d[2] = globalPose6D->points[0].y;

    Eigen::Matrix4f transform = ConvertToTransformMatrix(d, q);

    pcl::PointCloud<pcl::PointXYZI>::Ptr inputMapTransformed(
        new pcl::PointCloud<pcl::PointXYZI>);

    pcl::transformPointCloud(*inputMap, *inputMapTransformed, transform);

    // odometry info of current point cloud
    // taken from data file
    float x = -9.0, y = 23.0, z = 0.0;

    extract_local_map(x, y, z, radius);

    pcl::visualization::PCLVisualizer::Ptr viewer, viewer1;
    viewer = simpleVis(localMap, "Local Map");
    viewer1 = simpleVis(inputMapTransformed, "Input Cloud");
    cout << localMap->width << "\n";

    while (!viewer1->wasStopped()) {
      viewer->spinOnce(100);
      viewer1->spinOnce(100);
      std::this_thread::sleep_for(100ms);
    }
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "localise");

  GetSurroundingCloud obj;

  obj.SaveFullCloud();
}
