#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#ifdef USE_OPENCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Return the width and height of an image\n"
        "Usage:\n"
        "    get_image_size IMAGE_FILE\n");

  //gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 2) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/get_image_size");
    return 1;
  }
  
  cv::Mat cv_img = ReadImageToCVMat(argv[1]);
  
  if (cv_img.data) {
    std::cout << cv_img.cols << "," << cv_img.rows << std::endl;
  } else {
    LOG(FATAL) << "Could not read image";
    return 1;
  }

#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
