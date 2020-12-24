#include "util.hpp"

using namespace cv;

// [[Rcpp::export]]
XPtrMat cvmat_face(XPtrMat ptr, const char * facedata, const char * eyedata){
  /* load training data */
  CascadeClassifier face, eyes;
  if(!face.load( facedata ))
    throw std::runtime_error(std::string("Failed to load: ") + facedata);
  if(!eyes.load( eyedata ))
    throw std::runtime_error(std::string("Failed to load: ") + eyedata);

  //modify in place
  detectAndDraw(get_mat(ptr), face, eyes, 1, 0);
  return ptr;
}

// [[Rcpp::export]]
XPtrMat cvmat_eyemask(XPtrMat ptr, const char * facedata, const char * eyedata){
  /* load training data */
  CascadeClassifier face;
  if(!face.load( facedata ))
    throw std::runtime_error(std::string("Failed to load: ") + facedata);

  CascadeClassifier eye;
  if(!eye.load( eyedata ))
    throw std::runtime_error(std::string("Failed to load: ") + eyedata);

  Mat gray;
  Mat input = get_mat(ptr);
  Scalar color = Scalar(255,0,0);
  cvtColor( input, gray, COLOR_BGR2GRAY );
  equalizeHist(gray, gray);
  std::vector<Rect> faces;
  face.detectMultiScale( gray, faces, 1.1, 2, 0
                           //|CASCADE_FIND_BIGGEST_OBJECT
                           //|CASCADE_DO_ROUGH_SEARCH
                           |CASCADE_SCALE_IMAGE,
                           Size(30, 30) );

  Mat mask(gray.size(), gray.type(), Scalar::all(0));
  Rcpp::IntegerVector rvecFace(faces.size());
  Rcpp::IntegerVector xvecFace(faces.size());
  Rcpp::IntegerVector yvecFace(faces.size());
  Rcpp::IntegerVector rvecEye(faces.size() * 2);
  Rcpp::IntegerVector xvecEye(faces.size() * 2);
  Rcpp::IntegerVector yvecEye(faces.size() * 2);

  for (int i = 0; i < faces.size(); i++ ) {
    Point faceCenter;
    Rect r = faces.at(i);

    faceCenter.x = cvRound((r.x + r.width*0.5));
    faceCenter.y = cvRound((r.y + r.height*0.5));
    int radius = cvRound((r.width + r.height)*0.25);
    rvecFace.at(i) = radius;
    xvecFace.at(i) = faceCenter.x;
    yvecFace.at(i) = faceCenter.y;

    circle(input, faceCenter, radius, color, 3, 8, 0);
    Mat gray_face;
    std::vector<Rect> eyes;
    gray_face = gray( r );
    eye.detectMultiScale(gray_face, eyes, 1.1, 2, 0
                           |CASCADE_SCALE_IMAGE,
                           Size(30, 30) );
    Rprintf("Detected eyes: %i", eyes.size());
    for (int j = 0; j < eyes.size(); j++) {
      Point eyeCenter;
      Rect q = eyes.at(j);
      int radius = cvRound((q.width + q.height)*0.25);
      eyeCenter.x = cvRound((r.x + q.x + q.width*0.5));
      eyeCenter.y = cvRound((r.y + q.y + q.height*0.5));

      int eyeIndex = (((i + 1) * 2) - 1) + (j - 1);
      xvecEye.at(eyeIndex) = eyeCenter.x;
      yvecEye.at(eyeIndex) = eyeCenter.y;
      rvecEye.at(eyeIndex) = radius;

      circle( input, eyeCenter, radius, color, 3, 8, 0);
    }

  }
  XPtrMat out = cvmat_xptr(input);
  out.attr("faces") = Rcpp::DataFrame::create(
    Rcpp::_["radius"] = rvecFace,
    Rcpp::_["x"] = xvecFace,
    Rcpp::_["y"] = yvecFace
  );
  out.attr("eyes") = Rcpp::DataFrame::create(
    Rcpp::_["radius"] = rvecEye,
    Rcpp::_["x"] = xvecEye,
    Rcpp::_["y"] = yvecEye
  );
  return out;
}


// [[Rcpp::export]]
XPtrMat cvmat_facemask(XPtrMat ptr, const char * facedata){
  /* load training data */
  CascadeClassifier face;
  if(!face.load( facedata ))
    throw std::runtime_error(std::string("Failed to load: ") + facedata);

  Mat gray;
  Mat input = get_mat(ptr);
  Scalar color = Scalar(255,0,0);
  cvtColor( input, gray, COLOR_BGR2GRAY );
  equalizeHist(gray, gray);
  std::vector<Rect> faces;
  face.detectMultiScale( gray, faces, 1.1, 2, 0
                           //|CASCADE_FIND_BIGGEST_OBJECT
                           //|CASCADE_DO_ROUGH_SEARCH
                           |CASCADE_SCALE_IMAGE,
                           Size(30, 30) );

  Mat mask(gray.size(), gray.type(), Scalar::all(0));
  Rcpp::IntegerVector rvec(faces.size());
  Rcpp::IntegerVector xvec(faces.size());
  Rcpp::IntegerVector yvec(faces.size());
  for ( size_t i = 0; i < faces.size(); i++ ) {
    Point center;
    Rect r = faces.at(i);
    center.x = cvRound((r.x + r.width*0.5));
    center.y = cvRound((r.y + r.height*0.5));
    int radius = cvRound((r.width + r.height)*0.25);
    circle( input, center, radius, color, 3, 8, 0);
    rvec.at(i) = radius;
    xvec.at(i) = center.x;
    yvec.at(i) = center.y;
  }
  XPtrMat out = cvmat_xptr(input);
  out.attr("faces") = Rcpp::DataFrame::create(
    Rcpp::_["radius"] = rvec,
    Rcpp::_["x"] = xvec,
    Rcpp::_["y"] = yvec
  );
  return out;
}

// [[Rcpp::export]]
XPtrMat cvmat_mog2(XPtrMat ptr) {
#ifndef HAVE_OPENCV_3
  throw std::runtime_error("createBackgroundSubtractorMOG2 requires OpenCV 3 or newer");
#else
  static Ptr<BackgroundSubtractorMOG2> model = createBackgroundSubtractorMOG2();
  model->setVarThreshold(10);
  cv::Mat frame = get_mat(ptr);
  cv::Mat mask, out_frame;
  model->apply(frame, mask);
  //refineSegments(frame, mask, out_frame);
  return cvmat_xptr(mask);
#endif
}

// [[Rcpp::export]]
XPtrMat cvmat_knn(XPtrMat ptr) {
#ifndef HAVE_OPENCV_3
  throw std::runtime_error("createBackgroundSubtractorKNN requires OpenCV 3 or newer");
#else
  static Ptr<BackgroundSubtractorKNN> model = createBackgroundSubtractorKNN();
  cv::Mat frame = get_mat(ptr);
  cv::Mat mask, out_frame;
  model->apply(frame, mask);
  return cvmat_xptr(mask);
#endif
}

// [[Rcpp::export]]
XPtrMat cvmat_edges(XPtrMat ptr) {
  cv::Mat edges;
  cv::Mat frame = get_mat(ptr);
  cvtColor(frame, edges, COLOR_BGR2GRAY);
  GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
  Canny(edges, edges, 0, 30, 3);
  return cvmat_xptr(edges);
}
