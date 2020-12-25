#include "util.hpp"

// based on https://docs.opencv.org/4.5.1/db/d00/samples_2cpp_2squares_8cpp-example.html
using namespace cv;
using namespace std;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
  double dx1 = pt1.x - pt0.x;
  double dy1 = pt1.y - pt0.y;
  double dx2 = pt2.x - pt0.x;
  double dy2 = pt2.y - pt0.y;
  return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


// [[Rcpp::export]]
XPtrMat cvmat_rectangles(XPtrMat ptr){
  int thresh = 50, N = 11;
  vector<vector<Point> > rectangles;

  Mat input = get_mat(ptr);
  Scalar color = Scalar(255,0,0);

  Mat pyr, timg, gray0(input.size(), CV_8U), gray;
  // down-scale and upscale the input to filter out the noise
  pyrDown(input, pyr, Size(input.cols/2, input.rows/2));
  pyrUp(pyr, timg, input.size());
  vector<vector<Point> > contours;
  // find rectangles in every color plane of the input
  for( int c = 0; c < 3; c++ )
  {
    int ch[] = {c, 0};
    mixChannels(&timg, 1, &gray0, 1, ch, 1);
    // try several threshold levels
    for( int l = 0; l < N; l++ )
    {
      // hack: use Canny instead of zero threshold level.
      // Canny helps to catch rectangles with gradient shading
      if( l == 0 )
      {
        // apply Canny. Take the upper threshold from slider
        // and set the lower to 0 (which forces edges merging)
        Canny(gray0, gray, 0, thresh, 5);
        // dilate canny output to remove potential
        // holes between edge segments
        dilate(gray, gray, Mat(), Point(-1,-1));
      }
      else
      {
        // apply threshold if l!=0:
        //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
        gray = gray0 >= (l+1)*255/N;
      }
      // find contours and store them all as a list
      findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
      vector<Point> approx;
      // test each contour
      for( size_t i = 0; i < contours.size(); i++ )
      {
        // approximate contour with accuracy proportional
        // to the contour perimeter
        approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);
        // square contours should have 4 vertices after approximation
        // relatively large area (to filter out noisy contours)
        // and be convex.
        // Note: absolute value of an area is used because
        // area may be positive or negative - in accordance with the
        // contour orientation
        if( approx.size() == 4 &&
            fabs(contourArea(approx)) > 1000 &&
            isContourConvex(approx) )
        {
          double maxCosine = 0;
          for( int j = 2; j < 5; j++ )
          {
            // find the maximum cosine of the angle between joint edges
            double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
            maxCosine = MAX(maxCosine, cosine);
          }
          // if cosines of all angles are small
          // (all angles are ~90 degree) then write quandrange
          // vertices to resultant sequence
          if( maxCosine < 0.3 )
            rectangles.push_back(approx);
        }
      }
    }
  }

  polylines(input, rectangles, true, Scalar(0, 255, 0), 3, LINE_AA);
  int nRectangle = rectangles.size();
  //Rprintf("Number of rectangles: %i\n", nRectangle);
  //Rprintf("Rectangle 1 has %i points\n", rectangles[0].size());
  //Rprintf("Rectangle 2 has %i points\n", rectangles[1].size());
  Rcpp::List squareList(nRectangle);
  for (int i = 0; i < nRectangle; i++) {
    int nPoint = rectangles[i].size();
    Rcpp::NumericVector x(nPoint);
    Rcpp::NumericVector y(nPoint);
    for (int j = 0; j < nPoint; j++) {
      x[j] = rectangles[i][j].x;
      y[j] = rectangles[i][j].y;
    }
    rectangleList[i] = Rcpp::List::create(Rcpp::Named("x") = x, Rcpp::_["y"] = y);
  }
  //Rprintf("Length of list: %i \n", squareList.size());
  XPtrMat out = cvmat_xptr(input);
  out.attr("rectangles") = rectangleList;
  return out;
}
