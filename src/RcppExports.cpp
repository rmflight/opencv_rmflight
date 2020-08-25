// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "opencv_types.h"
#include <Rcpp.h>

using namespace Rcpp;

// cvmat_rect
XPtrMat cvmat_rect(XPtrMat ptr, int x, int y, int width, int height);
RcppExport SEXP _opencv_cvmat_rect(SEXP ptrSEXP, SEXP xSEXP, SEXP ySEXP, SEXP widthSEXP, SEXP heightSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< int >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type width(widthSEXP);
    Rcpp::traits::input_parameter< int >::type height(heightSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_rect(ptr, x, y, width, height));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_destroy
void cvmat_destroy(XPtrMat image);
RcppExport SEXP _opencv_cvmat_destroy(SEXP imageSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type image(imageSEXP);
    cvmat_destroy(image);
    return R_NilValue;
END_RCPP
}
// cvmat_dead
bool cvmat_dead(XPtrMat image);
RcppExport SEXP _opencv_cvmat_dead(SEXP imageSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type image(imageSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_dead(image));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_size
int cvmat_size(XPtrMat image);
RcppExport SEXP _opencv_cvmat_size(SEXP imageSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type image(imageSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_size(image));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_new
XPtrMat cvmat_new();
RcppExport SEXP _opencv_cvmat_new() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(cvmat_new());
    return rcpp_result_gen;
END_RCPP
}
// cvmat_dupe
XPtrMat cvmat_dupe(XPtrMat image);
RcppExport SEXP _opencv_cvmat_dupe(SEXP imageSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type image(imageSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_dupe(image));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_read
XPtrMat cvmat_read(Rcpp::String path);
RcppExport SEXP _opencv_cvmat_read(SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::String >::type path(pathSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_read(path));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_camera
XPtrMat cvmat_camera();
RcppExport SEXP _opencv_cvmat_camera() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(cvmat_camera());
    return rcpp_result_gen;
END_RCPP
}
// cvmat_write
std::string cvmat_write(XPtrMat image, std::string path);
RcppExport SEXP _opencv_cvmat_write(SEXP imageSEXP, SEXP pathSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type image(imageSEXP);
    Rcpp::traits::input_parameter< std::string >::type path(pathSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_write(image, path));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_resize
XPtrMat cvmat_resize(XPtrMat ptr, int width, int height);
RcppExport SEXP _opencv_cvmat_resize(SEXP ptrSEXP, SEXP widthSEXP, SEXP heightSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< int >::type width(widthSEXP);
    Rcpp::traits::input_parameter< int >::type height(heightSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_resize(ptr, width, height));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_raw_bgr
XPtrMat cvmat_raw_bgr(Rcpp::RawVector image, int width, int height);
RcppExport SEXP _opencv_cvmat_raw_bgr(SEXP imageSEXP, SEXP widthSEXP, SEXP heightSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::RawVector >::type image(imageSEXP);
    Rcpp::traits::input_parameter< int >::type width(widthSEXP);
    Rcpp::traits::input_parameter< int >::type height(heightSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_raw_bgr(image, width, height));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_raw_bw
XPtrMat cvmat_raw_bw(Rcpp::RawVector image, int width, int height);
RcppExport SEXP _opencv_cvmat_raw_bw(SEXP imageSEXP, SEXP widthSEXP, SEXP heightSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::RawVector >::type image(imageSEXP);
    Rcpp::traits::input_parameter< int >::type width(widthSEXP);
    Rcpp::traits::input_parameter< int >::type height(heightSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_raw_bw(image, width, height));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_bitmap
Rcpp::RawVector cvmat_bitmap(XPtrMat ptr);
RcppExport SEXP _opencv_cvmat_bitmap(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_bitmap(ptr));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_copyto
XPtrMat cvmat_copyto(XPtrMat from, XPtrMat to, XPtrMat mask);
RcppExport SEXP _opencv_cvmat_copyto(SEXP fromSEXP, SEXP toSEXP, SEXP maskSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type from(fromSEXP);
    Rcpp::traits::input_parameter< XPtrMat >::type to(toSEXP);
    Rcpp::traits::input_parameter< XPtrMat >::type mask(maskSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_copyto(from, to, mask));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_info
Rcpp::List cvmat_info(XPtrMat image);
RcppExport SEXP _opencv_cvmat_info(SEXP imageSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type image(imageSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_info(image));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_display
void cvmat_display(XPtrMat ptr);
RcppExport SEXP _opencv_cvmat_display(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    cvmat_display(ptr);
    return R_NilValue;
END_RCPP
}
// livestream
void livestream(Rcpp::Function filter);
RcppExport SEXP _opencv_livestream(SEXP filterSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::Function >::type filter(filterSEXP);
    livestream(filter);
    return R_NilValue;
END_RCPP
}
// data_prefix
Rcpp::String data_prefix();
RcppExport SEXP _opencv_data_prefix() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(data_prefix());
    return rcpp_result_gen;
END_RCPP
}
// set_num_threads
int set_num_threads(int n);
RcppExport SEXP _opencv_set_num_threads(SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(set_num_threads(n));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_blur
XPtrMat cvmat_blur(XPtrMat ptr, size_t ksize);
RcppExport SEXP _opencv_cvmat_blur(SEXP ptrSEXP, SEXP ksizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< size_t >::type ksize(ksizeSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_blur(ptr, ksize));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_sketch
XPtrMat cvmat_sketch(XPtrMat ptr, bool color);
RcppExport SEXP _opencv_cvmat_sketch(SEXP ptrSEXP, SEXP colorSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< bool >::type color(colorSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_sketch(ptr, color));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_stylize
XPtrMat cvmat_stylize(XPtrMat ptr);
RcppExport SEXP _opencv_cvmat_stylize(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_stylize(ptr));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_face
XPtrMat cvmat_face(XPtrMat ptr, const char * facedata, const char * eyedata);
RcppExport SEXP _opencv_cvmat_face(SEXP ptrSEXP, SEXP facedataSEXP, SEXP eyedataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< const char * >::type facedata(facedataSEXP);
    Rcpp::traits::input_parameter< const char * >::type eyedata(eyedataSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_face(ptr, facedata, eyedata));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_facemask
XPtrMat cvmat_facemask(XPtrMat ptr, const char * facedata);
RcppExport SEXP _opencv_cvmat_facemask(SEXP ptrSEXP, SEXP facedataSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< const char * >::type facedata(facedataSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_facemask(ptr, facedata));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_mog2
XPtrMat cvmat_mog2(XPtrMat ptr);
RcppExport SEXP _opencv_cvmat_mog2(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_mog2(ptr));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_knn
XPtrMat cvmat_knn(XPtrMat ptr);
RcppExport SEXP _opencv_cvmat_knn(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_knn(ptr));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_edges
XPtrMat cvmat_edges(XPtrMat ptr);
RcppExport SEXP _opencv_cvmat_edges(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_edges(ptr));
    return rcpp_result_gen;
END_RCPP
}
// cvkeypoints_fast
Rcpp::List cvkeypoints_fast(XPtrMat ptr, int threshold, bool nonmaxSuppression, int type);
RcppExport SEXP _opencv_cvkeypoints_fast(SEXP ptrSEXP, SEXP thresholdSEXP, SEXP nonmaxSuppressionSEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< int >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< bool >::type nonmaxSuppression(nonmaxSuppressionSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(cvkeypoints_fast(ptr, threshold, nonmaxSuppression, type));
    return rcpp_result_gen;
END_RCPP
}
// cvkeypoints_brief
Rcpp::List cvkeypoints_brief(XPtrMat ptr, int bytes, bool use_orientation);
RcppExport SEXP _opencv_cvkeypoints_brief(SEXP ptrSEXP, SEXP bytesSEXP, SEXP use_orientationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    Rcpp::traits::input_parameter< int >::type bytes(bytesSEXP);
    Rcpp::traits::input_parameter< bool >::type use_orientation(use_orientationSEXP);
    rcpp_result_gen = Rcpp::wrap(cvkeypoints_brief(ptr, bytes, use_orientation));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_hog
XPtrMat cvmat_hog(XPtrMat ptr);
RcppExport SEXP _opencv_cvmat_hog(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_hog(ptr));
    return rcpp_result_gen;
END_RCPP
}
// cvmat_markers
XPtrMat cvmat_markers(XPtrMat ptr);
RcppExport SEXP _opencv_cvmat_markers(SEXP ptrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< XPtrMat >::type ptr(ptrSEXP);
    rcpp_result_gen = Rcpp::wrap(cvmat_markers(ptr));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_opencv_cvmat_rect", (DL_FUNC) &_opencv_cvmat_rect, 5},
    {"_opencv_cvmat_destroy", (DL_FUNC) &_opencv_cvmat_destroy, 1},
    {"_opencv_cvmat_dead", (DL_FUNC) &_opencv_cvmat_dead, 1},
    {"_opencv_cvmat_size", (DL_FUNC) &_opencv_cvmat_size, 1},
    {"_opencv_cvmat_new", (DL_FUNC) &_opencv_cvmat_new, 0},
    {"_opencv_cvmat_dupe", (DL_FUNC) &_opencv_cvmat_dupe, 1},
    {"_opencv_cvmat_read", (DL_FUNC) &_opencv_cvmat_read, 1},
    {"_opencv_cvmat_camera", (DL_FUNC) &_opencv_cvmat_camera, 0},
    {"_opencv_cvmat_write", (DL_FUNC) &_opencv_cvmat_write, 2},
    {"_opencv_cvmat_resize", (DL_FUNC) &_opencv_cvmat_resize, 3},
    {"_opencv_cvmat_raw_bgr", (DL_FUNC) &_opencv_cvmat_raw_bgr, 3},
    {"_opencv_cvmat_raw_bw", (DL_FUNC) &_opencv_cvmat_raw_bw, 3},
    {"_opencv_cvmat_bitmap", (DL_FUNC) &_opencv_cvmat_bitmap, 1},
    {"_opencv_cvmat_copyto", (DL_FUNC) &_opencv_cvmat_copyto, 3},
    {"_opencv_cvmat_info", (DL_FUNC) &_opencv_cvmat_info, 1},
    {"_opencv_cvmat_display", (DL_FUNC) &_opencv_cvmat_display, 1},
    {"_opencv_livestream", (DL_FUNC) &_opencv_livestream, 1},
    {"_opencv_data_prefix", (DL_FUNC) &_opencv_data_prefix, 0},
    {"_opencv_set_num_threads", (DL_FUNC) &_opencv_set_num_threads, 1},
    {"_opencv_cvmat_blur", (DL_FUNC) &_opencv_cvmat_blur, 2},
    {"_opencv_cvmat_sketch", (DL_FUNC) &_opencv_cvmat_sketch, 2},
    {"_opencv_cvmat_stylize", (DL_FUNC) &_opencv_cvmat_stylize, 1},
    {"_opencv_cvmat_face", (DL_FUNC) &_opencv_cvmat_face, 3},
    {"_opencv_cvmat_facemask", (DL_FUNC) &_opencv_cvmat_facemask, 2},
    {"_opencv_cvmat_mog2", (DL_FUNC) &_opencv_cvmat_mog2, 1},
    {"_opencv_cvmat_knn", (DL_FUNC) &_opencv_cvmat_knn, 1},
    {"_opencv_cvmat_edges", (DL_FUNC) &_opencv_cvmat_edges, 1},
    {"_opencv_cvkeypoints_fast", (DL_FUNC) &_opencv_cvkeypoints_fast, 4},
    {"_opencv_cvkeypoints_brief", (DL_FUNC) &_opencv_cvkeypoints_brief, 3},
    {"_opencv_cvmat_hog", (DL_FUNC) &_opencv_cvmat_hog, 1},
    {"_opencv_cvmat_markers", (DL_FUNC) &_opencv_cvmat_markers, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_opencv(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
