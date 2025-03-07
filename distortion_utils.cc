//Modifications copyright (C) 2025 SOMATIC

#include "bcr_vision/distortion_utils.hh"

namespace bcr::vision {

UndistortionResult DistortionUtils::UndistortImageFishEye(double fx,
                                                          double fy,
                                                          double ppx,
                                                          double ppy,
                                                          double k1,
                                                          double k2,
                                                          double k3,
                                                          double k4,
                                                          const cv::Mat& inputImage) noexcept {
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3)
            <<
            fx, 0, ppx,
            0, fy, ppy,
            0, 0, 1);

    cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << k1, k2, k3, k4);

    cv::Size imageSize = inputImage.size();

    cv::Mat newCameraMatrix;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            cameraMatrix,
            distCoeffs,
            imageSize,
            cv::Matx33d::eye(),
            newCameraMatrix,
            0.0,                // Balance parameter (1.0 retains all pixels)
            imageSize,
            0
    );

    cv::Mat undistortedImage;
    cv::fisheye::undistortImage(
            inputImage,
            undistortedImage,
            cameraMatrix,
            distCoeffs,
            newCameraMatrix,
            imageSize
    );

    float newFx = static_cast<float>(newCameraMatrix.at<double>(0, 0));
    float newFy = static_cast<float>(newCameraMatrix.at<double>(1, 1));
    float newPpx = static_cast<float>(newCameraMatrix.at<double>(0, 2));
    float newPpy = static_cast<float>(newCameraMatrix.at<double>(1, 2));

    model::CameraIntrinsicParams newParams(
            newFx,
            newFy,
            newPpx,
            newPpy
    );

    return UndistortionResult{newParams, undistortedImage};
}

}