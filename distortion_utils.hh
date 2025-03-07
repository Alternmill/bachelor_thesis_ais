//Modifications copyright (C) 2025 SOMATIC

#ifndef BCR_VISION_DISTORTION_UTILS_HH
#define BCR_VISION_DISTORTION_UTILS_HH

#include <bcr_model/camera/camera_intrinsic_parameters.hh>
#include <bcr_model/camera/fisheye_camera_intrinsic_parameters.hh>

#include <opencv2/opencv.hpp>

namespace bcr::vision {

struct UndistortionResult {
    model::CameraIntrinsicParams NewIntrinsicParams;
    cv::Mat UndistortedImage;
};

class DistortionUtils {
public:
    static UndistortionResult UndistortImageFishEye(
            model::FisheyeCameraIntrinsicParams& params,
            const cv::Mat& inputImage) noexcept;
};

}

#endif // BCR_VISION_DISTORTION_UTILS_HH
