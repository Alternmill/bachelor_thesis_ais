//Modifications copyright (C) 2025 SOMATIC

#ifndef BCR_ROBOT_LOGIC_PERCEPTION_STALL_DOOR_PERCEPTOR_SERVICE_HH
#define BCR_ROBOT_LOGIC_PERCEPTION_STALL_DOOR_PERCEPTOR_SERVICE_HH

#include "perceptor/stall_door_perceptor.hh"

#include <bcr_core/logic/collision_validators/collision_engine_provider.hh>
#include <bcr_core/logic/world/world_handler.hh>

#include <image_segmentation_action.pb.h>

#include <bcr_transport/action/action_client.hh>
#include <bcr_transport/communication/camera/realsense_camera/realsense_camera_client_connector.hh>
#include <bcr_transport/communication/camera/usb_camera/usb_camera_client_connector.hh>

#include <bcr_vision/segmentation.hh>

namespace bcr::transport::agent {
class Agent;
}

namespace bcr::transport::core {
class TransportLayer;
}

namespace bcr::vision {
struct UndistortionResult;
}

namespace bcr::robot::logic::perception {

class StallDoorPerceptorAndUpdaterService
        : transport::action::ActionClientResponseHandlerBase<
                msgs::ImageSegmentationAction> {
    using ImageSegmentationAction = msgs::ImageSegmentationAction;
    using ImageSegmentationActionGoalHandle =
            transport::action::ClientGoalHandle<ImageSegmentationAction>;

public:
    StallDoorPerceptorAndUpdaterService(
            model::CameraName cameraName,
            transport::agent::Agent& mapPerceptorNode,
            const bcr::core::logic::world::WorldHandler* world,
            core::logic::collision::CollisionEngineProvider& collisionEngineProvider,
            transport::visualization::VisualLogger& visualLogger,
            transport::core::TransportLayer& transportLayer) noexcept;

public:
    [[nodiscard]] std::optional<vision::TimedColorImage> RequestRGBImage() const;
    void PerceptStallDoors();

private:
    void InitCameraConnector(transport::core::TransportLayer& transport_layer);
    void FeedbackCallback(
            ImageSegmentationActionGoalHandle& /*handle*/,
            const ImageSegmentationActionGoalHandle::Feedback& feedback) override;
    void ResultCallback(
            ImageSegmentationActionGoalHandle::WrappedResult& result) override;
private:
    void ShowMasksAndOverlay(std::vector<vision::SegmentedInstance> masks) noexcept;

    vision::UndistortionResult UndistortImage(model::GrayscaleImage mask,
                                              model::CameraIntrinsicParams cameraIntrinsicParams);
private:
    model::CameraName cameraName_;

    static const inline std::string baseLoggerName_ = "Stall door perceptor service ";
    const std::string loggerName_;
    detector::VisualLogHelper visualLogHelper_;

    transport::agent::Agent& mapPerceptorNode_;
    const bcr::core::logic::world::WorldHandler* world_;
    core::logic::collision::CollisionEngineProvider& collisionEngineProvider_;

    std::unique_ptr<transport::RealsenseCameraClientConnector>
            rgbdCameraConnector_;

    std::optional<model::ColorImage> image_;
    core::map::MapState mapState_;
    model::RobotState robotState_;
    std::shared_ptr<const core::map::Map> map_;
    math::Isometry3d cameraPositionInWorld_;

    const transport::action::ActionClient<ImageSegmentationAction>::SharedPtr
            imageSegmentationActionClient_;
    StallDoorPerceptor stallDoorPerceptor_;
    ImageSegmentationActionGoalHandle::UniquePtr imageSegmentationGoalHandle_;
    std::unique_ptr<transport::USBCameraClientConnector> usbCameraConnector_;
};
} // namespace bcr::robot::logic::perception
#endif // BCR_ROBOT_LOGIC_PERCEPTION_STALL_DOOR_PERCEPTOR_SERVICE_HH
