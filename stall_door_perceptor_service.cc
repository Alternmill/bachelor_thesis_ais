//Modifications copyright (C) 2025 SOMATIC

#include "bcr_robot/logic/perception/stall_door_perceptor_service.hh"

#include <bcr_robot/logic/detector/visual_log_helper.hh>

#include <bcr_transport/addresses/actions_servers.hh>
#include <bcr_transport/agent/agent.hh>
#include <bcr_transport/serialization/camera_serializator.hh>
#include <bcr_transport/serialization/segmentation_serialization.hh>
#include <bcr_transport/visualization/visual_logger.hh>

#include <bcr_vision/distortion_utils.hh>

#include <bcr_model/image/image.hh>

#include <bcr_core/logic/collision_validators/collision_engine.hh>
#include <bcr_core/logic/world/update_actions/map_state_update_action.hh>

#include <bcr_common/log/log.hh>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace bcr::robot::logic::perception {
StallDoorPerceptorAndUpdaterService::StallDoorPerceptorAndUpdaterService(
        model::CameraName cameraName, transport::agent::Agent& mapPerceptorNode,
        const bcr::core::logic::world::WorldHandler* world,
        core::logic::collision::CollisionEngineProvider& collisionEngineProvider,
        transport::visualization::VisualLogger& visualLogger,
        transport::core::TransportLayer& transportLayer) noexcept
        : cameraName_(cameraName),
          loggerName_(baseLoggerName_ + ToStdString(cameraName_)),
          visualLogHelper_(&visualLogger, loggerName_, true),
          mapPerceptorNode_(mapPerceptorNode),
          world_(world),
          collisionEngineProvider_(collisionEngineProvider),
          cameraPositionInWorld_(math::Isometry3d::Identity()),
          imageSegmentationActionClient_(
                  mapPerceptorNode_.CreateActionClient<ImageSegmentationAction>(
                          transport::addresses::action_server::ImageSegmentation())),
          stallDoorPerceptor_(cameraName, visualLogger) {

    InitCameraConnector(transportLayer);
}
std::optional<vision::TimedColorImage> StallDoorPerceptorAndUpdaterService::RequestRGBImage() const {
    if (cameraName_ == model::CameraName::arm_camera) {
        return rgbdCameraConnector_->RequestColorImage();
    } else {
        return usbCameraConnector_->RequestUSBCameraOriginalImage();
    }
}
void StallDoorPerceptorAndUpdaterService::PerceptStallDoors() {
    if (imageSegmentationGoalHandle_) {
        return;
    }

    auto imageOpt = RequestRGBImage();
    if (!imageOpt.has_value()) {
        return;
    }
    image_ = imageOpt->Image();
    if (!image_) {
        return;
    }
    double ts;
    robotState_ = world_->Proxy().RobotState(ts);
    mapState_ = world_->Proxy().MapState(ts);
    map_ = world_->Proxy().Map();
    auto cameraInRobotBase = world_->RobotModel()->GetCameraModels()
            .CameraModelByName(cameraName_)
            .ColorOriginInRobotBase(robotState_.ArmState());
    auto robotBaseInWorld = robotState_.RobotBaseInWorld();
    cameraPositionInWorld_ = robotBaseInWorld * cameraInRobotBase;
    //visualLogHelper_.VisualizeColorImage("Original image", image_.value());
    auto stallDoorLinks = StallDoorPerceptor::FindEligibleDoors(map_->Links(), robotState_, mapState_);

    if (stallDoorLinks.empty()) {
        return;
    }
    msgs::Segmentation_Request req;
    msgs::ImageMsg msg;
    transport::serialization::ToMsgRaw(image_.value(), &msg);
    *req.mutable_color_img() = msg;
    req.set_object_class(msgs::SegmentableClass::STALL_DOOR);

    msgs::ImageSegmentationAction::Goal goal;
    *goal.mutable_req() = req;

    imageSegmentationGoalHandle_ = imageSegmentationActionClient_->SendGoal(goal, *this);
}

void StallDoorPerceptorAndUpdaterService::InitCameraConnector(
        transport::core::TransportLayer& transport_layer) {
    if (cameraName_ == model::CameraName::arm_camera) {
        rgbdCameraConnector_ = transport_layer.CreateRealsenseCameraClientConnector();
    } else {
        usbCameraConnector_ = transport_layer.CreateUSBCameraClientConnector(cameraName_);
    }
}

void StallDoorPerceptorAndUpdaterService::FeedbackCallback(
        ImageSegmentationActionGoalHandle& /*handle*/,
        const ImageSegmentationActionGoalHandle::Feedback&  /*feedBack*/) {}

void StallDoorPerceptorAndUpdaterService::ResultCallback(
        ImageSegmentationActionGoalHandle::WrappedResult& result) {
    class ImageSegmentationActionGoalHandleRelease {
        using ImageSegmentationActionGoalHandle =
                transport::action::ClientGoalHandle<ImageSegmentationAction>;
    public:
        ImageSegmentationActionGoalHandleRelease(std::unique_ptr<ImageSegmentationActionGoalHandle> goalHandle) :
                goalHandle_(std::move(goalHandle)) {};
    private:
        std::unique_ptr<ImageSegmentationActionGoalHandle> goalHandle_;
    };
    ImageSegmentationActionGoalHandleRelease release(std::move(imageSegmentationGoalHandle_));

    if (result.code != msgs::GoalResultCode::SUCCEEDED) {
        return;
    }

    auto segmentationResponse = transport::serialization::FromProto(result.result.rep());
    auto instances = segmentationResponse.Instances();
    if (instances.empty()) {
        return;
    }
    //ShowMasksAndOverlay(instances);

    std::vector<model::GrayscaleImage> masksBcr;
    auto cameraParams = world_->RobotModel()->GetCameraModels().CameraModelByName(cameraName_).ColorIntrinsicParams();
    model::CameraIntrinsicParams undistortedParameters;
    for (auto& instance : instances) {
        auto undistortionResult = UndistortImage(instance.Mask(), cameraParams);
        undistortedParameters = undistortionResult.NewIntrinsicParams;
        masksBcr.push_back(vision::Conversions::MatrixToImage<1>(undistortionResult.UndistortedImage));
    }

    auto stallDoorLinks = StallDoorPerceptor::FindEligibleDoors(map_->Links(), robotState_, mapState_);

    if (stallDoorLinks.empty()) {
        return;
    }

    std::vector<std::pair<core::DoorLink*, math::Isometry3d>> linksAndPositions;

    linksAndPositions.reserve(stallDoorLinks.size());

    for (auto link : stallDoorLinks) {
        linksAndPositions.push_back({link, link->GlobalTransform(mapState_)});
    }

    StallDoorPerceptingData perceptingData{masksBcr,
                                           linksAndPositions,
                                           cameraPositionInWorld_,
                                           undistortedParameters};

    auto collisionEngine = collisionEngineProvider_.BorrowCollisionEngine(loggerName_);
    double ts;
    auto dummyMapState = world_->Proxy().MapState(ts);

    auto start = std::chrono::high_resolution_clock::now();

    auto perceptorModifiers = stallDoorPerceptor_.Percept(perceptingData, collisionEngine, dummyMapState, robotState_);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    BCR_INFO_STREAM(loggerName_, "Time elapsed for perception: " << duration.count() << " milliseconds");

    if (map_ != world_->Proxy().Map()) {
        BCR_INFO_STREAM(loggerName_, "Map changed during stall door perception! No results are applied");
        return;
    }
    auto newMapState = world_->Proxy().MapState(ts);
    for (const auto& [uuid, value] : perceptorModifiers) {
        if (newMapState.GetJointState(uuid).Value() != mapState_.GetJointState(uuid).Value()) {
            BCR_INFO_STREAM(loggerName_,
                            "Joint state changed for joint: {"
                                    << boost::lexical_cast<std::string>(uuid)
                                    << "} during perception, ignoring results");
        }
        mapState_.UpdateJointState(uuid, value);
    }
    //temporarily comment this out
    //world_->Proxy().Update<core::logic::world::MapStateUpdateAction>(perceptorModifiers);
    imageSegmentationGoalHandle_.reset();
}
vision::UndistortionResult
StallDoorPerceptorAndUpdaterService::UndistortImage(model::GrayscaleImage mask,
                                                    model::CameraIntrinsicParams cameraIntrinsicParams) {
    if (cameraName_ == model::CameraName::arm_camera) {
        auto undistortionResult = vision::UndistortionResult();
        undistortionResult.UndistortedImage = vision::Conversions::ImageToMatrix<1>(mask);
        undistortionResult.NewIntrinsicParams = cameraIntrinsicParams;
        return undistortionResult;
    } else {
        auto undistortionResult = vision::DistortionUtils::UndistortImageFishEye(
                506,
                506,
                640,
                360,
                -0.015,
                -0.0068,
                0.0036,
                -0.0013,
                vision::Conversions::ImageToMatrix<1>(mask));

        auto undistortionResultImage = vision::DistortionUtils::UndistortImageFishEye(
                506,
                506,
                640,
                360,
                -0.015,
                -0.0068,
                0.0036,
                -0.0013,
                vision::Conversions::ImageToMatrix<3>(image_.value()));
//        visualLogHelper_.VisualizeColorImage("Undistorted image",
//                                             vision::Conversions::MatrixToImage<3>(
//                                                     undistortionResultImage.UndistortedImage));
        return undistortionResult;

    }

}
void StallDoorPerceptorAndUpdaterService::ShowMasksAndOverlay(std::vector<vision::SegmentedInstance> masks) noexcept {
    for (size_t i = 0; i < masks.size(); i++) {
        auto mask3D = vision::Conversions::OneChannelToManyConversion(masks[i].Mask());
        visualLogHelper_.VisualizeColorImage((std::stringstream() << "Image: {" << i << "}").str(), mask3D);
        model::ColorImage overlayImage = mask3D;
        for (size_t row = 0; row < mask3D.Height(); row++) {
            for (size_t col = 0; col < mask3D.Width(); col++) {
                for (size_t channel = 0; channel < 3; channel++) {
                    auto location = model::PixelLocation{.Column = col, .Row = row};
                    int val1 = image_->GetValue(location, channel);
                    int val2 = mask3D.GetValue(location, channel);
                    if (channel != 0) {
                        val2 /= 2;
                    }
                    auto new_val = (val1 + val2) / 2;
                    overlayImage.SetValue(location, channel, new_val);
                }
            }
        }
        visualLogHelper_.VisualizeColorImage((std::stringstream() << "Overlayed image: {" << i << "}").str(),
                                             overlayImage);
    }
}

} // namespace bcr::robot::logic::perception
