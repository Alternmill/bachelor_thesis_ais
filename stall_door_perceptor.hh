//Modifications copyright (C) 2025 SOMATIC

#ifndef BCR_ROBOT_LOGIC_PERCEPTION_PERCEPTOR_STALL_DOOR_PERCEPTOR_HH_
#define BCR_ROBOT_LOGIC_PERCEPTION_PERCEPTOR_STALL_DOOR_PERCEPTOR_HH_

#include <bcr_robot/logic/detector/visual_log_helper.hh>

#include <bcr_core/logic/collision_validators/collision_engine_provider.hh>
#include <bcr_core/model/map/link/camera/camera_link.hh>
#include <bcr_core/model/map/link/door/door_link.hh>
#include <bcr_core/model/map/map_state.hh>

#include <bcr_model/image/image.hh>
#include <bcr_model/robot/type/camera_type.hh>

#include <opencv2/core/mat.hpp>

#include <chrono>

namespace bcr::math {
class Polygon3D;
}

namespace bcr::model {
struct CameraIntrinsicParams;
class RobotState;
}

namespace bcr::core::trajectory {
class MapStateDataPoint;
}

namespace bcr::transport::visualization {
class VisualLogger;
}

namespace bcr::robot::logic::detection {
class VisualLoggerHelper;
}

namespace bcr::robot::logic::perception {

struct StallDoorPerceptingData {
    std::vector<model::GrayscaleImage> Masks;
    std::vector<std::pair<core::DoorLink*, math::Isometry3d>> Doors;
    math::Isometry3d CameraInWorld;
    model::CameraIntrinsicParams CameraParams;
};

class StallDoorPerceptor {
public:
    explicit StallDoorPerceptor(
            model::CameraName cameraName,
            transport::visualization::VisualLogger& visualLogger) noexcept;

    [[nodiscard]] core::map::JointStateById Percept(const StallDoorPerceptingData& perceptingData,
                                                    core::logic::collision::CollisionEngine& collisionEngine,
                                                    core::map::MapState& mapState,
                                                    const model::RobotState& robotState) noexcept;
    [[nodiscard]] static std::vector<core::DoorLink*>
    FindEligibleDoors(const std::vector<core::Link*>& links,
                      const model::RobotState& robotState,
                      const core::map::MapState& mapState) noexcept;
private: // constants
    static constexpr std::array<std::array<size_t, 4>, 6> facesIndices_ = {{
                                                                                   {0, 1, 5, 4},
                                                                                   {0, 1, 3, 2},
                                                                                   {0, 2, 6, 4},
                                                                                   {7, 6, 2, 3},
                                                                                   {7, 5, 1, 3},
                                                                                   {7, 5, 4, 6}
                                                                           }};
    static constexpr double nearPlaneDistance_ = 0.01;
    static constexpr float maxSupportedDoorDistance_ = 3.5f;
    static const inline std::string baseLoggerName_ = "Stall door perceptor ";
    const std::string loggerName_;

private: // methods
    [[nodiscard]] std::vector<std::pair<const core::DoorLink*, double>>
    MatchDoorsAndMasks(const StallDoorPerceptingData& perceptingData,
                       core::logic::collision::CollisionEngine& collisionEngine,
                       core::map::MapState& mapState,
                       const model::RobotState& robotState) noexcept;
    void Draw(core::DoorLink* doorLink, math::Isometry3d doorPosition, double angle) noexcept;
    [[nodiscard]] std::vector<std::pair<double, double>> static
    GetBestAnglesForStallDoor(const core::DoorLink* doorLink,
                              const math::Isometry3d& doorPosition,
                              const std::vector<model::GrayscaleImage>& masks,
                              const math::Isometry3d& cameraInWorld,
                              const model::CameraIntrinsicParams& cameraParams,
                              core::logic::collision::CollisionEngine& collisionEngine,
                              core::map::MapState& mapState,
                              const model::RobotState& robotState) noexcept;
    [[nodiscard]] static std::set<boost::uuids::uuid> GetAllSublinksOfDoorLink(const core::DoorLink* doorLink) noexcept;
    [[nodiscard]] static bool CheckAngleForCollision(const core::DoorLink* doorLink,
                                                     std::set<boost::uuids::uuid> doorChildLinksUuids,
                                                     core::logic::collision::CollisionEngine& collisionEngine,
                                                     core::map::MapState& mapState,
                                                     const model::RobotState& robotState,
                                                     double angle) noexcept;
    [[nodiscard]] std::vector<double> static
    ScoreAngleOfStallDoor(const core::DoorLink* doorLink,
                          const math::Isometry3d& doorPositionInWorld,
                          const math::Isometry3d& cameraPositionInWorld,
                          const std::vector<cv::Mat>& masks,
                          const model::CameraIntrinsicParams& cameraParams,
                          double angle) noexcept;
    [[nodiscard]] static std::array<math::Vector3d, 8>
    CreateDoorCornersInCoordinateSystem(const model::DoorData& doorData,
                                        const math::Isometry3d& doorPositionInWorld,
                                        const math::Isometry3d& cameraPositionInWorld,
                                        double angle) noexcept;

    [[nodiscard]] static cv::Mat CreateDoorMask(const std::array<math::Vector3d, 8>& doorCornersInCamera,
                                                const model::CameraIntrinsicParams& cameraParams,
                                                size_t imageWidth,
                                                size_t imageHeight) noexcept;

    [[nodiscard]] static std::vector<math::Polygon3D>
    CreateDoorFaces(const std::array<math::Vector3d, 8>& doorCornersInCamera);

    void ShowResultForDoor(const bcr::core::DoorLink* doorLink, model::GrayscaleImage mask) noexcept;

    [[nodiscard]] static bool IsDoorLink(core::Link::UPtr::pointer link) noexcept;
    [[nodiscard]] static bool IsRegularDoor(core::Link::UPtr::pointer link) noexcept;
    [[nodiscard]] static bool IsOutOfObservationRange(core::Link::UPtr::pointer link,
                                                      const model::RobotState& robotState,
                                                      const core::map::MapState& mapState) noexcept;

private: // immutable state
    transport::visualization::VisualLogger& visualLogger_;
    detector::VisualLogHelper visualLogHelper_;
    const model::CameraName cameraName_;

private: // mutable state
    std::chrono::high_resolution_clock::time_point lastPerceptionTime_;
};
} // namespace bcr::robot::logic::perception

#endif // BCR_ROBOT_LOGIC_PERCEPTION_PERCEPTOR_STALL_DOOR_PERCEPTOR_HH_
