//Modifications copyright (C) 2025 SOMATIC

#include "bcr_robot/logic/perception/perceptor/stall_door_perceptor.hh"

#include <bcr_vision/conversions.hh>

#include <bcr_transport/visualization/graphics.hh>
#include <bcr_transport/visualization/visual_logger.hh>

#include <bcr_model/camera/camera_intrinsic_parameters.hh>
#include <bcr_model/map/joint/joint_state.hh>
#include <bcr_model/robot/state/robot_state.hh>

#include <bcr_common/log/log.hh>

#include <somatic_math/Plane.hh>
#include <somatic_math/Polygon3D.hh>

#include <opencv2/imgproc.hpp>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <chrono>
#include <iomanip>
#include <set>
#include <stack>
#include <vector>

namespace bcr::robot::logic::perception {

StallDoorPerceptor::StallDoorPerceptor(
        model::CameraName cameraName,
        transport::visualization::VisualLogger& visualLogger
) noexcept:
        loggerName_(baseLoggerName_ + ToStdString(cameraName)),
        visualLogger_(visualLogger),
        visualLogHelper_(&visualLogger, loggerName_, true),
        cameraName_(cameraName),
        lastPerceptionTime_(std::chrono::high_resolution_clock::now()) {}

std::vector<bcr::core::DoorLink*> StallDoorPerceptor::FindEligibleDoors(const std::vector<core::Link*>& links,
                                                                        const model::RobotState& robotState,
                                                                        const core::map::MapState& mapState) noexcept {
    std::vector<bcr::core::DoorLink*> eligibleDoors;
    for (const auto& link : links) {
        if (!IsDoorLink(link) or IsRegularDoor(link) or IsOutOfObservationRange(link, robotState, mapState))
            continue;

        eligibleDoors.push_back(dynamic_cast<bcr::core::DoorLink*>(link));

    }
    return eligibleDoors;
}

bool StallDoorPerceptor::IsDoorLink(core::Link::UPtr::pointer link) noexcept {
    return link->Type() == core::LinkType::DoorRoot;
}

bool StallDoorPerceptor::IsRegularDoor(core::Link::UPtr::pointer link) noexcept {
    auto doorLink = dynamic_cast<bcr::core::DoorLink*>(link);
    if (!doorLink) {
        return false;
    }
    auto doorData = doorLink->GetDoorData();
    return doorData.HandleData(bcr::model::DoorSide::Counterclockwise).has_value()
            or doorData.HandleData(bcr::model::DoorSide::Clockwise).has_value();
}

//todo add if door in camera view

bool StallDoorPerceptor::IsOutOfObservationRange(core::Link::UPtr::pointer link,
                                                 const model::RobotState& robotState,
                                                 const core::map::MapState& mapState) noexcept {
    auto robotBaseInWorld = robotState.RobotBaseInWorld();
    auto doorLinkInWorld = link->GlobalTransform(mapState);
    return (robotBaseInWorld.Translation() - doorLinkInWorld.Translation()).Norm() > maxSupportedDoorDistance_;
}

void StallDoorPerceptor::Draw(core::DoorLink* doorLink, math::Isometry3d doorPosition, double angle) noexcept {
    transport::visualization::Graphics g;
    for (const auto& polygon : CreateDoorFaces(CreateDoorCornersInCoordinateSystem(doorLink->GetDoorData(),
                                                                                   doorPosition,
                                                                                   math::Isometry3d::Identity(),
                                                                                   angle))) {
        g.DrawLines3D(polygon.Sides(), transport::visualization::color::OLIVE);
    }
    visualLogger_.Draw("Perception result for door: {}" + boost::uuids::to_string(doorLink->Uuid()),
                       "Perception result for door: {}" + boost::uuids::to_string(doorLink->Uuid()),
                       g);
}

core::map::JointStateById StallDoorPerceptor::Percept(
        const StallDoorPerceptingData& perceptingData,
        core::logic::collision::CollisionEngine& collisionEngine,
        core::map::MapState& mapState,
        const model::RobotState& robotState) noexcept {
    auto masks = perceptingData.Masks;

    core::map::JointStateById modifiers;

    auto results = MatchDoorsAndMasks(perceptingData, collisionEngine, mapState, robotState);

    for (auto [doorLink, doorAngle] : results) {
        modifiers.insert_or_assign(
                doorLink->DoorBoxJoint()->Uuid(), model::OneDOFJointState::CreateRevolute(doorAngle));
    }

    return modifiers;
}

std::vector<std::pair<const bcr::core::DoorLink*, double>>
StallDoorPerceptor::MatchDoorsAndMasks(const StallDoorPerceptingData& data,
                                       core::logic::collision::CollisionEngine& collisionEngine,
                                       core::map::MapState& mapState,
                                       const model::RobotState& robotState) noexcept {
    if (data.Doors.empty() || data.Masks.empty()) {
        return {};
    }
    std::vector<std::vector<std::pair<double, double>>> resultMatrix;
    for (auto [door, doorPosition] : data.Doors) {
        auto angles = GetBestAnglesForStallDoor(door,
                                                doorPosition,
                                                data.Masks,
                                                data.CameraInWorld,
                                                data.CameraParams,
                                                collisionEngine,
                                                mapState,
                                                robotState);
        resultMatrix.push_back(angles);
    }
    std::set<size_t> usedDoors, usedMasks;
    size_t totalDoors = data.Doors.size();
    size_t totalMasks = data.Masks.size();
    for (size_t i = 0; i < totalDoors; i++) {
        std::stringstream s;
        s << std::fixed << std::setprecision(3);
        for (size_t j = 0; j < totalMasks; j++) {
            s << resultMatrix[i][j].first << ' ';
        }
    }
    std::vector<std::pair<const bcr::core::DoorLink*, double>> result;
    while (true) {
        double bestScore = 0, bestAngle = 0;
        size_t bestDoor = 0, bestMask = 0;
        for (size_t i = 0; i < totalDoors; i++) {
            for (size_t j = 0; j < totalMasks; j++) {
                if (bestScore < resultMatrix[i][j].first && !usedDoors.contains(i) && !usedMasks.contains(j)) {
                    bestScore = resultMatrix[i][j].first;
                    bestAngle = resultMatrix[i][j].second;
                    bestDoor = i;
                    bestMask = j;
                }
            }
        }
        if (bestScore < 0.5) {
            break;
        }
        result.push_back({data.Doors[bestDoor].first, resultMatrix[bestDoor][bestMask].second});
        Draw(data.Doors[bestDoor].first, data.Doors[bestDoor].second, bestAngle);
        usedDoors.insert(bestDoor);
        usedMasks.insert(bestMask);
        BCR_DEBUG_STREAM(loggerName_,
                         "Chosen door: {" << bestDoor << "} and mask: {" << bestMask << "}" << "at angle: {"
                                          << resultMatrix[bestDoor][bestMask].second << "}");

    }
    return result;
}

std::vector<std::pair<double, double>> StallDoorPerceptor::GetBestAnglesForStallDoor(
        const bcr::core::DoorLink* doorLink,
        const math::Isometry3d& doorPosition,
        const std::vector<model::GrayscaleImage>& masks,
        const math::Isometry3d& cameraInWorld,
        const model::CameraIntrinsicParams& cameraParams,
        core::logic::collision::CollisionEngine& collisionEngine,
        core::map::MapState& mapState,
        const model::RobotState& robotState
) noexcept {

    auto doorData = doorLink->GetDoorData();

    std::vector<cv::Mat> doorMasks;
    for (auto& mask : masks) {
        doorMasks.push_back(vision::Conversions::ImageToMatrix(mask));
    }

    auto childLinksUUIDS = GetAllSublinksOfDoorLink(doorLink);

    std::vector<std::pair<double, double>> bestScores(masks.size());
    for (double angle = doorData.MinAngle; angle <= doorData.MaxAngle; angle += 3._deg) {
        auto isAngleInCollision = CheckAngleForCollision(doorLink,
                                                         childLinksUUIDS,
                                                         collisionEngine,
                                                         mapState,
                                                         robotState,
                                                         angle);
        if (isAngleInCollision) {
            continue;
        }
        auto angleScore = ScoreAngleOfStallDoor(
                doorLink,
                doorPosition,
                cameraInWorld,
                doorMasks,
                cameraParams,
                angle);

        for (size_t i = 0; i < masks.size(); i++) {
            if (angleScore[i] > bestScores[i].first) {
                bestScores[i].
                        first = angleScore[i];
                bestScores[i].
                        second = angle;
            }
        }

    }
    return bestScores;
}

bool StallDoorPerceptor::CheckAngleForCollision(const core::DoorLink* doorLink,
                                                std::set<boost::uuids::uuid> doorChildLinksUuids,
                                                core::logic::collision::CollisionEngine& collisionEngine,
                                                core::map::MapState& mapState,
                                                const model::RobotState& robotState,
                                                double angle) noexcept {

    mapState.UpdateJoint(doorLink->DoorBoxJoint(), angle);
    collisionEngine.UpdateMapStateFromLink(core::logic::collision::WorldType::NoMargins, doorLink, mapState);

    auto collisionsList = collisionEngine.CollisionsList(robotState, core::logic::collision::WorldType::NoMargins);

    for (auto collision : collisionsList) {
        if (doorChildLinksUuids.contains(collision.ObjectData1.LinkUuid)
                or doorChildLinksUuids.contains(collision.ObjectData2.LinkUuid)) {
            return true;
        }
    }

    return false;
}

std::set<boost::uuids::uuid> StallDoorPerceptor::GetAllSublinksOfDoorLink(const core::DoorLink* doorLink) noexcept {
    std::set<boost::uuids::uuid> childrenOfDoor;
    std::stack<core::Link*> currentChildren;
    for (const auto& joint : doorLink->Joints()) {
        currentChildren.push(joint->Link());
    }
    while (!currentChildren.empty()) {
        auto child = currentChildren.top();
        currentChildren.pop();
        childrenOfDoor.insert(child->Uuid());
        for (const auto& joint : child->Joints()) {
            currentChildren.push(joint->Link());
        }
    }
    return childrenOfDoor;
}

std::vector<double>
StallDoorPerceptor::ScoreAngleOfStallDoor(const core::DoorLink* doorLink,
                                          const math::Isometry3d& doorPositionInWorld,
                                          const math::Isometry3d& cameraPositionInWorld,
                                          const std::vector<cv::Mat>& masks,
                                          const model::CameraIntrinsicParams& cameraParams,
                                          double angle
) noexcept {
    auto doorData = doorLink->GetDoorData();

    auto doorCorners = CreateDoorCornersInCoordinateSystem(doorData, doorPositionInWorld, cameraPositionInWorld, angle);

    cv::Mat doorMask = CreateDoorMask(doorCorners, cameraParams, masks[0].cols, masks[0].rows);

    std::vector<double> scores(masks.size());
    for (size_t i = 0; i < masks.size(); i++) {
        auto mask = masks[i];
        double overlap = cv::countNonZero(doorMask & mask);
        double unionOfMasks = cv::countNonZero(doorMask | mask);
        double score = static_cast<double>(overlap) / unionOfMasks;
        scores[i] = score;
    }

    return scores;
}

std::array<math::Vector3d, 8>
StallDoorPerceptor::CreateDoorCornersInCoordinateSystem(const model::DoorData& doorData,
                                                        const math::Isometry3d& doorPositionInWorld,
                                                        const math::Isometry3d& cameraPositionInWorld,
                                                        double angle) noexcept {
    std::array<math::Vector3d, 8> doorCornersInLocal;
    for (int i = 0; i < 8; i++) {
        doorCornersInLocal.at(i) +=
                {doorData.BoxToHingesOffsetX,
                 doorData.BoxToHingesOffsetY - doorData.Thickness / 2.0,
                 doorData.FloorOffset};
        if (i & 1) {
            doorCornersInLocal.at(i) += {doorData.Width, 0, 0};
        }
        if (i & 2) {
            doorCornersInLocal.at(i) += {0, 0, doorData.Height};
        }
        if (i & 4) {
            doorCornersInLocal.at(i) += {0, doorData.Thickness, 0};
        }
    }
    for (int i = 0; i < 8; i++) {
        doorCornersInLocal.at(i) = math::Isometry3d::XyzRpy(0, 0, 0, 0, 0, angle) * doorCornersInLocal.at(i);
    }

    std::array<math::Vector3d, 8> doorCornersInWorld;

    for (int i = 0; i < 8; i++) {
        doorCornersInWorld.at(i) = doorPositionInWorld * doorCornersInLocal.at(i);
    }

    std::array<math::Vector3d, 8> doorCornersInCamera;

    for (int i = 0; i < 8; i++) {
        doorCornersInCamera.at(i) = cameraPositionInWorld.Inverse() * doorCornersInWorld.at(i);
    }

    return doorCornersInCamera;
}
cv::Mat StallDoorPerceptor::CreateDoorMask(const std::array<math::Vector3d, 8>& doorCornersInCamera,
                                           const model::CameraIntrinsicParams& cameraParams,
                                           size_t imageWidth,
                                           size_t imageHeight) noexcept {

    std::vector<math::Polygon3D> faces = CreateDoorFaces(doorCornersInCamera);

    double fX = cameraParams.Fx;
    double fY = cameraParams.Fy;
    double ppX = cameraParams.Ppx;
    double ppY = cameraParams.Ppy;

    std::vector<math::Plane> cameraFrustrum{
            math::Plane::CreateByOrigin({0, 0, 0}, {fX, 0, ppX}), //left plane
            math::Plane::CreateByOrigin({0, 0, 0}, {-fX, 0, imageWidth - ppX}), //right plane
            math::Plane::CreateByOrigin({0, 0, 0}, {0, fY, ppY}), //top plane
            math::Plane::CreateByOrigin({0, 0, 0}, {0, -fY, imageHeight - ppY}), //bot plane
            math::Plane::CreateByOrigin({0, 0, nearPlaneDistance_}, {0, 0, 1}) //near plane
    };

    for (auto& face : faces) {
        for (auto cuttingPlane : cameraFrustrum) {
            face = face.CutByPlane(cuttingPlane);
        }
    }

    auto projectToImage = [&](const math::Vector3d& pt) -> cv::Point {
        double x = pt.x();
        double y = pt.y();
        double z = pt.z();

        if (z <= 0) {
            sassert_fail("All points need to bew visible from the camera!");
        }

        double imageX = (fX * x / z) + ppX;
        double imageY = (fY * y / z) + ppY;

        return cv::Point(static_cast<int>(imageX), static_cast<int>(imageY));
    };

    std::vector<std::vector<cv::Point>> polygons;

    for (auto& face : faces) {
        if (face.Points().size() <= 2) {
            continue;
        }
        std::vector<cv::Point> polygon;
        for (auto point : face.Points()) {
            polygon.push_back(projectToImage(point));
        }
        polygon.push_back(polygon[0]);
        polygons.push_back(polygon);
    }

    auto imageSize = cv::Size(imageWidth, imageHeight);
    cv::Mat doorCast = cv::Mat::zeros(imageSize, CV_8UC1);
    for (auto& polygon : polygons) {
        cv::fillPoly(doorCast, {polygon}, cv::Scalar(255));
    }
    return doorCast;
}

std::vector<math::Polygon3D> StallDoorPerceptor::CreateDoorFaces(const std::array<math::Vector3d,
                                                                                  8>& doorCornersInCamera) {
    std::vector<math::Polygon3D> faces;

    for (auto indices : facesIndices_) {

        auto plane_opt = math::Plane::CreateByPoints(doorCornersInCamera[indices[0]],
                                                     doorCornersInCamera[indices[1]],
                                                     doorCornersInCamera[indices[2]]);
        sassert_has_value(plane_opt, "Door was constructed wrong. Probably corrupted door data");
        auto plane = plane_opt.value();

        std::vector<math::Vector2d> pointsForPolygon2D;
        pointsForPolygon2D.reserve(4);
        for (auto index : indices) {
            pointsForPolygon2D.push_back(plane.To2D(doorCornersInCamera[index]));
        }
        faces.push_back(math::Polygon3D(math::Polygon(pointsForPolygon2D), plane));
    }

    return faces;
}

}
