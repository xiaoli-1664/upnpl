#pragma once

#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

struct Camera {
    double fx, fy, cx, cy;
    string distortion_model;
    vector<double> distortion_coeffs;
    vector<double> image_size;
    Eigen::Isometry3d Tbc;

    cv::Mat map1, map2;

    bool need_rictified = false;

    void backProjectPixel(double u, double v, Eigen::Vector3d &point) const {
        point(0) = (u - cx) / fx;
        point(1) = (v - cy) / fy;
        point(2) = 1.0;    // Assuming unit depth for back-projection
        point.normalize(); // Normalize to make it a unit vector
    }

    void getRectifiedFisheyeCamera() {
        cv::Size image_size_cv(image_size[0], image_size[1]);

        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat D = cv::Mat(distortion_coeffs);

        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

        cv::Mat newK;

        double alpha = 1.0;

        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            K, D, image_size_cv, R, newK, alpha);

        fx = newK.at<double>(0, 0);
        fy = newK.at<double>(1, 1);
        cx = newK.at<double>(0, 2);
        cy = newK.at<double>(1, 2);

        cv::fisheye::initUndistortRectifyMap(K, D, R, newK, image_size_cv,
                                             CV_32FC1, map1, map2);
        need_rictified = true;
    }

    void getRecitifiedPinholeCamera() {
        cv::Size image_size_cv(image_size[0], image_size[1]);
        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat D = cv::Mat(distortion_coeffs);

        cv::Mat newK;
        double alpha = 1.0;

        newK = cv::getOptimalNewCameraMatrix(K, D, image_size_cv, alpha,
                                             image_size_cv);

        cv::initUndistortRectifyMap(K, distortion_coeffs, cv::Mat(), newK,
                                    image_size_cv, CV_32FC1, map1, map2);
        fx = newK.at<double>(0, 0);
        fy = newK.at<double>(1, 1);
        cx = newK.at<double>(0, 2);
        cy = newK.at<double>(1, 2);
        need_rictified = true;
    }
};

Eigen::MatrixXd cvMatToEigen(const cv::Mat &mat) {
    Eigen::MatrixXd eigen_mat(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            eigen_mat(i, j) = mat.at<double>(i, j);
        }
    }
    return eigen_mat;
}

cv::Mat eigenToCvMat(const Eigen::MatrixXd &eigen_mat) {
    cv::Mat mat(eigen_mat.rows(), eigen_mat.cols(), CV_64F);
    for (int i = 0; i < eigen_mat.rows(); ++i) {
        for (int j = 0; j < eigen_mat.cols(); ++j) {
            mat.at<double>(i, j) = eigen_mat(i, j);
        }
    }
    return mat;
}

Eigen::Vector3d rotToCGR(const Eigen::Matrix3d &R) {
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d RpI = R + I;
    if (RpI.determinant() < 1e-6) {
        cerr << "Error: Rotation matrix is singular." << endl;
        return Eigen::Vector3d::Zero();
    }

    Eigen::Matrix3d A = (R - I) * RpI.inverse();

    Eigen::Vector3d caley;
    caley << A(2, 1), A(0, 2), A(1, 0);
    return caley;
}

void triangulateLines(const vector<Eigen::Vector3d> &normals_c1,
                      const vector<Eigen::Vector3d> &normals_c2,
                      Eigen::Isometry3d T_cw1, Eigen::Isometry3d T_cw2,
                      vector<Eigen::VectorXd> &lines_w, vector<bool> &valid) {
    Eigen::Isometry3d T_wc1 = T_cw1.inverse();
    Eigen::Isometry3d T_wc2 = T_cw2.inverse();

    Eigen::Matrix3d R_wc1 = T_wc1.linear();
    Eigen::Matrix3d R_wc2 = T_wc2.linear();
    Eigen::Vector3d t_wc1 = T_wc1.translation();
    Eigen::Vector3d t_wc2 = T_wc2.translation();

    for (int i = 0; i < normals_c1.size(); ++i) {
        Eigen::Vector3d normal_w1 = R_wc1 * normals_c1[i];
        Eigen::Vector3d normal_w2 = R_wc2 * normals_c2[i];

        Eigen::Vector4d pi1;
        pi1.head<3>() = normal_w1;
        pi1(3) = -normal_w1.dot(t_wc1);

        Eigen::Vector4d pi2;
        pi2.head<3>() = normal_w2;
        pi2(3) = -normal_w2.dot(t_wc2);

        Eigen::Matrix<double, 2, 4> W;
        W.row(0) = pi1;
        W.row(1) = pi2;

        Eigen::JacobiSVD<Eigen::Matrix<double, 2, 4>> svd(W,
                                                          Eigen::ComputeFullV);
        Eigen::MatrixXd V = svd.matrixV();
        Eigen::VectorXd S = svd.singularValues();

        Eigen::Vector4d p1 = V.col(3);
        Eigen::Vector4d p2 = V.col(2);
        p1 /= p1(3); // Normalize by w
        p2 /= p2(3); // Normalize by w

        Eigen::VectorXd line(6);
        line.head<3>() = p1.head<3>();
        line.tail<3>() = p2.head<3>();
        lines_w.push_back(line);

        if (fabs(S(1)) < 1e-6)
            valid.push_back(false);
        else
            valid.push_back(true);
    }
}

void generatePnPLData(int index, const vector<vector<string>> &image_files,
                      const vector<vector<double>> &times,
                      const vector<vector<Eigen::Isometry3d>> &poses_gt,
                      const vector<Camera> &cameras,
                      vector<Eigen::Vector3d> &points_w,
                      vector<Eigen::VectorXd> &lines_w,
                      vector<Eigen::Vector3d> &uv_c,
                      vector<Eigen::Vector3d> &normals_c,
                      vector<int> &points_cam, vector<int> &lines_cam) {
    int num_cameras = cameras.size();
    const int num_features = 200;

    for (int cam_id = 1; cam_id < num_cameras; ++cam_id) {
        const Camera &camera = cameras[cam_id];
        const string &image_file = image_files[cam_id][index];
        const string &image_file_next = image_files[cam_id][index + 1];
        Eigen::Isometry3d pose_gt = poses_gt[cam_id][index];
        double time = times[cam_id][index];

        cv::Mat image = cv::imread(image_file, cv::IMREAD_GRAYSCALE);
        cv::Mat image_next = cv::imread(image_file_next, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "Error: Could not read image " << image_file << endl;
            continue;
        }
        if (image_next.empty()) {
            cerr << "Error: Could not read image " << image_file_next << endl;
            continue;
        }

        int j = index + 1;
        int cam_j_id = cam_id;
        if (cam_id > 1) {
            while (j < image_files[cam_id].size()) {
                Eigen::Isometry3d pose_gt_next = poses_gt[cam_id][j];
                Eigen::Isometry3d deltaT = pose_gt_next.inverse() * pose_gt;
                if (deltaT.translation().norm() > 0.025) {
                    break;
                }
                ++j;
            }
            cam_j_id = cam_id;
        } else {
            j = index;
            cam_j_id = 0;
        }

        if (camera.need_rictified) {
            cv::Mat undistorted_image;
            cv::remap(image, undistorted_image, camera.map1, camera.map2,
                      cv::INTER_LINEAR);
            image = undistorted_image;

            cv::Mat undistorted_image_next;
            cv::remap(image_next, undistorted_image_next, camera.map1,
                      camera.map2, cv::INTER_LINEAR);
            image_next = undistorted_image_next;
        }

        cv::Mat image1;
        Eigen::Isometry3d pose_gt1;

        image1 = cv::imread(image_files[cam_j_id][j], cv::IMREAD_GRAYSCALE);
        if (image1.empty()) {
            cerr << "Error: Could not read image " << image_files[cam_j_id][j]
                 << endl;
            continue;
        }
        pose_gt1 = poses_gt[cam_j_id][j];
        if (cameras[cam_j_id].need_rictified) {
            cv::Mat undistorted_image1;
            cv::remap(image1, undistorted_image1, cameras[cam_j_id].map1,
                      cameras[cam_j_id].map2, cv::INTER_LINEAR);
            image1 = undistorted_image1;
        }

        cv::Mat K = (cv::Mat_<double>(3, 3) << camera.fx, 0, camera.cx, 0,
                     camera.fy, camera.cy, 0, 0, 1);
        cv::Mat K1 = (cv::Mat_<double>(3, 3) << cameras[cam_j_id].fx, 0,
                      cameras[cam_j_id].cx, 0, cameras[cam_j_id].fy,
                      cameras[cam_j_id].cy, 0, 0, 1);
        Eigen::Isometry3d T_cw = (pose_gt * camera.Tbc).inverse();
        cv::Mat R_cw = eigenToCvMat(T_cw.linear());
        cv::Mat t_cw = eigenToCvMat(T_cw.translation());

        Eigen::Isometry3d T_cw1 = (pose_gt1 * cameras[cam_j_id].Tbc).inverse();
        cv::Mat R_cw1 = eigenToCvMat(T_cw1.linear());
        cv::Mat t_cw1 = eigenToCvMat(T_cw1.translation());

        cv::Mat P = cv::Mat::zeros(3, 4, CV_64F);
        R_cw.copyTo(P.rowRange(0, 3).colRange(0, 3));
        t_cw.copyTo(P.rowRange(0, 3).col(3));
        P = K * P;

        cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
        R_cw1.copyTo(P1.rowRange(0, 3).colRange(0, 3));
        t_cw1.copyTo(P1.rowRange(0, 3).col(3));
        P1 = K1 * P1;

        cv::Ptr<cv::ORB> orb = cv::ORB::create(num_features);
        vector<cv::KeyPoint> keypoints1, keypoints2, keypoints_next;
        cv::Mat descriptors1, descriptors2, descriptors_next;
        orb->detectAndCompute(image, cv::noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(image1, cv::noArray(), keypoints2, descriptors2);
        orb->detectAndCompute(image_next, cv::noArray(), keypoints_next,
                              descriptors_next);

        cv::BFMatcher matcher(cv::NORM_HAMMING);
        vector<vector<cv::DMatch>> matches, matches_next;
        matcher.knnMatch(descriptors1, descriptors2, matches, 2);
        matcher.knnMatch(descriptors1, descriptors_next, matches_next, 2);

        vector<cv::DMatch> good_matches;
        for (const auto &match : matches) {
            if (match.size() > 1 &&
                match[0].distance < 0.75 * match[1].distance) {
                good_matches.push_back(match[0]);
            }
        }

        vector<cv::DMatch> good_matches_next;
        for (const auto &match : matches_next) {
            if (match.size() > 1 &&
                match[0].distance < 0.75 * match[1].distance) {
                good_matches_next.push_back(match[0]);
            }
        }

        vector<cv::Point2f> points1, points2;
        vector<int> indexes1(keypoints1.size(), -1);
        vector<int> indexes2(keypoints2.size(), -1);
        int match_i = 0;
        for (const auto &match : good_matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            indexes1[match.queryIdx] = match_i;
            points2.push_back(keypoints2[match.trainIdx].pt);
            indexes2[match.trainIdx] = match_i;
            match_i++;
        }
        cv::Mat points4D;
        cv::Mat pts1_h, pts2_h;
        cv::Mat(points1).convertTo(pts1_h, CV_64F);
        cv::Mat(points2).convertTo(pts2_h, CV_64F);
        pts1_h = pts1_h.t();
        pts2_h = pts2_h.t();

        cv::triangulatePoints(P, P1, pts1_h, pts2_h, points4D);

        for (const auto &match : good_matches_next) {
            int col = indexes1[match.queryIdx];
            if (col != -1) {
                Eigen::Vector3d point_w;
                point_w(0) =
                    points4D.at<double>(0, col) / points4D.at<double>(3, col);
                point_w(1) =
                    points4D.at<double>(1, col) / points4D.at<double>(3, col);
                point_w(2) =
                    points4D.at<double>(2, col) / points4D.at<double>(3, col);

                if (point_w.norm() < 0.1 || point_w.norm() > 100.0) {
                    continue; // Skip invalid points
                }

                points_w.push_back(point_w);
                points_cam.push_back(cam_id);
                Eigen::Vector3d uv;
                camera.backProjectPixel(keypoints_next[match.trainIdx].pt.x,
                                        keypoints_next[match.trainIdx].pt.y,
                                        uv);
                uv_c.push_back(uv);
            }
        }

        const string &image_file0_next = image_files[0][index + 1];
        cv::Mat image0_next;

        if (cam_id == 1) {
            image0_next = cv::imread(image_file0_next, cv::IMREAD_GRAYSCALE);
            if (image0_next.empty()) {
                cerr << "Error: Could not read image " << image_file0_next
                     << endl;
                continue;
            }

            if (cameras[0].need_rictified) {
                cv::Mat undistorted_image0_next;
                cv::remap(image0_next, undistorted_image0_next, cameras[0].map1,
                          cameras[0].map2, cv::INTER_LINEAR);
                image0_next = undistorted_image0_next;
            }

            vector<cv::KeyPoint> keypoints0_next;
            cv::Mat descriptors0_next;
            orb->detectAndCompute(image0_next, cv::noArray(), keypoints0_next,
                                  descriptors0_next);
            vector<vector<cv::DMatch>> matches0_next;
            matcher.knnMatch(descriptors2, descriptors0_next, matches0_next, 2);
            vector<cv::DMatch> good_matches0_next;
            for (const auto &match : matches0_next) {
                if (match.size() > 1 &&
                    match[0].distance < 0.75 * match[1].distance) {
                    good_matches0_next.push_back(match[0]);
                }
            }

            for (const auto &match : good_matches0_next) {
                int col = indexes2[match.queryIdx];
                if (col != -1) {
                    Eigen::Vector3d point_w;
                    point_w(0) = points4D.at<double>(0, col) /
                                 points4D.at<double>(3, col);
                    point_w(1) = points4D.at<double>(1, col) /
                                 points4D.at<double>(3, col);
                    point_w(2) = points4D.at<double>(2, col) /
                                 points4D.at<double>(3, col);
                    if (point_w.norm() < 0.1 || point_w.norm() > 100.0) {
                        continue; // Skip invalid points
                    }

                    points_w.push_back(point_w);
                    points_cam.push_back(0);
                    Eigen::Vector3d uv;
                    cameras[0].backProjectPixel(
                        keypoints0_next[match.trainIdx].pt.x,
                        keypoints0_next[match.trainIdx].pt.y, uv);
                    uv_c.push_back(uv);
                }
            }
        }

        cv::Ptr<cv::line_descriptor::LSDDetector> lsd =
            cv::line_descriptor::LSDDetector::createLSDDetector();
        vector<cv::line_descriptor::KeyLine> keylines1, keylines2,
            keylines_next;
        cv::Mat lbd_descriptors1, lbd_descriptors2, lbd_descriptors_next;
        lsd->detect(image, keylines1, 2, 1);
        lsd->detect(image1, keylines2, 2, 1);
        lsd->detect(image_next, keylines_next, 2, 1);

        cv::Ptr<cv::line_descriptor::BinaryDescriptor> bd =
            cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
        bd->compute(image, keylines1, lbd_descriptors1);
        bd->compute(image1, keylines2, lbd_descriptors2);
        bd->compute(image_next, keylines_next, lbd_descriptors_next);

        if (keylines1.empty() || keylines2.empty() || keylines_next.empty()) {
            cerr << "No keylines detected in images." << endl;
            continue;
        }

        cv::BFMatcher lbd_matcher(cv::NORM_HAMMING);
        vector<vector<cv::DMatch>> lbd_matches;

        lbd_matcher.knnMatch(lbd_descriptors1, lbd_descriptors2, lbd_matches,
                             2);

        vector<cv::DMatch> good_lbd_matches;
        for (const auto &match : lbd_matches) {
            if (match.size() > 1 &&
                match[0].distance < 0.75 * match[1].distance) {
                good_lbd_matches.push_back(match[0]);
            }
        }

        vector<Eigen::Vector3d> normals_c_tmp, normals_c1_tmp;
        vector<int> keyline_indexes1(keylines1.size(), -1);
        vector<int> keyline_indexes2(keylines2.size(), -1);
        match_i = 0;
        for (const auto &match : good_lbd_matches) {
            const auto &keyline1 = keylines1[match.queryIdx];
            const auto &keyline2 = keylines2[match.trainIdx];

            if (keyline1.lineLength < 25 || keyline2.lineLength < 25) {
                continue; // Skip short keylines
            }

            Eigen::Vector3d uv_start;
            camera.backProjectPixel(keyline1.startPointX, keyline1.startPointY,
                                    uv_start);
            Eigen::Vector3d uv_end;
            camera.backProjectPixel(keyline1.endPointX, keyline1.endPointY,
                                    uv_end);
            Eigen::Vector3d normal_c = uv_start.cross(uv_end);

            Eigen::Vector3d uv_start1;
            cameras[cam_j_id].backProjectPixel(keyline2.startPointX,
                                               keyline2.startPointY, uv_start1);
            Eigen::Vector3d uv_end1;
            cameras[cam_j_id].backProjectPixel(keyline2.endPointX,
                                               keyline2.endPointY, uv_end1);
            Eigen::Vector3d normal_c1 = uv_start1.cross(uv_end1);

            normals_c_tmp.push_back(normal_c.normalized());
            keyline_indexes1[match.queryIdx] = match_i;
            normals_c1_tmp.push_back(normal_c1.normalized());
            keyline_indexes2[match.trainIdx] = match_i;
            match_i++;
        }

        vector<Eigen::VectorXd> lines_w_temp;
        vector<bool> valid;
        triangulateLines(normals_c_tmp, normals_c1_tmp, T_cw, T_cw1,
                         lines_w_temp, valid);

        cv::BFMatcher lbd_matcher_next(cv::NORM_HAMMING);
        vector<vector<cv::DMatch>> lbd_matches_next;

        lbd_matcher_next.knnMatch(lbd_descriptors1, lbd_descriptors_next,
                                  lbd_matches_next, 2);

        vector<cv::DMatch> good_lbd_matches_next;
        for (const auto &match : lbd_matches_next) {
            if (match.size() > 1 &&
                match[0].distance < 0.75 * match[1].distance) {
                good_lbd_matches_next.push_back(match[0]);
            }
        }

        for (const auto &match : good_lbd_matches_next) {
            int col = keyline_indexes1[match.queryIdx];
            if (col != -1 && valid[col]) {
                const auto &keyline_next = keylines_next[match.trainIdx];
                lines_w.push_back(lines_w_temp[col]);
                lines_cam.push_back(cam_id);

                Eigen::Vector3d uv_start;
                camera.backProjectPixel(keyline_next.startPointX,
                                        keyline_next.startPointY, uv_start);
                Eigen::Vector3d uv_end;
                camera.backProjectPixel(keyline_next.endPointX,
                                        keyline_next.endPointY, uv_end);
                Eigen::Vector3d normal_c = uv_start.cross(uv_end);

                normals_c.push_back(normal_c.normalized());
            }
        }

        if (cam_id == 1) {
            vector<cv::line_descriptor::KeyLine> keylines0_next;
            cv::Mat lbd_descriptors0_next;
            lsd->detect(image0_next, keylines0_next, 2, 1);
            bd->compute(image0_next, keylines0_next, lbd_descriptors0_next);
            if (keylines0_next.empty()) {
                cerr << "No keylines detected in image0_next." << endl;
                continue;
            }

            vector<vector<cv::DMatch>> lbd_matches0_next;
            lbd_matcher_next.knnMatch(lbd_descriptors2, lbd_descriptors0_next,
                                      lbd_matches0_next, 2);
            vector<cv::DMatch> good_lbd_matches0_next;
            for (const auto &match : lbd_matches0_next) {
                if (match.size() > 1 &&
                    match[0].distance < 0.75 * match[1].distance) {
                    good_lbd_matches0_next.push_back(match[0]);
                }
            }

            for (const auto &match : good_lbd_matches0_next) {
                int col = keyline_indexes2[match.queryIdx];
                if (col != -1 && valid[col]) {
                    const auto &keyline_next = keylines0_next[match.trainIdx];
                    lines_w.push_back(lines_w_temp[col]);
                    lines_cam.push_back(0);

                    Eigen::Vector3d uv_start;
                    cameras[0].backProjectPixel(keyline_next.startPointX,
                                                keyline_next.startPointY,
                                                uv_start);
                    Eigen::Vector3d uv_end;
                    cameras[0].backProjectPixel(keyline_next.endPointX,
                                                keyline_next.endPointY, uv_end);
                    Eigen::Vector3d normal_c = uv_start.cross(uv_end);

                    normals_c.push_back(normal_c.normalized());
                }
            }
        }
    }
}
