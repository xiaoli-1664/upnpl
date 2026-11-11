#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace simulation {
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::VectorXd;

struct Config {
    std::string name;
    int cameras = 4, points = 12, lines = 8;
    // Negative means ordinary forward-facing pinhole observations.
    double h = -1;
    bool normalize = true;
    double coordinate_scale = 1, bearing_scale = 1;
};

struct Scene {
    std::vector<Vector3d> points, bearings, tbc;
    std::vector<VectorXd> lines, endpoints;
    std::vector<Matrix3d> Rbc;
    std::vector<int> point_cameras, line_cameras;
    Matrix3d R;
    Vector3d t;
};

inline Scene makeScene(const Config &config, unsigned seed, double sigma,
                       int outliers) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(-1, 1);
    std::normal_distribution<double> gaussian;
    auto randomVector = [&]() {
        Vector3d v;
        for (int k = 0; k < 3; ++k) v(k) = uniform(rng);
        return v;
    };
    Scene s;
    s.R = Eigen::AngleAxisd(0.43 + 0.1 * uniform(rng),
                          Vector3d(1, -2, 0.5).normalized()).toRotationMatrix();
    const Vector3d offset(2.3, -1.7, 0.8);
    s.t = config.coordinate_scale * Vector3d(0.7, -0.4, 1.1) - s.R * offset;
    const Vector3d centers[] = {{0, 0, 0}, {1.1, 0.2, 0.5},
                                {-0.4, 0.9, -0.6}, {0.3, -0.8, 1.0}};
    for (int c = 0; c < config.cameras; ++c) {
        s.Rbc.emplace_back(config.h >= 0 ? Matrix3d::Identity() :
            Eigen::AngleAxisd(0.22 * c, Vector3d(1, 2, -1).normalized()).toRotationMatrix());
        s.tbc.emplace_back(config.coordinate_scale * centers[c]);
    }
    auto observed = [&](const Vector3d &camera_ray, bool pinhole) -> Vector3d {
        if (pinhole) {
            Vector3d obs = camera_ray / camera_ray.z();
            obs.x() += sigma * gaussian(rng);
            obs.y() += sigma * gaussian(rng);
            return obs;
        }
        Vector3d f = camera_ray.normalized(), noise;
        for (int k = 0; k < 3; ++k) noise(k) = sigma * gaussian(rng);
        return (f + noise - f * f.dot(noise)).normalized();
    };
    for (int i = 0; i < config.points; ++i) {
        const int c = i % config.cameras;
        Vector3d camera_ray;
        if (config.h >= 0) {
            const double theta = 6.283185307179586 * i / config.points + 0.15 * uniform(rng);
            camera_ray = Vector3d(std::cos(theta), std::sin(theta), config.h).normalized();
        } else {
            camera_ray = randomVector();
            camera_ray.z() = 1;
            camera_ray.head<2>() *= 0.65;
        }
        const double depth = config.coordinate_scale * (5 + 2 * uniform(rng));
        const Vector3d pb = s.tbc[c] + depth * s.Rbc[c] * camera_ray;
        s.points.emplace_back(s.R.transpose() * (pb - s.t));
        s.bearings.emplace_back(config.bearing_scale * observed(camera_ray, config.h < 0));
        s.point_cameras.push_back(c);
    }
    for (int i = 0; i < config.lines; ++i) {
        const int c = i % config.cameras;
        Vector3d a = randomVector(), b;
        a.z() = 4.5 + uniform(rng);
        a.head<2>() *= 2;
        b = a + randomVector();
        a *= config.coordinate_scale;
        b *= config.coordinate_scale;
        VectorXd world(6), image(6);
        world.head<3>() = s.R.transpose() * (s.tbc[c] + s.Rbc[c] * a - s.t);
        world.tail<3>() = s.R.transpose() * (s.tbc[c] + s.Rbc[c] * b - s.t);
        image.head<3>() = observed(a, true);
        image.tail<3>() = observed(b, true);
        s.lines.push_back(world);
        s.endpoints.push_back(image);
        s.line_cameras.push_back(c);
    }
    // Deterministic mismatches, independent of solver sampling. Alternate
    // points and lines when both exist; keep the outlier count exact.
    int p = 0, l = 0;
    for (int i = 0; i < outliers; ++i) {
        if ((i % 2 == 0 && p < config.points) || l >= config.lines) {
            s.bearings[p].x() += config.bearing_scale * 0.7;
            s.bearings[p].y() -= config.bearing_scale * 0.5;
            ++p;
        } else {
            s.endpoints[l](0) += 0.6;
            s.endpoints[l](3) += 0.6;
            s.endpoints[l](1) -= 0.4;
            s.endpoints[l](4) -= 0.4;
            ++l;
        }
    }
    return s;
}

// Fingerprint the actual inputs; comparison checks that both executables saw
// identical generated data, including observations, extrinsics and truth.
inline std::uint64_t fingerprint(const Scene &s) {
    std::uint64_t hash = 14695981039346656037ULL;
    auto matrix = [&](const auto &m) {
        for (int i = 0; i < m.size(); ++i) {
            std::uint64_t bits;
            const double value = m.data()[i];
            std::memcpy(&bits, &value, sizeof(bits));
            for (int k = 0; k < 8; ++k) {
                hash ^= (bits >> (8 * k)) & 255;
                hash *= 1099511628211ULL;
            }
        }
    };
    for (const auto &m : s.points) matrix(m);
    for (const auto &m : s.bearings) matrix(m);
    for (const auto &m : s.lines) matrix(m);
    for (const auto &m : s.endpoints) matrix(m);
    for (const auto &m : s.Rbc) matrix(m);
    for (const auto &m : s.tbc) matrix(m);
    matrix(s.R);
    matrix(s.t);
    return hash;
}

inline double pointResidual(const Scene &s, const Matrix3d &R, const Vector3d &t) {
    double error = 0;
    for (size_t i = 0; i < s.points.size(); ++i) {
        const int c = s.point_cameras[i];
        const Vector3d f = (s.Rbc[c] * s.bearings[i]).normalized();
        error += (R * s.points[i] + t - s.tbc[c]).cross(f).squaredNorm();
    }
    return error;
}
} // namespace simulation
