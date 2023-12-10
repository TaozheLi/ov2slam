/**
*    This file is part of OV²SLAM.
*    
*    Copyright (C) 2020 ONERA
*
*    For more information see <https://github.com/ov2slam/ov2slam>
*
*    OV²SLAM is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    OV²SLAM is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with OV²SLAM.  If not, see <https://www.gnu.org/licenses/>.
*
*    Authors: Maxime Ferrera     <maxime.ferrera at gmail dot com> (ONERA, DTIS - IVA),
*             Alexandre Eudes    <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Julien Moras       <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Martial Sanfourche <first.last at onera dot fr>      (ONERA, DTIS - IVA)
*/
#pragma once


#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class FeatureTracker {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // FeatureTracker() {}
    FeatureTracker(int nmax_iter, float fmax_px_precision, cv::Ptr<cv::CLAHE> pclahe) 
        : klt_convg_crit_(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, nmax_iter, fmax_px_precision)
        , pclahe_(pclahe)
    {}

    // Forward-Backward KLT Tracking
    void fbKltTracking(const std::vector<cv::Mat> &vprevpyr, const std::vector<cv::Mat> &vcurpyr, int nwinsize, int nbpyrlvl, float ferr, float fmax_fbklt_dist,
        std::vector<cv::Point2f> &vpts, std::vector<cv::Point2f> &vpriorkps, std::vector<bool> &vkpstatus) const;

    void fbKltTrackingWithDepth(const std::vector<double> &kp_depth, const std::vector<cv::Mat> &vprevpyr, const std::vector<cv::Mat> &vcurpyr, int nwinsize, int nbpyrlvl, float ferr, float fmax_fbklt_dist,
        std::vector<cv::Point2f> &vpts, std::vector<cv::Point2f> &vpriorkps, std::vector<bool> &vkpstatus) const;

    double FeatureTracker::avg_double(const std::vector<double> &v);
    double FeatureTracker::std_double(const std::vector<double> & v);
    std::vector<int> classifyBasedOnDepth(const int & classes, const std::vector<double> &featureDepth);
    void RemovePointsThroughDepth(const int & classes, const std::vector<int> & groups, const std::vector<cv::Point2f> &prevFeatures,
                                                          const std::vector<cv::Point2f> &currentFeatures, const double &a, const double & b, std::vector<bool> &status);
    
    void getLineMinSAD(const cv::Mat &iml, const cv::Mat &imr, const cv::Point2f &pt, const int nwinsize, float &xprior, float &l1err, bool bgoleft) const;

    bool inBorder(const cv::Point2f &pt, const cv::Mat &im) const;

    // added by Taozhe Li
    void ClassifyBasedOnXY(const int &classes, const double &a, const double &b, const std::vector<int> & groups, const std::vector<cv::Point2f> &featurePointsPrev, const std::vector<cv::Point2f> &featurePointsCurrent, std::vector<bool> &status, const int &parts, const bool& useGlobalInformation);
    void ClassifyBasedOnXYAndRemovePoint(const double &a, const double &b,const std::vector<int> &IndexOfOneGroup, const std::vector<cv::Point2f> &featurePointPrev, const std::vector<cv::Point2f> &featurePointCurrent, std::vector<bool> &status, const std::vector<double> &orientation, const int &parts, const bool & useGlobalInformation);
    std::vector<double> ComputingGlobalOrientation(const int &parts, const std::vector<bool> & status, const std::vector<cv::Point2f> &featurePointsPrev, const std::vector<cv::Point2f> &featurePointsCurrent);
    double ComputeAngle(const cv::Point2f & pPrev, const cv::Point2f & pCurrent);
    bool Converse(std::vector<double> &angles, const double & threshRatio);
    bool RemovedConditionOnlyLength(const double &opticalFlowLength, const double & mean, const double & std, const double & a, const double &b);
    bool RemovedConditionOnlyOrientation(const double &a, const double & b, const double & theta, const double &meanTheta, const double & stdTheta, const bool &case2);
    bool RemovedCondition(const double &opticalFlowLength, const double & mean, const double & std, const double & a, const double &b, const double & theta, const double &meanTheta, const double &stdTheta, const double &globalOrientation, const bool & useGlobalInformation);

    // KLT optim. parameter
    cv::TermCriteria klt_convg_crit_;

    cv::Ptr<cv::CLAHE> pclahe_;
};