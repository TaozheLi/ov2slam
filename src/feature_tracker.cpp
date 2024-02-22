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

#include "feature_tracker.hpp"
#include <iostream>
#include <unordered_map>
#include <opencv2/video/tracking.hpp>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "multi_view_geometry.hpp"

void FeatureTracker::fbKltTracking(const std::vector<cv::Mat> &vprevpyr, const std::vector<cv::Mat> &vcurpyr, 
        int nwinsize, int nbpyrlvl, float ferr, float fmax_fbklt_dist, std::vector<cv::Point2f> &vkps, 
        std::vector<cv::Point2f> &vpriorkps, std::vector<bool> &vkpstatus) const
{
    // std::cout << "\n \t >>> Forward-Backward kltTracking with Pyr of Images and Motion Prior! \n";

    assert(vprevpyr.size() == vcurpyr.size());

    if( vkps.empty() ) {
        std::cout << "\n \t >>> No kps were provided to kltTracking()!\n";
        return;
    }

    cv::Size klt_win_size(nwinsize, nwinsize);

    if( (int)vprevpyr.size() < 2*(nbpyrlvl+1) ) {
        nbpyrlvl = vprevpyr.size() / 2 - 1;
    }
    // Objects for OpenCV KLT
    size_t nbkps = vkps.size();
    vkpstatus.reserve(nbkps);

    std::vector<uchar> vstatus;
    std::vector<float> verr;
    std::vector<int> vkpsidx;
    vstatus.reserve(nbkps);
    verr.reserve(nbkps);
    vkpsidx.reserve(nbkps);

    // Tracking Forward
    cv::calcOpticalFlowPyrLK(vprevpyr, vcurpyr, vkps, vpriorkps, 
                vstatus, verr, klt_win_size,  nbpyrlvl, klt_convg_crit_, 
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS) 
                );

    std::vector<cv::Point2f> vnewkps;
    std::vector<cv::Point2f> vbackkps;
    vnewkps.reserve(nbkps);
    vbackkps.reserve(nbkps);

    size_t nbgood = 0;
    std::cout<<"ijrierjeirje"<<std::endl;
    // Init outliers vector & update tracked kps
    for( size_t i = 0 ; i < nbkps ; i++ ) 
    {
        if( !vstatus.at(i) ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( verr.at(i) > ferr ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( !inBorder(vpriorkps.at(i), vcurpyr.at(0)) ) {
            vkpstatus.push_back(false);
            continue;
        }

        vnewkps.push_back(vpriorkps.at(i));
        vbackkps.push_back(vkps.at(i));
        vkpstatus.push_back(true);
        vkpsidx.push_back(i);
        nbgood++;
    }  

    if( vnewkps.empty() ) {
        return;
    }
    
    vstatus.clear();
    verr.clear();

    std::cout << "\n \t >>> Forward kltTracking : #" << nbgood << " out of #" << nbkps << " \n";

    // Tracking Backward
    cv::calcOpticalFlowPyrLK(vcurpyr, vprevpyr, vnewkps, vbackkps, 
                vstatus, verr, klt_win_size,  0, klt_convg_crit_,
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS) 
                );
    
    nbgood = 0;
    for( int i = 0, iend=vnewkps.size() ; i < iend ; i++ )
    {
        int idx = vkpsidx.at(i);

        if( !vstatus.at(i) ) {
           vkpstatus.at(idx) = false;
           continue;
        }

        if( cv::norm(vkps.at(idx) - vbackkps.at(i)) > fmax_fbklt_dist ) {
            vkpstatus.at(idx) = false;
            continue;
       }

        nbgood++;
    }
    std::cout << "\n \t >>> Backward kltTracking : #" << nbgood << " out of #" << vkpsidx.size() << " \n";
    // classify feature points through its depth
    // author: Taozhe Li

    std::vector<cv::Point2f> selectedPrevFeatures, selectedCurrentFeatures;
    std::vector<int> originalIndex;
    selectedPrevFeatures.reserve(vkpstatus.size());
    selectedCurrentFeatures.reserve(vkpstatus.size());
    originalIndex.reserve(vkpstatus.size());

 
    for(int i=0; i<vkpstatus.size(); ++i){
        if(vkpstatus[i]){
            selectedPrevFeatures.push_back(vkps[i]);
            selectedCurrentFeatures.push_back(vpriorkps[i]);
            originalIndex.push_back(i);
        }
    }
    std::cout<<"original feature nums: "<<selectedPrevFeatures.size()<<std::endl;
    // std::cout<<"original index size:"<<originalIndex.size()<<std::endl;
    if(originalIndex.size() <= 5) return;
    int maxIterations = 1000;
    float fx, fy, cx, cy, thresholdEM;

    //00-02
    // fx = 718.856;
    // fy = 718.856;
    // cx = 607.1928;
    // cy = 185.2157;

    // 04-12
    // fx = 707.0912;
    // fy = 707.0912;
    // cx = 601.8873;
    // cy = 183.1104;

    //03
    fx = 721.5377;
    fy = 721.5377;
    cx = 609.1928;
    cy = 172.854;
    thresholdEM = 3.0f;
    std::vector<bool> Mask;
    Mask.resize(selectedPrevFeatures.size());
    CalEssentialMatrix(vkpstatus, originalIndex, selectedPrevFeatures, selectedCurrentFeatures, thresholdEM, maxIterations, fx, fy, cx, cy, Mask);
    // std::cout<<"Mask.size(): "<<Mask.size()<<std::endl;

    nbgood = 0;
    for( int i = 0; i<vkpstatus.size() ; i++ )
    {
        if(vkpstatus[i]) nbgood++;
    }
    std::cout << "\n \t >>>after EM remaining : #" << nbgood << " out of #" << vkpsidx.size() << " \n";
    
}
bool FeatureTracker::FeatureSelectionThroughEssentialMatrix(const double &threshold, const cv::Mat & E, const std::vector<cv::Point2f> &prevs, const std::vector<cv::Point2f> &currents, std::vector<bool> &mask) const{
    // default method is selected EssentialMatrix randomly
    cv::Mat essentialMatrix;
    bool success = SelectedEssentialMatrixThroughMaxRatio(threshold, E, prevs, currents, essentialMatrix);
    // selected essentialmatrix successfully
    bool originalE = E.rows == 3;
    if(originalE) essentialMatrix = E.clone();
    bool apply = success || originalE;
    // apply feature selection through mask
    if(apply){
        for(int i=0; i<prevs.size(); i++){
            cv::Point2f prev = prevs[i];
            cv::Point2f cur = currents[i];
            cv::Mat prev_ = cv::Mat::ones(3, 1, CV_64F);
            prev_.at<double>(0, 0) = prevs[i].x;
            prev_.at<double>(1, 0) = prevs[i].y;

            cv::Mat cur_ = cv::Mat::ones(3, 1, CV_64F);
            cur_.at<double>(0, 0) = currents[i].x;
            cur_.at<double>(1, 0) = currents[i].y;
            cv::transpose(cur_, cur_);
            cv::Mat l = cur_ * essentialMatrix;
            double A, B;
            A = l.at<double>(0, 0);
            B = l.at<double>(0, 1);
            double norm = pow(A * A + B * B, 0.5);
            cv::Mat _ = l * prev_;
            if(_.at<double>(0,0) / norm <= threshold){
                mask[i] = true;
            }
            else{
                mask[i] = false;
            }

        }
        // apply this method successfully
        return true;
    }
    // doesn't apply
    else{
        return false;
    }
}

bool FeatureTracker::SelectedEssentialMatrixRandomly(const double &threshold, const cv::Mat & E, const std::vector<cv::Point2f> &prevs, const std::vector<cv::Point2f> &currents, cv::Mat & outputE) const{
	if(E.empty() || E.rows == 3)
            return false;
        // it means that row of essential matrix is not 3
        const int rows = E.rows;
        for(int i=0; i< rows / 3; ++i){
            cv::Mat e = E.rowRange(3*i, 3 * (i+1));
            double ratio = InlierRatio(threshold, e, prevs, currents);
            if(ratio > 0.5){
                outputE = e.clone();
                return true;
            }
        }
        return false;
}

bool FeatureTracker::SelectedEssentialMatrixThroughMaxRatio(const double &threshold, const cv::Mat & E, const std::vector<cv::Point2f> &prevs, const std::vector<cv::Point2f> &currents, cv::Mat & outputE) const{
	if(E.empty() || E.rows == 3)
            return false;
        // it means that row of essential matrix is not 3
        int maxIndex = -1;
        double maxRatio = -1.0f;
        const int rows = E.rows;
        for(int i=0; i< rows / 3; ++i){
            cv::Mat e = E.rowRange(3*i, 3 * (i+1));
            double ratio = InlierRatio(threshold, e, prevs, currents);
            if(ratio > 0.5 && ratio > maxRatio){
                maxRatio = ratio;
                maxIndex = i;
            }
        }
        if(maxIndex != -1){
            outputE = E.rowRange(maxIndex * 3, (maxIndex + 1) * 3);
            return true;
        }
    return false;
}

double FeatureTracker::InlierRatio(const double &threshold, const cv::Mat & e, const std::vector<cv::Point2f> &prevs, const std::vector<cv::Point2f> &currents) const{
        double inlier = 0.0;
        auto length = double(prevs.size());
        for(int i=0; i<prevs.size(); i++){
            cv::Point2f prev = prevs[i];
            cv::Point2f cur = currents[i];
            cv::Mat prev_ = cv::Mat::ones(3, 1, CV_64F);
            prev_.at<double>(0, 0) = prevs[i].x;
            prev_.at<double>(1, 0) = prevs[i].y;

            cv::Mat cur_ = cv::Mat::ones(3, 1, CV_64F);
            cur_.at<double>(0, 0) = currents[i].x;
            cur_.at<double>(1, 0) = currents[i].y;
            cv::transpose(cur_, cur_);
            // std::cout<<"get epipolar line ------------"<<std::endl;
            // std::cout<<cur_<<std::endl;
            // std::cout<<e<<std::endl;
            cv::Mat l = cur_ * e;
            // std::cout<<"get epipolar line ============"<<std::endl;
            double A, B;
            A = l.at<double>(0, 0);
            B = l.at<double>(0, 1);
            double norm = pow(A * A + B * B, 0.5);
            cv::Mat _ = l * prev_;
            if(_.at<double>(0,0) / norm <= threshold){
                inlier++;
            }

        }
        return inlier / length;
}

void FeatureTracker::CalEssentialMatrix(std::vector<bool> &vStatus, const std::vector<int> &originalIndex, const std::vector<cv::Point2f> &prevKeyPoints, const std::vector<cv::Point2f> &currentKeyPoints, const float & threshold, const int & maxIterations, const float &fx, const float &fy, const float &cx, const float & cy, std::vector<bool> &mask) const{
    cv::Mat K = cv::Mat::zeros(cv::Size(3, 3), CV_32FC1);
    // std::cout<<"generate K matrix successfully"<<std::endl;
    K.at<float>(0, 0) = fx; K.at<float>(0, 2) = cx;
    K.at<float>(1, 1) = fy; K.at<float>(1, 2) = cy;
    K.at<float>(2, 2) = 1.0f;
    // std::cout<<K<<std::endl;
    // std::cout<<"threshold is "<<threshold<<std::endl;
    // std::cout<<prevKeyPoints.size()<<" "<<currentKeyPoints.size()<<std::endl;
    cv::Mat EssentialMatrix = cv::findEssentialMat(prevKeyPoints, currentKeyPoints, K, cv::RANSAC,  0.999, threshold, maxIterations);
    // std::cout<<EssentialMatrix<<std::endl;
    // std::cout<<EssentialMatrix.size()<<std::endl;
    cv::Mat outputE;
    bool apply = FeatureSelectionThroughEssentialMatrix(threshold, EssentialMatrix, prevKeyPoints, currentKeyPoints, mask);
    if(!apply) {
        std::cout<<"didn't apply essential matrix based method"<<std::endl;
        return;
    }
    int length = 0;
    // std::cout<<"mask.size(): "<<mask.size()<<std::endl;
    //if(mask2.size() == 0){
    //std::cout<<"exit when mask size is 0"<<std::endl;
    //exit(-1);
    //}
    for(int i=0; i<mask.size(); ++i){
    	if(mask[i]) length+=1;
    }
    std::cout<<"after EM there are "<<length<<" feature points"<<std::endl;
    for(int x=0; x<mask.size(); ++x){
        if(mask[x]){
            continue;
        }
        int index = originalIndex[x];
        //std::cout<<"EM removing Index: "<<index<<std::endl;
        vStatus[index] = false;
    }
    
    
}

double FeatureTracker::avg_double(const std::vector<double> &v) const{
    double sum;
    sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / double(v.size());
}

double FeatureTracker::std_double(const std::vector<double> & v) const{
    double mean = avg_double(v);
    double sum = 0;
    for(const auto & x: v){
        sum += (x - mean) * (x- mean);
    }
    return sqrt(sum) / double(v.size());
}

double FeatureTracker::max_double(const std::vector<double> &v) const{
    double max = v[0];
    for(const auto &x:v){
        if(x > max){
            max = x;
        }
    }
    return max;
}

std::vector<int> FeatureTracker::classifyBasedOnDepth(const int  &cls, const std::vector<double> &featureDepth) const{
    double max_depth = max_double(featureDepth);
    double segment = (max_depth + 1.0) / double(cls);
    std::vector<int> groups;
    for(int i=0; i<featureDepth.size(); ++i){
        //
        double depth = featureDepth[i];
        if(depth <= 0) continue;
        int group = -1;
        for(int s = 0; s<cls; ++s){
            if(depth > double(s + 1) * segment) continue;
            if(depth > double(s * segment)) group = s;
        }
        if(group == -1){
            std::cout<<"not assign group \n";
            std::cout<<"segment: "<<segment<<std::endl;
            std::cout<<"depth: "<<depth<<std::endl;
            std::cout<<"max dpeth: "<<max_depth<<std::endl;
            exit(-1);
        }
        groups.push_back(group);
    }
    return groups;
}

void FeatureTracker::RemovePointsThroughDepth(const int & classes, const std::vector<int> & groups, const std::vector<cv::Point2f> &prevFeatures,
                                                          const std::vector<cv::Point2f> &currentFeatures, const double &a, const double & b, std::vector<bool> &status){
    assert(prevFeatures.size() == currentFeatures.size());
    std::vector<std::pair<double, double>> AvgAndStdArray;
    std::vector<double> Lengths;
    std::vector<std::vector<double>> LengthOfEachGroups(classes);
    for(int i=0; i<prevFeatures.size(); ++i){
        int group = groups[i];
        double length = std::pow((currentFeatures[i].x - prevFeatures[i].x) * (currentFeatures[i].x - prevFeatures[i].x) +
                                 (currentFeatures[i].y - prevFeatures[i].y) * (currentFeatures[i].y - prevFeatures[i].y), 0.5);
        LengthOfEachGroups[group].push_back(length);
        Lengths.push_back(length);
    }
    for(int i=0; i<classes; ++i){
        std::pair<double, double> _;
        if(LengthOfEachGroups[i].size() != 0) {
            _.first = avg_double(LengthOfEachGroups[i]);
            _.second = std_double(LengthOfEachGroups[i]);
        }
        else{
            _.first = 0;
            _.second = 0;
        }
        AvgAndStdArray.push_back(_);
    }
//    for(int i=0; i<classes; ++i){
//        std::cout<<"group :"<<i<<" avg :"<<AvgAndStdArray[i].first<<" std: "<<AvgAndStdArray[i].second<<std::endl;
//    }
    for(int i=0; i<prevFeatures.size(); ++i){
        int group = groups[i];
        double avg = AvgAndStdArray[group].first;
        double std = AvgAndStdArray[group].second;
        double length = Lengths[i];
        if(length < (avg - a*std) || length > (avg + b * std)){
            status[i] = false;
        }
    }

}



void FeatureTracker::fbKltTrackingWithDepth(const std::vector<double> &kp_depth, const std::vector<cv::Mat> &vprevpyr, const std::vector<cv::Mat> &vcurpyr, 
        int nwinsize, int nbpyrlvl, float ferr, float fmax_fbklt_dist, std::vector<cv::Point2f> &vkps, std::vector<cv::Point2f> &vpriorkps, std::vector<bool> &vkpstatus) const
{
    // std::cout << "\n \t >>> Forward-Backward kltTracking with Pyr of Images and Motion Prior! \n";

    assert(vprevpyr.size() == vcurpyr.size());

    if( vkps.empty() ) {
        // std::cout << "\n \t >>> No kps were provided to kltTracking()!\n";
        return;
    }

    cv::Size klt_win_size(nwinsize, nwinsize);

    if( (int)vprevpyr.size() < 2*(nbpyrlvl+1) ) {
        nbpyrlvl = vprevpyr.size() / 2 - 1;
    }

    // Objects for OpenCV KLT
    size_t nbkps = vkps.size();
    vkpstatus.reserve(nbkps);

    std::vector<uchar> vstatus;
    std::vector<float> verr;
    std::vector<int> vkpsidx;
    vstatus.reserve(nbkps);
    verr.reserve(nbkps);
    vkpsidx.reserve(nbkps);

    // Tracking Forward
    cv::calcOpticalFlowPyrLK(vprevpyr, vcurpyr, vkps, vpriorkps, 
                vstatus, verr, klt_win_size,  nbpyrlvl, klt_convg_crit_, 
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS) 
                );

    std::vector<cv::Point2f> vnewkps;
    std::vector<cv::Point2f> vbackkps;
    vnewkps.reserve(nbkps);
    vbackkps.reserve(nbkps);

    size_t nbgood = 0;

    // Init outliers vector & update tracked kps
    for( size_t i = 0 ; i < nbkps ; i++ ) 
    {
        if( !vstatus.at(i) ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( verr.at(i) > ferr ) {
            vkpstatus.push_back(false);
            continue;
        }

        if( !inBorder(vpriorkps.at(i), vcurpyr.at(0)) ) {
            vkpstatus.push_back(false);
            continue;
        }

        vnewkps.push_back(vpriorkps.at(i));
        vbackkps.push_back(vkps.at(i));
        vkpstatus.push_back(true);
        vkpsidx.push_back(i);
        nbgood++;
    }  
    
    if( vnewkps.empty() ) {
        return;
    }
    // taozhe li part
    std::vector<int> groups;
    int classes = 10;
    double a = 2.5;
    double b = 2.5;
    int parts = 5;
    bool useGlobalInformation = false;
    // std::cout<<"run calssifyBased on Depth"<<std::endl;
    groups = classifyBasedOnDepth(classes, kp_depth);
    // std::cout<<"run calssifyBased on depth function successfully"<<std::endl;
    ClassifyBasedOnXY(classes, a, b, groups, vbackkps, vnewkps, vkpstatus, parts, useGlobalInformation);
    // std::cout<<"run classifyBasedOnXY successfully"<<std::endl;
    // RemovePointsThroughDepth(classes, groups, vnewkps, vbackkps, a, b, vkpstatus);
    
    vstatus.clear();
    verr.clear();

    // std::cout << "\n \t >>> Forward kltTracking : #" << nbgood << " out of #" << nbkps << " \n";

    // Tracking Backward
    cv::calcOpticalFlowPyrLK(vcurpyr, vprevpyr, vnewkps, vbackkps, 
                vstatus, verr, klt_win_size,  0, klt_convg_crit_,
                (cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS) 
                );
    
    nbgood = 0;
    for( int i = 0, iend=vnewkps.size() ; i < iend ; i++ )
    {
        int idx = vkpsidx.at(i);

        if( !vstatus.at(i) ) {
            vkpstatus.at(idx) = false;
            continue;
        }

        if( cv::norm(vkps.at(idx) - vbackkps.at(i)) > fmax_fbklt_dist ) {
            vkpstatus.at(idx) = false;
            continue;
        }

        nbgood++;
    }
    // classify feature points through its depth
    

    // std::cout << "\n \t >>> Backward kltTracking : #" << nbgood << " out of #" << vkpsidx.size() << " \n";
}


void FeatureTracker::getLineMinSAD(const cv::Mat &iml, const cv::Mat &imr, 
    const cv::Point2f &pt,  const int nwinsize, float &xprior, 
    float &l1err, bool bgoleft) const
{
    xprior = -1;

    if( nwinsize % 2 == 0 ) {
        std::cerr << "\ngetLineMinSAD requires an odd window size\n";
        return;
    }

    const float x = pt.x;
    const float y = pt.y;
    int halfwin = nwinsize / 2;

    if( x - halfwin < 0 ) 
        halfwin += (x-halfwin);
    if( x + halfwin >= imr.cols )
        halfwin += (x+halfwin - imr.cols - 1);
    if( y - halfwin < 0 )
        halfwin += (y-halfwin);
    if( y + halfwin >= imr.rows )
        halfwin += (y+halfwin - imr.rows - 1);
    
    if( halfwin <= 0 ) {
        return;
    }

    cv::Size winsize(2 * halfwin + 1, 2 * halfwin + 1);

    int nbwinpx = (winsize.width * winsize.height);

    float minsad = 255.;
    // int minxpx = -1;

    cv::Mat patch, target;

    cv::getRectSubPix(iml, winsize, pt, patch);

    if( bgoleft ) {
        for( float c = x ; c >= halfwin ; c-=1. )
        {
            cv::getRectSubPix(imr, winsize, cv::Point2f(c, y), target);
            l1err = cv::norm(patch, target, cv::NORM_L1);
            l1err /= nbwinpx;

            if( l1err < minsad ) {
                minsad = l1err;
                xprior = c;
            }
        }
    } else {
        for( float c = x ; c < imr.cols - halfwin ; c+=1. )
        {
            cv::getRectSubPix(imr, winsize, cv::Point2f(c, y), target);
            l1err = cv::norm(patch, target, cv::NORM_L1);
            l1err /= nbwinpx;

            if( l1err < minsad ) {
                minsad = l1err;
                xprior = c;
            }
        }
    }

    l1err = minsad;
}



/**
 * \brief Perform a forward-backward calcOpticalFlowPyrLK() tracking with OpenCV.
 *
 * \param[in] pt  Opencv 2D point.
 * \return True if pt is within image borders, False otherwise
 */
bool FeatureTracker::inBorder(const cv::Point2f &pt, const cv::Mat &im) const
{
    const float BORDER_SIZE = 1.;

    return BORDER_SIZE <= pt.x && pt.x < im.cols - BORDER_SIZE && BORDER_SIZE <= pt.y && pt.y < im.rows - BORDER_SIZE;
}

void FeatureTracker::ClassifyBasedOnXY(const int &classes, const double &a, const double &b,const std::vector<int> & groups, const std::vector<cv::Point2f> &featurePointsPrev, const std::vector<cv::Point2f> &featurePointsCurrent, std::vector<bool> &status, const int &parts, const bool& useGlobalInformation)const{
    int cout = 0;
    for(int i=0; i<status.size(); ++i){
        if(status[i]) ++cout;
    }
    // std::cout<<"count 1 successfully"<<std::endl;
    std::vector<std::vector<int>> ClassifyThroughDepth;
    ClassifyThroughDepth.resize(classes);
    for(int i=0; i<groups.size(); ++i){
        if(groups[i] < classes){
            ClassifyThroughDepth[groups[i]].push_back(i);
        }
        else{
            std::cout<<"error group: "<<groups[i]<<std::endl;
            std::cout<<"error group: "<<groups[i]<<std::endl;
            exit(-1);
        }
    }
    // std::cout<<"generate classifythroughdepth successfully"<<std::endl;
    std::vector<double> globalOrientation = ComputingGlobalOrientation(parts, status, featurePointsPrev, featurePointsCurrent);
    // std::cout<<"run ComputingGlobaOrientation successfully"<<std::endl;
    for(int i=0; i<ClassifyThroughDepth.size(); ++i){
        if(ClassifyThroughDepth[i].empty()) continue;
        // std::cout<<"group :"<<i<<" size: "<<ClassifyThroughDepth[i].size()<<std::endl;
        ClassifyBasedOnXYAndRemovePoint(a, b, ClassifyThroughDepth[i], featurePointsPrev, featurePointsCurrent, status, globalOrientation, parts, useGlobalInformation);
    }
    // std::cout<<"run classifyBasedOnXYAndRemovePoint successfully"<<std::endl;
    int _cout = 0;
    for(int i=0; i<status.size(); ++i){
        if(status[i]) ++_cout;
    }
    std::cout<<"before process there are "<<cout<<" feautre points after process, there are "<<_cout<<" feature points"<<std::endl;
    std::cout<<"ratio: "<<double(_cout) / double(cout)<<std::endl;

}

void FeatureTracker::ClassifyBasedOnXYAndRemovePoint(const double &a, const double &b,const std::vector<int> &IndexOfOneGroup, const std::vector<cv::Point2f> &featurePointPrev,
    const std::vector<cv::Point2f> &featurePointCurrent, std::vector<bool> &status, const std::vector<double> &globalOrientation, const int &parts, const bool & useGlobalInformation) const{
    // std::cout<<"no error in ClassifyBasedOnXYAndRemovePoint"<<std::endl;
    // std::cout<<"start to run classifybasedonXYandRemovePoint"<<std::endl;
    const int width = 1231;
    const int height = 376;
    const int minimumNumber = 5;
    const int nb = 10;
    const int mb = 3;
    const int totalClasses = nb * mb;
    const int mw = height / mb;
    const int nw = width / nb;
    int type;
    if(IndexOfOneGroup.size() <= 5) return;
    // std::cout<<"start to based on x and y"<<std::endl;
    // if number of points is too small, don't remove points
    std::vector<std::vector<int>> newGroups;
    std::vector<std::vector<double>> length;
    std::vector<std::vector<double>> angle;
    length.resize(totalClasses);
    newGroups.resize(totalClasses);
    angle.resize(totalClasses);
    // std::cout<<"run here 1"<<std::endl;
    double x_cor = 0.0;
    std::vector<int> count_n(30, 0);
    // std::cout<<"start to run first loop"<<std::endl;
    for(int i=0; i<IndexOfOneGroup.size(); ++i){
        // it's good points
        // std::cout<<"this part can run"<<std::endl;
        int originalIndex = IndexOfOneGroup[i];
        if(std::isnan(featurePointPrev[originalIndex].y ) || std::isnan(featurePointPrev[originalIndex].x) || featurePointPrev[originalIndex].y >= height || featurePointPrev[originalIndex].y <0 || featurePointPrev[originalIndex].x >= width || featurePointPrev[originalIndex].x < 0){
            status[originalIndex] = false;
            continue;
            
        }
        if(std::isnan(featurePointCurrent[originalIndex].y ) || std::isnan(featurePointCurrent[originalIndex].x) || featurePointCurrent[originalIndex].y <0 || featurePointCurrent[originalIndex].y >= height || featurePointCurrent[originalIndex].x <0 || featurePointCurrent[originalIndex].x >= width){
            status[originalIndex] = false;
            continue;

        }
        // std::cout<<"satisfy prev coordinates y: "<<featurePointPrev[originalIndex].y<<" x: "<<featurePointPrev[originalIndex].x<<std::endl;
        // std::cout<<"cuurent coordinates y: "<<featurePointCurrent[originalIndex].y<<" x: "<<featurePointCurrent[originalIndex].x<<std::endl;
        if(status[originalIndex]){
            int _row = featurePointPrev[originalIndex].y / mw;
            int _col = featurePointPrev[originalIndex].x / nw;
            int group = _row * nb + _col;
//            std::cout<<"run here 3"<<std::endl;
//            std::cout<<"run this too: "<<group<<std::endl;
            if(group >= totalClasses){
                std::cout<<"x: "<<featurePointPrev[originalIndex].x<<" y: "<<featurePointPrev[originalIndex].y<<std::endl;
                std::cout<<"mv: "<<mw<<" nw: "<<nw<<std::endl;
                std::cout<<"_row: "<<_row<<" _col: "<<_col<<std::endl;
                std::cout<<"error group: "<<group<<std::endl;
                exit(-1);
            }
            newGroups[group].push_back(originalIndex);
            length[group].push_back(cv::norm(featurePointCurrent[originalIndex] - featurePointPrev[originalIndex]));
//            std::cout<<"group : "<<group<<std::endl;
//            std::cout<<"run here 4"<<std::endl;
//            if(group == 3){
//                std::cout<<"i: "<<ComputeAngle(featurePointPrev[originalIndex], featurePointCurrent[originalIndex])<<std::endl;
//            }
            double _ = ComputeAngle(featurePointPrev[originalIndex], featurePointCurrent[originalIndex]);
            if(_ < -180.0 || _ > 180.0){
                std::cout<<featurePointPrev[originalIndex]<<" "<<featurePointCurrent[originalIndex]<<std::endl;
                std::exit(-1);
            }
            angle[group].push_back(_);
            // std::cout<<"it push_back successfully"<<std::endl;
//            std::cout<<_<<std::endl;
//            count_n[group] += 1;
            x_cor += featurePointPrev[originalIndex].x;
            // std::cout<<"run here !!!!"<<std::endl;
        }
    }
    // std::cout<<"finished first loop !!!!!!!"<<std::endl;
//    std::cout<<"run here 2"<<std::endl;
//    for(int i=0; i<angle[3].size(); i++){
//        std::cout<<"group 3 4 i: "<<i<<" angle: "<<angle[3][i]<<std::endl;
//    }

    x_cor = x_cor / double(IndexOfOneGroup.size());
    for(int i=0; i<parts; ++i){
        if(x_cor > (i+1) * double(width + 1) / double(parts)) continue;
        type = i;
        break;
    }
    // std::cout<<"global regin: "<<type<<std::endl;
    // check
//    for(int i=0; i<newGroups.size(); ++i){
//        if(newGroups[i].empty()) continue;
//        std::cout<<"xy group: "<<i<<" size: "<<newGroups[i].size()<<std::endl;
//    }
    // check it again
//    for(int i=0; i<angle[3].size(); i++){
//        std::cout<<"group 3 i: "<<i<<" angle: "<<angle[3][i]<<std::endl;
//    }
    // converse
    double threshRatio = 0.7;
    int count = 0;
    for(int i=0; i<angle.size(); ++i) {
        if (angle[i].empty()) continue;
//        std::cout<<"not empty group: "<<i<<std::endl;
        if (Converse(angle[i], threshRatio)) count+=1;
    }
//    std::cout<<"there are total "<<count<<" conversed group"<<std::endl;
    for(int i=0; i<totalClasses; ++i){
        // no element in these group
        if(newGroups[i].empty()) continue;
        if(newGroups[i].size() < minimumNumber) continue;
        double avg_optflow = avg_double(length[i]);
        double std_optflow = std_double(length[i]);
        double avg_angle = avg_double(angle[i]);
        double std_angle = std_double(angle[i]);
        // std::cout<<" xy_group: "<<i<<" mean_optflow: "<<avg_optflow<<" std_optflow: "<<std_optflow;
        // std::cout<<" mean_angle: "<<avg_angle<<std::endl;
        for(int each_idx=0; each_idx < newGroups[i].size(); ++each_idx){
            // detect angle rangle
            double optflow = length[i][each_idx];
            double theta = angle[i][each_idx];
//            std::cout<<"optflow: "<<optflow<<" theta: "<<theta<<std::endl;
            if(RemovedCondition(optflow, avg_optflow, std_optflow, a, b, theta, avg_angle, std_angle, globalOrientation[type], useGlobalInformation)){
                int originalIndex = newGroups[i][each_idx];
                status[originalIndex] = false;
            }
        }
    }               
}

std::vector<double> FeatureTracker::ComputingGlobalOrientation(const int &parts, const std::vector<bool> & status, const std::vector<cv::Point2f> &featurePointsPrev, const std::vector<cv::Point2f> &featurePointsCurrent) const{
    const int width = 1231;
    double segment = double(width+1) / double(parts);
    std::vector<double> globalOrientation(parts, 0.0);
    std::vector<double> count(parts, 0.0);
    for(int i=0; i<status.size(); ++i){
        if(status[i]){
            int b;
            for(int k=0; k<parts; k++){
                if(featurePointsPrev[i].x > (k+1) * segment ) continue;
                b = k;
                break;
            }
            globalOrientation[b] += ComputeAngle(featurePointsCurrent[i], featurePointsPrev[i]);
//            if(b == 1)
//            std::cout<<ComputeAngle(featurePointsCurrent[i], featurePointsPrev[i])<<std::endl;
            count[b]++;
        }
    }
    for(int i=0; i<parts; ++i){
        globalOrientation[i] = (globalOrientation[i] / count[i]);
        // std::cout<<"region: "<<i<<" orientation: "<<globalOrientation[i]<<std::endl;
    }
    return globalOrientation;
}

double FeatureTracker::ComputeAngle(const cv::Point2f & pPrev, const cv::Point2f & pCurrent) const{
    return atan2((pCurrent.y - pPrev.y),  (pCurrent.x - pPrev.x) ) * 180.0 / M_PI;
}

bool FeatureTracker::Converse(std::vector<double> &angles, const double & threshRatio) const{
    double total = double(angles.size());
    double count = 0.0;
    // std::cout<<"start to run converse !!!! "<<std::endl;
    // std::cout<<angles.size()<<std::endl;
    // for(int i=0; i<angles.size(); ++i){
    //     std::cout<<"i:"<<i<<" angle: "<<angles[i]<<std::endl;
    // }
    for(const auto & angle: angles){
        if(abs(angle) > 90)  count+=1;
    }
    // std::cout<<"count: "<<count<<std::endl;

    // need to converse
    if((count / total) > threshRatio) {
        for(auto angle: angles){
            if(angle < 0) angle = -angle - 180;
            else{
                angle = -angle + 180;
            }
        }
        return true;
    }
    else
        return false;
}

bool FeatureTracker::RemovedConditionOnlyLength(const double &opticalFlowLength, const double & mean, const double & std, const double & a, const double &b) const{
    if(opticalFlowLength > mean + a * std || opticalFlowLength < mean - b * std)
        return true;
    else{
        return false;
    }
}

bool FeatureTracker::RemovedConditionOnlyOrientation(const double &a, const double & b, const double & theta, const double &meanTheta, const double & stdTheta, const bool &case2) const{
    if(!case2) {
        double theta_threshold = 15.0;
        if (abs(theta - meanTheta) > theta_threshold) return true;
        else
            return false;
    }
    else{
        double upperBound = meanTheta + a * stdTheta;
        double lowerBound = meanTheta - b * stdTheta;
        if(theta < lowerBound || theta > upperBound){
            return true;
        }
        else{
            return false;
        }
    }
}

bool FeatureTracker::RemovedCondition(const double &opticalFlowLength, const double & mean, const double & std, const double & a, const double &b,
                      const double & theta, const double &meanTheta, const double &stdTheta, const double &globalOrientation, const bool & useGlobalInformation) const{
    double alpha;
    double beta;
    if(useGlobalInformation){
        alpha = 0.3;
    }
    else{
        alpha = 1.0;
    }
    beta = 1 - alpha;
    if(RemovedConditionOnlyLength(opticalFlowLength, mean, std, a, b) || RemovedConditionOnlyOrientation(a, b, theta, alpha * meanTheta + beta * globalOrientation, stdTheta, false))
        return true;
    else
        return false;
}