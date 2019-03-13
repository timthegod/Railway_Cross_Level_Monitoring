//
//  SceneDetector.hpp
//  RailCrossing
//
//  Created by 楊廷禹 on 2019/1/21.
//  Copyright © 2019 Yang Ting Yu. All rights reserved.
//

#ifndef SceneDetector_hpp
#define SceneDetector_hpp

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

#include "opencv.hpp"
#include "highgui.hpp"
#include "core.hpp"
#include "imgproc.hpp"

using namespace std;
using namespace cv;

#endif /* SceneDetector_hpp */


class SceneDetector {
    Mat background;
    Mat diffContour;
    Mat maskA;
    Mat maskB_L;
    Mat maskB_R;
    Mat maskC_L;
    Mat maskC_R;
    Mat maskBarrier;
    vector<vector<Point>> globalContours; //for drawing contour at the end
    vector<Point> contourA;
    vector<Point> contourB_Left;
    vector<Point> contourB_Right;
    vector<Point> contourC_Left;
    vector<Point> contourC_Right;
    vector<Point> contourBarrier;
    Point *ptsA;
    Point *ptsB_Left;
    Point *ptsB_Right;
    Point *ptsC_Left;
    Point *ptsC_Right;
    Point *ptsBarrier;
    int nptsA;
    int nptsB_Left;
    int nptsB_Right;
    int nptsC_Left;
    int nptsC_Right;
    int nptsBarrier;
    bool train = false;
    bool barrier = false;
    
public:
    SceneDetector();
    SceneDetector(Mat);
    SceneDetector(vector<Mat>&, Mat&);
    ~SceneDetector();
    
    void setDiffContour(Mat);
    void setZones();
    void drawZones(Mat&);
    void drawGlobalContours(Mat&);
    void createZonesMask(int, int);
    void fixDetectAccuracy(vector<vector<Point>>&, int, int);
    bool detectZoneA();
    bool detectZoneB_L();
    bool detectZoneB_R();
    bool detectZoneC_L();
    bool detectZoneC_R();
    bool detectZoneBarrier();
    void OutputSceneResult();
    double getZoneContourArea(vector<vector<Point>>&);
};
