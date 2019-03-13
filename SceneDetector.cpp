//
//  SceneDetector.cpp
//  RailCrossing
//
//  Created by 楊廷禹 on 2019/1/21.
//  Copyright © 2019 Yang Ting Yu. All rights reserved.
//

#include "SceneDetector.hpp"

SceneDetector::SceneDetector() {
    setZones();
}

SceneDetector::SceneDetector(Mat srcEmpty) {
    setZones();
    cvtColor(srcEmpty, srcEmpty, COLOR_BGR2GRAY);
    background = srcEmpty.clone();
}

SceneDetector::SceneDetector(vector<Mat>& srcEmpty, Mat& srcSize) {
    setZones();
    if(srcEmpty.empty())
        cerr<<"No training data for empty scene!"<<endl;
    else{
        Mat mean(srcEmpty[0].rows, srcEmpty[0].cols, CV_64FC3);
        mean.setTo(Scalar(0,0,0,0));
        
        Mat temp;
        for (int i = 0; i < srcEmpty.size(); ++i)
        {
            // Convert the input images to CV_64FC3 ...
            srcEmpty[i].convertTo(temp, CV_64FC3);
            
            //accumulate
            mean += temp;
        }
        // Convert back to CV_8U type, division to get actual mean
        mean.convertTo(mean, CV_8U, 1. / srcEmpty.size());
        background = mean.clone();
        imshow("back", background);
        cvtColor(background, background, COLOR_BGR2GRAY);
    }
    
}

SceneDetector::~SceneDetector() {}

void SceneDetector::setZones()
{
    //zone A
    contourA.push_back(Point(693,88));
    contourA.push_back(Point(693,195));
    contourA.push_back(Point(302,463));
    contourA.push_back(Point(24,403));
    
    //zone B left
    contourB_Left.push_back(Point(24,56));
    contourB_Left.push_back(Point(24,168));
    contourB_Left.push_back(Point(175,280));
    contourB_Left.push_back(Point(400,210));
    contourB_Left.push_back(Point(112,56));
    
    //zone B right
    contourB_Right.push_back(Point(693,222));
    contourB_Right.push_back(Point(616,285));
    contourB_Right.push_back(Point(693,321));
    
    //zone C right
    contourC_Right.push_back(Point(295,55));
    contourC_Right.push_back(Point(230,54));
    contourC_Right.push_back(Point(459,186));
    contourC_Right.push_back(Point(535,156));
    
    //zone C left
    contourC_Left.push_back(Point(631,296));
    contourC_Left.push_back(Point(402,463));
    contourC_Left.push_back(Point(693,463));
    contourC_Left.push_back(Point(693,323));
    
    //zone Barrier
    contourBarrier.push_back(Point(24,275));
    contourBarrier.push_back(Point(24,285));
    contourBarrier.push_back(Point(359,168));
    contourBarrier.push_back(Point(342,156));
    
    ptsA = (Point*) Mat(contourA).data;
    nptsA = Mat(contourA).rows;
    
    ptsB_Left = (Point*) Mat(contourB_Left).data;
    nptsB_Left = Mat(contourB_Left).rows;
    
    ptsB_Right = (Point*) Mat(contourB_Right).data;
    nptsB_Right = Mat(contourB_Right).rows;
    
    ptsC_Left = (Point*) Mat(contourC_Left).data;
    nptsC_Left = Mat(contourC_Left).rows;
    
    ptsC_Right = (Point*) Mat(contourC_Right).data;
    nptsC_Right = Mat(contourC_Right).rows;
    
    ptsBarrier = (Point*) Mat(contourBarrier).data;
    nptsBarrier = Mat(contourBarrier).rows;
}

void SceneDetector::drawZones(Mat& dst)
{
    polylines(dst, &ptsA, &nptsA, 1, true, Scalar(255,255,0), 2, LINE_AA, 0);
    polylines(dst, &ptsB_Left, &nptsB_Left, 1, true, Scalar(255,0,0), 2, LINE_AA, 0);
    polylines(dst, &ptsB_Right, &nptsB_Right, 1, true, Scalar(255,0,0), 2, LINE_AA, 0);
    polylines(dst, &ptsC_Left, &nptsC_Left, 1, true, Scalar(255,0,255), 2, LINE_AA, 0);
    polylines(dst, &ptsC_Right, &nptsC_Right, 1, true, Scalar(255,0,255), 2, LINE_AA, 0);
    polylines(dst, &ptsBarrier, &nptsBarrier, 1, true, Scalar(255,255,255), 2, LINE_AA, 0);
}

void SceneDetector::drawGlobalContours(Mat& dst)
{
    if(!train){
        int cnt = 0;
        vector<Rect> r;
        vector<vector<Point>>::
        const_iterator itc = globalContours.begin();
        while (itc != globalContours.end()) {
            if (  itc->size() > 123){//79
                Rect tmp = boundingRect(Mat(globalContours[cnt]));
                r.push_back(tmp);
            }
            ++cnt;
            ++itc;
        }
        if(r.size() >= 2){
            vector<Rect>::const_iterator it = r.begin();
            while (it != (r.end() - 1)) {
                Point center1(it->x + it->width/2, it->y + it->height/2);
                Point center2((it + 1)->x + (it + 1)->width/2, (it + 1)->y + (it + 1)->height/2);
//                cout<<"norm: "<<norm(center2 - center1)<<" point "<<center1<<center2<<endl;
                if(norm(center2 - center1) <= 200){
                    //erase the smaller one
                    if(it->area()>(it+1)->area())
                        r.erase(it+1);
                    else{
                        r.erase(it);
                    }
                }
                else{
                    //if nothing deleted ++it
                    ++it;
                }
            }
        }
        for (int i = 0; i < r.size(); i++) {
            rectangle(dst, r[i], Scalar(0, 0 ,255), 4);
        }
    }
}

void SceneDetector::createZonesMask(int R, int C)
{
    maskA = Mat::zeros(R, C, CV_8U);
    maskB_L = Mat::zeros(R, C, CV_8U);
    maskB_R = Mat::zeros(R, C, CV_8U);
    maskC_L = Mat::zeros(R, C, CV_8U);
    maskC_R = Mat::zeros(R, C, CV_8U);
    maskBarrier = Mat::zeros(R, C, CV_8U);
    fillConvexPoly(maskA, contourA, Scalar(255), LINE_AA, 0);
    fillConvexPoly(maskB_L, contourB_Left, Scalar(255), LINE_AA, 0);
    fillConvexPoly(maskB_R, contourB_Right, Scalar(255), LINE_AA, 0);
    fillConvexPoly(maskC_L, contourC_Left, Scalar(255), LINE_AA, 0);
    fillConvexPoly(maskC_R, contourC_Right, Scalar(255), LINE_AA, 0);
    fillConvexPoly(maskBarrier, contourBarrier, Scalar(255), LINE_AA, 0);
    
//    Mat tmp;
//    tmp = maskA.clone();
//    bitwise_or(tmp, maskB_L, tmp);
//    bitwise_or(tmp, maskB_R, tmp);
//    bitwise_or(tmp, maskC_L, tmp);
//    bitwise_or(tmp, maskC_R, tmp);
//    bitwise_or(tmp, maskBarrier, tmp);
//    imshow("tmp", tmp);
}

void SceneDetector::fixDetectAccuracy(vector<vector<Point>>& src, int min, int max)
{
    vector<vector<Point>>::
    const_iterator itc = src.begin();
    while (itc != src.end()) {
        if (itc->size() < min || itc->size() > max)
            itc = src.erase(itc);
        else
            ++itc;
    }
}
double SceneDetector::getZoneContourArea(vector<vector<Point>>& src)
{
    double AreaSUM = 0;
    vector<vector<Point>>::const_iterator itc = src.begin();
    while (itc != src.end()) {
        AreaSUM += itc->size();
        ++itc;
    }
//    cout<<AreaSUM<<endl;
    return AreaSUM;
}
void SceneDetector::setDiffContour(Mat src)
{
    absdiff(background, src, diffContour);
    imshow("abs", diffContour);
    GaussianBlur(diffContour, diffContour, Size(11,11), 3.5);
    imshow("gau", diffContour);
    threshold(diffContour, diffContour, 60, 255, THRESH_BINARY);
//    adaptiveThreshold(diffContour, diffContour, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 105, -40);
    imshow("thr", diffContour);
    
    findContours(diffContour, globalContours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//    fixDetectAccuracy(contours, 40, 2000);
    
    diffContour = Mat(src.size(), CV_8UC1, Scalar(0));
    drawContours(diffContour, globalContours, -1, Scalar(255), -1);
}

bool SceneDetector::detectZoneA()
{
    Mat srcANDmask;
    bool result = false;
    int trainAreaThreshold = 0;
    bitwise_and(diffContour, maskA, srcANDmask);
    vector<vector<Point>> contours;
    findContours(srcANDmask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    fixDetectAccuracy(contours, 123, 10000);
    if (contours.size() >=1) {
//        cout<<"A: "<<contours.size()<<endl;
        result = true;
    }
    if(detectZoneBarrier()){
        cout<<" Barrier lowered!";
        barrier = true;
        trainAreaThreshold = 1000;
    }
    else if(barrier){
        trainAreaThreshold = 1000;
    }
    else{
        barrier = false;
        trainAreaThreshold = 1460;
    }
    
    if(getZoneContourArea(contours) >= trainAreaThreshold){
        train = true;
    }
    else{
        train = false;
    }
    
    if (!train &&!detectZoneBarrier()){
        barrier = false;
    }
    cout<<getZoneContourArea(contours);
    Mat out(diffContour.size(),diffContour.type(),Scalar(0));
    for(int i = 0; i<contours.size(); i++){
        drawContours(out, contours, i, Scalar(255));
    }
    drawZones(out);
    imshow("A", out);
    return result;
}

bool SceneDetector::detectZoneB_L()
{
    Mat srcANDmask;
    bool result = false;
    bitwise_and(diffContour, maskB_L, srcANDmask);
    vector<vector<Point>> contours;
    findContours(srcANDmask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    fixDetectAccuracy(contours, 105, 2000);
    if (contours.size() >=1) {
//        cout<<"B_L: "<<contours.size()<<endl;
        result = true;
    }
    Mat out(diffContour.size(),diffContour.type(),Scalar(0));
    for(int i = 0; i<contours.size(); i++){
        drawContours(out, contours, i, Scalar(255));
    }
//    cout<<"B_L size: "<<getZoneContourArea(contours)<<endl;
    imshow("B_L",  out);
    return result;
}

bool SceneDetector::detectZoneB_R()
{
    Mat srcANDmask;
    bool result = false;
    bitwise_and(diffContour, maskB_R, srcANDmask);
    vector<vector<Point>> contours;
    findContours(srcANDmask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    fixDetectAccuracy(contours, 80, 2000);
    if (contours.size() >=1) {
//        cout<<"B_R: "<<contours.size()<<endl;
        result = true;
    }
    Mat out(diffContour.size(),diffContour.type(),Scalar(0));
    for(int i = 0; i<contours.size(); i++){
        drawContours(out, contours, i, Scalar(255));
    }
    imshow("B_R", out);
    return result;
}

bool SceneDetector::detectZoneC_L()
{
    Mat srcANDmask;
    bool result = false;
    bitwise_and(diffContour, maskC_L, srcANDmask);
    vector<vector<Point>> contours;
    findContours(srcANDmask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    fixDetectAccuracy(contours, 85, 2000);
//    cout<<"C_L Area size: "<<getZoneContourArea(contours)<<endl;
    if (contours.size() >=1) {
//        cout<<"C_L: "<<contours.size()<<endl;
        result = true;
    }
    Mat out(diffContour.size(),diffContour.type(),Scalar(0));
    for(int i = 0; i<contours.size(); i++){
        drawContours(out, contours, i, Scalar(255));
    }
    imshow("C_L", out);
    return result;
}

bool SceneDetector::detectZoneC_R()
{
    Mat srcANDmask;
    bool result = false;
    bitwise_and(diffContour, maskC_R, srcANDmask);
    vector<vector<Point>> contours;
    findContours(srcANDmask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    fixDetectAccuracy(contours, 40, 2000);
    if (contours.size() >=1) {
//        cout<<"C_R: "<<contours.size()<<endl;
        result = true;
    }
    Mat out(diffContour.size(),diffContour.type(),Scalar(0));
    for(int i = 0; i<contours.size(); i++){
        drawContours(out, contours, i, Scalar(255));
    }
    imshow("C_R", out);
    return result;
}

bool SceneDetector::detectZoneBarrier()
{
    Mat srcANDmask;
//    Mat lineTMP;
    bool result = false;
    bitwise_and(diffContour, maskBarrier, srcANDmask);
    vector<vector<Point>> contours;
    findContours(srcANDmask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    fixDetectAccuracy(contours, 1, 50);
    Mat out(diffContour.size(),diffContour.type(),Scalar(0));
    for(int i = 0; i<contours.size(); i++){
        drawContours(out, contours, i, Scalar(255));
    }
//    lineTMP = out.clone();
    vector<Vec4i> lines;
    HoughLinesP(out, lines, 1, CV_PI/180, 0, 175, 200 );
//    for( size_t i = 0; i < lines.size(); i++ )
//    {
//        Vec4i l = lines[i];
//        line( lineTMP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(155), 3, LINE_AA);
//    }
//    imshow("LINE", lineTMP);
    if(lines.size()>0 && getZoneContourArea(contours) >= 140){
        // if there is a line in the zone, means the barrier is lowered
        result = true;
    }
    
    
    imshow("Barrier", out);
    return result;
}


void SceneDetector::OutputSceneResult()
{
    bool isEmpty = true;
    if(detectZoneA()){
        cout<<" Track not clear!";
        isEmpty = false;
    }
    if(!train)
    {
        if(detectZoneB_L()){
            cout<<" Left lane entering!";
            isEmpty = false;
        }
        if(detectZoneB_R()){
            cout<<" Right lane entering!";
            isEmpty = false;
        }
        if(detectZoneC_L()){
            cout<<" Left lane leaving!";
            isEmpty = false;
        }
        if(detectZoneC_R()){
            cout<<" Right lane leaving!";
            isEmpty = false;
        }
    }
    else{
        cout<<" Trian using rail!";
        isEmpty = false;
        
        if(detectZoneB_L()){
            cout<<" Left lane entering!";
            isEmpty = false;
        }
    }
    
    if(isEmpty){
        cout<<" Empty!";
    }
    drawZones(diffContour);
    imshow("D", diffContour);
}
