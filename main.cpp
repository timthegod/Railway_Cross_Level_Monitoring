//
//  main.cpp
//  RailCrossing
//
//  Created by 楊廷禹 on 2019/1/16.
//  Copyright © 2019 Yang Ting Yu. All rights reserved.
//
#define testPNG 1
//#define testVIDEO 1

#include <iostream>
#include <string>
#include <vector>


#include "SceneDetector.hpp"

#include "opencv.hpp"
#include "highgui.hpp"
#include "core.hpp"
#include "imgproc.hpp"

using namespace std;
using namespace cv;

int g_run = 1;

int main(int argc, const char * argv[]) {
    Mat srcEmpty, frame, frameGray;
    VideoCapture cap;
    cap.open("levelcrossing.mp4");
    
    vector<String> fn;
    glob("Individual/empty/*.png", fn, false);
    vector<Mat> emptyScene;
    size_t fileCount = fn.size();
    for (size_t i=0; i<fileCount; i++)
        emptyScene.push_back(imread(fn[i]));
    
    string srcEmptyPath = "Individual/empty/lc-00290.png";
    srcEmpty = imread(srcEmptyPath, -1);
    
    SceneDetector SD(emptyScene, srcEmpty);
    
//    SceneDetector SD(srcEmpty);
    SD.createZonesMask(srcEmpty.rows, srcEmpty.cols);
#ifdef testVIDEO
    int cnt=0;
    for (int k=0;;k++)
    {
        if (g_run != 0)
        {
            cout<<"----------------------"<<endl;
            cout<<"frame: "<<cnt<<endl;
            cap >> frame;
            if (frame.empty()){
                cout<<"Video end!"<<endl;
                break;
            }
            cvtColor(frame, frameGray, COLOR_BGR2GRAY);
            SD.setDiffContour(frameGray);
            SD.OutputSceneResult();
            
            SD.drawZones(frame);
            SD.drawGlobalContours(frame);
            imshow("F", frame);

            cnt++;
            g_run -= 1;
        }
        char c = (char)waitKey(33);
        if (c == 's')  // single step
            g_run = 1;
        
        if (c == 'r')  // run mode
            g_run = -1;
        
    }
#endif
    
#ifdef testPNG
    vector<string> fn1;
    int SceneCount = 0;
    vector<Mat> scene;
    for(int i = 1; i < argc; i++)
    {
        fn1.push_back(argv[i]);
        SceneCount++;
    }
    for (size_t i=0; i<SceneCount; i++)
        scene.push_back(imread(fn1[i]));
    
    
    for (int i = 0; i < scene.size(); i++) {
//        cout<<"----------------------"<<endl;
//        cout<<"frame: "<<i<<endl;
        cout<<fn1[i]<<":";
        frame = scene[i].clone();
        if (frame.empty()){
            cout<<"PNG end!"<<endl;
            break;
        }
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        SD.setDiffContour(frameGray);
        SD.OutputSceneResult();
        
        SD.drawZones(frame);
        SD.drawGlobalContours(frame);
        imshow("F", frame);
        
        char c = waitKey(100);
        if(c == 's'){
            waitKey(100000);
        }
        cout<<endl;
    }
    
#endif
    waitKey(33);
    return 0;
}
