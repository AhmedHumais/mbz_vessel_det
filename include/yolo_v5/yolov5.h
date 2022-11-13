#ifndef YOLOV5_H
#define YOLOV5_H

#include <iostream>
#include <chrono>
#include <cmath>
#include "yolo_v5/cuda_utils.h"
#include "yolo_v5/logging.h"
#include "yolo_v5/common.h"
#include "yolo_v5/yoloparam.h"
#include "yolo_v5/coco_names.hpp"

using namespace cv;
using namespace std;

typedef struct
{
    float probability;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    uint id;
    uint img_width;
    uint img_height;
    int center_dist;
    float x_center;
    float y_center;
    float w_pixel;
    float h_pixel;
    uint flag;
    string class_id;
} _bbox;

class Yolov5
{
public:
    Yolov5(void *YParam);
    ~Yolov5();
    void doInference(IExecutionContext &context, cudaStream_t &stream,
                     void **buffers, float *input, float *output, int batchSize);
    void initEngine();
    void objDetection(Mat img, Mat &dst, vector<_bbox> &bbox); //vector<DetectBox>& detect_info
    Mat preprocessImg(Mat &img, int input_w, int input_h);

private:
    IExecutionContext *context;
    ICudaEngine *engine;
    IRuntime *runtime;
    void *buffers[2];
    cudaStream_t stream;

    const char *INPUT_BLOB_NAME = "data";
    const char *OUTPUT_BLOB_NAME = "prob";
    Logger gLogger;
    signed int inputIndex;
    signed int outputIndex;

public:
    yoloparam YP;
    Common *com;
};

#endif // YOLOV5_H
