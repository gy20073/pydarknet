#include "bridge.h"

#if USE_CV == 1

#include <iostream>

#include <stdio.h>



#ifdef __cplusplus
extern "C" {
#endif
// Include darknet as a C Library
image ipl_to_image(IplImage* src);

#ifdef __cplusplus
}
#endif

using namespace cv;

image get_darknet_image(const Mat &input){
    // Darknet requires BGR order
    Mat flipped;
    cvtColor(input, flipped, CV_RGB2BGR);

    // Darknet uses IPL Image
    IplImage* iplImage;
    iplImage = cvCreateImage(cvSize(flipped.cols,flipped.rows),8,3);

    IplImage ipltemp=flipped;
    cvCopy(&ipltemp,iplImage);

    flipped.release();

    // Convert to Darknet Image
    image darknet_image = ipl_to_image(iplImage);

    // Free memory
    cvReleaseImage(&iplImage);
    return darknet_image;
}

#endif


int get_yolo_logits(network *net, int *batch, int *w, int *h, int *n, int *classp5, float** data){
    int i;
    int yolo_i = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO){
            batch[yolo_i] = l.batch;
            w[yolo_i] = l.w;
            h[yolo_i] = l.h;
            n[yolo_i] = l.n;
            classp5[yolo_i] = l.classes + 4 + 1;
            data[yolo_i] = net->layers[i].output;
            yolo_i ++;
        }
    }
    return yolo_i;
}
