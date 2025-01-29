#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include "image.h"

using namespace std;

Image laplacian_detector(const Image& im, float sigma) {
    Image gray = im.c == 1 ? im : rgb_to_grayscale(im);
    
    int size = ceil(sigma * 6);
    if (!(size % 2)) size++;
    Image kernel(size, size, 1);
    
    float sigma2 = sigma * sigma;
    float sigma4 = sigma2 * sigma2;
    for(int y = 0; y < size; y++) {
        for(int x = 0; x < size; x++) {
            float rx = x - size/2;
            float ry = y - size/2;
            float r2 = rx*rx + ry*ry;
            kernel(x,y,0) = (r2 - 2*sigma2) * exp(-r2/(2*sigma2)) / (2*M_PI*sigma4);
        }
    }
    
    float sum = 0;
    for(int i = 0; i < size*size; i++) sum += fabs(kernel.data[i]);
    for(int i = 0; i < size*size; i++) kernel.data[i] /= sum;
    
    Image response = convolve_image(gray, kernel, true);
    
    for(int i = 0; i < response.w * response.h; i++) {
        response.data[i] = fabs(response.data[i]);
    }
    
    return response;
}

vector<Descriptor> log_keypoint_detector(const Image& im, float sigma, float thresh, int window, int nms) {
    Image response = laplacian_detector(im, sigma);
    
    float max_val = 0;
    for(int i = 0; i < response.w * response.h; i++) {
        max_val = max(max_val, response.data[i]);
    }
    for(int i = 0; i < response.w * response.h; i++) {
        response.data[i] /= max_val;
    }
    
    Image nms_response = nms_image(response, nms);
    
    return detect_corners(im, nms_response, thresh, window);
}

Image draw_keypoints(const Image& im, const vector<Descriptor>& keypoints) {
    Image marked = im;
    for(auto& d : keypoints) {
        int x = d.p.x;
        int y = d.p.y;
        
        for(int i = -4; i < 5; ++i) {
            marked.set_pixel(x+i, y, 0, 0);
            marked.set_pixel(x, y+i, 0, 0);
            marked.set_pixel(x+i, y, 1, 1);
            marked.set_pixel(x, y+i, 1, 1);
            marked.set_pixel(x+i, y, 2, 0);
            marked.set_pixel(x, y+i, 2, 0);
        }
    }
    return marked;
}