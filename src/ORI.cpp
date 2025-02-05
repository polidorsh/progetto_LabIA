#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include "image.h"

vector<Descriptor> oriented_detector(const Image& im, float sigma, float thresh, int window) {
    Image gray = (im.c == 3) ? rgb_to_grayscale(im) : im;
    
    const int num_angles = 8;
    const float PI = 3.14159265358979323846;
    vector<Image> responses;
    
    for(int i = 0; i < num_angles; i++) {
        float angle = i * PI / num_angles;
        float dx = cos(angle);
        float dy = sin(angle);
        
        int size = ceil(sigma * 6);
        if(!(size % 2)) size++; 
        Image filter(size, size, 1);
        
        for(int y = 0; y < size; y++) {
            for(int x = 0; x < size; x++) {
                float rx = x - size/2;
                float ry = y - size/2;
                float proj = rx*dx + ry*dy;
                float perp = rx*(-dy) + ry*dx;
                filter(x,y,0) = exp(-(perp*perp)/(2*sigma*sigma)) * 
                               (1 - (proj*proj)/(sigma*sigma)) * 
                               exp(-(proj*proj)/(2*sigma*sigma));
            }
        }
        
        Image response = convolve_image(gray, filter, true);
        responses.push_back(response);
    }
    
    Image maxResponse(gray.w, gray.h, 1);
    for(int y = 0; y < gray.h; y++) {
        for(int x = 0; x < gray.w; x++) {
            float maxVal = 0;
            for(auto& resp : responses) {
                maxVal = max(maxVal, fabs(resp(x,y,0)));
            }
            maxResponse(x,y,0) = maxVal;
        }
    }
    
    Image nms = nms_image(maxResponse, window);
    
    return detect_corners(gray, nms, thresh, window);
}

vector<Descriptor> oriented_corner_detector(const Image& im, float sigma, float thresh, int window, int nms_window) {
    return oriented_detector(im, sigma, thresh, window);
}

Image oriented_corners(const Image& im, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> corners = oriented_corner_detector(im, sigma, thresh, window, nms);
    printf("Numero di Descrittori: %ld\n", corners.size());
    return mark_corners(im, corners);
}

Image oriented_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> ad = oriented_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = oriented_corner_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());

    
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    return draw_matches(A, B, m, {});
}

Image oriented_inliers(const Image& a, const Image& b, float sigma, float thresh, int window, 
                      int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = oriented_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = oriented_corner_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    
    Matrix H = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, H, m, inlier_thresh);
}

Image oriented_panorama(const Image& a, const Image& b, float sigma, float thresh, int window, 
                       int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = oriented_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = oriented_corner_detector(b, sigma, thresh, window, nms);
    
    vector<Match> matches = match_descriptors(ad, bd);
    
    Matrix H = RANSAC(matches, inlier_thresh, iters, cutoff);
    
    return combine_images(a, b, H, 0.5);
}
