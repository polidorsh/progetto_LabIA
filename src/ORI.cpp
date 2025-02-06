#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include "image.h"

Image make_steerable_filter(float sigma, float theta) {
    int w = ceil(sigma * 6);
    if (!(w % 2)) w++;
    Image filter(w, w, 1);
    
    float cos_t = cos(theta);
    float sin_t = sin(theta);
    
    for(int y = 0; y < w; y++) {
        for(int x = 0; x < w; x++) {
            float rx = x - (w/2);
            float ry = y - (w/2);
            
            float xr = rx * cos_t + ry * sin_t;
            float yr = -rx * sin_t + ry * cos_t;
            
            float g = exp(-(xr*xr + yr*yr)/(2*sigma*sigma));
            float h = (xr*xr - sigma*sigma) / (sigma*sigma*sigma*sigma);
            filter(x,y,0) = h * g;
        }
    }
    filter.l1_normalize();
    return filter;
}

Image steerable_response(const Image& im, float sigma) {
    Image response(im.w, im.h, 1);
    
    const int n_orientations = 8;
    const float pi = M_PI;
    
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            float max_response = -INFINITY;
            
            for(int i = 0; i < n_orientations; i++) {
                float theta = i * pi / n_orientations;
                Image filter = make_steerable_filter(sigma, theta);
                
                float r = 0;
                for(int fy = 0; fy < filter.h; fy++) {
                    for(int fx = 0; fx < filter.w; fx++) {
                        int ix = x + fx - filter.w/2;
                        int iy = y + fy - filter.h/2;
                        r += im.clamped_pixel(ix, iy, 0) * filter(fx, fy, 0);
                    }
                }
                max_response = fmax(max_response, fabs(r));
            }
            response(x,y,0) = max_response;
        }
    }
    return response;
}

vector<Descriptor> steerable_keypoint_detector(const Image& im2, float sigma, float thresh, int window, int nms_window) {
    Image im;
    if(im2.c == 1) im = im2;
    else im = rgb_to_grayscale(im2);
    Image response = steerable_response(im, sigma);
    Image nms = nms_image(response, nms_window);
    return detect_corners(im, nms, thresh, window);
}

Image detect_and_draw_keypoints(const Image& im, float sigma, float thresh, int window, int nms_window) {
    TIME(1);
    vector<Descriptor> d = steerable_keypoint_detector(im, sigma, thresh, window, nms_window);
    printf("Numero di Descrittori: %ld\n", d.size());
    return mark_corners(im, d);
}

Image oriented_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> ad = steerable_keypoint_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = steerable_keypoint_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());

    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    return draw_matches(A, B, m, {});
}

Image oriented_inliers(const Image& a, const Image& b, float sigma, float thresh, int window, 
                      int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = steerable_keypoint_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = steerable_keypoint_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    
    Matrix H = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, H, m, inlier_thresh);
}

Image oriented_panorama(const Image& a, const Image& b, float sigma, float thresh, int window, 
                       int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = steerable_keypoint_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = steerable_keypoint_detector(b, sigma, thresh, window, nms);
    vector<Match> matches = match_descriptors(ad, bd);
    Matrix H = RANSAC(matches, inlier_thresh, iters, cutoff);

    return combine_images(a, b, H, 0.5);
}
