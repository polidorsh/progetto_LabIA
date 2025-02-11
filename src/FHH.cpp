#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include "image.h"

using namespace std;

Image compute_hessian(const Image& im, float sigma) {
    Image smoothed = smooth_image(im, sigma);
    
    Image fx = make_gx_filter();
    Image fy = make_gy_filter();
    
    Image Ix = convolve_image(smoothed, fx, true);
    Image Ixx = convolve_image(Ix, fx, true);
    
    Image Iy = convolve_image(smoothed, fy, true);
    Image Iyy = convolve_image(Iy, fy, true);
    
    Image Ixy = convolve_image(Ix, fy, true);
    
    Image H(im.w, im.h, 3);
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            H(x,y,0) = Ixx(x,y,0);  // Ixx
            H(x,y,1) = Iyy(x,y,0);  // Iyy
            H(x,y,2) = Ixy(x,y,0);  // Ixy
        }
    }
    return H;
}

void normalize_response(Image& R) {
    float min_val = INFINITY;
    float max_val = -INFINITY;
    
    for(int y = 0; y < R.h; y++) {
        for(int x = 0; x < R.w; x++) {
            float val = R(x,y,0);
            if(val < min_val) min_val = val;
            if(val > max_val) max_val = val;
        }
    }
    
    float range = max_val - min_val;
    if(range < 1e-8f) range = 1e-8f;
    
    for(int y = 0; y < R.h; y++) {
        for(int x = 0; x < R.w; x++) {
            R(x,y,0) = (R(x,y,0) - min_val) / range;
        }
    }
}

vector<Descriptor> fhh_detector(const Image& im, int method, float sigma, 
                                          float thresh, int window, int nms_window) {
    Image R(im.w, im.h, 1);
    
    if (method == 0) {  
        Image H = compute_hessian(im, sigma);
        
        for(int y = 0; y < H.h; y++) {
            for(int x = 0; x < H.w; x++) {
                float Ixx = H(x,y,0);
                float Iyy = H(x,y,1);
                float Ixy = H(x,y,2);
                
                float det = Ixx * Iyy - Ixy * Ixy;
                
                R(x,y,0) = det ;
            }
        }
    } else {
        Image S = structure_matrix(im, sigma);
        
        for(int y = 0; y < S.h; y++) {
            for(int x = 0; x < S.w; x++) {
                float a = S(x,y,0);  // Ix^2
                float b = S(x,y,1);  // Iy^2
                float c = S(x,y,2);  // IxIy
                
                float trace = a + b;
                float det = a*b - c*c;
                
                switch(method) {
                    case 1:  // Förstner
                        R(x,y,0) = det / (trace + 1e-8f);
                        break;
                        
                    case 2:  // Harris
                        R(x,y,0) = det - 0.04f * powf(trace, 2);
                        break;
                        
                    case 3:  // Ibrido
                        float forstner_weight = det / (trace + 1e-8f);
                        float harris_weight = det - 0.04f * powf(trace, 2);
                        R(x,y,0) = 0.5f * (forstner_weight + harris_weight);
                        // Alternativa: R(x,y,0) = sqrtf(forstner_weight * harris_weight);
                        break;
                }
            }
        }
    }

    normalize_response(R);

    Image Rnms = nms_image(R, nms_window);
    return detect_corners(im, Rnms, thresh, window);
}

Image detect_and_draw_fhh(const Image& im, int method, float sigma, 
                                    float thresh, int window, int nms_window) {
    TIME(1);
    vector<Descriptor> corners = fhh_detector(im, method, sigma, 
                                                        thresh, window, nms_window);
    printf("Numero di Descrittori: %ld\n", corners.size());
    return mark_corners(im, corners);
}

Image find_and_draw_fhh_matches(const Image& a, const Image& b,
                                          int method, float sigma, float thresh, 
                                          int window, int nms_window) {
    TIME(1);
    vector<Descriptor> ad = fhh_detector(a, method, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = fhh_detector(b, method, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image draw_fhh_inliers(const Image& a, const Image& b,
                                   int method, float sigma, float thresh, 
                                   int window, int nms_window,
                                   float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = fhh_detector(a, method, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = fhh_detector(b, method, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_fhh(const Image& a, const Image& b,
                                     int method, float sigma, float thresh, 
                                     int window, int nms_window,
                                     float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = fhh_detector(a, method, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = fhh_detector(b, method, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}
