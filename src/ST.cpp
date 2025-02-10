#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <thread>

#include "image.h"
#include "matrix.h"

using namespace std;

vector<Descriptor> shi_tomasi_detector(const Image& im, float sigma, float thresh, int window, int nms_size) {
    Image S = structure_matrix(im, sigma);
    
    Image R(S.w, S.h, 1);
    for (int y = 0; y < S.h; y++) {
        for (int x = 0; x < S.w; x++) {
            float Ixx = S(x, y, 0); 
            float Iyy = S(x, y, 1); 
            float Ixy = S(x, y, 2); 
            
            float trace = Ixx + Iyy;
            float determinant = (Ixx * Iyy) - (Ixy * Ixy);
            
            float temp = (trace / 2) * (trace / 2) - determinant;
            if (temp < 0) temp = 0;
            float sqrt_term = sqrt(temp);
            
            float lambda1 = trace / 2 + sqrt_term;
            float lambda2 = trace / 2 - sqrt_term;

            R(x, y, 0) = std::min(lambda1, lambda2);
        }
    }
    
    Image Rnms = nms_image(R, nms_size);
    
    return detect_corners(im, Rnms, thresh, window);
}


Image detect_and_draw_corners_shi_tomasi(const Image& im, float sigma, float thresh, int window, int nms_size) {
    TIME(1);
    vector<Descriptor> d = shi_tomasi_detector(im, sigma, thresh, window, nms_size);
    printf("Numero di Descrittori: %zu\n", d.size());
    return mark_corners(im, d);
}


Image find_and_draw_shi_tomasi_matches(const Image& a, const Image& b,
                                       float sigma, float thresh, int window, int nms_window) {
    TIME(1);
    vector<Descriptor> ad = shi_tomasi_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = shi_tomasi_detector(b, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image draw_shi_tomasi_inliers(const Image& a, const Image& b,
                              float sigma, float thresh, int window, int nms_window,
                              float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = shi_tomasi_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = shi_tomasi_detector(b, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_shi_tomasi(const Image& a, const Image& b,
                                float sigma, float thresh, int window, int nms_window,
                                float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = shi_tomasi_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = shi_tomasi_detector(b, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}