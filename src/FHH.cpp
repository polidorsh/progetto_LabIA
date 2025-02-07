#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include "image.h"


float forstner_interest_measure(const Image& S) {
    float det = S(0, 0, 0) * S(1, 1, 0) - S(0, 1, 0) * S(1, 0, 0);
    float tr = S(0, 0, 0) + S(1, 1, 0);
    
    float qqq = det / (tr + 1e-6); 
    
    return qqq;
}

float harris_response(const Image& S) {
    float det = S(0, 0, 0) * S(1, 1, 0) - S(0, 1, 0) * S(1, 0, 0);
    float tr = S(0, 0, 0) + S(1, 1, 0);
    
    const float k = 0.04; 
    return det - k * tr * tr;
}

vector<Descriptor> forstner_harris_hessian_detector(const Image& im, float sigma, float thresh,
int window, int nms) {
    Image S = structure_matrix(im, sigma);
    Image R(im.w, im.h, 1);
    float max_forstner = -std::numeric_limits<float>::max();
    float max_harris = -std::numeric_limits<float>::max();
    
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            Image localS(2, 2, 1);
            localS(0, 0, 0) = S(x, y, 0);
            localS(1, 1, 0) = S(x, y, 1);
            localS(0, 1, 0) = localS(1, 0, 0) = S(x, y, 2);
            
            float forstner = forstner_interest_measure(localS);
            float harris = harris_response(localS);
            
            max_forstner = std::max(max_forstner, std::abs(forstner));
            max_harris = std::max(max_harris, std::abs(harris));
        }
    }
    
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            Image localS(2, 2, 1);
            localS(0, 0, 0) = S(x, y, 0);
            localS(1, 1, 0) = S(x, y, 1);
            localS(0, 1, 0) = localS(1, 0, 0) = S(x, y, 2);
            
            float forstner = forstner_interest_measure(localS);
            float harris = harris_response(localS);
            
            R(x, y, 0) = (forstner / max_forstner + harris / max_harris) / 2.0;
        }
    }
    
    float mean_response = 0, std_response = 0;
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            mean_response += R(x, y, 0);
        }
    }
    mean_response /= (im.w * im.h);
    
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            std_response += pow(R(x, y, 0) - mean_response, 2);
        }
    }
    std_response = sqrt(std_response / (im.w * im.h));
    
    float adaptive_thresh = mean_response + thresh * std_response;
    Image Rnms = nms_image(R, nms);
    return detect_corners(im, Rnms, adaptive_thresh, window);
}

Image detect_and_draw_forstner_corners(const Image& im, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> d = forstner_harris_hessian_detector(im, sigma, thresh, window, nms);
    printf("Numero di Descrittori: %ld\n", d.size());
    return mark_corners(im, d);
}


Image find_and_draw_forstner_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> ad = forstner_harris_hessian_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = forstner_harris_hessian_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image find_and_draw_forstner_inliers(const Image& a, const Image& b, float sigma, float thresh, int window,
int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = forstner_harris_hessian_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = forstner_harris_hessian_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_forstner(const Image& a, const Image& b, float sigma, float thresh, int window,
int nms, float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = forstner_harris_hessian_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = forstner_harris_hessian_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}