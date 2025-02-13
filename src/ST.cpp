#include <vector>
#include <algorithm>
#include "image.h"

vector<Descriptor> shi_tomasi_detector(const Image& im, bool is_adaptive, float sigma, float thresh, 
                                     int window, int nms_size) {
    Image S = structure_matrix(im, sigma);
    
    Image R(S.w, S.h, 1);
    float max_response = -INFINITY;
    float mean_response = 0.0;
    int valid_points = 0;
    
    for(int y = 0; y < S.h; y++) {
        for(int x = 0; x < S.w; x++) {
            float Ixx = S(x,y,0);
            float Iyy = S(x,y,1);
            float Ixy = S(x,y,2);
            
            float term1 = (Ixx + Iyy) / 2.0f;
            float term2 = sqrt(pow(Ixx - Iyy, 2) / 4.0f + pow(Ixy, 2));
            
            R(x,y,0) = term1 - term2; 
            
        }
    }
    
    for(int y = 0; y < R.h; y++) {
        for(int x = 0; x < R.w; x++) {
            if(R(x,y,0) > 0) {
                max_response = std::max(max_response, R(x,y,0));
                mean_response += R(x,y,0);
                valid_points++;
            }
        }
    }
    mean_response /= valid_points;
    
    float final_thresh;
    if(is_adaptive) {
        final_thresh = thresh * (0.5f * max_response + 0.5f * mean_response);
    } else {
        final_thresh = thresh;  
    }
    
    Image Rnms = nms_image(R, nms_size);
    
    return detect_corners(im, Rnms, final_thresh, window);
}

Image detect_and_draw_shi_tomasi(const Image& im, bool is_adaptive, float sigma, float thresh, 
                                int window, int nms_size) {
    TIME(1);
    vector<Descriptor> keypoints = shi_tomasi_detector(im, is_adaptive, sigma, thresh, window, nms_size);
    printf("Numero di Descrittori: %zu\n", keypoints.size());
    return mark_corners(im, keypoints);
}
Image find_and_draw_shi_tomasi_matches(const Image& a, const Image& b,
                                       bool is_adaptive, float sigma, float thresh,
                                       int window, int nms_size) {
    TIME(1);
    vector<Descriptor> ad = shi_tomasi_detector(a, is_adaptive, sigma, thresh, window, nms_size);
    vector<Descriptor> bd = shi_tomasi_detector(b, is_adaptive, sigma, thresh, window, nms_size);
    
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %zu\n", m.size());
    
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image draw_shi_tomasi_inliers(const Image& a, const Image& b,
                              bool is_adaptive, float sigma, float thresh,
                              int window, int nms_size,
                              float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = shi_tomasi_detector(a, is_adaptive, sigma, thresh, window, nms_size);
    vector<Descriptor> bd = shi_tomasi_detector(b, is_adaptive, sigma, thresh, window, nms_size);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_shi_tomasi(const Image& a, const Image& b,
                                bool is_adaptive, float sigma, float thresh,
                                int window, int nms_size,
                                float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = shi_tomasi_detector(a, is_adaptive, sigma, thresh, window, nms_size);
    vector<Descriptor> bd = shi_tomasi_detector(b, is_adaptive, sigma, thresh, window, nms_size);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}
