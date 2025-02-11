#include <vector>
#include <algorithm>
#include "image.h"

// Calcola la soglia adattiva per una finestra locale
float compute_adaptive_threshold(const Image& response_map, int x, int y, int window_size) {
    float max_val = 0.0f;
    float mean_val = 0.0f;
    int count = 0;
    
    // Calcola il massimo e la media nella finestra
    for(int dy = -window_size/2; dy <= window_size/2; ++dy) {
        for(int dx = -window_size/2; dx <= window_size/2; ++dx) {
            float val = response_map.clamped_pixel(x + dx, y + dy, 0);
            max_val = std::max(max_val, val);
            mean_val += val;
            count++;
        }
    }
    mean_val /= count;
    
    // Soglia adattiva: media tra il massimo locale e la media locale
    return (max_val + mean_val) / 2.0f;
}

// Applica NMS adattivo
Image adaptive_nms_image(const Image& im, int w) {
    Image r = im;
    
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            float adaptive_thresh = compute_adaptive_threshold(im, x, y, w);
            float current_val = im(x,y,0);
            
            // Sopprime i punti sotto la soglia adattiva
            if(current_val < adaptive_thresh) {
                r(x,y,0) = -0.00001;
                continue;
            }
            
            // Controllo NMS tradizionale
            for(int ny = y-w; ny <= y+w; ny++) {
                for(int nx = x-w; nx <= x+w; nx++) {
                    if(im.clamped_pixel(nx,ny,0) > current_val) {
                        r(x,y,0) = -0.00001;
                        goto next_pixel;
                    }
                }
            }
            next_pixel:;
        }
    }
    return r;
}

vector<Descriptor> shi_tomasi_detector(const Image& im, float sigma, float thresh, 
                                     int window, int nms_window) {
    Image S = structure_matrix(im, sigma);
    
    Image R(S.w, S.h, 1);
    for(int y = 0; y < S.h; y++) {
        for(int x = 0; x < S.w; x++) {
            float a = S(x,y,0); // Ixx
            float b = S(x,y,1); // Iyy
            float c = S(x,y,2); // Ixy
            
            float trace = a + b;
            float det = a*b - c*c;
            float lambda1 = (trace + sqrt(trace*trace - 4*det)) / 2.0f;
            float lambda2 = (trace - sqrt(trace*trace - 4*det)) / 2.0f;
            
            R(x,y,0) = std::min(lambda1, lambda2);
        }
    }
    
    Image Rnms = adaptive_nms_image(R, nms_window);
    
    return detect_corners(im, Rnms, thresh, window);
}

Image detect_and_draw_shi_tomasi(const Image& im, float sigma, float thresh, 
                                int window, int nms_window) {
    vector<Descriptor> d = shi_tomasi_detector(im, sigma, thresh, window, nms_window);
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