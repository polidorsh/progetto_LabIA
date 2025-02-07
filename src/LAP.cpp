#include "image.h"
#include <vector>
#include <cmath>

Image make_log_filter(float sigma) {
    int w = ceil(sigma * 6);
    if (!(w % 2)) w++;
    Image filter(w, w, 1);
    int center = w / 2;
    float sigma2 = sigma * sigma;
    float norm_factor = -1.0 / (M_PI * sigma2 * sigma2);
    for (int y = 0; y < w; y++) {
        for (int x = 0; x < w; x++) {
            float dx = x - center;
            float dy = y - center;
            float r2 = dx * dx + dy * dy;
            float r2_term = r2 / (2 * sigma2);
            filter(x, y, 0) = norm_factor * (1 - r2_term) * exp(-r2_term) * (sigma2); // Added sigma^2 scaling
        }
    }
    return filter;
}

Image multi_scale_log(const Image& im, float initial_sigma, int num_scales, float scale_factor) {
    Image response(im.w, im.h, 1);
    Image gray = (im.c == 3) ? rgb_to_grayscale(im) : im;
    float sigma = initial_sigma;
    for (int s = 0; s < num_scales; s++) {
        Image filter = make_log_filter(sigma);
        Image scale_response = convolve_image(gray, filter, true);
        for (int y = 0; y < im.h; y++) {
            for (int x = 0; x < im.w; x++) {
                float val = fabs(scale_response(x, y, 0));
                response(x, y, 0) = std::max(val, response(x, y, 0));
            }
        }
        sigma *= scale_factor;
    }
    return response;
}

vector<Descriptor> log_detector(const Image& im, float sigma, int num_scales,
    float scale_factor, float thresh, int nms_w, int window) {
    Image response = multi_scale_log(im, sigma, num_scales, scale_factor);
    Image nms = nms_image(response, nms_w);
    return detect_corners(im, nms, thresh, window);
}

Image detect_and_draw_log_keypoints(const Image& im, float sigma, int num_scales,
    float scale_factor, float thresh, int nms, int window) {
    vector<Descriptor> keypoints = log_detector(im, sigma, num_scales, scale_factor, thresh, nms, window);
    printf("Numero di Descrittori: %ld\n", keypoints.size());
    return mark_corners(im, keypoints);
}


Image find_and_draw_laplacian_matches(const Image& a, const Image& b, float sigma, int num_scales, 
                                  float scale_factor, float thresh, int nms, int window) {
    TIME(1);
    vector<Descriptor> ad = log_detector(a, sigma, num_scales, scale_factor, 
                                        thresh, nms, window);
    vector<Descriptor> bd = log_detector(b, sigma, num_scales, scale_factor, 
                                        thresh, nms, window);
    
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image draw_laplacian_inliers(const Image& a, const Image& b, float sigma, int num_scales, 
                                  float scale_factor, float thresh, int nms, int window, 
                                  float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = log_detector(a, sigma, num_scales, scale_factor, 
                                        thresh, nms, window);
    vector<Descriptor> bd = log_detector(b, sigma, num_scales, scale_factor, 
                                        thresh, nms, window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_laplacian(const Image& a, const Image& b, float sigma, int num_scales, 
                                  float scale_factor, float thresh, int nms, int window, 
                                  float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = log_detector(a, sigma, num_scales, scale_factor, 
                                        thresh, nms, window);
    vector<Descriptor> bd = log_detector(b, sigma, num_scales, scale_factor, 
                                        thresh, nms, window);

    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}
