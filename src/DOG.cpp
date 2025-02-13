#include "image.h"
#include <vector>
#include <cmath>

vector<Descriptor> dog_detector(const Image& im, float sigma, float thresh, int window, int nms_window) {
    Image working_image = (im.c == 1) ? im : rgb_to_grayscale(im);
    
    float k = 1.6f;
    
    Image gaussian1 = smooth_image(working_image, sigma);
    Image gaussian2 = smooth_image(working_image, sigma * k);
    
    Image dog(gaussian1.w, gaussian1.h, 1);
    for (int y = 0; y < dog.h; y++) {
        for (int x = 0; x < dog.w; x++) {
            dog(x, y, 0) = gaussian2(x, y, 0) - gaussian1(x, y, 0);
        }
    }
    
    Image nms_result = nms_image(dog, nms_window);
    
    return detect_corners(working_image, nms_result, thresh, window);
}

Image detect_and_draw_dog(const Image& im, float sigma, float thresh, int window, int nms_window) {
    TIME(1);  
    vector<Descriptor> keypoints = dog_detector(im, sigma, thresh, window, nms_window);
    printf("Numero di Descrittori: %zu\n", keypoints.size());  
    return mark_corners(im, keypoints);
}

Image find_and_draw_dog_matches(const Image& a, const Image& b,
                                float sigma, float thresh, int window, int nms_window) {
    TIME(1);
    vector<Descriptor> ad = dog_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %zu\n", m.size());
    
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image draw_dog_inliers(const Image& a, const Image& b,
                       float sigma, float thresh, int window, int nms_window,
                       float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = dog_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_dog(const Image& a, const Image& b,
                         float sigma, float thresh, int window, int nms_window,
                         float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = dog_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}