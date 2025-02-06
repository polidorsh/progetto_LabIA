#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>

#include "image.h"

using namespace std;

Descriptor describe_index(const Image& im, int x, int y, int w) {
    Descriptor d;
    d.p = {(double)x, (double)y};
    d.data.reserve(w * w * im.c);
    
    for (int c = 0; c < im.c; c++) {
        float cval = im.clamped_pixel(x, y, c);
        for (int dx = -w/2; dx <= w/2; dx++) {
            for (int dy = -w/2; dy <= w/2; dy++) {
                d.data.push_back(im.clamped_pixel(x + dx, y + dy, c) - cval);
            }
        }
    }
    return d;
}

vector<vector<Image>> create_dog_pyramids(const Image& im, float sigma, int octaves) {
    vector<vector<Image>> dog_pyramids;
    
    Image current = im;
    for (int o = 0; o < octaves; o++) {
        vector<Image> gaussian_levels;
        vector<Image> dog_levels;
        
        for (int s = 0; s < 5; s++) {
            float current_sigma = sigma * pow(2.0f, s / 2.0f);
            gaussian_levels.push_back(smooth_image(current, current_sigma));
        }
        
        for (int s = 1; s < gaussian_levels.size(); s++) {
            Image dog = gaussian_levels[s] - gaussian_levels[s - 1];
            dog_levels.push_back(dog);
        }
        
        dog_pyramids.push_back(dog_levels);
        
        current = bilinear_resize(gaussian_levels[2], gaussian_levels[2].w / 2, gaussian_levels[2].h / 2);
    }
    
    return dog_pyramids;
}

vector<Descriptor> detect_dog_keypoints(const vector<vector<Image>>& dog_pyramids, float thresh, int window, int nms) {
    vector<Descriptor> keypoints;
    
    for (const auto& dog_levels : dog_pyramids) {
        int levels = dog_levels.size();
        Image keypoint_response(dog_levels[0].w, dog_levels[0].h, 1);
        
        for (int level = 1; level < levels - 1; level++) {
            const Image& current = dog_levels[level];
            
            for (int y = 2; y < current.h - 2; y++) {
                for (int x = 2; x < current.w - 2; x++) {
                    float pixel_val = current.clamped_pixel(x, y, 0);
                    bool is_max = true, is_min = true;
                    
                    for (int dl = -1; dl <= 1; dl++) {
                        int neighbor_level = level + dl;
                        if (neighbor_level < 0 || neighbor_level >= levels) continue;
                        const Image& neighbor = dog_levels[neighbor_level];
                        
                        for (int dx = -1; dx <= 1; dx++) {
                            for (int dy = -1; dy <= 1; dy++) {
                                if (dl == 0 && dx == 0 && dy == 0) continue;
                                float neighbor_val = neighbor.clamped_pixel(x + dx, y + dy, 0);
                                if (pixel_val <= neighbor_val) is_max = false;
                                if (pixel_val >= neighbor_val) is_min = false;
                            }
                        }
                    }
                    
                    if ((is_max || is_min) && fabs(pixel_val) > thresh) {
                        keypoint_response(x, y, 0) = fabs(pixel_val);
                    }
                }
            }
            
            Image nms_response = nms_image(keypoint_response, nms);
            
            for (int y = 2; y < current.h - 2; y++) {
                for (int x = 2; x < current.w - 2; x++) {
                    if (nms_response(x, y, 0) > 0) {
                        keypoints.push_back(describe_index(current, x, y, window));
                    }
                }
            }
        }
    }
    
    return keypoints;
}

vector<Descriptor> dog_corner_detector(const Image& im, float sigma, float thresh, int window, int nms) {
    // Costruiamo le piramidi DoG separate per ogni ottava (in questo esempio utilizziamo 4 ottave)
    vector<vector<Image>> dog_pyramids = create_dog_pyramids(im, sigma, 4);
    return detect_dog_keypoints(dog_pyramids, thresh, window, nms);
}

Image detect_and_draw_dog_corners(const Image& im, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> d = dog_corner_detector(im, sigma, thresh, window, nms);
    printf("Numero di Descrittori: %ld\n", d.size());
    return mark_corners(im, d);
}

Image find_and_draw_dog_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> ad = dog_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = dog_corner_detector(b, sigma, thresh, window, nms);
    vector<Match> matches = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", matches.size());
    return draw_matches(mark_corners(a, ad), mark_corners(b, bd), matches, {});
}

Image draw_dog_inliers(const Image& a, const Image& b, float sigma, float thresh, int window, int nms, float inlier_thresh, int ransac_iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = dog_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = dog_corner_detector(b, sigma, thresh, window, nms);
    vector<Match> matches = match_descriptors(ad, bd);
    Matrix H = RANSAC(matches, inlier_thresh, ransac_iters, cutoff);
    return draw_inliers(a, b, H, matches, inlier_thresh);
}

Image panorama_image_dog(const Image& a, const Image& b, float sigma, float thresh, int window, int nms, float inlier_thresh, int ransac_iters, int cutoff, float blend_coeff) {
    TIME(1);
    vector<Descriptor> ad = dog_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = dog_corner_detector(b, sigma, thresh, window, nms);
    vector<Match> matches = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(matches, inlier_thresh, ransac_iters, cutoff);
    return combine_images(a, b, Hba, blend_coeff);
}
