#include <vector>
#include <cmath>
#include "image.h"

Descriptor describe_index(const Image& im, int x, int y, int w) {
    Descriptor d;
    d.p = {(double)x, (double)y};
    d.data.reserve(w * w * im.c);

    for (int c = 0; c < im.c; c++) {
        float cval = im.clamped_pixel(x, y, c);
        for (int dx = -w / 2; dx <= w / 2; dx++) {
            for (int dy = -w / 2; dy <= w / 2; dy++) {
                d.data.push_back(im.clamped_pixel(x + dx, y + dy, c) - cval);
            }
        }
    }
    return d;
}

std::vector<Descriptor> detect_scale_space_keypoints(const Image& im,
                                                     float base_sigma,
                                                     float thresh,
                                                     int window,
                                                     int nms,
                                                     int num_octaves = 4,
                                                     int scales_per_octave = 3) {
    std::vector<Descriptor> keypoints;
    std::vector<Image> scale_space;
    std::vector<Image> responses;

    float scale_factor = pow(2.0f, 1.0f / scales_per_octave);

    Image current = im;
    float current_sigma = base_sigma;

    for (int octave = 0; octave < num_octaves; ++octave) {
        for (int scale = 0; scale < scales_per_octave; ++scale) {
            Image smoothed = smooth_image(current, current_sigma);
            scale_space.push_back(smoothed);

            Image S = structure_matrix(smoothed, current_sigma);
            Image R = cornerness_response(S, 1);
            responses.push_back(R);

            current_sigma *= scale_factor;
        }

        if (octave < num_octaves - 1) {
            int new_w = current.w / 2;
            int new_h = current.h / 2;
            current = bilinear_resize(current, new_w, new_h);
            current_sigma = base_sigma;
        }
    }

    for (int octave = 0; octave < num_octaves; ++octave) {
        for (int scale = 0; scale < scales_per_octave; ++scale) {
            int idx = octave * scales_per_octave + scale;
            if (idx == 0 || idx >= responses.size() - 1) continue;

            Image current_response = responses[idx];
            Image prev_response = responses[idx - 1];
            Image next_response = responses[idx + 1];

            float scale_multiplier = pow(2.0f, octave);

            Image nms_result = nms_image(current_response, nms);

            for (int y = 0; y < nms_result.h; ++y) {
                for (int x = 0; x < nms_result.w; ++x) {
                    float current_val = nms_result(x, y, 0);

                    if (current_val <= thresh) continue;

                    bool is_scale_maximum = true;

                    float prev_val = prev_response.clamped_pixel(x, y, 0);
                    if (current_val <= prev_val) {
                        is_scale_maximum = false;
                    }

                    float next_val = next_response.clamped_pixel(x, y, 0);
                    if (current_val <= next_val) {
                        is_scale_maximum = false;
                    }

                    if (is_scale_maximum) {
                        Descriptor d = describe_index(scale_space[idx], x, y, window);
                        d.p.x = x * scale_multiplier;
                        d.p.y = y * scale_multiplier;
                        keypoints.push_back(d);
                    }
                }
            }
        }
    }

    return keypoints;
}

Image detect_and_draw_scale_space_keypoints(const Image& im, float base_sigma, float thresh, int window, int nms, int num_octaves = 4, int scales_per_octave = 3) {
    TIME(1);
    std::vector<Descriptor> keypoints = detect_scale_space_keypoints(im, base_sigma, thresh, window, nms, num_octaves, scales_per_octave);
    printf("Numero di Descrittori: %zu\n", keypoints.size());
    return mark_corners(im, keypoints);
}


Image find_and_draw_scale_matches(const Image& a, const Image& b,
                                  float sigma, float thresh, int window, int nms_window,
                                  int num_octaves = 4, int scales_per_octave = 3) {
    TIME(1);
    std::vector<Descriptor> ad = detect_scale_space_keypoints(a, sigma, thresh, window, nms_window,
                                                              num_octaves, scales_per_octave);
    std::vector<Descriptor> bd = detect_scale_space_keypoints(b, sigma, thresh, window, nms_window,
                                                              num_octaves, scales_per_octave);

    std::vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());

    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);

    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image draw_scale_inliers(const Image& a, const Image& b,
                        float sigma, float thresh, int window, int nms_window,
                        float inlier_thresh, int iters, int cutoff,
                        int num_octaves=4, int scales_per_octave=3) {
    vector<Descriptor> ad = detect_scale_space_keypoints(a, sigma, thresh, window, nms_window, 
                                                         num_octaves, scales_per_octave);
    vector<Descriptor> bd = detect_scale_space_keypoints(b, sigma, thresh, window, nms_window, 
                                                         num_octaves, scales_per_octave);
    
    vector<Match> m = match_descriptors(ad, bd);
    
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_scale(const Image& a, const Image& b,
                          float sigma, float thresh, int window, int nms_window,
                          float inlier_thresh, int iters, int cutoff, float acoeff,
                          int num_octaves=4, int scales_per_octave=3) {
    vector<Descriptor> ad = detect_scale_space_keypoints(a, sigma, thresh, window, nms_window, 
                                                         num_octaves, scales_per_octave);
    vector<Descriptor> bd = detect_scale_space_keypoints(b, sigma, thresh, window, nms_window, 
                                                         num_octaves, scales_per_octave);
    
    vector<Match> m = match_descriptors(ad, bd);
    
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    
    return combine_images(a, b, Hba, acoeff);
}
