#include <vector>
#include <cmath>
#include "image.h"

// Crea il descrittore per il keypoint in x,y
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

// Rileva punti chiave nello spazio delle scale
vector<Descriptor> detect_scale_space_keypoints(const Image& im, float base_sigma, float thresh, int window, int nms, int num_octaves = 4, int scales_per_octave = 3) {
    vector<Descriptor> keypoints;
    vector<Image> scale_space, responses;
    float scale_factor = pow(2.0f, 1.0f / scales_per_octave);
    Image current = im;
    float current_sigma = base_sigma;

    // Costruisce lo spazio delle scale
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
            current = bilinear_resize(current, current.w / 2, current.h / 2);
            current_sigma = base_sigma;
        }
    }

    // Selezione dei keypoint
    for (int octave = 0; octave < num_octaves; ++octave) {
        for (int scale = 0; scale < scales_per_octave; ++scale) {
            int idx = octave * scales_per_octave + scale;
            if (idx == 0 || idx >= responses.size() - 1) continue;

            Image current_response = responses[idx];
            Image prev_response = responses[idx - 1];
            Image next_response = responses[idx + 1];

            Image nms_result = nms_image(current_response, nms);

            for (int y = 0; y < nms_result.h; ++y) {
                for (int x = 0; x < nms_result.w; ++x) {
                    float current_val = nms_result(x, y, 0);
                    if (current_val <= thresh) continue;
                    if (current_val <= prev_response.clamped_pixel(x, y, 0) || current_val <= next_response.clamped_pixel(x, y, 0)) continue;
                    
                    Descriptor d = describe_index(scale_space[idx], x, y, window);
                    float scale_multiplier = pow(2.0f, octave);
                    d.p.x = x * scale_multiplier;
                    d.p.y = y * scale_multiplier;
                    keypoints.push_back(d);
                }
            }
        }
    }
    return keypoints;
}

// Rileva e disegna i punti chiave
Image detect_and_draw_scale_space_keypoints(const Image& im, float base_sigma, float thresh, int window, int nms, int num_octaves = 4, int scales_per_octave = 3) {
    TIME(1);
    std::vector<Descriptor> keypoints = detect_scale_space_keypoints(im, base_sigma, thresh, window, nms, num_octaves, scales_per_octave);
    printf("Numero di Descrittori: %zu\n", keypoints.size());
    return mark_corners(im, keypoints);
}

// Trova e disegna i match tra due immagini
Image find_and_draw_scale_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms_window, int num_octaves = 4, int scales_per_octave = 3) {
    TIME(1);
    auto ad = detect_scale_space_keypoints(a, sigma, thresh, window, nms_window, num_octaves, scales_per_octave);
    auto bd = detect_scale_space_keypoints(b, sigma, thresh, window, nms_window, num_octaves, scales_per_octave);
    auto m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    return draw_matches(mark_corners(a, ad), mark_corners(b, bd), m, {});
}

// Disegna gli inliers trovati con RANSAC
Image draw_scale_inliers(const Image& a, const Image& b, float sigma, float thresh, int window, int nms_window, float inlier_thresh, int iters, int cutoff, int num_octaves=4, int scales_per_octave=3) {
    auto ad = detect_scale_space_keypoints(a, sigma, thresh, window, nms_window, num_octaves, scales_per_octave);
    auto bd = detect_scale_space_keypoints(b, sigma, thresh, window, nms_window, num_octaves, scales_per_octave);
    auto m = match_descriptors(ad, bd);
    auto Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

// Crea un panorama usando i keypoint dello spazio delle scale
Image panorama_image_scale(const Image& a, const Image& b, float sigma, float thresh, int window, int nms_window, float inlier_thresh, int iters, int cutoff, float acoeff, int num_octaves=4, int scales_per_octave=3) {
    auto ad = detect_scale_space_keypoints(a, sigma, thresh, window, nms_window, num_octaves, scales_per_octave);
    auto bd = detect_scale_space_keypoints(b, sigma, thresh, window, nms_window, num_octaves, scales_per_octave);
    auto m = match_descriptors(ad, bd);
    auto Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}
