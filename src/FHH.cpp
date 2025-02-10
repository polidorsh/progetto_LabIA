#include <vector>
#include <cmath>
#include <limits>
#include <cstdio>
#include "image.h" 

using namespace std;

float forstner_interest_measure(float Sxx, float Syy, float Sxy) {
    float det = Sxx * Syy - Sxy * Sxy;
    float tr = Sxx + Syy;
    return det / (tr + 1e-6f); 
}

float harris_response(float Sxx, float Syy, float Sxy) {
    float det = Sxx * Syy - Sxy * Sxy;
    float tr = Sxx + Syy;
    const float k = 0.04f;
    return det - k * tr * tr;
}

vector<Descriptor> fhh_detector(const Image& im, float sigma, float thresh, int window, int nms) {
    Image gray = (im.c == 1) ? im : rgb_to_grayscale(im);
    
    Image S = structure_matrix(gray, sigma); 
    Image R(im.w, im.h, 1);
    float max_forstner = -numeric_limits<float>::max();
    float max_harris   = -numeric_limits<float>::max();

    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float Sxx = S(x, y, 0);
            float Syy = S(x, y, 1);
            float Sxy = S(x, y, 2);
            float forstner = forstner_interest_measure(Sxx, Syy, Sxy);
            float harris   = harris_response(Sxx, Syy, Sxy);
            max_forstner = max(max_forstner, fabs(forstner));
            max_harris   = max(max_harris, fabs(harris));
        }
    }

    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            float Sxx = S(x, y, 0);
            float Syy = S(x, y, 1);
            float Sxy = S(x, y, 2);
            float forstner = forstner_interest_measure(Sxx, Syy, Sxy);
            float harris   = harris_response(Sxx, Syy, Sxy);
            R(x, y, 0) = ( (forstner / max_forstner) + (harris / max_harris) ) / 2.0f;
        }
    }

    float final_thresh;
    if (fabs(thresh - 1.0f) < 1e-6) {
        float mean_response = 0.0f, std_response = 0.0f;
        int total_pixels = im.w * im.h;
        for (int y = 0; y < im.h; y++) {
            for (int x = 0; x < im.w; x++) {
                mean_response += R(x, y, 0);
            }
        }
        mean_response /= total_pixels;
        
        for (int y = 0; y < im.h; y++) {
            for (int x = 0; x < im.w; x++) {
                float diff = R(x, y, 0) - mean_response;
                std_response += diff * diff;
            }
        }
        std_response = sqrt(std_response / total_pixels);
        final_thresh = mean_response + std_response;
    } else {
        final_thresh = thresh;
    }

    Image Rnms = nms_image(R, nms);
    return detect_corners(im, Rnms, final_thresh, window);
}

Image detect_and_draw_fhh_corners(const Image& im, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> d = fhh_detector(im, sigma, thresh, window, nms);
    printf("Numero di Descrittori: %zu\n", d.size());
    return mark_corners(im, d);
}


Image find_and_draw_fhh_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> ad = fhh_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = fhh_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image find_and_draw_fhh_inliers(const Image& a, const Image& b, float sigma, float thresh, int window,
int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = fhh_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = fhh_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_fhh(const Image& a, const Image& b, float sigma, float thresh, int window,
int nms, float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = fhh_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = fhh_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}