#include <vector>
#include <cmath>
#include <limits>
#include <cstdio>
#include "image.h" 

using namespace std;

// Calcola la misura di interesse di Förstner
float forstner_interest_measure(const Image& S) {
    float det = S(0, 0, 0) * S(1, 1, 0) - S(0, 1, 0) * S(1, 0, 0);
    float tr = S(0, 0, 0) + S(1, 1, 0);
    return det / (tr + 1e-6); // Evita divisioni per zero
}

// Calcola la risposta di Harris
float harris_response(const Image& S) {
    float det = S(0, 0, 0) * S(1, 1, 0) - S(0, 1, 0) * S(1, 0, 0);
    float tr = S(0, 0, 0) + S(1, 1, 0);
    const float k = 0.04; 
    return det - k * tr * tr;
}

// Rilevatore Förstner-Harris-Hessian
vector<Descriptor> fhh_detector(const Image& im, float sigma, float thresh, int window, int nms) {
    Image S = structure_matrix(im, sigma);
    Image R(im.w, im.h, 1);
    float max_forstner = -numeric_limits<float>::max();
    float max_harris = -numeric_limits<float>::max();
    
    // Calcola i valori massimi di Förstner e Harris
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            Image localS(2, 2, 1);
            localS(0, 0, 0) = S(x, y, 0);
            localS(1, 1, 0) = S(x, y, 1);
            localS(0, 1, 0) = localS(1, 0, 0) = S(x, y, 2);
            
            float forstner = forstner_interest_measure(localS);
            float harris = harris_response(localS);
            
            max_forstner = max(max_forstner, abs(forstner));
            max_harris = max(max_harris, abs(harris));
        }
    }
    
    // Calcola la risposta combinata normalizzata
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            Image localS(2, 2, 1);
            localS(0, 0, 0) = S(x, y, 0);
            localS(1, 1, 0) = S(x, y, 1);
            localS(0, 1, 0) = localS(1, 0, 0) = S(x, y, 2);
            
            float forstner = forstner_interest_measure(localS);
            float harris = harris_response(localS);
            
            R(x, y, 0) = (forstner / max_forstner + harris / max_harris) / 2.0;
        }
    }
    
    // Calcolo della soglia adattiva basata su media e deviazione standard
    float mean_response = 0, std_response = 0;
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            mean_response += R(x, y, 0);
        }
    }
    mean_response /= (im.w * im.h);
    
    for (int y = 0; y < im.h; y++) {
        for (int x = 0; x < im.w; x++) {
            std_response += pow(R(x, y, 0) - mean_response, 2);
        }
    }
    std_response = sqrt(std_response / (im.w * im.h));
    float adaptive_thresh = mean_response + thresh * std_response;
    
    // Soppressione dei non-massimi
    Image Rnms = nms_image(R, nms);
    return detect_corners(im, Rnms, adaptive_thresh, window);
}

// Funzione wrapper per rilevare e disegnare i corner
Image detect_and_draw_fhh_corners(const Image& im, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> d = fhh_detector(im, sigma, thresh, window, nms);
    printf("Numero di Descrittori: %ld\n", d.size());
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