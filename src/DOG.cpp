#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include "image.h"

// Funzione per descrivere un punto nell'immagine con una finestra di dimensione w
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

// Calcola la risposta della Differenza di Gaussiana (DoG)
Image compute_dog_response(const Image& im, float sigma1, float sigma2) {
    assert(sigma2 > sigma1 && "sigma2 deve essere maggiore di sigma1");
    
    Image gray = (im.c == 3) ? rgb_to_grayscale(im) : im;
    
    // Sfocature gaussiane
    Image blur1 = smooth_image(gray, sigma1);
    Image blur2 = smooth_image(gray, sigma2);
    
    // Calcolo della differenza tra le due sfocature
    Image dog(gray.w, gray.h, 1);
    for (int y = 0; y < gray.h; ++y) {
        for (int x = 0; x < gray.w; ++x) {
            dog(x, y, 0) = blur1(x, y, 0) - blur2(x, y, 0);
        }
    }
    return dog;
}

// Rileva punti chiave utilizzando la DoG e restituisce i descrittori
vector<Descriptor> dog_detector(const Image& im, float sigma1, float sigma2, float thresh, int window_size, int nms_window) {
    Image response = compute_dog_response(im, sigma1, sigma2);
    Image nms = nms_image(response, nms_window);
    vector<Descriptor> descriptors;
    
    for (int y = 0; y < im.h; ++y) {
        for (int x = 0; x < im.w; ++x) {
            if (nms(x, y, 0) > thresh) {
                descriptors.push_back(describe_index(im, x, y, window_size));
            }
        }
    }
    return descriptors;
}

// Rileva e disegna i punti chiave DoG
Image detect_and_draw_dog(const Image& im, float sigma1, float sigma2, float thresh, int window_size, int nms_window) {
    TIME(1);
    vector<Descriptor> descriptors = dog_detector(im, sigma1, sigma2, thresh, window_size, nms_window);
    printf("Numero di Descrittori: %ld\n", descriptors.size());
    return mark_corners(im, descriptors);
}

// Trova e disegna corrispondenze tra immagini usando DoG
Image find_and_draw_dog_matches(const Image& a, const Image& b, float sigma1, float sigma2, float thresh, int window_size, int nms_window) {
    TIME(1);
    vector<Descriptor> ad = dog_detector(a, sigma1, sigma2, thresh, window_size, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma1, sigma2, thresh, window_size, nms_window);
    vector<Match> matches = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", matches.size());
    return draw_matches(mark_corners(a, ad), mark_corners(b, bd), matches, {});
}

// Disegna corrispondenze tra immagini evidenziando inlier e outlier
Image draw_dog_inliers(const Image& a, const Image& b, float sigma1, float sigma2, float thresh, int window_size, int nms_window, float inlier_thresh, int ransac_iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = dog_detector(a, sigma1, sigma2, thresh, window_size, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma1, sigma2, thresh, window_size, nms_window);
    vector<Match> matches = match_descriptors(ad, bd);
    Matrix H = RANSAC(matches, inlier_thresh, ransac_iters, cutoff);
    return draw_inliers(a, b, H, matches, inlier_thresh);
}

// Crea un panorama utilizzando DoG per la rilevazione dei punti chiave
Image panorama_image_dog(const Image& a, const Image& b, float sigma1, float sigma2, float thresh, int window_size, int nms_window, float inlier_thresh, int ransac_iters, int cutoff, float blend_coeff) {
    TIME(1);
    vector<Descriptor> ad = dog_detector(a, sigma1, sigma2, thresh, window_size, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma1, sigma2, thresh, window_size, nms_window);
    vector<Match> matches = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(matches, inlier_thresh, ransac_iters, cutoff);
    return combine_images(a, b, Hba, blend_coeff);
}
