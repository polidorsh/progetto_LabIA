#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include "image.h"

using namespace std;

// Calcola la matrice Hessiana
Image compute_hessian(const Image& im, float sigma) {
    Image smoothed = smooth_image(im, sigma); // Sfocatura

    Image fx = make_gx_filter(); // Filtro sobel x
    Image fy = make_gy_filter(); // Filtro sobel y

    Image Ix = convolve_image(smoothed, fx, true); // Derivata prima rispetto a x
    Image Ixx = convolve_image(Ix, fx, true); // Derivata seconda rispetto a x

    Image Iy = convolve_image(smoothed, fy, true); // Derivata prima rispetto a y
    Image Iyy = convolve_image(Iy, fy, true); // Derivata seconda rispetto a y

    Image Ixy = convolve_image(Ix, fy, true); // Derivata mista

    Image H(im.w, im.h, 3);
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            H(x,y,0) = Ixx(x,y,0);  // Componente Ixx
            H(x,y,1) = Iyy(x,y,0);  // Componente Iyy
            H(x,y,2) = Ixy(x,y,0);  // Componente Ixy
        }
    }
    return H;
}

// Rileva punti caratteristici utilizzando diversi metodi
vector<Descriptor> fhh_detector(const Image& im, int method, float sigma, 
                                float thresh, int window, int nms_window) {
    Image R(im.w, im.h, 1);

    if (method == 0) {  
        // Metodo Hessian
        Image H = compute_hessian(im, sigma);
        for(int y = 0; y < H.h; y++) {
            for(int x = 0; x < H.w; x++) {
                float Ixx = H(x,y,0);
                float Iyy = H(x,y,1);
                float Ixy = H(x,y,2);
                float det = Ixx * Iyy - Ixy * Ixy; // Determinante dell'Hessiana
                R(x,y,0) = det;
            }
        }
    } else {
        // Metodo Förstner, Harris o Ibrido
        Image S = structure_matrix(im, sigma);
        for(int y = 0; y < S.h; y++) {
            for(int x = 0; x < S.w; x++) {
                float a = S(x,y,0);  // Ix^2
                float b = S(x,y,1);  // Iy^2
                float c = S(x,y,2);  // IxIy
                float trace = a + b;
                float det = a * b - c * c;
                float forstner_weight, harris_weight;

                switch(method) {
                    case 1:  // Förstner
                        R(x,y,0) = det / (trace + 1e-8f);
                        break;
                    case 2:  // Harris
                        R(x,y,0) = det - 0.04f * powf(trace, 2);
                        break;
                    case 3:  // Ibrido
                        forstner_weight = det / (trace + 1e-8f);
                        harris_weight = det - 0.04f * powf(trace, 2);
                        R(x,y,0) = 0.5f * (forstner_weight + harris_weight);
                        break;
                    default:
                        fprintf(stderr, "Errore: metodo non valido. Metodi: 0, 1, 2, 3\n");
                        exit(EXIT_FAILURE);
                }
            }
        }
    }

    Image Rnms = nms_image(R, nms_window); 
    return detect_corners(im, Rnms, thresh, window); 
}

// Rileva e disegna i punti caratteristici sull'immagine
Image detect_and_draw_fhh(const Image& im, int method, float sigma, 
                          float thresh, int window, int nms_window) {
    TIME(1);
    vector<Descriptor> corners = fhh_detector(im, method, sigma, thresh, window, nms_window);
    printf("Numero di Descrittori: %ld\n", corners.size());
    return mark_corners(im, corners);
}

// Trova e disegna le corrispondenze tra due immagini
Image find_and_draw_fhh_matches(const Image& a, const Image& b,
                                int method, float sigma, float thresh, 
                                int window, int nms_window) {
    TIME(1);
    vector<Descriptor> ad = fhh_detector(a, method, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = fhh_detector(b, method, sigma, thresh, window, nms_window);
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

// Disegna le corrispondenze inlier tra due immagini
Image draw_fhh_inliers(const Image& a, const Image& b,
                       int method, float sigma, float thresh, 
                       int window, int nms_window,
                       float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = fhh_detector(a, method, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = fhh_detector(b, method, sigma, thresh, window, nms_window);
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_fhh(const Image& a, const Image& b,
                                     int method, float sigma, float thresh, 
                                     int window, int nms_window,
                                     float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = fhh_detector(a, method, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = fhh_detector(b, method, sigma, thresh, window, nms_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}
