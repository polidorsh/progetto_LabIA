#include "image.h"
#include <vector>
#include <cmath>

// Implementa il filtro Laplaciano di Gaussiana
Image make_log_filter(float sigma) {
    // Il filtro LoG richiede una dimensione sufficiente per catturare la forma della funzione
    int w = ceil(sigma * 6);
    if (!(w % 2)) w++; // Assicura dimensione dispari
    
    Image filter(w, w, 1);
    int center = w/2;
    
    for(int y = 0; y < w; y++) {
        for(int x = 0; x < w; x++) {
            float dx = x - center;
            float dy = y - center;
            float r2 = dx*dx + dy*dy;
            
            // Formula del LoG: -1/(pi*sigma^4) * (1 - r^2/(2*sigma^2)) * e^(-r^2/(2*sigma^2))
            float exp_term = exp(-r2/(2*sigma*sigma));
            float log_term = 1 - (r2 / (2 * sigma * sigma));
            filter(x, y, 0) = -log_term * exp_term / (M_PI * pow(sigma, 4));

        }
    }
    
    return filter;
}

// Rileva keypoint usando il Laplaciano di Gaussiane
vector<Descriptor> log_keypoint_detector(const Image& im, float sigma, float thresh, int window, int nms_size) {
    // Converti l'immagine in scala di grigi se necessario
    Image gray;
    if(im.c == 1) gray = im;
    else gray = rgb_to_grayscale(im);
    
    // Applica il filtro LoG
    Image log_filter = make_log_filter(sigma);
    Image response = convolve_image(gray, log_filter, true);
    
    // Trova i massimi locali usando la funzione NMS esistente
    Image nms_response = nms_image(response, nms_size);
    
    // Rileva i keypoint usando la funzione detect_corners esistente
    return detect_corners(im, nms_response, thresh, window);
}

// Funzione wrapper per rilevare e disegnare i keypoint
Image detect_and_draw_log_keypoints(const Image& im, float sigma, float thresh, int window, int nms_size) {
    TIME(1);
    vector<Descriptor> keypoints = log_keypoint_detector(im, sigma, thresh, window, nms_size);
    printf("Numero di Keypoints: %ld\n", keypoints.size());
    return mark_corners(im, keypoints);
}


// Funzione per rilevare keypoint con LoG, accoppiare le descrizioni e disegnare i match
Image find_and_draw_log_matches(const Image& a, const Image& b,
                                float sigma, float thresh, int window, int nms_size) {
    TIME(1);
    // Rilevamento dei keypoint in ciascuna immagine usando il detector LoG
    vector<Descriptor> ad = log_keypoint_detector(a, sigma, thresh, window, nms_size);
    vector<Descriptor> bd = log_keypoint_detector(b, sigma, thresh, window, nms_size);
    
    // Associa le descrizioni (match)
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    
    // Disegna i keypoint sulle immagini originali e li visualizza
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

// Funzione per disegnare gli inlier a partire dalla stima di una trasformazione robusta (es. RANSAC)
Image draw_log_inliers(const Image& a, const Image& b,
                       float sigma, float thresh, int window, int nms_size,
                       float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    // Rileva keypoint con il metodo LoG
    vector<Descriptor> ad = log_keypoint_detector(a, sigma, thresh, window, nms_size);
    vector<Descriptor> bd = log_keypoint_detector(b, sigma, thresh, window, nms_size);
    
    // Trova i match tra le descrizioni
    vector<Match> m = match_descriptors(ad, bd);
    // Stima la matrice omografica con RANSAC
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    // Disegna gli inlier sulla base della trasformazione stimata
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

// Funzione per generare un'immagine panoramica combinando due immagini utilizzando il detector LoG
Image panorama_image_log(const Image& a, const Image& b,
                         float sigma, float thresh, int window, int nms_size,
                         float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    // Rileva i keypoint nelle due immagini con il detector LoG
    vector<Descriptor> ad = log_keypoint_detector(a, sigma, thresh, window, nms_size);
    vector<Descriptor> bd = log_keypoint_detector(b, sigma, thresh, window, nms_size);
    
    // Trova i match fra i descrittori
    vector<Match> m = match_descriptors(ad, bd);
    // Stima la trasformazione con RANSAC
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    // Combina le immagini usando la trasformazione stimata
    return combine_images(a, b, Hba, acoeff);
}
