#include "image.h"
#include <vector>
#include <cmath>

// Crea il filtro Laplacian of Gaussian (LoG) per il valore di sigma dato
Image make_log_filter(float sigma) {
    
    // Calcola la dimensione della finestra: 6*sigma, arrotondato al prossimo numero dispari
    int w = ceil(sigma * 6);
    if (!(w % 2)) w++; 

    Image filter(w, w, 1);
    int center = w / 2;
    
    // Calcola il valore del filtro per ogni pixel della finestra
    for (int y = 0; y < w; y++) {
        for (int x = 0; x < w; x++) {
            float dx = x - center;
            float dy = y - center;
            float r2 = dx * dx + dy * dy;
            
            float exp_term = exp(-r2 / (2 * sigma * sigma));
            float log_term = 1 - (r2 / (2 * sigma * sigma));
            // Calcola e normalizza il valore del filtro LoG
            filter(x, y, 0) = -log_term * exp_term / (M_PI * pow(sigma, 4));
            //-(1-(r2/(2*sigma*sigma)))*(e^(-r2/(2*sigma*sigma))) / PI*sigma^4
            //Laplaciano: Somma delle derivate seconde parziali rispetto a x e y
        }
    }
    
    return filter;
}

// Rileva i keypoints usando il filtro LoG
vector<Descriptor> log_keypoint_detector(const Image& im, float sigma, float thresh, int window, int nms_size) {
    
    // Converte l'immagine in scala di grigi se necessario
    Image gray = (im.c == 1) ? im : rgb_to_grayscale(im);
    
    // Crea il filtro LoG e lo applica tramite convoluzione
    Image log_filter = make_log_filter(sigma);
    Image response = convolve_image(gray, log_filter, true);
    
    // Applica la Non-Maximum Suppression 
    Image nms_response = nms_image(response, nms_size);
    
    // Rileva i corner 
    return detect_corners(gray, nms_response, thresh, window);
}

// Rileva e disegna i keypoints LoG sull'immagine originale
Image detect_and_draw_log_keypoints(const Image& im, float sigma, float thresh, int window, int nms_size) {
    TIME(1);  
    vector<Descriptor> keypoints = log_keypoint_detector(im, sigma, thresh, window, nms_size);
    printf("Numero di Descrittori: %zu\n", keypoints.size());
    return mark_corners(im, keypoints); 
}

// Trova e disegna le corrispondenze (matches) tra due immagini 
Image find_and_draw_log_matches(const Image& a, const Image& b,
                                float sigma, float thresh, int window, int nms_size) {
    TIME(1); 
    vector<Descriptor> ad = log_keypoint_detector(a, sigma, thresh, window, nms_size);
    vector<Descriptor> bd = log_keypoint_detector(b, sigma, thresh, window, nms_size);
    
    // Trova le corrispondenze tra i descrittori delle due immagini
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    
    // Disegna i corner e le linee che collegano i match trovati
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

// Disegna gli inlier delle corrispondenze dopo aver applicato RANSAC
Image draw_log_inliers(const Image& a, const Image& b,
                       float sigma, float thresh, int window, int nms_size,
                       float inlier_thresh, int iters, int cutoff) {
    TIME(1);  
    vector<Descriptor> ad = log_keypoint_detector(a, sigma, thresh, window, nms_size);
    vector<Descriptor> bd = log_keypoint_detector(b, sigma, thresh, window, nms_size);
    
    // Trova le corrispondenze e calcola l'omografia tramite RANSAC
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

// Crea un'immagine combinando le due in input
Image panorama_image_log(const Image& a, const Image& b,
                         float sigma, float thresh, int window, int nms_size,
                         float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);  
    vector<Descriptor> ad = log_keypoint_detector(a, sigma, thresh, window, nms_size);
    vector<Descriptor> bd = log_keypoint_detector(b, sigma, thresh, window, nms_size);
    
    // Trova le corrispondenze e stima l'omografia con RANSAC
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);  
}
