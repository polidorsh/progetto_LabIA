#include "image.h"
#include <vector>
#include <cmath>

// Funzione per rilevare keypoints usando il metodo DoG (Difference of Gaussians)
vector<Descriptor> dog_detector(const Image& im, float sigma, float thresh, int window, int nms_window) {
    
    // Converte in scala di grigi se l'immagine non lo è già
    Image working_image = (im.c == 1) ? im : rgb_to_grayscale(im);
    
    float k = 1.6f; // Fattore di scala per il secondo filtro gaussiano
    
    // Applica il filtro gaussiano con sigma e sigma * k
    Image gaussian1 = smooth_image(working_image, sigma);
    Image gaussian2 = smooth_image(working_image, sigma * k);
    
    // Calcola la DoG: differenza tra le due immagini gaussiane
    Image dog(gaussian1.w, gaussian1.h, 1);
    for (int y = 0; y < dog.h; y++) {
        for (int x = 0; x < dog.w; x++) {
            dog(x, y, 0) = gaussian2(x, y, 0) - gaussian1(x, y, 0);
        }
    }
    
    // Applica la non-maximum suppression 
    Image nms_result = nms_image(dog, nms_window);
    
    // Rileva i corner 
    return detect_corners(working_image, nms_result, thresh, window);
}

// Funzione per rilevare e disegnare i keypoints DoG sull'immagine
Image detect_and_draw_dog(const Image& im, float sigma, float thresh, int window, int nms_window) {
    TIME(1);  
    vector<Descriptor> keypoints = dog_detector(im, sigma, thresh, window, nms_window);
    printf("Numero di Descrittori: %zu\n", keypoints.size());
    return mark_corners(im, keypoints); 
}

// Funzione per trovare e disegnare le corrispondenze tra due immagini
Image find_and_draw_dog_matches(const Image& a, const Image& b,
                                float sigma, float thresh, int window, int nms_window) {
    TIME(1);  
    vector<Descriptor> ad = dog_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma, thresh, window, nms_window);
    
    // Trova le corrispondenze tra i descrittori delle due immagini
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %zu\n", m.size());
    
    // Disegna i corner e le linee di corrispondenza
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

// Funzione per disegnare gli inlier dopo RANSAC sulle corrispondenze tra immagini
Image draw_dog_inliers(const Image& a, const Image& b,
                       float sigma, float thresh, int window, int nms_window,
                       float inlier_thresh, int iters, int cutoff) {
    TIME(1);  
    vector<Descriptor> ad = dog_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma, thresh, window, nms_window);
    
    // Trova le corrispondenze e stima l'omografia con RANSAC
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

// Funzione per creare una combinazione delle due immagini
Image panorama_image_dog(const Image& a, const Image& b,
                         float sigma, float thresh, int window, int nms_window,
                         float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);  
    vector<Descriptor> ad = dog_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma, thresh, window, nms_window);
    
    // Trova le corrispondenze e stima l'omografia con RANSAC
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff); 
}
