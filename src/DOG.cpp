#include "image.h"
#include <vector>
#include <cmath>

Descriptor describe_index(const Image& im, int x, int y, int w){
    Descriptor d;
    d.p={(double)x,(double)y};
    d.data.reserve(w*w*im.c);
    
    for(int c=0; c<im.c; c++){
        float cval = im.clamped_pixel(x,y,c);
        for(int dx=-w/2; dx<=w/2; dx++)
            for(int dy=-w/2; dy<=w/2; dy++)
                d.data.push_back(im.clamped_pixel(x+dx,y+dy,c)-cval);
    }
    return d;
}

// Implementa il rilevatore di keypoint basato su Differenza di Gaussiane
vector<Descriptor> dog_detector(const Image& im, float sigma, float thresh, int window, int nms_window) {
    // Convertiamo l'immagine in scala di grigi se necessario
    Image working_image;
    if (im.c == 1) working_image = im;
    else working_image = rgb_to_grayscale(im);
    
    // Creiamo due versioni sfocate dell'immagine
    // La seconda usa un sigma k volte più grande (tipicamente k = 1.6)
    float k = 1.6f;
    Image gaussian1 = smooth_image(working_image, sigma);
    Image gaussian2 = smooth_image(working_image, sigma * k);
    
    // Calcoliamo la Differenza di Gaussiane
    Image dog(gaussian1.w, gaussian1.h, 1);
    for(int y = 0; y < dog.h; y++) {
        for(int x = 0; x < dog.w; x++) {
            // La sottrazione ci dà i punti di interesse potenziali
            dog(x,y,0) = gaussian2(x,y,0) - gaussian1(x,y,0);
        }
    }
    
    // Applichiamo la soppressione dei non-massimi
    Image nms_result = nms_image(dog, nms_window);
    
    // Individuiamo i keypoint sopra la soglia
    vector<Descriptor> keypoints;
    for(int y = 0; y < nms_result.h; y++) {
        for(int x = 0; x < nms_result.w; x++) {
            if(nms_result(x,y,0) > thresh) {
                // Per ogni keypoint, creiamo un descrittore
                keypoints.push_back(describe_index(working_image, x, y, window));
            }
        }
    }
    
    return keypoints;
}

// Funzione wrapper per visualizzare i keypoint trovati
Image detect_and_draw_dog(const Image& im, float sigma, float thresh, int window, int nms_window) {
    TIME(1);
    vector<Descriptor> keypoints = dog_detector(im, sigma, thresh, window, nms_window);
    return mark_corners(im, keypoints);
}


// Funzione per rilevare keypoint con DOG, accoppiare le descrizioni e disegnare i match
Image find_and_draw_dog_matches(const Image& a, const Image& b,
                                float sigma, float thresh, int window, int nms_window) {
    TIME(1);
    // Rilevamento dei keypoint in ciascuna immagine usando il detector DOG
    vector<Descriptor> ad = dog_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma, thresh, window, nms_window);
    
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
// utilizzando il detector DOG per la rilevazione dei keypoint
Image draw_dog_inliers(const Image& a, const Image& b,
                       float sigma, float thresh, int window, int nms_window,
                       float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    // Rileva keypoint con il metodo DOG
    vector<Descriptor> ad = dog_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma, thresh, window, nms_window);
    
    // Trova i match tra le descrizioni
    vector<Match> m = match_descriptors(ad, bd);
    // Stima la matrice omografica con RANSAC
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    // Disegna gli inlier sulla base della trasformazione stimata
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

// Funzione per generare un'immagine panoramica combinando due immagini
// utilizzando il detector DOG per la rilevazione dei keypoint
Image panorama_image_dog(const Image& a, const Image& b,
                         float sigma, float thresh, int window, int nms_window,
                         float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    // Rileva i keypoint nelle due immagini con il detector DOG
    vector<Descriptor> ad = dog_detector(a, sigma, thresh, window, nms_window);
    vector<Descriptor> bd = dog_detector(b, sigma, thresh, window, nms_window);
    
    // Trova i match fra i descrittori
    vector<Match> m = match_descriptors(ad, bd);
    // Stima la trasformazione con RANSAC
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    // Combina le immagini usando la trasformazione stimata
    return combine_images(a, b, Hba, acoeff);
}
