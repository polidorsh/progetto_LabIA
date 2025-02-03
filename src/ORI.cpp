#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include "image.h"

// Implementa un rilevatore di caratteristiche basato su filtri orientati (Koethe 2003)
vector<Descriptor> oriented_detector(const Image& im, float sigma, float thresh, int window) {
    // Converti l'immagine in scala di grigi se necessario
    Image gray = (im.c == 3) ? rgb_to_grayscale(im) : im;
    
    // Definisce il numero di orientazioni del filtro
    const int num_angles = 8;
    const float PI = 3.14159265358979323846;
    vector<Image> responses;
    
    // Crea filtri gaussiani orientati in diverse direzioni
    for(int i = 0; i < num_angles; i++) {
        float angle = i * PI / num_angles;
        float dx = cos(angle);
        float dy = sin(angle);
        
        // Determina la dimensione del filtro in base a sigma
        int size = ceil(sigma * 6);
        if(!(size % 2)) size++; // Assicura una dimensione dispari
        Image filter(size, size, 1);
        
        // Popola il filtro con una Gaussiana orientata
        for(int y = 0; y < size; y++) {
            for(int x = 0; x < size; x++) {
                float rx = x - size/2;
                float ry = y - size/2;
                // Proietta il punto lungo la direzione del filtro
                float proj = rx*dx + ry*dy;
                float perp = rx*(-dy) + ry*dx;
                // Calcola la risposta Gaussiana lungo la direzione perpendicolare
                filter(x,y,0) = exp(-(perp*perp)/(2*sigma*sigma)) * 
                               // Seconda derivata lungo la direzione del filtro
                               (1 - (proj*proj)/(sigma*sigma)) * 
                               exp(-(proj*proj)/(2*sigma*sigma));
            }
        }
        
        // Convolvi l'immagine con il filtro orientato e salva la risposta
        Image response = convolve_image(gray, filter, true);
        responses.push_back(response);
    }
    
    // Determina la risposta massima tra tutte le orientazioni
    Image maxResponse(gray.w, gray.h, 1);
    for(int y = 0; y < gray.h; y++) {
        for(int x = 0; x < gray.w; x++) {
            float maxVal = 0;
            for(auto& resp : responses) {
                maxVal = max(maxVal, fabs(resp(x,y,0)));
            }
            maxResponse(x,y,0) = maxVal;
        }
    }
    
    // Applica la soppressione non massima
    Image nms = nms_image(maxResponse, window);
    
    // Rileva gli angoli e genera i descrittori
    return detect_corners(gray, nms, thresh, window);
}

// Interfaccia principale simile al rilevatore di Harris
vector<Descriptor> oriented_corner_detector(const Image& im, float sigma, float thresh, int window, int nms_window) {
    return oriented_detector(im, sigma, thresh, window);
}

// Disegna gli angoli rilevati dal filtro orientato
Image oriented_corners(const Image& im, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> corners = oriented_corner_detector(im, sigma, thresh, window, nms);
    printf("Numero di Descrittori: %ld\n", corners.size());
    return mark_corners(im, corners);
}

// Trova e disegna le corrispondenze tra due immagini utilizzando il rilevatore orientato
Image oriented_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> ad = oriented_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = oriented_corner_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());

    
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    return draw_matches(A, B, m, {});
}

// Disegna gli inlier trovati con il rilevatore orientato
Image oriented_inliers(const Image& a, const Image& b, float sigma, float thresh, int window, 
                      int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = oriented_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = oriented_corner_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    
    Matrix H = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, H, m, inlier_thresh);
}

// Crea un panorama utilizzando il rilevatore orientato
Image oriented_panorama(const Image& a, const Image& b, float sigma, float thresh, int window, 
                       int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    // Estrai i punti chiave con il rilevatore orientato
    vector<Descriptor> ad = oriented_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = oriented_corner_detector(b, sigma, thresh, window, nms);
    
    // Trova le corrispondenze tra i descrittori
    vector<Match> matches = match_descriptors(ad, bd);
    
    // Stima l'omografia usando RANSAC
    Matrix H = RANSAC(matches, inlier_thresh, iters, cutoff);
    
    // Unisce le immagini per creare il panorama
    return combine_images(a, b, H, 0.5);
}
