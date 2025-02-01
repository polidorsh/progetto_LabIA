#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>

#include "image.h"
#include "matrix.h"

using namespace std;

// Creazione del filtro Laplaciano
Image make_laplacian_filter() {
    Image f(3,3,1);
    f(0,0,0) = f(0,2,0) = f(2,0,0) = f(2,2,0) = 0.25;
    f(0,1,0) = f(1,0,0) = f(1,2,0) = f(2,1,0) = 0.5;
    f(1,1,0) = -4;
    return f;
}

// Calcolo della risposta Laplaciana per la rilevazione dei punti chiave
Image laplacian_response(const Image& im, float sigma) {
    // Convertiamo l'immagine in scala di grigi se necessario
    Image gray;
    if(im.c == 3) gray = rgb_to_grayscale(im);
    else gray = im;
    
    // Applichiamo una sfocatura gaussiana per ridurre il rumore
    Image smoothed = smooth_image(gray, sigma);
    
    // Applichiamo il filtro Laplaciano
    Image f = make_laplacian_filter();
    Image response = convolve_image(smoothed, f, true);
    
    // Eleviamo al quadrato la risposta per ottenere la magnitudo del contrasto
    for(int y = 0; y < response.h; y++) {
        for(int x = 0; x < response.w; x++) {
            response(x,y,0) = response(x,y,0) * response(x,y,0);
        }
    }
    
    return response;
}

// Rilevazione dei punti chiave utilizzando il Laplaciano
vector<Descriptor> laplacian_detector(const Image& im, float sigma, float thresh, int window, int nms) {
    // Calcoliamo la risposta Laplaciana
    Image response = laplacian_response(im, sigma);
    
    // Applichiamo la soppressione dei massimi locali per ottenere solo i punti più significativi
    Image nms_response = nms_image(response, nms);
    
    // Rileviamo i punti chiave e creiamo i descrittori
    return detect_corners(im, nms_response, thresh, window);
}

// Disegna i punti chiave rilevati con il metodo Laplaciano
Image detect_and_draw_laplacian(const Image& im, float sigma, float thresh, int window, int nms) {
    vector<Descriptor> d = laplacian_detector(im, sigma, thresh, window, nms);
    return mark_corners(im, d);
}

// Trova e disegna i match tra due immagini utilizzando il rilevatore Laplaciano
Image find_and_draw_laplacian_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    // Rileviamo i punti chiave in entrambe le immagini
    vector<Descriptor> ad = laplacian_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = laplacian_detector(b, sigma, thresh, window, nms);
    
    // Troviamo le corrispondenze tra i punti chiave
    vector<Match> m = match_descriptors(ad, bd);
    
    // Disegniamo i punti chiave e le linee di corrispondenza
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    
    return lines;
}

// Disegna i match evidenziando gli inlier tramite RANSAC
Image draw_laplacian_inliers(const Image& a, const Image& b, float sigma, float thresh, int window, 
                            int nms, float inlier_thresh, int iters, int cutoff) {
    // Rileviamo i punti chiave
    vector<Descriptor> ad = laplacian_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = laplacian_detector(b, sigma, thresh, window, nms);
    
    // Troviamo le corrispondenze tra i punti chiave
    vector<Match> m = match_descriptors(ad, bd);
    
    // Applichiamo RANSAC per identificare gli inlier
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    
    // Disegniamo gli inlier (verde) e gli outlier (rosso)
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

// Creazione del panorama utilizzando il rilevatore Laplaciano
Image panorama_image_laplacian(const Image& a, const Image& b, float sigma, float thresh, int window, 
                             int nms, float inlier_thresh, int iters, int cutoff, float acoeff) {
    // Rileviamo i punti chiave
    vector<Descriptor> ad = laplacian_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = laplacian_detector(b, sigma, thresh, window, nms);
    
    // Troviamo le corrispondenze tra i punti chiave
    vector<Match> m = match_descriptors(ad, bd);
    
    // Applichiamo RANSAC per stimare la trasformazione omografica
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    
    // Combiniamo le immagini per creare il panorama
    return combine_images(a, b, Hba, acoeff);
}
