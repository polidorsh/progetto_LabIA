#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../ORI.cpp"

using namespace std;

int main() {
    // Carica le immagini da elaborare
    Image a = load_image("pano/cse/1.jpg");
    Image b = load_image("pano/cse/2.jpg");
    save_image(a, "output/a");
    save_image(b, "output/b");
    
    // Definizione dei parametri per il rilevamento delle caratteristiche
    float sigma = 3;          
    float thresh = 0.08;      
    int window = 15;         
    int nms = 24;            

    // Parametri per RANSAC nella stima dell'omografia
    float inlier_thresh = 8; 
    int iters = 1000;         
    int cutoff = 250;        
    
    // Rilevamento degli angoli nell'immagine e salvataggio del risultato
    Image corners = oriented_corners(a, sigma, thresh, window, nms);
    save_image(corners, "output/oriented_corners");

    // Individuazione e visualizzazione delle corrispondenze tra le immagini
    Image matches = oriented_matches(a, b, sigma, thresh, window, nms);
    save_image(matches, "output/oriented_matches");

    // Identificazione degli inlier tramite RANSAC e salvataggio del risultato
    Image inliers = oriented_inliers(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff);
    save_image(inliers, "output/oriented_inliers");

    // Creazione del panorama combinando le due immagini
    Image pano = oriented_panorama(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff);
    save_image(pano, "output/oriented_panorama");

    return 0;
}
