#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <thread>

#include "image.h"
#include "matrix.h"

using namespace std;

Descriptor describe_index(const Image& im, int x, int y, int w){
  Descriptor d;
  d.p={(double)x,(double)y};
  d.data.reserve(w*w*im.c);
  
  for(int c=0;c<im.c;c++){
    float cval = im.clamped_pixel(x,y,c);
    for(int dx=-w/2;dx<=w/2;dx++)for(int dy=-w/2;dy<=w/2;dy++)
      d.data.push_back(im.clamped_pixel(x+dx,y+dy,c)-cval);
  }
  return d;
}

// Shi-Tomasi corner detection
Image shi_tomasi_response(const Image& im, float sigma) {
    // Compute structure matrix similar to harris corner detector
    Image S = structure_matrix(im, sigma);
    
    Image R(S.w, S.h);
    for(int y = 0; y < S.h; y++) {
        for(int x = 0; x < S.w; x++) {
            // Compute eigenvalues of structure matrix
            float a = S(x,y,0);  // Ix^2
            float b = S(x,y,1);  // Iy^2
            float c = S(x,y,2);  // IxIy
            
            // Compute minimum eigenvalue (Shi-Tomasi criterion)
            float lambda1 = (a + b - sqrt(pow(a - b, 2) + 4*c*c)) / 2;
            float lambda2 = (a + b + sqrt(pow(a - b, 2) + 4*c*c)) / 2;
            
            // Use minimum eigenvalue as corner response
            R(x,y,0) = min(lambda1, lambda2);
        }
    }
    return R;
}


// Rilevazione dei punti chiave utilizzando il Laplaciano
vector<Descriptor> shi_tomasi_corner_detector(const Image& im, float sigma, float thresh, int window, int nms) {
    // Calcoliamo la risposta Laplaciana
    Image response = shi_tomasi_response(im, sigma);
    
    // Applichiamo la soppressione dei massimi locali per ottenere solo i punti più significativi
    Image nms_response = nms_image(response, nms);
    
    // Rileviamo i punti chiave e creiamo i descrittori
    return detect_corners(im, nms_response, thresh, window);
}

// Disegna i punti chiave rilevati con il metodo Laplaciano
Image detect_and_draw_shi_tomasi_corners(const Image& im, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> d = shi_tomasi_corner_detector(im, sigma, thresh, window, nms);
    printf("Numero di Descrittori: %ld\n", d.size());
    return mark_corners(im, d);
}

// Trova e disegna i match tra due immagini utilizzando il rilevatore Laplaciano
Image find_and_draw_shi_tomasi_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    TIME(1);
    // Rileviamo i punti chiave in entrambe le immagini
    vector<Descriptor> ad = shi_tomasi_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = shi_tomasi_corner_detector(b, sigma, thresh, window, nms);
    
    // Troviamo le corrispondenze tra i punti chiave
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());

    
    // Disegniamo i punti chiave e le linee di corrispondenza
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    
    return lines;
}

// Disegna i match evidenziando gli inlier tramite RANSAC
Image find_and_draw_shi_tomasi_inliers(const Image& a, const Image& b, float sigma, float thresh, int window, 
                            int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    // Rileviamo i punti chiave
    vector<Descriptor> ad = shi_tomasi_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = shi_tomasi_corner_detector(b, sigma, thresh, window, nms);
    
    // Troviamo le corrispondenze tra i punti chiave
    vector<Match> m = match_descriptors(ad, bd);
    
    // Applichiamo RANSAC per identificare gli inlier
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    
    // Disegniamo gli inlier (verde) e gli outlier (rosso)
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

// Creazione del panorama utilizzando il rilevatore Laplaciano
Image panorama_image_shi_tomasi(const Image& a, const Image& b, float sigma, float thresh, int window, 
                             int nms, float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    // Rileviamo i punti chiave
    vector<Descriptor> ad = shi_tomasi_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = shi_tomasi_corner_detector(b, sigma, thresh, window, nms);
    
    // Troviamo le corrispondenze tra i punti chiave
    vector<Match> m = match_descriptors(ad, bd);
    
    // Applichiamo RANSAC per stimare la trasformazione omografica
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    
    // Combiniamo le immagini per creare il panorama
    return combine_images(a, b, Hba, acoeff);
}
