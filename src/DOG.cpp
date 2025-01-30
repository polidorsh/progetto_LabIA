#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include "image.h"

using namespace std;

// Crea un livello della piramide DoG (Differenza di Gaussiane)
// sigma1: valore di sigma più piccolo
// sigma2: valore di sigma più grande (tipicamente sigma1 * k, dove k è solitamente √2)
// restituisce: immagine della differenza tra le due gaussiane
Image dog_create_level(const Image& im, float sigma1, float sigma2) {
    // Applica il filtro gaussiano con sigma1 e sigma2
    Image smooth1 = smooth_image(im, sigma1);
    Image smooth2 = smooth_image(im, sigma2);
    
    // Crea l'immagine DoG sottraendo le due immagini smussate
    Image dog(im.w, im.h, 1);
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            dog(x,y,0) = smooth2(x,y,0) - smooth1(x,y,0); // Differenza tra le due immagini smussate
        }
    }
    return dog;
}

// Verifica se un punto è un estremo locale in un vicinato 3x3x3 nello spazio delle scale
bool dog_is_extremum(const vector<Image>& dog_pyramid, int level, int x, int y) {
    float center_val = dog_pyramid[level](x,y,0); // Valore del centro nel livello corrente
    bool is_max = true; // Flag per il massimo
    bool is_min = true; // Flag per il minimo
    
    // Controllo del vicinato 3x3x3
    for(int l = max(0, level-1); l <= min((int)dog_pyramid.size()-1, level+1); l++) {
        for(int dy = -1; dy <= 1; dy++) {
            for(int dx = -1; dx <= 1; dx++) {
                if(l == level && dx == 0 && dy == 0) continue; // Salta il centro
                float neighbor = dog_pyramid[l].clamped_pixel(x+dx, y+dy, 0);
                if(neighbor >= center_val) is_max = false; // Se il vicino è maggiore, non è un massimo
                if(neighbor <= center_val) is_min = false; // Se il vicino è minore, non è un minimo
            }
        }
    }
    return is_max || is_min; // Se è un massimo o un minimo, è un estremo
}

// Crea un marcatore visivo per la posizione e la scala di un keypoint
void dog_mark_spot(Image& im, const Point& p, float scale) {
    int x = p.x;
    int y = p.y;
    int radius = round(scale * 3); // Scala il marcatore con la scala del keypoint
    
    for(int i = -radius; i <= radius; ++i) {
        // Disegna una croce centrata sul keypoint
        im.set_pixel(x+i, y, 0, 1); // Linea orizzontale
        im.set_pixel(x, y+i, 0, 1); // Linea verticale
        // Disegna una circonferenza approssimata
        for(int j = -radius; j <= radius; ++j) {
            if(abs(i*i + j*j - radius*radius) < radius) { // Approccio per approssimare un cerchio
                im.set_pixel(x+i, y+j, 2, 1); // Colore blu per il cerchio
            }
        }
    }
}

// Funzione per segnare e visualizzare i keypoints sull'immagine
Image dog_mark_keypoints(const Image& im, const vector<Descriptor>& d, const vector<float>& scales) {
    Image im2 = im;
    for(size_t i = 0; i < d.size(); ++i) {
        dog_mark_spot(im2, d[i].p, scales[i]); // Disegna ogni keypoint sull'immagine
    }
    return im2;
}

// Funzione principale per rilevare i keypoints usando la Differenza di Gaussiane (DoG)
// octaves: numero di ottave nello spazio delle scale
// intervals: numero di intervalli per ottava
// sigma: valore iniziale di sigma
// contrast_threshold: soglia per filtrare i keypoints deboli
vector<Descriptor> dog_detect_keypoints(const Image& im2, int octaves, int intervals, 
                                      float sigma, float contrast_threshold) {
    // Converte l'immagine in scala di grigio se necessario
    Image im;
    if(im2.c == 1) im = im2;
    else im = rgb_to_grayscale(im2);
    
    vector<Descriptor> keypoints; // Lista dei keypoints rilevati
    vector<float> scales; // Memorizza le scale per la visualizzazione
    float k = pow(2.0f, 1.0f/intervals); // Fattore di scala tra gli intervalli
    
    // Per ogni ottava
    Image current_im = im;
    for(int o = 0; o < octaves; ++o) {
        vector<Image> dog_pyramid;
        
        // Crea la piramide DoG per questa ottava
        for(int i = 0; i < intervals + 2; ++i) {
            float sigma1 = sigma * pow(k, i); // Calcola sigma per il primo livello
            float sigma2 = sigma * pow(k, i+1); // Calcola sigma per il secondo livello
            dog_pyramid.push_back(dog_create_level(current_im, sigma1, sigma2)); // Crea il livello DoG
        }
        
        // Trova gli estremi nella piramide DoG
        for(int i = 1; i < intervals + 1; ++i) {
            for(int y = 1; y < current_im.h-1; ++y) {
                for(int x = 1; x < current_im.w-1; ++x) {
                    if(fabs(dog_pyramid[i](x,y,0)) < contrast_threshold) continue; // Scarta i valori sotto la soglia
                    
                    if(dog_is_extremum(dog_pyramid, i, x, y)) { // Verifica se è un estremo
                        // Calcola le coordinate del keypoint tenendo conto della scala
                        float scale_factor = pow(2.0f, o); // Fattore di scala per l'ottava
                        Point p = {(double)(x * scale_factor), (double)(y * scale_factor)};
                        
                        // Crea un descrittore per il keypoint
                        Descriptor d;
                        d.p = p;
                        
                        // Memorizza il keypoint e la sua scala
                        keypoints.push_back(d);
                        scales.push_back(sigma * pow(k, i) * scale_factor);
                    }
                }
            }
        }
        
        // Riduci la risoluzione dell'immagine per la prossima ottava
        if(o < octaves-1) {
            Image smaller(current_im.w/2, current_im.h/2, 1);
            for(int y = 0; y < smaller.h; ++y) {
                for(int x = 0; x < smaller.w; ++x) {
                    smaller(x,y,0) = current_im(2*x,2*y,0); // Subsampling
                }
            }
            current_im = smaller; // Aggiorna l'immagine per la prossima ottava
        }
    }
    
    return keypoints; // Restituisce la lista dei keypoints trovati
}

// Funzione principale per rilevare e disegnare i keypoints DoG sull'immagine
Image dog_detect_and_draw_keypoints(const Image& im, int octaves, int intervals, 
                                  float sigma, float contrast_threshold) {
    // Rileva i keypoints
    vector<Descriptor> keypoints = dog_detect_keypoints(im, octaves, intervals, sigma, contrast_threshold);
    
    // Calcola le scale per la visualizzazione (necessario per ridisegnare i keypoints)
    vector<float> scales;
    float k = pow(2.0f, 1.0f/intervals);
    for(size_t i = 0; i < keypoints.size(); ++i) {
        int octave = log2(keypoints[i].p.x / (int)keypoints[i].p.x); // Calcola l'ottava in base alla posizione
        scales.push_back(sigma * pow(k, octave) * pow(2.0f, octave)); // Calcola la scala per ogni keypoint
    }
    
    // Disegna i keypoints sull'immagine
    return dog_mark_keypoints(im, keypoints, scales);
}
