#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include "image.h"

using namespace std;

// Struttura per rappresentare i candidati keypoint
struct KeypointCandidate {
    int x, y;            // Coordinate del punto
    int scale_index;     // Indice della scala a cui il punto è stato rilevato
    float response;      // Risposta del punto in base al filtro Laplaciano
    KeypointCandidate(int x_, int y_, int s_, float r_) : 
        x(x_), y(y_), scale_index(s_), response(r_) {}
};

// Funzione per creare il filtro Laplaciano del Gaussiano (LoG) con una data sigma e dimensione
Image lap_make_log_filter(float sigma, int size) {
    // Se la dimensione è pari, la incrementiamo per renderla dispari (centrata)
    if (size % 2 == 0) size++;
    Image filter(size, size, 1);  // Immagine filtro (monocanale)
    int center = size / 2;       // Centro del filtro
    
    // Calcoliamo alcuni termini predefiniti per l'efficienza
    float sigma2 = sigma * sigma; 
    float sigma4 = sigma2 * sigma2;
    
    // Fattori di normalizzazione e varianza per il filtro
    float norm_factor = -1.0 / (M_PI * sigma4);  // Normalizzazione del filtro
    float var_factor = 1.0 / (2 * sigma2);      // Fattore della varianza
    
    // Costruiamo il filtro LoG con una convoluzione basata sulla formula
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center;  // Distanza orizzontale dal centro
            float dy = y - center;  // Distanza verticale dal centro
            float r2 = dx * dx + dy * dy;  // Distanza quadratica dal centro
            float exp_term = exp(-r2 * var_factor);  // Esponenziale per la parte Gaussiana
            filter(x, y, 0) = norm_factor * (1 - r2 * var_factor) * exp_term;  // Formula finale
        }
    }
    
    return filter;
}

// Funzione per applicare la soppressione dei massimi non locali e nelle scale
vector<KeypointCandidate> lap_non_maximum_suppression(const vector<Image>& responses, 
                                                     float threshold, int window_size) {
    vector<KeypointCandidate> candidates;  // Lista di keypoints trovati
    int num_scales = responses.size();    // Numero di scale
    int half_window = window_size / 2;     // Dimensione della finestra di soppressione
    
    // Scansione attraverso tutte le scale, escludendo la prima e l'ultima
    for (int s = 1; s < num_scales - 1; s++) {
        const Image& curr_response = responses[s];  // Risposta alla scala corrente
        const Image& prev_response = responses[s - 1];  // Risposta alla scala precedente
        const Image& next_response = responses[s + 1];  // Risposta alla scala successiva
        
        // Scansione su ogni pixel della risposta
        for (int y = half_window; y < curr_response.h - half_window; y++) {
            for (int x = half_window; x < curr_response.w - half_window; x++) {
                float center_val = fabs(curr_response(x, y, 0));  // Valore assoluto al centro
                
                // Controllo rapido sulla soglia
                if (center_val < threshold) continue;
                
                bool is_maximum = true;  // Flag per determinare se il punto è un massimo
                
                // Controllo se è un massimo locale nella finestra spaziale
                for (int dy = -half_window; dy <= half_window && is_maximum; dy++) {
                    for (int dx = -half_window; dx <= half_window && is_maximum; dx++) {
                        if (dx == 0 && dy == 0) continue;  // Non confrontiamo il punto con se stesso
                        if (fabs(curr_response(x + dx, y + dy, 0)) >= center_val) {
                            is_maximum = false;  // Non è un massimo se troviamo un valore maggiore
                        }
                    }
                }
                
                // Controllo nelle scale adiacenti
                if (is_maximum) {
                    if (fabs(prev_response(x, y, 0)) >= center_val || 
                        fabs(next_response(x, y, 0)) >= center_val) {
                        is_maximum = false;  // Non è un massimo se il valore nelle scale adiacenti è maggiore
                    }
                }
                
                // Se il punto è un massimo, lo aggiungiamo ai candidati
                if (is_maximum) {
                    candidates.emplace_back(x, y, s, center_val);
                }
            }
        }
    }
    
    return candidates;
}

// Funzione per segnare visivamente un keypoint sull'immagine
void lap_mark_spot(Image& im, int x, int y, float strength) {
    // Determiniamo la dimensione della croce in base alla forza del keypoint
    int size = max(2, min(5, int(strength * 10)));  // Dimensione tra 2 e 5
    
    // Disegniamo una croce centrata nel punto (x, y)
    for (int i = -size; i <= size; ++i) {
        // Linea orizzontale
        if (x + i >= 0 && x + i < im.w) {
            if (y >= 0 && y < im.h) {
                im(x + i, y, 0) = 1;  // Componente rossa
                im(x + i, y, 1) = 0;  // Componente verde
                im(x + i, y, 2) = 0;  // Componente blu
            }
        }
        // Linea verticale
        if (y + i >= 0 && y + i < im.h) {
            if (x >= 0 && x < im.w) {
                im(x, y + i, 0) = 1;  // Componente rossa
                im(x, y + i, 1) = 0;  // Componente verde
                im(x, y + i, 2) = 0;  // Componente blu
            }
        }
    }
}

// Funzione principale per rilevare i keypoints usando il metodo Laplaciano
vector<Descriptor> lap_detect_keypoints(const Image& im2, float sigma_start, float sigma_end, 
                                         int num_scales, float threshold, int nms_window) {
    Image im = (im2.c == 1) ? im2 : rgb_to_grayscale(im2);  // Convertiamo in scala di grigi se necessario
    vector<Image> responses;  // Risposte per ogni scala
    float k = pow(sigma_end / sigma_start, 1.0f / (num_scales - 1));  // Fattore di scala
    
    // Calcoliamo la risposta per tutte le scale
    #pragma omp parallel for
    for (int i = 0; i < num_scales; ++i) {
        float sigma = sigma_start * pow(k, i);  // Calcoliamo la sigma per questa scala
        int filter_size = 2 * ceil(3 * sigma) + 1;  // Dimensione del filtro
        Image log_filter = lap_make_log_filter(sigma, filter_size);  // Creiamo il filtro LoG
        
        // Applichiamo la convoluzione con il filtro
        Image response = convolve_image(im, log_filter, true);
        
        // Normalizziamo la risposta in base alla sigma
        float scale_factor = sigma * sigma;
        for (int y = 0; y < response.h; ++y) {
            for (int x = 0; x < response.w; ++x) {
                response(x, y, 0) *= scale_factor;  // Normalizzazione
            }
        }
        
        #pragma omp critical
        responses.push_back(response);  // Aggiungiamo la risposta alla lista
    }
    
    // Applichiamo la soppressione dei massimi non locali
    vector<KeypointCandidate> candidates = lap_non_maximum_suppression(responses, threshold, nms_window);
    
    // Convertiamo i candidati in descrittori
    vector<Descriptor> keypoints;
    keypoints.reserve(candidates.size());
    for (const auto& c : candidates) {
        Descriptor d;
        d.p = {(double)c.x, (double)c.y};  // Creiamo il descrittore per ogni keypoint
        keypoints.push_back(d);
    }
    
    return keypoints;
}

// Funzione per rilevare e disegnare keypoints sull'immagine
Image lap_detect_and_draw_keypoints(const Image& im, float sigma_start, float sigma_end, 
                                    int num_scales, float threshold, int nms_window) {
    // Rileviamo i keypoints
    vector<Descriptor> keypoints = lap_detect_keypoints(im, sigma_start, sigma_end, 
                                                        num_scales, threshold, nms_window);
    
    Image marked = im;  // Copia dell'immagine originale
    // Disegniamo ogni keypoint come una croce rossa
    for (const auto& kp : keypoints) {
        lap_mark_spot(marked, kp.p.x, kp.p.y, 0.5);  // Usando una forza fissa per la visualizzazione
    }
    
    return marked;
}
