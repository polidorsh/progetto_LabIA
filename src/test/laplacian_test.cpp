#include "../image.h"
#include "../utils.h"
#include "../laplacian_image.cpp"
#include <string>

using namespace std;

int main(int argc, char **argv) {
    Image im = load_image("pano/rainier/0.jpg");  // o qualsiasi altra immagine
    
    // Parametri del rilevatore LoG
    float sigma = 2.0;      // scala del Laplaciano
    float thresh = 0.1;    // soglia per i keypoint 
    int window = 5;         // dimensione finestra descrittore
    int nms = 3;           // dimensione finestra per non-maximum suppression
    
    // Rileva keypoint
    vector<Descriptor> keypoints = log_keypoint_detector(im, sigma, thresh, window, nms);
    printf("Found %zu keypoints\n", keypoints.size());
    
    // Visualizza keypoint
    Image marked = draw_keypoints(im, keypoints);
    save_image(marked, "output/log_points.png");
    
    return 0;
}