#include "../image.h"
#include "../utils.h"
#include "../matrix.h"
#include <string>
#include "../ST.cpp"

using namespace std;

//TESTIAMO IL RILEVATORE:
//Columbia 01 23 45 67 89 || sigma=1.5, thresh=0.05, window=11, nms=7
//Rainier  01 23 45       || 
//Field    32 45 67       || 
//Helens   01 23 45       || 
//Sun      01 23          || 
//Wall     12             || 
//Cse      12 34 56 78    || 

int main() {
    // Carica le immagini
    Image a = load_image("pano/rainier/1.jpg");
    Image b = load_image("pano/rainier/2.jpg");

    float sigma = 1.5;
    float thresh = 0.3;
    int window = 11;
    int nms = 7;  
    
    float inlier_thresh = 5;
    int iters = 10000;
    int cutoff = 150;
    float acoeff = 0.5;

    Image shi_tomasi_points_A = detect_and_draw_shi_tomasi(a, sigma, thresh, window, nms);
    save_image(shi_tomasi_points_A, "output/shi_tomasi_keypoints_A");

    Image shi_tomasi_points_B = detect_and_draw_shi_tomasi(b, sigma, thresh, window, nms);
    save_image(shi_tomasi_points_B, "output/shi_tomasi_keypoints_B");

    Image shi_tomasi_matches = find_and_draw_shi_tomasi_matches(a, b, sigma, thresh, window, nms);
    save_image(shi_tomasi_matches, "output/shi_tomasi_matches");

    Image shi_tomasi_inliers = draw_shi_tomasi_inliers(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff);
    save_image(shi_tomasi_inliers, "output/shi_tomasi_inliers");

    Image shi_tomasi_panorama = panorama_image_shi_tomasi(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff, acoeff);
    save_image(shi_tomasi_panorama, "output/shi_tomasi_panorama");

    return 0;
}