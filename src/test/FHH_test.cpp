#include "../image.h"
#include "../utils.h"
#include "../matrix.h"
#include <string>
#include "../FHH.cpp"

using namespace std;

//TESTIAMO IL RILEVATORE: sigma=1.5, window=10, nms=10
//columbia 01 23 45 67 89 || METODO 0: thresh=3   || METODO 123: thresh=0.2
//rainier  01 23 45       ||                      ||
//field    32 45 67       ||                      ||                    
//helens   10 23 45       ||                      ||                    
//sun      01 23          ||                      ||
//wall     01 12          ||                      ||
//cse      12 34 56 78    ||                      ||

int main() {
    Image a = load_image("pano/cse/1.jpg");
    Image b = load_image("pano/cse/2.jpg");

    int method = 3; 

    float sigma = 1.5;
    float thresh = 0.2;
    int window = 10;
    int nms = 10;  
    
    float inlier_thresh = 5;
    int iters = 10000;
    int cutoff = 150;
    float acoeff = 0.5;

    Image fh_points_A = detect_and_draw_fhh(a, method, sigma, thresh, window, nms);
    save_image(fh_points_A, "output/fhh_keypoints_A");

    Image fh_points_B = detect_and_draw_fhh(b, method, sigma, thresh, window, nms);
    save_image(fh_points_B, "output/fhh_keypoints_B");

    Image fh_matches = find_and_draw_fhh_matches(a, b, method, sigma, thresh, window, nms);
    save_image(fh_matches, "output/fhh_matches");

    Image fh_inliers = draw_fhh_inliers(a, b, method, sigma, thresh, window, nms, inlier_thresh, iters, cutoff);
    save_image(fh_inliers, "output/fhh_inliers");

    Image fh_panorama = panorama_image_fhh(a, b, method, sigma, thresh, window, nms, inlier_thresh, iters, cutoff, acoeff);
    save_image(fh_panorama, "output/fhh_panorama");

    return 0;
}
