#include "../image.h"
#include "../utils.h"
#include "../matrix.h"
#include <string>
#include "../ST.cpp"

using namespace std;

//TESTIAMO IL RILEVATORE:
//columbia 01 23 45 67 89 || sigma=1.5, thresh=0.3, window=10, nms=10
//rainier  01 23 45       || 
//field    32 45 67       || 
//helens   01 23 45       || 
//sun      23 10          || 
//wall     12             || 
//cse      12 34 56 78    || 

int main() {

    Image a = load_image("pano/rainier/0.jpg");
    Image b = load_image("pano/rainier/1.jpg");

    bool is_adaptive = true;  

    float sigma = 1.5;
    float thresh = 0.3;
    int window = 10;
    int nms_size = 10;

    float inlier_thresh = 5;
    int iters = 10000;
    int cutoff = 150;
    float acoeff = 0.5;

    Image shi_tomasi_points_A = detect_and_draw_shi_tomasi(a, is_adaptive, sigma, thresh, window, nms_size);
    save_image(shi_tomasi_points_A, "output/shi_tomasi_keypoints_A");

    Image shi_tomasi_points_B = detect_and_draw_shi_tomasi(b, is_adaptive, sigma, thresh, window, nms_size);
    save_image(shi_tomasi_points_B, "output/shi_tomasi_keypoints_B");

    Image shi_tomasi_matches = find_and_draw_shi_tomasi_matches(a, b, is_adaptive, sigma, thresh, window, nms_size);
    save_image(shi_tomasi_matches, "output/shi_tomasi_matches");

    Image shi_tomasi_inliers = draw_shi_tomasi_inliers(a, b, is_adaptive, sigma, thresh, window, nms_size, inlier_thresh, iters, cutoff);
    save_image(shi_tomasi_inliers, "output/shi_tomasi_inliers");

    Image shi_tomasi_panorama = panorama_image_shi_tomasi(a, b, is_adaptive, sigma, thresh, window, nms_size, inlier_thresh, iters, cutoff, acoeff);
    save_image(shi_tomasi_panorama, "output/shi_tomasi_panorama");

    return 0;
}
