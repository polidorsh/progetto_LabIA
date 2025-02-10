#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../LAP.cpp"

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
    Image a = load_image("pano/rainier/0.jpg");
    Image b = load_image("pano/rainier/1.jpg");

    float sigma = 1.5;
    float thresh = 0.05;
    int window = 11;
    int nms = 7;
    
    float inlier_thresh = 5;
    int iters = 10000;
    int cutoff = 150;
    float acoeff = 0.5;

    Image log_points_A = detect_and_draw_log_keypoints(a, sigma, thresh, window, nms);
    save_image(log_points_A, "output/log_keypoints_A");

    Image log_points_B = detect_and_draw_log_keypoints(b, sigma, thresh, window, nms);
    save_image(log_points_B, "output/log_keypoints_B");

    Image log_matches = find_and_draw_log_matches(a, b, sigma, thresh, window, nms);
    save_image(log_matches, "output/log_matches");

    Image log_inliers = draw_log_inliers(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff);
    save_image(log_inliers, "output/log_inliers");

    Image log_panorama = panorama_image_log(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff, acoeff);
    save_image(log_panorama, "output/log_panorama");

    return 0;
}