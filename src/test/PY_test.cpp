#include "../image.h"
#include "../utils.h"
#include "../matrix.h"
#include <string>
#include "../PY.cpp"

using namespace std;

//TESTIAMO IL RILEVATORE:
//columbia 01 23 45 67 89 || sigma=1.5, thresh=0.03, window=10, nms=10
//rainier  01 23 45       || 
//field    32 45 67       || 
//helens   01 23 45       || 
//sun      01 23          || 
//wall     01 12          || 
//cse      12 34 56 78    || 

int main() {

    Image a = load_image("pano/rainier/0.jpg");
    Image b = load_image("pano/rainier/1.jpg");

    float sigma = 1.5;
    float thresh = 0.03;
    int window = 10;
    int nms = 10;  

    int num_octaves=4;
    int scales_per_octave=3;
    
    float inlier_thresh = 5;
    int iters = 10000;
    int cutoff = 150;
    float acoeff = 0.5;

    Image scale_points_A = detect_and_draw_scale_space_keypoints(a, sigma, thresh, window, nms, num_octaves, scales_per_octave);
    save_image(scale_points_A, "output/scale_keypoints_A");

    Image scale_points_B = detect_and_draw_scale_space_keypoints(b, sigma, thresh, window, nms, num_octaves, scales_per_octave);
    save_image(scale_points_B, "output/scale_keypoints_B");

    Image scale_matches = find_and_draw_scale_matches(a, b, sigma, thresh, window, nms, num_octaves, scales_per_octave);
    save_image(scale_matches, "output/scale_matches");

    Image scale_inliers = draw_scale_inliers(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff, num_octaves, scales_per_octave);
    save_image(scale_inliers, "output/scale_inliers");

    Image scale_panorama = panorama_image_scale(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff, acoeff, num_octaves, scales_per_octave);
    save_image(scale_panorama, "output/scale_panorama");

    return 0;
}
