#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../FHH.cpp"

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
    Image a = load_image("pano/columbia/0.jpg");
    Image b = load_image("pano/columbia/1.jpg");

    float sigma = 1.5;
    float thresh = 0.07;
    int window = 11;
    int nms = 7;
    float inlier_thresh = 5;
    int ransac_iters = 10000;
    int cutoff = 150;
    float blend = 0.5;

    Image forstner_points_A = detect_and_draw_fhh_corners(a, sigma, thresh, window, nms);
    save_image(forstner_points_A, "output/forstner_keypoint_A");

    Image forstner_points_B = detect_and_draw_fhh_corners(b, sigma, thresh, window, nms);
    save_image(forstner_points_B, "output/forstner_keypoint_B");

    Image matches = find_and_draw_fhh_matches(a, b, sigma, thresh, window, nms);
    save_image(matches, "output/forstner_matches");

    Image inliers = find_and_draw_fhh_inliers(a, b, sigma, thresh, window, nms,
        inlier_thresh, ransac_iters, cutoff);
    save_image(inliers, "output/forstner_inliers");

    Image pano = panorama_image_fhh(a, b, sigma, thresh, window, nms,
        inlier_thresh, ransac_iters, cutoff, blend);
    save_image(pano, "output/forstner_panorama");

    return 0;
}