#include "../image.h"
#include "../utils.h"
#include "../matrix.h"
#include <string>
#include "../FHH.cpp"

using namespace std;

int main() {
    Image a = load_image("pano/columbia/0.jpg");
    Image b = load_image("pano/columbia/1.jpg");

    int method = 2; 

    float sigma = 1.5;
    float thresh = 0.5;
    int window = 11;
    int nms = 7;  
    
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
