#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../LAP.cpp"

using namespace std;

int main() {
    Image a = load_image("pano/cse/1.jpg");
    Image b = load_image("pano/cse/2.jpg");
    save_image(a, "output/a");
    save_image(b, "output/b");

    float sigma = 1;
    int num_scales=4;
    float scale_factor=1.5;
    float thresh = 0.05;
    int window = 15;
    int nms = 20;

    float inlier_thresh = 7;
    int ransac_iters = 10000;
    int cutoff = 180;

    Image laplacian_points = detect_and_draw_log_keypoints(a, sigma, num_scales, scale_factor, 
                                                thresh, nms, window);
    save_image(laplacian_points, "output/laplacian_keypoints");

    Image matches = find_and_draw_laplacian_matches(a, b, sigma, num_scales, scale_factor, 
                                                thresh, nms, window);
    save_image(matches, "output/laplacian_matches");

    Image inliers = draw_laplacian_inliers(a, b, sigma, num_scales, scale_factor, thresh, 
                                        nms, window, inlier_thresh, ransac_iters, cutoff);
    save_image(inliers, "output/laplacian_inliers");

    Image pano = panorama_image_laplacian(a, b, sigma, num_scales, scale_factor, thresh, 
                                        nms, window, inlier_thresh, ransac_iters, cutoff, 0.5);
    save_image(pano, "output/laplacian_panorama");

    return 0;
}
