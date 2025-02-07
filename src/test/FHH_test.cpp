#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../FHH.cpp"

using namespace std;

int main() {
    Image a = load_image("pano/rainier/1.jpg");
    Image b = load_image("pano/rainier/2.jpg");
    save_image(a, "output/a");
    save_image(b, "output/b");

    float sigma = 2;
    float thresh = 0.07;
    int window = 7;
    int nms = 10;
    float inlier_thresh = 5;
    int ransac_iters = 50000;
    int cutoff = 100;
    float blend = 0.5;

    Image forstner_points = detect_and_draw_forstner_corners(a, sigma, thresh, window, nms);
    save_image(forstner_points, "output/forstner_keypoint");

    Image matches = find_and_draw_forstner_matches(a, b, sigma, thresh, window, nms);
    save_image(matches, "output/forstner_matches");

    Image inliers = find_and_draw_forstner_inliers(a, b, sigma, thresh, window, nms,
        inlier_thresh, ransac_iters, cutoff);
    save_image(inliers, "output/forstner_inliers");

    Image pano = panorama_image_forstner(a, b, sigma, thresh, window, nms,
        inlier_thresh, ransac_iters, cutoff, blend);
    save_image(pano, "output/forstner_panorama");

    return 0;
}