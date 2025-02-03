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

    // Parametri per il rilevatore Laplaciano
    float sigma = 2.7;
    float thresh = 0.08;
    int window_size = 13;
    int nms_window = 20;

    // Parametri per RANSAC
    float inlier_thresh = 7;
    int ransac_iters = 600;
    int cutoff = 180;

    // Visualizza i keypoint
    Image laplacian_points = detect_and_draw_laplacian(a, sigma, thresh, window_size, nms_window);
    save_image(laplacian_points, "output/laplacian_keypoints");

    // Visualizza tutti i matches
    Image matches = find_and_draw_laplacian_matches(a, b, sigma, thresh, window_size, nms_window);
    save_image(matches, "output/laplacian_matches");

    // Visualizza inliers e outliers
    Image inliers = draw_laplacian_inliers(a, b, sigma, thresh, window_size, nms_window,
                                           inlier_thresh, ransac_iters, cutoff);
    save_image(inliers, "output/laplacian_inliers");

    // Crea il panorama
    Image pano = panorama_image_laplacian(a, b, sigma, thresh, window_size, nms_window,
                                          inlier_thresh, ransac_iters, cutoff, 0.5);
    save_image(pano, "output/laplacian_panorama");

    return 0;
}
