#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../ST.cpp"

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
    int ransac_iters = 10000;
    int cutoff = 180;

    // Visualizza i keypoint
    Image laplacian_points = detect_and_draw_shi_tomasi_corners(a, sigma, thresh, window_size, nms_window);
    save_image(laplacian_points, "output/shi_tomasi_keypoints");

    // Visualizza tutti i matches
    Image matches = find_and_draw_shi_tomasi_matches(a, b, sigma, thresh, window_size, nms_window);
    save_image(matches, "output/shi_tomasi_matches");

    // Visualizza inliers e outliers
    Image inliers = find_and_draw_shi_tomasi_inliers(a, b, sigma, thresh, window_size, nms_window,
                                           inlier_thresh, ransac_iters, cutoff);
    save_image(inliers, "output/shi_tomasi_inliers");

    // Crea il panorama
    Image pano = panorama_image_shi_tomasi(a, b, sigma, thresh, window_size, nms_window,
                                          inlier_thresh, ransac_iters, cutoff, 0.5);
    save_image(pano, "output/shi_tomasi_panorama");

    return 0;
}