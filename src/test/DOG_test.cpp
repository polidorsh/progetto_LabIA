#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../DOG.cpp"

using namespace std;

int main(){
    Image a = load_image("pano/cse/1.jpg");
    Image b = load_image("pano/cse/2.jpg");
    save_image(a, "output/a");
    save_image(b, "output/b");


    // Parametri per il rilevatore DoG
    float sigma1 = 1.8;
    float sigma2 = 3.2;
    float thresh = 0.08;
    int window_size = 14;
    int nms_window = 20;

    // Parametri per RANSAC
    float inlier_thresh = 7;
    int ransac_iters = 600;
    int cutoff = 180;

    // Visualizza i keypoint
    Image dog_points = detect_and_draw_dog(a, sigma1, sigma2, thresh, window_size, nms_window);
    save_image(dog_points, "output/dog_keypoints");

    // Visualizza tutti i matches
    Image matches = find_and_draw_dog_matches(a, b, sigma1, sigma2, thresh, window_size, nms_window);
    save_image(matches, "output/dog_matches");

    // Visualizza inliers e outliers
    Image inliers = draw_dog_inliers(a, b, sigma1, sigma2, thresh, window_size, nms_window,
                                    inlier_thresh, ransac_iters, cutoff);
    save_image(inliers, "output/dog_inliers");

    // Crea il panorama
    Image pano = panorama_image_dog(a, b, sigma1, sigma2, thresh, window_size, nms_window,
                                    inlier_thresh, ransac_iters, cutoff, 0.5);
    save_image(pano, "output/dog_panorama");
}