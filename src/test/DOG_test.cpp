#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../DOG.cpp"

using namespace std;

int main(){
Image a = load_image("pano/rainier/0.jpg");
Image b = load_image("pano/rainier/2.jpg");
save_image(a, "output/a.jpg");
save_image(b, "output/b.jpg");


// Parametri per il rilevatore DoG
float sigma1 = 1.0;
float sigma2 = 2.0;
float thresh = 0.03;
int window_size = 5;
int nms_window = 3;

// Parametri per RANSAC
float inlier_thresh = 5.0;
int ransac_iters = 1000;
int cutoff = 50;

// Visualizza i keypoint
Image dog_points = detect_and_draw_dog(a, sigma1, sigma2, thresh, window_size, nms_window);
save_image(dog_points, "output/dog_keypoints.jpg");

// Visualizza tutti i matches
Image matches = find_and_draw_dog_matches(a, b, sigma1, sigma2, thresh, window_size, nms_window);
save_image(matches, "output/dog_matches.jpg");

// Visualizza inliers e outliers
Image inliers = draw_dog_inliers(a, b, sigma1, sigma2, thresh, window_size, nms_window,
                                inlier_thresh, ransac_iters, cutoff);
save_image(inliers, "output/dog_inliers.jpg");

// Crea il panorama
Image pano = panorama_image_dog(a, b, sigma1, sigma2, thresh, window_size, nms_window,
                               inlier_thresh, ransac_iters, cutoff, 0.5);
save_image(pano, "output/dog_panorama.jpg");
}