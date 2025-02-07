#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../DOG.cpp"

using namespace std;

int main() {
    Image a = load_image("pano/cse/1.jpg");
    Image b = load_image("pano/cse/2.jpg");
    save_image(a, "output/a");
    save_image(b, "output/b");
    
    float sigma = 1.5;
    int num_octaves=4;
    float scales_per_octave=5;
    float thresh = 0.05;
    int window = 7;
    int nms = 7;
    
    float inlier_thresh = 5;
    int ransac_iters = 50000;
    int cutoff = 100;
    float blend = 0.5;
    
    Image dog_points = detect_and_draw_keypoints_dog(a, sigma, num_octaves, scales_per_octave, thresh, nms, window);
    save_image(dog_points, "output/dog_keypoints");

    Image matches = find_and_draw_dog_matches(a, b, sigma, num_octaves, scales_per_octave, thresh, nms, window);
    save_image(matches, "output/dog_matches");

    Image inliers = draw_dog_inliers(a, b, sigma, num_octaves, scales_per_octave, thresh, nms, window, inlier_thresh, ransac_iters, cutoff);
    save_image(inliers, "output/dog_inliers");

    Image pano = panorama_image_dog(a, b, sigma, num_octaves, scales_per_octave, thresh, 
                                        nms, window, inlier_thresh, ransac_iters, cutoff, 0.5);
    save_image(pano, "output/dog_panorama");


    return 0;
}