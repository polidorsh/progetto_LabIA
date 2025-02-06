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
    
    float sigma = 1;
    float thresh = 0.05;
    int window = 15;
    int nms = 20;
    
    float inlier_thresh = 7;
    int ransac_iters = 10000;
    int cutoff = 180;
    float blend = 0.5;
    
    
    Image dog_points = detect_and_draw_dog_corners(a, sigma, thresh, window, nms);
    save_image(dog_points, "output/dog_keypoints");
    
    Image matches = find_and_draw_dog_matches(a, b, sigma, thresh, window, nms);
    save_image(matches, "output/dog_matches");
    
    Image inliers = draw_dog_inliers(a, b, sigma, thresh, window, nms, 
                                   inlier_thresh, ransac_iters, cutoff);
    save_image(inliers, "output/dog_inliers");
    
    Image pano = panorama_image_dog(a, b, sigma, thresh, window, nms,
                                  inlier_thresh, ransac_iters, cutoff, blend);
    save_image(pano, "output/dog_panorama");
    
    return 0;
}