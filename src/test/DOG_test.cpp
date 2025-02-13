#include "../image.h"
#include "../utils.h"
#include "../matrix.h"
#include <string>
#include "../DOG.cpp"

using namespace std;

//TESTIAMO IL RILEVATORE:
//columbia 01 23 45 67 89 || sigma=1.5, thresh=0.05, window=10, nms=10
//rainier  01 23 45       || 
//field    10 32 45 67    || 
//helens   01 23 45       || 
//sun      01 23          || 
//wall     01 12          || 
//cse      12 34 56 78    || 

int main() {

    Image a = load_image("pano/columbia/0.jpg");
    Image b = load_image("pano/columbia/1.jpg");

    float sigma = 1.5;
    float thresh = 0.05;
    int window = 10;
    int nms = 10;  
    
    float inlier_thresh = 5;
    int iters = 10000;
    int cutoff = 150;
    float acoeff = 0.5;

    Image dog_points_A = detect_and_draw_dog(a, sigma, thresh, window, nms);
    save_image(dog_points_A, "output/dog_keypoints_A");

    Image dog_points_B = detect_and_draw_dog(b, sigma, thresh, window, nms);
    save_image(dog_points_B, "output/dog_keypoints_B");

    Image dog_matches = find_and_draw_dog_matches(a, b, sigma, thresh, window, nms);
    save_image(dog_matches, "output/dog_matches");

    Image dog_inliers = draw_dog_inliers(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff);
    save_image(dog_inliers, "output/dog_inliers");

    Image dog_panorama = panorama_image_dog(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff, acoeff);
    save_image(dog_panorama, "output/dog_panorama");

    return 0;
}