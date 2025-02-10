#include "../image.h"
#include "../utils.h"
#include "../matrix.h"
#include <string>
#include "../DOG.cpp"

using namespace std;

//TESTIAMO IL RILEVATORE:
//Columbia 01 23 45 67 89 || sigma=1.5, thresh=0.05, window=11, nms=7
//Rainier  01 23 45       || 
//Field    32 45 67       || 
//Helens   01 23 45       || 
//Sun      01 23          || 
//Wall     12             || 
//Cse      12 34 56 78    || 

int main() {
    // Carica le immagini
    Image a = load_image("pano/rainier/0.jpg");
    Image b = load_image("pano/rainier/1.jpg");

    float sigma = 1.5;
    float thresh = 0.05;
    int window = 11;
    int nms = 7;  
    
    float inlier_thresh = 5;
    int iters = 10000;
    int cutoff = 150;
    float acoeff = 0.5;

    Image dog_points_A = detect_and_draw_dog(a, sigma, thresh, window, nms);
    save_image(dog_points_A, "output/dog_keypoints_A");

    Image dog_points_B = detect_and_draw_dog(a, sigma, thresh, window, nms);
    save_image(dog_points_B, "output/dog_keypoints_B");

    Image dog_matches = find_and_draw_dog_matches(a, b, sigma, thresh, window, nms);
    save_image(dog_matches, "output/dog_matches");

    Image dog_inliers = draw_dog_inliers(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff);
    save_image(dog_inliers, "output/dog_inliers");

    Image dog_panorama = panorama_image_dog(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff, acoeff);
    save_image(dog_panorama, "output/dog_panorama");

    return 0;
}