#include "../image.h"
#include "../utils.h"
#include "../matrix.h"

#include <string>
#include "../ORI.cpp"

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
    int iters = 10000;         
    int cutoff = 180;        
    
    Image corners = oriented_corners(a, sigma, thresh, window, nms);
    save_image(corners, "output/oriented_corners");

    Image matches = oriented_matches(a, b, sigma, thresh, window, nms);
    save_image(matches, "output/oriented_matches");

    Image inliers = oriented_inliers(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff);
    save_image(inliers, "output/oriented_inliers");

    Image pano = oriented_panorama(a, b, sigma, thresh, window, nms, inlier_thresh, iters, cutoff);
    save_image(pano, "output/oriented_panorama");

    return 0;
}
