#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include "image.h"

Image compute_hessian_matrix(const Image& im, int x, int y, float sigma) {
    Image fx = make_gx_filter();
    Image fy = make_gy_filter();

    Image Ix = convolve_image(im, fx, true);
    Image Iy = convolve_image(im, fy, true);
    Image Ixx = convolve_image(Ix, fx, true);
    Image Iyy = convolve_image(Iy, fy, true);
    Image Ixy = convolve_image(Ix, fy, true);

    Image H(2, 2, 1);
    H(0, 0, 0) = Ixx(x, y, 0);     
    H(1, 1, 0) = Iyy(x, y, 0);     
    H(0, 1, 0) = H(1, 0, 0) = Ixy(x, y, 0);  

    return H;
}

float forstner_interest_measure(const Image& S) {
    float det = S(0, 0, 0) * S(1, 1, 0) - S(0, 1, 0) * S(1, 0, 0);
    float tr = S(0, 0, 0) + S(1, 1, 0);
    
    float w = det / tr;
    return w;
}

vector<Descriptor> forstner_harris_hessian_detector(
    const Image& im, 
    float sigma = 1.0, 
    float thresh = 0.01, 
    int window = 5, 
    int nms = 3
) {
    Image S = structure_matrix(im, sigma);
    
    Image R(im.w, im.h, 1);
    for(int y = 0; y < im.h; y++) {
        for(int x = 0; x < im.w; x++) {
            Image localS(2, 2, 1);
            localS(0, 0, 0) = S(x, y, 0);
            localS(1, 1, 0) = S(x, y, 1);
            localS(0, 1, 0) = localS(1, 0, 0) = S(x, y, 2);
            
            R(x, y, 0) = forstner_interest_measure(localS);
        }
    }
    
    Image Rnms = nms_image(R, nms);
    
    return detect_corners(im, Rnms, thresh, window);
}

Image detect_and_draw_forstner_corners(const Image& im, float sigma, float thresh, int window, int nms){
    TIME(1);
    vector<Descriptor> d = forstner_harris_hessian_detector(im, sigma, thresh, window, nms);
    printf("Numero di Descrittori: %ld\n", d.size());
    return mark_corners(im, d);
}



Image find_and_draw_forstner_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> ad = forstner_harris_hessian_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = forstner_harris_hessian_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image find_and_draw_forstner_inliers(const Image& a, const Image& b, float sigma, float thresh, int window,
int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = forstner_harris_hessian_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = forstner_harris_hessian_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_forstner(const Image& a, const Image& b, float sigma, float thresh, int window,
int nms, float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = forstner_harris_hessian_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = forstner_harris_hessian_detector(b, sigma, thresh, window, nms);
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}