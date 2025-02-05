#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <thread>

#include "image.h"
#include "matrix.h"

using namespace std;

Descriptor describe_index(const Image& im, int x, int y, int w){
  Descriptor d;
  d.p={(double)x,(double)y};
  d.data.reserve(w*w*im.c);
  
  for(int c=0;c<im.c;c++){
    float cval = im.clamped_pixel(x,y,c);
    for(int dx=-w/2;dx<=w/2;dx++)for(int dy=-w/2;dy<=w/2;dy++)
      d.data.push_back(im.clamped_pixel(x+dx,y+dy,c)-cval);
  }
  return d;
}

Image shi_tomasi_response(const Image& im, float sigma) {
    Image S = structure_matrix(im, sigma);
    
    Image R(S.w, S.h);
    for(int y = 0; y < S.h; y++) {
        for(int x = 0; x < S.w; x++) {
            float a = S(x,y,0);  // Ix^2
            float b = S(x,y,1);  // Iy^2
            float c = S(x,y,2);  // IxIy
            
            float lambda1 = (a + b - sqrt(pow(a - b, 2) + 4*c*c)) / 2;
            float lambda2 = (a + b + sqrt(pow(a - b, 2) + 4*c*c)) / 2;
            
            R(x,y,0) = min(lambda1, lambda2);
        }
    }
    return R;
}


vector<Descriptor> shi_tomasi_corner_detector(const Image& im, float sigma, float thresh, int window, int nms) {
    Image response = shi_tomasi_response(im, sigma);
    
    Image nms_response = nms_image(response, nms);
    
    return detect_corners(im, nms_response, thresh, window);
}

Image detect_and_draw_shi_tomasi_corners(const Image& im, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> d = shi_tomasi_corner_detector(im, sigma, thresh, window, nms);
    printf("Numero di Descrittori: %ld\n", d.size());
    return mark_corners(im, d);
}

Image find_and_draw_shi_tomasi_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms) {
    TIME(1);
    vector<Descriptor> ad = shi_tomasi_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = shi_tomasi_corner_detector(b, sigma, thresh, window, nms);
    
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());

    
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    
    return lines;
}

Image find_and_draw_shi_tomasi_inliers(const Image& a, const Image& b, float sigma, float thresh, int window, 
                            int nms, float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = shi_tomasi_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = shi_tomasi_corner_detector(b, sigma, thresh, window, nms);
    
    vector<Match> m = match_descriptors(ad, bd);
    
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_shi_tomasi(const Image& a, const Image& b, float sigma, float thresh, int window, 
                             int nms, float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = shi_tomasi_corner_detector(a, sigma, thresh, window, nms);
    vector<Descriptor> bd = shi_tomasi_corner_detector(b, sigma, thresh, window, nms);
    
    vector<Match> m = match_descriptors(ad, bd);
    
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    
    return combine_images(a, b, Hba, acoeff);
}
