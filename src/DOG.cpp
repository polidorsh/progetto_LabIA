#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>

#include "image.h"

using namespace std;

Descriptor describe_index(const Image& im, int x, int y, int w){
    Descriptor d;
    d.p={(double)x,(double)y};
    d.data.reserve(w*w*im.c);
    
    for(int c=0; c<im.c; c++){
        float cval = im.clamped_pixel(x,y,c);
        for(int dx=-w/2; dx<=w/2; dx++)
            for(int dy=-w/2; dy<=w/2; dy++)
                d.data.push_back(im.clamped_pixel(x+dx,y+dy,c)-cval);
    }
    return d;
}

struct OctaveLevel {
    Image gaussian;  
    Image dog;      
    float sigma;    
};

vector<OctaveLevel> create_gaussian_pyramid(const Image& im, float initial_sigma, int scales_per_octave, float k = sqrt(2)) {
    vector<OctaveLevel> pyramid;
    
    OctaveLevel first;
    first.sigma = initial_sigma;
    first.gaussian = smooth_image(im, initial_sigma);
    pyramid.push_back(first);
    
    for(int i = 1; i < scales_per_octave + 2; ++i) {
        OctaveLevel level;
        level.sigma = initial_sigma * pow(k, i);
        level.gaussian = smooth_image(im, level.sigma);
        pyramid.push_back(level);
    }
    
    for(size_t i = 0; i < pyramid.size() - 1; ++i) {
        Image dog(pyramid[i].gaussian.w, pyramid[i].gaussian.h, 1);
        
        for(int y = 0; y < dog.h; ++y) {
            for(int x = 0; x < dog.w; ++x) {
                dog(x,y,0) = pyramid[i+1].gaussian(x,y,0) - pyramid[i].gaussian(x,y,0);
            }
        }
        pyramid[i].dog = dog;
    }
    
    return pyramid;
}

vector<Point> find_local_extrema(const vector<OctaveLevel>& pyramid, float contrast_threshold, int nms_window) {
    vector<Point> keypoints;
    
    for(size_t i = 1; i < pyramid.size() - 2; ++i) {
        const Image& curr_dog = pyramid[i].dog;
        const Image& prev_dog = pyramid[i-1].dog;
        const Image& next_dog = pyramid[i+1].dog;
        
        Image response(curr_dog.w, curr_dog.h, 1);
        
        for(int y = 1; y < curr_dog.h - 1; ++y) {
            for(int x = 1; x < curr_dog.w - 1; ++x) {
                float val = curr_dog(x,y,0);
                
                if(fabs(val) < contrast_threshold) continue;
                
                bool is_max = true;
                bool is_min = true;
                
                for(int dy = -1; dy <= 1 && (is_max || is_min); ++dy) {
                    for(int dx = -1; dx <= 1 && (is_max || is_min); ++dx) {
                        if(dx != 0 || dy != 0) {
                            float neighbor = curr_dog(x+dx, y+dy, 0);
                            if(neighbor >= val) is_max = false;
                            if(neighbor <= val) is_min = false;
                        }
                        
                        float prev = prev_dog(x+dx, y+dy, 0);
                        if(prev >= val) is_max = false;
                        if(prev <= val) is_min = false;
                        
                        float next = next_dog(x+dx, y+dy, 0);
                        if(next >= val) is_max = false;
                        if(next <= val) is_min = false;
                    }
                }
                
                if(is_max || is_min) {
                    response(x,y,0) = fabs(val);
                }
            }
        }
        
        Image nms = nms_image(response, nms_window);
        
        for(int y = 0; y < nms.h; ++y) {
            for(int x = 0; x < nms.w; ++x) {
                if(nms(x,y,0) > 0) {  
                    Point p = {(double)x, (double)y};
                    keypoints.push_back(p);
                }
            }
        }
    }
    
    return keypoints;
}

vector<Descriptor> detect_keypoints_dog(const Image& im, float initial_sigma, 
                                      int num_octaves, int scales_per_octave,
                                      float contrast_threshold,
                                      int nms_window,
                                      int descriptor_window) {
    vector<Descriptor> all_descriptors;
    Image current = im;
    
    for(int o = 0; o < num_octaves; ++o) {
        vector<OctaveLevel> pyramid = create_gaussian_pyramid(current, initial_sigma, 
                                                            scales_per_octave);
        
        vector<Point> keypoints = find_local_extrema(pyramid, contrast_threshold, nms_window);
        
        for(const auto& p : keypoints) {
            Descriptor d = describe_index(current, p.x, p.y, descriptor_window);
            
            d.p.x *= pow(2, o);
            d.p.y *= pow(2, o);
            
            all_descriptors.push_back(d);
        }
        
        if(o < num_octaves - 1) {
            current = bilinear_resize(current, current.w/2, current.h/2);
        }
    }
    
    return all_descriptors;
}

Image detect_and_draw_keypoints_dog(const Image& im, float initial_sigma ,
                                  int num_octaves , int scales_per_octave ,
                                  float contrast_threshold ,
                                  int nms_window ,
                                  int descriptor_window ) {
    TIME(1);
    vector<Descriptor> descriptors = detect_keypoints_dog(im, initial_sigma, 
                                                        num_octaves, scales_per_octave,
                                                        contrast_threshold, nms_window,
                                                        descriptor_window);
    printf("Numero di Descrittori: %ld\n", descriptors.size());
    return mark_corners(im, descriptors);
}



Image find_and_draw_dog_matches(const Image& a, const Image& b,
                                float initial_sigma, int num_octaves, int scales_per_octave,
                                float contrast_threshold, int nms_window, int descriptor_window) {
    TIME(1);
    vector<Descriptor> ad = detect_keypoints_dog(a, initial_sigma, num_octaves, scales_per_octave,
                                                  contrast_threshold, nms_window, descriptor_window);
    vector<Descriptor> bd = detect_keypoints_dog(b, initial_sigma, num_octaves, scales_per_octave,
                                                  contrast_threshold, nms_window, descriptor_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    printf("Numero di Match: %ld\n", m.size());
    
    Image A = mark_corners(a, ad);
    Image B = mark_corners(b, bd);
    Image lines = draw_matches(A, B, m, {});
    return lines;
}

Image draw_dog_inliers(const Image& a, const Image& b,
                       float initial_sigma, int num_octaves, int scales_per_octave,
                       float contrast_threshold, int nms_window, int descriptor_window,
                       float inlier_thresh, int iters, int cutoff) {
    TIME(1);
    vector<Descriptor> ad = detect_keypoints_dog(a, initial_sigma, num_octaves, scales_per_octave,
                                                  contrast_threshold, nms_window, descriptor_window);
    vector<Descriptor> bd = detect_keypoints_dog(b, initial_sigma, num_octaves, scales_per_octave,
                                                  contrast_threshold, nms_window, descriptor_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return draw_inliers(a, b, Hba, m, inlier_thresh);
}

Image panorama_image_dog(const Image& a, const Image& b,
                         float initial_sigma, int num_octaves, int scales_per_octave,
                         float contrast_threshold, int nms_window, int descriptor_window,
                         float inlier_thresh, int iters, int cutoff, float acoeff) {
    TIME(1);
    vector<Descriptor> ad = detect_keypoints_dog(a, initial_sigma, num_octaves, scales_per_octave,
                                                  contrast_threshold, nms_window, descriptor_window);
    vector<Descriptor> bd = detect_keypoints_dog(b, initial_sigma, num_octaves, scales_per_octave,
                                                  contrast_threshold, nms_window, descriptor_window);
    
    vector<Match> m = match_descriptors(ad, bd);
    Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
    return combine_images(a, b, Hba, acoeff);
}
