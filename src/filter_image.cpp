#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"

#define M_PI 3.14159265358979323846

void l1_normalize(Image& im)
{
    for (int c = 0; c < im.c; c ++) {
        float sum = 0.0;
        for (int y = 0; y < im.h; y ++) {
            for (int x = 0; x < im.w; x ++) {
                sum += im(x,y,c);
            }
        }

        for (int y = 0; y < im.h; y ++) {
            for (int x = 0; x < im.w; x ++) {
                im(x,y,c) = (float) im(x,y,c) / sum;
            }
        }
    }
    return;
}

void l2_normalize(Image& im)
{
    for (int c = 0; c < im.c; c ++) {
        float sum = 0.0;
        for (int y = 0; y < im.h; y ++) {
            for (int x = 0; x < im.w; x ++) {
                sum += im(x,y,c);
            }
        }

        for (int y = 0; y < im.h; y ++) {
            for (int x = 0; x < im.w; x ++) {
                im(x,y,c) = (float) im(x,y,c) / sum;
            }
        }
    }
    return;
}


// returns the filter Image of size WxW
Image make_box_filter(int w)
{
    assert(w%2); // w needs to be odd

    Image box(w,w,1);
    float val = 1 / (float) (w * w);

    for (int y = 0; y < box.h; y++) {
        for (int x = 0; x < box.w; x ++) {
            box(x,y,0) = val;
        }
    }

    return box;
}

// returns the convolved image
Image convolve_image(const Image& im, const Image& filter, bool preserve)
{
    assert(filter.c==1);
    Image conv(im.w, im.h, im.c);

    for (int c = 0; c < im.c; c ++) {
        for (int y = 0; y < im.h; y ++) {
            for (int x = 0; x < im.w; x ++) {

                // The starting coordinates in image
                int sx = x - (filter.w / 2);
                int sy = y - (filter.h / 2);

                float sum = 0.0;

                for (int fy = 0; fy < filter.h; fy ++) {
                    for (int fx = 0; fx < filter.w; fx ++) {
                        sum += filter(fx, fy, 0) * im.clamped_pixel(sx + fx, sy + fy, c);
                    }
                }

                conv(x,y,c) = sum;
            }
        }
    }


    if (!preserve) {
        for (int y = 0; y < conv.h; y ++) {
            for (int x = 0; x < conv.w; x ++) {
                float sum = 0;
                for (int c = 0; c < conv.c; c ++) {
                    sum += conv(x,y,c);
                }
                conv(x,y,0) = sum;
            }
        }
        conv.c = 1;
    }
    return conv;
}

// returns basic 3x3 high-pass filter
Image make_highpass_filter()
{
    Image f (3,3,1);

    f(0,0,0) = 0;
    f(1,0,0) = -1;
    f(2,0,0) = 0;
    f(0,1,0) = -1;
    f(1,1,0) = 4;
    f(2,1,0) = -1;
    f(0,2,0) = 0;
    f(1,2,0) = -1;
    f(2,2,0) = 0;

    return f;
}

// returns basic 3x3 sharpen filter
Image make_sharpen_filter()
{
    Image f (3,3,1);

    f(0,0,0) = 0;
    f(1,0,0) = -1;
    f(2,0,0) = 0;
    f(0,1,0) = -1;
    f(1,1,0) = 5;
    f(2,1,0) = -1;
    f(0,2,0) = 0;
    f(1,2,0) = -1;
    f(2,2,0) = 0;

    return f;
}

// returns basic 3x3 emboss filter
Image make_emboss_filter()
{
    Image f (3,3,1);

    f(0,0,0) = -2;
    f(1,0,0) = -1;
    f(2,0,0) = 0;
    f(0,1,0) = -1;
    f(1,1,0) = 1;
    f(2,1,0) = 1;
    f(0,2,0) = 0;
    f(1,2,0) = 1;
    f(2,2,0) = 2;

    return f;
}


// returns basic gaussian filter
Image make_gaussian_filter(float sigma)
{
    // TODO: Implement the filter

    int w = ceil (sigma * 6);
    if (!(w % 2))
        w ++;

    Image f (w,w,1);

    for (int y = 0; y < f.h; y ++) {
        for (int x = 0; x < f.w; x ++) {

            int rx = x - (w/2);
            int ry = y - (w/2);

            float var = powf(sigma, 2);
            float c = 2 * M_PI * var;
            float p = -(powf(rx,2) + powf(ry,2)) / (2 * var);
            float e = expf(p);
            float val = e / c;
            f(x,y,0) = val;
        }
    }

    l2_normalize(f);

    return f;
}


// returns their sum
Image add_image(const Image& a, const Image& b)
{
    assert(a.w==b.w && a.h==b.h && a.c==b.c); // assure images are the same size

    Image res (a.w, a.h, a.c);
    for (int c = 0; c < res.c; c++) {
        for (int y = 0; y < res.h; y ++) {
            for (int x = 0; x < res.w; x ++) {
                res(x,y,c) = a(x,y,c) + b(x,y,c);
            }
        }
    }

    return res;
}


// returns their difference res=a-b
Image sub_image(const Image& a, const Image& b)
{
    assert(a.w==b.w && a.h==b.h && a.c==b.c); // assure images are the same size

    Image res (a.w, a.h, a.c);
    for (int c = 0; c < res.c; c++) {
        for (int y = 0; y < res.h; y ++) {
            for (int x = 0; x < res.w; x ++) {
                res(x,y,c) = a(x,y,c) - b(x,y,c);
            }
        }
    }

    return res;
}

// returns basic GX filter
Image make_gx_filter()
{
    Image f (3,3,1);

    f(0,0,0) = -1;
    f(1,0,0) = 0;
    f(2,0,0) = 1;
    f(0,1,0) = -2;
    f(1,1,0) = 0;
    f(2,1,0) = 2;
    f(0,2,0) = -1;
    f(1,2,0) = 0;
    f(2,2,0) = 1;

    return f;
}

// returns basic GY filter
Image make_gy_filter()
{
    Image f (3,3,1);

    f(0,0,0) = -1;
    f(1,0,0) = -2;
    f(2,0,0) = -1;
    f(0,1,0) = 0;
    f(1,1,0) = 0;
    f(2,1,0) = 0;
    f(0,2,0) = 1;
    f(1,2,0) = 2;
    f(2,2,0) = 1;

    return f;
}

void feature_normalize(Image& im)
{
    assert(im.w*im.h); // assure we have non-empty image

    for (int c = 0; c < im.c; c ++) {

        float min_val = im(0,0,c);
        float max_val = im(0,0,c);

        for (int y = 0; y < im.h; y ++) {
            for (int x = 0; x < im.w; x ++) {
                min_val = min (min_val, im(x,y,c));
                max_val = max (max_val, im(x,y,c));
            }
        }

        float range = max_val - min_val;

        for (int y = 0; y < im.h; y ++) {
            for (int x = 0; x < im.w; x++) {
                if (range) {
                    im(x,y,c) = (im(x,y,c) - min_val) / range;
                }
                else
                    im(x,y,c) = 0;
            }
        }
    }

    return;
}


// Normalizes features across all channels
void feature_normalize_total(Image& im)
{
    assert(im.w*im.h*im.c); // assure we have non-empty image

    int nc=im.c;
    im.c=1;im.w*=nc;

    feature_normalize(im);

    im.w/=nc;im.c=nc;

}


pair<Image,Image> sobel_image(const Image& im)
{

    Image fx = make_gx_filter();
    Image fy = make_gy_filter();

    Image Gx = convolve_image (im, fx, false);
    Image Gy = convolve_image (im, fy, false);


    Image G(im.w, im.h, 1);
    Image T(im.w, im.h, 1);


    for (int y = 0; y < im.h; y ++) {
        for (int x = 0; x < im.w; x ++) {
            G(x,y,0) = sqrtf ( pow(Gx(x,y,0), 2) + pow(Gy(x,y,0), 2));
            T(x,y,0) = atan2f( Gy(x,y,0) , Gx(x,y,0));
        }
    }


    return {G,T};
}

Image colorize_sobel(const Image& im)
{

    Image f = make_gaussian_filter(4);
    Image blur = convolve_image(im, f, true);
    blur.clamp();

    pair<Image, Image> sobel = sobel_image(blur);

    Image mag = sobel.first;
    Image theta = sobel.second;


    feature_normalize(mag);

    for (int y = 0; y < im.h; y ++) {
        for (int x = 0; x < im.w; x ++) {
            theta(x,y,0) = theta(x,y,0) / (2 * M_PI) + 0.5;
        }
    }


    Image hsv (im.w, im.h, 3);

    for (int y = 0; y < im.h; y ++) {
        for (int x = 0; x < im.w; x ++) {
            hsv(x,y,0) = theta(x,y,0);
            hsv(x,y,1) = mag(x,y,0);
            hsv(x,y,2) = mag(x,y,0);
        }
    }

    hsv_to_rgb(hsv);

    return hsv;
}



Image make_bilateral_filter (const Image &im, const Image &sgf, int cx, int cy, int cc, float sigma) {


    // Color gaussian filter
    Image cgf (sgf.w, sgf.h, 1);

    for (int y = 0; y < sgf.w; y ++) {
        for (int x = 0; x < sgf.w; x ++) {
            int ax = cx - sgf.w/2 + x;
            int ay = cy - sgf.w/2 + y;

            float diff = im.clamped_pixel(ax ,ay ,cc) - im.clamped_pixel(cx, cy, cc);

            float var = powf(sigma, 2);
            float c = 2 * M_PI * var;
            float p = - powf(diff,2) / (2 * var);
            float e = expf(p);
            float val = e / c;

            cgf(x,y,0) = val;

        }
    }


    Image bf (sgf.w, sgf.h, 1);

    // Multiply space gaussian by color gaussian
    for (int y = 0; y < bf.h; y ++) {
        for (int x = 0; x < bf.w; x ++) {
            bf (x,y,0) = sgf(x,y,0) * cgf(x,y,0);
        }
    }


    l1_normalize(bf);


    return bf;
}


Image bilateral_filter(const Image& im, float sigma1, float sigma2)
{

    Image gf = make_gaussian_filter(sigma1);

    Image res(im.w, im.h, im.c);


    for (int c = 0; c < res.c; c ++) {
        for (int y = 0; y < im.h; y ++) {
            for (int x = 0; x < im.w; x ++) {

                // Get bilateral filter
                Image bf = make_bilateral_filter(im, gf, x, y, c, sigma2);

                float sum = 0.0;
                // Convolve for pixel x,y,c
                for (int fy = 0; fy < gf.w; fy ++) {
                    for (int fx = 0; fx < gf.w; fx ++) {

                        int ax = x - bf.w/2 + fx;
                        int ay = y - bf.w/2 + fy;

                        sum += bf(fx,fy,0) * im.clamped_pixel( ax, ay, c);
                    }
                }

                res (x,y,c) = sum;

            }
        }
    }

    return res;
}



float *compute_histogram(const Image &im, int ch, int num_bins) {
    float *hist = (float *) malloc(sizeof(float) * num_bins);
    for (int i = 0; i < num_bins; ++i) {
        hist[i] = 0;
    }
    int bin_val;
    int N = im.w * im.h;
    float eps=1.0/(num_bins*1000);
    for (int x = 0; x < im.w; x++) {
        for (int y = 0; y < im.h; y++) {
            bin_val = (im(x,y,ch)-eps)*(float)num_bins;
            hist[bin_val]++;
        }
    }
    for (int i = 0; i < num_bins; i++) {
        hist[i] /= N; // Normalize histogram
    }
    return hist;
}

float *compute_CDF(float *hist, int num_bins) {
    float *cdf = (float *) malloc(sizeof(float) * num_bins);
    cdf[0] = hist[0];
    for (int i = 1; i < num_bins; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    return cdf;
}

Image histogram_equalization_hsv(const Image &im, int num_bins) {
    // Creazione di una copia dell'immagine e conversione da RGB a HSV
    Image new_im(im);
    float eps=1.0/(num_bins*1000);
    rgb_to_hsv(new_im);
    // Calcolo dell'istogramma del canale V (luminosità)
    float* hist = compute_histogram(new_im, 2, num_bins);
    // Calcolo della CDF e normalizzazione
    float* cdf = compute_CDF(hist, num_bins);
    for(int x=0; x<new_im.w; x++){
        for(int y=0; y<new_im.h; y++){
            unsigned int val=(unsigned int)((new_im(x,y,2)-eps)*num_bins);
            new_im(x,y,2)=cdf[val];
        }
    }
    // Conversione da HSV a RGB e pulizia della memoria
    hsv_to_rgb(new_im);
    delete hist;
    delete cdf;
    return new_im;
}


Image histogram_equalization_rgb(const Image &im, int num_bins) {
    Image new_im(im);
    for (int c = 0; c < im.c; ++c) {
        float *hist = compute_histogram(new_im, c, num_bins);
        float *cdf = compute_CDF(hist, num_bins);
        for (int x = 0; x < new_im.w; x++) {
            for (int y = 0; y < new_im.h; y++) {
                unsigned int val = (unsigned int)(new_im(x, y, c) * num_bins);
                val = std::min(std::max(val, 0U), (unsigned int)(num_bins - 1)); // Clamp
                new_im(x, y, c) = cdf[val];
            }
        }
        delete hist;
        delete cdf;
    }
    return new_im;
}






// HELPER MEMBER FXNS

void Image::feature_normalize(void) { ::feature_normalize(*this); }
void Image::feature_normalize_total(void) { ::feature_normalize_total(*this); }
void Image::l1_normalize(void) { ::l1_normalize(*this); }
void Image::l2_normalize(void) { ::l2_normalize(*this); }

Image operator-(const Image& a, const Image& b) { return sub_image(a,b); }
Image operator+(const Image& a, const Image& b) { return add_image(a,b); }
