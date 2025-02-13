#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>

#include "image.h"
//#include "matrix.h"

using namespace std;

// returns: Descriptor for that index.
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

void mark_spot(Image& im, const Point& p){
  int x = p.x;
  int y = p.y;
  
  for(int i = -9; i < 10; ++i){
    im.set_pixel(x+i, y, 0, 1);
    im.set_pixel(x, y+i, 0, 1);
    im.set_pixel(x+i, y, 1, 0);
    im.set_pixel(x, y+i, 1, 0);
    im.set_pixel(x+i, y, 2, 1);
    im.set_pixel(x, y+i, 2, 1);
  }
}

Image mark_corners(const Image& im, const vector<Descriptor>& d){
  Image im2=im;
  for(auto&e1:d)mark_spot(im2,e1.p);
  return im2;
}


// returns: single row Image of the filter.
Image make_1d_gaussian(float sigma){
  int w=ceil(sigma*6);
  if(!(w%2))w++;
  Image lin(w,1,1); 
  for(int x=0; x<lin.w; x++){
    int rx=x-(w/2);
    float var=powf(sigma,2);
    float c=sqrtf(2*M_PI)*sigma;
    float p=-(powf(rx,2))/(2*var);
    float e=expf(p);
    float val=e/c;
    lin(x,0,0)=val;
  }
  lin.l1_normalize();
  return lin;
}


// returns: smoothed Image.
Image smooth_image(const Image& im, float sigma){
  Image f=make_1d_gaussian(sigma);
  Image conv=convolve_image(im,f,true);
  swap(f.h, f.w);
  conv=convolve_image(conv,f,true);
  return conv;
}


// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2, third channel is IxIy.
Image structure_matrix(const Image& im2, float sigma){
  assert((im2.c==1 || im2.c==3));
  Image im;
  if(im2.c==1)im=im2;
  else im=rgb_to_grayscale(im2);
  
  Image S(im.w, im.h, 3);
  Image fx=make_gx_filter();
  Image fy=make_gy_filter();
  Image Ix=convolve_image(im,fx,true);
  Image Iy=convolve_image(im,fy,true);

  for(int y=0; y<im.h; y++){
    for(int x=0; x<im.w; x++){
      S(x,y,0)=pow(Ix(x,y,0),2);
      S(x,y,1)=pow(Iy(x,y,0),2);
      S(x,y,2)=Ix(x,y,0)*Iy(x,y,0);
    }
  }

  Image f=make_gaussian_filter(sigma);
  S=convolve_image(S,f,true);

  return S;
}


// returns: a response map of cornerness calculations.
Image cornerness_response(const Image& S, int method){
  Image R(S.w, S.h);
  for(int y=0; y<S.h; y++){
    for(int x=0; x<S.w; x++){
      float det=S(x,y,0)*S(x,y,1)-S(x,y,2)*S(x,y,2);
      float tr=S(x,y,0)+S(x,y,1);
      if(!method)R(x,y,0)=det/tr;
      else R(x,y,0)=(tr-(sqrtf(powf(tr,2)-4*det)))/2;
    }
  }
  return R;
}


// returns: Image with only local-maxima responses within w pixels.
Image nms_image(const Image& im, int w){
  Image r=im;

  for(int y=0; y<im.h; y++){
    for(int x=0; x<im.w; x++){
      for(int ny=y-w; ny<=y+w; ny++){
        for(int nx=x-w; nx<=x+w; nx++){
          if(im.clamped_pixel(nx,ny,0)>im(x,y,0))
            r(x,y,0)=-0.00001;
        }
      }
    }
  }  
  return r;
}


// returns: vector of descriptors of the corners in the image.
vector<Descriptor> detect_corners(const Image& im, const Image& nms, float thresh, int window){
  vector<Descriptor> d;
  for(int y=0; y<im.h; y++){
    for(int x=0; x<im.w; x++){
      if(nms(x,y,0)>thresh)
        d.push_back(describe_index(im,x,y,window));
    }
  }
  return d;
}


// Perform harris corner detection and extract features from the corners.
vector<Descriptor> harris_corner_detector(const Image& im, float sigma, float thresh, int window, int nms, int corner_method){
  Image S = structure_matrix(im, sigma);
  Image R = cornerness_response(S,corner_method);
  Image Rnms = nms_image(R, nms);
  return detect_corners(im, Rnms, thresh, window);
}

// Find and draw corners on an image.
Image detect_and_draw_corners(const Image& im, float sigma, float thresh, int window, int nms, int corner_method){
  vector<Descriptor> d = harris_corner_detector(im, sigma, thresh, window, nms, corner_method);
  return mark_corners(im, d);
}
