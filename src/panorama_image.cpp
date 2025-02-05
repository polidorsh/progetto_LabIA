#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>

#include "image.h"
#include "matrix.h"

#include <set>
#include <thread>

using namespace std;

// returns: result of comparison, 0 if same, 1 if a > b, -1 if a < b.
int match_compare(const void *a, const void *b){
  Match *ra = (Match *)a;
  Match *rb = (Match *)b;
  if (ra->distance < rb->distance) return -1;
  else if (ra->distance > rb->distance) return  1;
  else return 0;
}


// returns: image with both a and b side-by-side.
Image both_images(const Image& a, const Image& b){
  assert(a.c==b.c);
  Image both(a.w + b.w, a.h > b.h ? a.h : b.h, a.c);
  
  for(int k = 0; k < both.c; ++k)
    for(int j = 0; j < a.h; ++j)
      for(int i = 0; i < a.w; ++i)
        both(i, j, k) = a(i, j, k);
  
  for(int k = 0; k < both.c; ++k)
    for(int j = 0; j < b.h; ++j)
      for(int i = 0; i < b.w; ++i)
        both(i+a.w, j, k) = b(i, j, k);
  return both;
}

// returns: image with matches drawn between a and b on same canvas.
Image draw_matches(const Image& a, const Image& b, const vector<Match>& matches, const vector<Match>& inliers){
  Image both = both_images(a, b);
  
  for(int i = 0; i < (int)matches.size(); ++i)
    {
    int bx = matches[i].a->p.x; 
    int ex = matches[i].b->p.x; 
    int by = matches[i].a->p.y;
    int ey = matches[i].b->p.y;
    for(int j = bx; j < ex + a.w; ++j)
      {
      int r = (float)(j-bx)/(ex+a.w - bx)*(ey - by) + by;
      both.set_pixel(j, r, 0, 1);
      both.set_pixel(j, r, 1, 0);
      both.set_pixel(j, r, 2, 0);
      }
    }
  for(int i = 0; i < (int)inliers.size(); ++i)
    {
    int bx = inliers[i].a->p.x; 
    int ex = inliers[i].b->p.x; 
    int by = inliers[i].a->p.y;
    int ey = inliers[i].b->p.y;
    for(int j = bx; j < ex + a.w; ++j)
      {
      int r = (float)(j-bx)/(ex+a.w - bx)*(ey - by) + by;
      both.set_pixel(j, r, 0, 0);
      both.set_pixel(j, r, 1, 1);
      both.set_pixel(j, r, 2, 0);
      }
    }
  return both;
}

// Draw the matches with inliers in green between two images.
Image draw_inliers(const Image& a, const Image& b, const Matrix& H, const vector<Match>& m, float thresh){
  vector<Match> inliers = model_inliers(H, m, thresh);
  Image lines = draw_matches(a, b, m, inliers);
  printf("Numero di Inliers: %ld\n", inliers.size());
  return lines;
}

// Find corners, match them, and draw them between two images.
Image find_and_draw_matches(const Image& a, const Image& b, float sigma, float thresh, int window, int nms, int corner_method){
  vector<Descriptor> ad= harris_corner_detector(a, sigma, thresh, window, nms, corner_method);
  vector<Descriptor> bd= harris_corner_detector(b, sigma, thresh, window, nms, corner_method);
  vector<Match> m = match_descriptors(ad, bd);
  
  
  Image A=mark_corners(a, ad);
  Image B=mark_corners(b, bd);
  Image lines = draw_matches(A, B, m, {});
  
  return lines;
}

// returns: l1 distance between arrays (sum of absolute differences).
float l1_distance(const vector<float>& a,const vector<float>& b){
  assert(a.size()==b.size() && "Arrays must have same size\n");
  float sum=0;
  for(int i=0; i<a.size(); i++){
    sum+=fabs(a[i]-b[i]);
  }  
  return sum;
}


// returns: best matches found. For each element in a[] find the index of best match in b[]
vector<int> match_descriptors_a2b(const vector<Descriptor>& a, const vector<Descriptor>& b){
  vector<int> ind;
  for(int j=0;j<(int)a.size();j++){
    int bind = -1; // <- find the best match (-1: no match)
    float best_distance=1e10f;  // <- best distance
    for(int k=0; k<b.size(); k++){
      float dist=l1_distance(a[j].data, b[k].data);
      if(dist<=best_distance){
        best_distance=dist;
        bind=k;
      }
    }
    ind.push_back(bind);
  }
  return ind;
  
}


// returns: best matches found. each descriptor in a should match with at most one other descriptor in b.
vector<Match> match_descriptors(const vector<Descriptor>& a, const vector<Descriptor>& b){
  if(a.size()==0 || b.size()==0)return {};
  vector<Match> m;
  vector<int> match_a2b=match_descriptors_a2b(a,b);
  vector<int> match_b2a=match_descriptors_a2b(b,a);
  for(int i=0; i<a.size();i++){
    int mb=match_a2b[i];
    if(match_b2a[mb]==i){
      m.push_back(Match(&a[i],&b[mb],l1_distance(a[i].data, b[mb].data)));
    }
  }
  return m;
}


// returns: point projected using the homography.
Point project_point(const Matrix& H, const Point& p){
  Point pp(0,0);
  double div=H(2,0)*p.x+H(2,1)*p.y+1;
  pp.x=(H(0,0)*p.x+H(0,1)*p.y+H(0,2))/div;
  pp.y=(H(1,0)*p.x+H(1,1)*p.y+H(1,2))/div;
  return pp;
}

// returns: L2 distance between them.
double point_distance(const Point& p, const Point& q){
  double dist=sqrt(pow((p.x-q.x),2)+pow((p.y-q.y),2));
  return dist;
}

// returns: inliers whose projected point falls within thresh of their match in the other image.
vector<Match> model_inliers(const Matrix& H, const vector<Match>& m, float thresh){
  vector<Match> inliers;
  for(int i=0; i<m.size(); i++){
    Point pp=project_point(H,m[i].a->p);
    if(point_distance(pp,m[i].b->p)<thresh)
      inliers.push_back(m[i]);
  }
  return inliers;
}

// Randomly shuffle matches for RANSAC.
// vector<Match>& m: matches to shuffle in place.
void randomize_matches(vector<Match>& m){
  for(int i=m.size()-1; i>0; i--){
    int j=rand()%(i+1);
    swap(m[i],m[j]);
  }
}


// returns: matrix representing homography H that maps image a to image b.
Matrix compute_homography_ba(const vector<Match>& matches){
  if(matches.size()<4)printf("Need at least 4 points for homography! %zu supplied\n",matches.size());
  if(matches.size()<4)return Matrix::identity(3,3);
  
  Matrix M(matches.size()*2,8);
  Matrix b(matches.size()*2);
  
  for(int i = 0; i < (int)matches.size(); ++i){
    double mx = matches[i].a->p.x;
    double my = matches[i].a->p.y;
    
    double nx = matches[i].b->p.x;
    double ny = matches[i].b->p.y;
    
    M(i*2, 0)=mx;
    M(i*2, 1)=my;
    M(i*2, 2)=1;
    M(i*2, 3)=0;
    M(i*2, 4)=0;
    M(i*2, 5)=0;
    M(i*2, 6)=-nx*mx;
    M(i*2, 7)=-nx*my;

    M(i*2+1, 0)=0;
    M(i*2+1, 1)=0;
    M(i*2+1, 2)=0;
    M(i*2+1, 3)=mx;
    M(i*2+1, 4)=my;
    M(i*2+1, 5)=1;
    M(i*2+1, 6)=-ny*mx;
    M(i*2+1, 7)=-ny*my;

    b(i*2, 0)=nx;
    b(i*2+1,0)=ny;
  }
  
  Matrix a = solve_system(M, b);
  Matrix Hba(3, 3);

  Hba(0,0)=a(0,0);
  Hba(0,1)=a(1,0);
  Hba(0,2)=a(2,0);

  Hba(1,0)=a(3,0);
  Hba(1,1)=a(4,0);
  Hba(1,2)=a(5,0);

  Hba(2,0)=a(6,0);
  Hba(2,1)=a(7,0);
  Hba(2,2)=1;
  
  return Hba;
}


// returns: matrix representing most common homography between matches.
Matrix RANSAC(vector<Match> m, float thresh, int k, int cutoff){
  if(m.size()<4)
    return Matrix::identity(3,3);
  
  Matrix Hba = Matrix::translation_homography(256, 0);
  vector<Match> best_inliers;

  for(int i=0; i<k; i++){
    randomize_matches(m);
    vector<Match> sample;
    sample.assign(m.begin(),m.begin()+4);
    Matrix H=compute_homography_ba(sample);
    vector<Match> inliers=model_inliers(H,m,thresh);
    if(inliers.size()>best_inliers.size()){
      sample.insert(sample.end(),inliers.begin(),inliers.end());
      H=compute_homography_ba(sample);
      Hba=H;
      best_inliers=inliers;
    }
    if(best_inliers.size()>cutoff)break;
  }
  return Hba;
}


Image trim_image(const Image& a)
  {
  int minx=a.w-1;
  int maxx=0;
  int miny=a.h-1;
  int maxy=0;
  
  for(int q3=0;q3<a.c;q3++)for(int q2=0;q2<a.h;q2++)for(int q1=0;q1<a.w;q1++)if(a(q1,q2,q3))
    {
    minx=min(minx,q1);
    maxx=max(maxx,q1);
    miny=min(miny,q2);
    maxy=max(maxy,q2);
    }
  
  if(maxx<minx || maxy<miny)return a;
  
  Image b(maxx-minx+1,maxy-miny+1,a.c);
  
  for(int q3=0;q3<a.c;q3++)for(int q2=miny;q2<=maxy;q2++)for(int q1=minx;q1<=maxx;q1++)
    b(q1-minx,q2-miny,q3)=a(q1,q2,q3);
  
  return b;
  }


// returns: combined image stitched together.
Image combine_images(const Image& a, const Image& b, const Matrix& Hba, float ablendcoeff){
  Matrix Hinv=Hba.inverse();
  
  // Project the corners of image b into image a coordinates.
  Point c1 = project_point(Hinv, Point(0,0));
  Point c2 = project_point(Hinv, Point(b.w-1, 0));
  Point c3 = project_point(Hinv, Point(0, b.h-1));
  Point c4 = project_point(Hinv, Point(b.w-1, b.h-1));
  
  // Find top left and bottom right corners of image b warped into image a.
  Point topleft, botright;
  botright.x = max(c1.x, max(c2.x, max(c3.x, c4.x)));
  botright.y = max(c1.y, max(c2.y, max(c3.y, c4.y)));
  topleft.x = min(c1.x, min(c2.x, min(c3.x, c4.x)));
  topleft.y = min(c1.y, min(c2.y, min(c3.y, c4.y)));
  
  // Find how big our new image should be and the offsets from image a.
  int dx = min(0, (int)topleft.x);
  int dy = min(0, (int)topleft.y);
  int w = max(a.w, (int)botright.x) - dx;
  int h = max(a.h, (int)botright.y) - dy;
    
  // Can disable this if you are making very big panoramas.
  // Usually this means there was an error in calculating H.
  if(w > 15000 || h > 4000)
    {
    printf("Can't make such big panorama :/ (%d %d)\n",w,h);
    return Image(100,100,1);
    }
  
  Image c(w, h, a.c);
  
  // Paste image a into the new image offset by dx and dy.
  for(int k = 0; k < a.c; ++k)
    for(int j = 0; j < a.h; ++j)
      for(int i = 0; i < a.w; ++i){
        c(i - dx, j - dy, k) = a(i, j, k);
      }
  for (int j = 0; j < h; ++j)
      for (int i = 0; i < w; ++i){
          if (!c.is_nonempty_patch(i, j)) {

              Point projected = project_point(Hba, Point(i + dx, j + dy));
              if (projected.x >= 0 && projected.y >= 0 && projected.x < b.w && projected.y < b.h) {
                  for (int k = 0; k < b.c; k++) {
                      float prev = c(i, j, k);
                      float pixel = b.pixel_bilinear(projected.x, projected.y, k);

                      if (prev > 0) {
                          c(i, j, k) = ablendcoeff * c(i, j, k) + (1 - ablendcoeff) * pixel;
                      }
                      else c(i, j, k) = pixel;
                  }

              }
          }
      }
  return trim_image(c);
  }

// Create a panoramam between two images.
Image panorama_image(const Image& a, const Image& b, float sigma, int corner_method, float thresh, int window, int nms, float inlier_thresh, int iters, int cutoff, float acoeff){
  // Calculate corners and descriptors
  vector<Descriptor> ad;
  vector<Descriptor> bd;
  
  // doing it multithreading...
  thread tha([&](){ad = harris_corner_detector(a, sigma, thresh, window, nms, corner_method);});
  thread thb([&](){bd = harris_corner_detector(b, sigma, thresh, window, nms, corner_method);});
  tha.join();
  thb.join();
  
  // Find matches
  vector<Match> m = match_descriptors(ad, bd);
  
  // Run RANSAC to find the homography
  Matrix Hba = RANSAC(m, inlier_thresh, iters, cutoff);
  
  // Stitch the images together with the homography
  return combine_images(a, b, Hba, acoeff);
}

// returns: image projected onto cylinder, then flattened.
Image cylindrical_project(const Image& im, float f){
    Image c(im.w, im.h, im.c);
    int xc = im.w / 2;
    int yc = im.h / 2;

    for (int j = 0; j < im.h; j++) {
        for (int i = 0; i < im.w; i++) {
            float theta = (i - xc) / f;
            float h = (j - yc) / f;
            float X = sin(theta);
            float Y = h;
            float Z = cos(theta);
            float new_x = ((f * X) / Z) + xc;
            float new_y = ((f * Y) / Z) + yc;
            int a = (int)new_x;
            int b = (int)new_y;
            if (a >= 0 && a < im.w && b >= 0 && b < im.h)
                for (int k = 0; k < im.c; k++) {
                    c(i, j, k) = im(a, b, k);
                }

        }
    }
    return c;
}

// returns: image projected onto cylinder, then flattened.
Image spherical_project(const Image& im, float f){
  double hfov=atan(im.w/(2*f));
  double vfov=atan(im.h/(2*f));
  Image c(im.w, im.h, im.c);
  int xc = im.w / 2;
  int yc = im.h / 2;
  for (int j = 0; j < im.h; j++) {
      for (int i = 0; i < im.w; i++) {
          float theta = (i - xc) / f;
          float h = (j - yc) / f;
          float X = sin(theta) * cos(h);
          float Y = sin(h);
          float Z = cos(theta) * cos(h);
          float new_x = f * (X / Z) + xc;
          float new_y = f * (Y / Z) + yc;
          int a = (int)new_x;
          int b = (int)new_y;
          if (a >= 0 && a < im.w && b >= 0 && b < im.h)
              for (int k = 0; k < im.c; k++) {
                  c(i, j, k) = im(a, b, k);
              }
      }
  }
  return c;
}
