#include "../image.h"
#include "../utils.h"
#include "../matrix.h"
#include "../DOG.cpp"

#include <string>

using namespace std;

int main(){
    
    Image im = load_image("pano/rainier/0.jpg");
    Image dog = dog_detect_and_draw_keypoints(im, 4, 3, 1.6, 0.03);
    save_image(dog, "output/DOG");

    return 0;
}