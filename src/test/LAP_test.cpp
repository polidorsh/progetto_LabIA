#include "../image.h"
#include "../utils.h"
#include "../LAP.cpp"
#include <string>

using namespace std;

int main(int argc, char **argv) {

    Image im = load_image("pano/rainier/0.jpg");
    Image lap = lap_detect_and_draw_keypoints(im, 1.0, 4.0, 5, 0.10, 5);
    save_image(lap, "output/LAP");
    
    return 0;
}