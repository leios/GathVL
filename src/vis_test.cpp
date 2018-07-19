#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <fftw3.h>

#include "../include/camera.h"
#include "../include/scene.h"

void update_gaussian(std::vector<vec>& fft_points,
                     std::vector<vec>& gaussian_points, int step,
                     int xrange, int yrange, int res){

    // FFT'ing gaussian curve for momentum-space image
    std::vector<fftw_complex> gaussian(res), fft_out(res);

    double deviation = 0.5 / ((double)step*0.5);
    for (size_t i = 0; i < gaussian.size(); ++i){
        double xpos = -5 + 10.0*(i / (double)gaussian.size());
        gaussian[i][0] = std::exp(-(xpos*xpos)/deviation);
        gaussian[i][1] = 0;
    }

    // Creating FFTW plans and such
    fftw_plan plan = fftw_plan_dft_1d(res, gaussian.data(), fft_out.data(),
                                      FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    // Finding renormalization factor
    double renorm_factor = 0;
    double max = 0;
    for (int i = 0; i < res; ++i){
        double magnitude = (fft_out[i][0]*fft_out[i][0]
                            +fft_out[i][1]*fft_out[i][1]);

        if (magnitude > max){
             max = magnitude;
        }
/*
        renorm_factor += (fft_out[i][0]*fft_out[i][0]
                         +fft_out[i][1]*fft_out[i][1]);
*/
    }

    for (int i = 0; i < res; ++i) {
        int j =(i + res /2) % res;
        double fft_yval = (fft_out[j][0]*fft_out[j][0]
                           +fft_out[j][1]*fft_out[j][1]) / max;
        fft_points[i] = {i * xrange/res, fft_yval * yrange};
        gaussian_points[i] = {i * xrange/res, gaussian[i][0]*yrange};
    }

}

void first_scene(camera& cam, scene& world) {
    int xrange = 1200;
    int yrange = 300;
    int res = 500;

    auto y_axis = std::make_shared<line>(vec{50, cam.size.y-50},
                                         vec{50, cam.size.y-50});
    auto x_axis = std::make_shared<line>(vec{50, cam.size.y-50},
                                         vec{50, cam.size.y-50});

    int center_loc = xrange/2 + 50;
    auto arrow_1 = std::make_shared<arrow>(vec{center_loc, 250}, 100, M_PI / 2);
    auto arrow_2 = std::make_shared<arrow>(vec{center_loc, 250}, 100, M_PI / 2);
    std::vector<vec> exp_points;

    vec origin = {50, cam.size.y-50};

    for (int i = 0; i < res; ++i) {
        double xpos = i/(double)res*20 - 10;
        exp_points.emplace_back(i * xrange/res,
                                std::exp(-(xpos*xpos)/5.0)*yrange);
    }

    auto exp_curve =
        std::make_shared<curve>(std::vector<vec>(), origin);

    x_axis->add_animator<vec_animator>(0, 50, &x_axis->end, x_axis->start,
                                       x_axis->start + vec(xrange, 0));
    y_axis->add_animator<vec_animator>(0, 50, &y_axis->end, y_axis->start,
                                       y_axis->start + vec(0, -yrange));

    exp_curve->add_animator<vector_animator<vec>>(0, 50, 0, exp_points,
                                                  &exp_curve->points);
    arrow_1->add_animator<vec_animator>(50,101, &arrow_1->location,
                                        vec{center_loc, 250},
                                        vec{center_loc-xrange/4, 250});
    arrow_2->add_animator<vec_animator>(50,101, &arrow_2->location,
                                        vec{center_loc, 250},
                                        vec{center_loc+xrange/4, 250});

    arrow_1->add_animator<vec_animator>(100,151, &arrow_1->location,
                                        vec{center_loc-xrange/4, 250},
                                        vec{center_loc, 250});
    arrow_2->add_animator<vec_animator>(100,151, &arrow_2->location,
                                        vec{center_loc+xrange/4, 250},
                                        vec{center_loc, 250});

    world.add_shape(y_axis, 0);
    world.add_shape(x_axis, 0);
    world.add_shape(exp_curve, 0);

    for (int i = 0; i < 151; ++i) {
        if (i == 50){
            world.add_shape(arrow_1, 0);
            world.add_shape(arrow_2, 0);
        }
        if (i >= 51 && i < 101){
            double scale = (101-i)/50.0;
            for (int j = 0; j < exp_curve->points.size(); ++j){
                double xpos = j/(double)res*20 - 10;
                exp_curve->points[j] = {j * xrange/res,
                    scale*std::exp(-(xpos*xpos)/5.0)*yrange
                    + (1 - scale)*pow(sin(2*M_PI*xpos/20),2)*yrange};

            }
        }
        if (i >= 101 && i < 151){
            for (int j = 0; j < exp_curve->points.size(); ++j){
                double xpos = j/(double)res*20 - 10;
                double scale = (151-i)/50.0;
                exp_curve->points[j] = {j * xrange/res,
                    (1 - scale)*std::exp(-(xpos*xpos)/5.0)*yrange
                    + scale*pow(sin(2*M_PI*xpos/20),2)*yrange};

            }

        }
            world.update(i);
            cam.encode_frame(world);
    }
}

void second_scene(camera& cam, scene& world) {
    world.clear();

    // Creating axis for plotting
    int xrange = 500;
    int yrange = 500;
    int res = 500;

    auto y_axis1 = std::make_shared<line>(vec{50, cam.size.y-50},
                                          vec{50, cam.size.y-50});
    auto x_axis1 = std::make_shared<line>(vec{50, cam.size.y-50},
                                          vec{50, cam.size.y-50});

    auto y_axis2 = std::make_shared<line>(vec{690, cam.size.y-50},
                                          vec{690, cam.size.y-50});
    auto x_axis2 = std::make_shared<line>(vec{690, cam.size.y-50},
                                          vec{690, cam.size.y-50});

    x_axis1->add_animator<vec_animator>(0, 50, &x_axis1->end, x_axis1->start,
                                        x_axis1->start + vec(xrange, 0));
    y_axis1->add_animator<vec_animator>(0, 50, &y_axis1->end, y_axis1->start,
                                        y_axis1->start + vec(0, -yrange));

    x_axis2->add_animator<vec_animator>(0, 50, &x_axis2->end, x_axis2->start,
                                        x_axis2->start + vec(xrange, 0));
    y_axis2->add_animator<vec_animator>(0, 50, &y_axis2->end, y_axis2->start,
                                        y_axis2->start + vec(0, -yrange));

    // FFT'ing gaussian curve for momentum-space image
    std::vector<fftw_complex> gaussian(res), fft_out(res);

    for (size_t i = 0; i < gaussian.size(); ++i){
        double xpos = -5 + 10.0*(i / (double)gaussian.size());
        gaussian[i][0] = std::exp(-(xpos*xpos)/.0005);
        gaussian[i][1] = 0;
    }

    // Creating FFTW plans and such
    fftw_plan plan = fftw_plan_dft_1d(res, gaussian.data(), fft_out.data(),
                                      FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    // Finding renormalization factor
    double renorm_factor = 0;
    double max = 0;
    for (int i = 0; i < res; ++i){
        double magnitude = (fft_out[i][0]*fft_out[i][0]
                            +fft_out[i][1]*fft_out[i][1]);

        if (magnitude > max){
             max = magnitude;
        }
    }

    // Adding two gaussian curves
    std::vector<vec> gaussian_points, fft_out_points;
    vec origin1 = {50, cam.size.y-50};
    vec origin2 = {690, cam.size.y-50};

    auto gaussian_curve =
        std::make_shared<curve>(std::vector<vec>(), origin1);
    auto fft_out_curve =
        std::make_shared<curve>(std::vector<vec>(), origin2);

    for (int i = 0; i < res; ++i) {
        int j =(i + res /2) % res;
        double fft_yval = (fft_out[j][0]*fft_out[j][0]
                           +fft_out[j][1]*fft_out[j][1]) / max;
        fft_out_points.emplace_back(i * xrange/res, fft_yval * yrange);
        gaussian_points.emplace_back(i * xrange/res, gaussian[i][0]*yrange);
    }

    
    gaussian_curve->add_animator<vector_animator<vec>>(0,50,0,gaussian_points,
                                                       &gaussian_curve->points);

    fft_out_curve->add_animator<vector_animator<vec>>(0,50,0,fft_out_points,
                                                       &fft_out_curve->points);
    world.add_shape(x_axis1, 0);
    world.add_shape(y_axis1, 0);
    world.add_shape(x_axis2, 0);
    world.add_shape(y_axis2, 0);
    world.add_shape(gaussian_curve, 0);
    world.add_shape(fft_out_curve, 0);

    for (int i = 0; i < 251; ++i) {
        if (i >= 51){
            update_gaussian(fft_out_curve->points, gaussian_curve->points,
                            i - 51, xrange, yrange, res);
        }
        world.update(i);
        cam.encode_frame(world);
    }

}

int main() {
    camera cam(vec{1280, 720});
    scene world = scene({1280, 720}, {0, 0, 0, 1});

    //cam.add_encoder<png_encoder>();
    cam.add_encoder<video_encoder>("/tmp/video.mp4", cam.size, 50);

    first_scene(cam, world);

    second_scene(cam, world);

    cam.clear_encoders();

    return 0;
}
