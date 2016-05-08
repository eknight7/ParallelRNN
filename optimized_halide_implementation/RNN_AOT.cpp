#include <stdio.h>
#include "Halide.h"
#include "CycleTimer.h"
#include "halide_image_io.h"
#include "init_h0.h"
#include "fprop_h_t.h"
#include "fprop_y_t.h"
#include "loss.h"
#include "init_dEdh_in_tp1.h"
#include "bprop_dEdy_in.h"
#include "bprop_dEdh_in.h"
#include "bprop_Ghy.h"
#include "bprop_Ghh.h"
#include "bprop_Gxh.h"
#include "grad_descent.h"

using namespace Halide;

#define T (10)
#define ALL_DIMS (1024)
#define NUM_INPUT (ALL_DIMS)
#define NUM_HIDDEN (ALL_DIMS)
#define NUM_OUTPUT (ALL_DIMS)
#define BATCH_SIZE (ALL_DIMS)
#define ROW_STRIDE (16)

#define PARALLEL(func) \
    func \
        .tile(i, j, i_outer, j_outer, i_inner, j_inner, ALL_DIMS, ROW_STRIDE) \
        .fuse(i_outer, j_outer, tile_index) \
        .parallel(tile_index); \
    func \
        .tile(i_inner, j_inner, i_inner_outer, j_inner_outer, i_inner_inner, j_inner_inner, 16, 1) \
        .vectorize(i_inner_inner)

int main(int argc, char **argv) {
    /* Name of Input Files */
    char name[100];
    /* Set the variables i,j,k */
    Var i, i_outer, i_inner, i_inner_outer, i_inner_inner;
    Var j, j_outer, j_inner, j_inner_outer, j_inner_inner;
    Var k, tile_index;
    /* Loading Image for Input X */
    Func load_x; 
    load_x(i,j,k) = 0.f;
    for (int a = 1; a <= T; a++){
        sprintf(name, "images/X_%d.png", a);
        Image<uint8_t> img_x_t = Tools::load_image(name);
        load_x(i, j, a) = cast<float>(img_x_t(i,j)) / 255.f - 0.5f;
    }
    Image<float> x = load_x.realize(NUM_INPUT, BATCH_SIZE, T + 1);

    /* Loading Image for Target */
    Func load_target;
    load_target(i,j,k) = 0.f;
    for (int a = 1; a <= T; a++){
        sprintf(name, "images/T_%d.png", a);
        Image<uint8_t> img_target_t = Tools::load_image(name);
        load_target(i, j, a) = cast<float>(img_target_t(i,j)) / 255.f - 0.5f;
    }
    Image<float> target = load_target.realize(NUM_OUTPUT, BATCH_SIZE, T + 1);

    /* Load h0 */
    Func load_h;
    load_h(i,j,k) = 0.f;
    sprintf(name, "images/h0.png");
    Image<uint8_t> img_h0 = Tools::load_image(name);
    load_h(i,j,0) = cast<float>(img_h0(i,j)) / 255.f - 0.5f;
    Image<float> h = load_h.realize(NUM_HIDDEN, BATCH_SIZE, T + 1);

    /* Load y */
    Image<float> y(NUM_OUTPUT, BATCH_SIZE, T+1);

    /* Load Wxh */
    Func load_Wxh;
    sprintf(name, "images/Wxh.png");
    Image<uint8_t> img_Wxh = Tools::load_image(name);
    load_Wxh(i,j) = cast<float>(img_Wxh(i,j)) / 255.f - 0.5f;
    Image<float> Wxh = load_Wxh.realize(NUM_HIDDEN,NUM_INPUT);

    /* Load Whh */
    Func load_Whh;
    sprintf(name, "images/Whh.png");
    Image<uint8_t> img_Whh = Tools::load_image(name);
    load_Whh(i,j) = cast<float>(img_Whh(i,j)) / 255.f - 0.5f;
    Image<float> Whh = load_Whh.realize(NUM_HIDDEN,NUM_HIDDEN);

    /* Load Why */
    Func load_Why;
    sprintf(name, "images/Why.png");
    Image<uint8_t> img_Why = Tools::load_image(name);
    load_Why(i,j) = cast<float>(img_Why(i,j)) / 255.f - 0.5f;
    Image<float> Why = load_Why.realize(NUM_OUTPUT,NUM_HIDDEN);

    /* Training */
    double start_time = CycleTimer::currentSeconds();
    for (int epoch_num = 0; epoch_num < 10; epoch_num++) {


        /* Forward propagation for all time */
        Func fprop_h, fprop_y; 
        fprop_h(i,j,k) = h(i,j,k); /* Initialize */
        fprop_y(i,j,k) = 0.f; /* Initialize */

        /* Set up h_tm1 */
        Image<float> h_tm1(NUM_HIDDEN,BATCH_SIZE);
        init_h0(h.raw_buffer(), h_tm1.raw_buffer());

        /* Forward propagation */
        for (int t = 1; t <= T; t++){
            /* Calculate h_t */
            Image<float> h_t(NUM_HIDDEN,BATCH_SIZE);
            fprop_h_t(t, x.raw_buffer(), h_tm1.raw_buffer(), 
                Wxh.raw_buffer(), Whh.raw_buffer(), h_t.raw_buffer());
            /* Calculate y_t */
            Image<float> y_t(NUM_OUTPUT,BATCH_SIZE);
            fprop_y_t(h_t.raw_buffer(), Why.raw_buffer(), y_t.raw_buffer());
            /* Save for next iteration */
            h_tm1 = h_t;
            /* Save in all */
            fprop_h(i,j,t) = h_t(i,j);
            fprop_y(i,j,t) = y_t(i,j);
        }
        PARALLEL(fprop_h);
        fprop_h.realize(h);
        PARALLEL(fprop_y);
        fprop_y.realize(y);
        //printf("Fprop Time: %f\n", CycleTimer::currentSeconds() - start_time);

        /* Calculate Loss */
        Image<float> avg_loss(1,1,1);
        loss(y.raw_buffer(), target.raw_buffer(), avg_loss.raw_buffer());
        printf("Average Loss: %f\n", avg_loss(0,0,0) / (T * BATCH_SIZE * NUM_INPUT));

        /* Setup Gradients for Backprop */
        Image<float> Gxh(NUM_HIDDEN, NUM_INPUT);
        Image<float> Ghh(NUM_HIDDEN, NUM_HIDDEN);
        Image<float> Ghy(NUM_OUTPUT, NUM_HIDDEN);
        
        /* Initialize tp1 dEdh */
        Image<float> dEdh_in_tp1(NUM_HIDDEN, BATCH_SIZE);
        init_dEdh_in_tp1(dEdh_in_tp1.raw_buffer());

        /* Backward propagation */
        for (int t = T; t > 0; t--){
            Image<float> dEdy_in(NUM_OUTPUT, BATCH_SIZE);
            bprop_dEdy_in(t, y.raw_buffer(), target.raw_buffer(), dEdy_in.raw_buffer());
            Image<float> dEdh_in(NUM_OUTPUT, BATCH_SIZE);
            bprop_dEdh_in(t, h.raw_buffer(), dEdh_in_tp1.raw_buffer(), dEdy_in.raw_buffer(), 
                Whh.raw_buffer(), Why.raw_buffer(), dEdh_in.raw_buffer());
            /* Preserve the variable for next iteration */
            dEdh_in_tp1 = dEdh_in;
            /* Add up the gradient */
            bprop_Ghy(t, h.raw_buffer(), dEdy_in.raw_buffer(), Ghy.raw_buffer(), Ghy.raw_buffer());
            bprop_Ghh(t, h.raw_buffer(), dEdh_in.raw_buffer(), Ghh.raw_buffer(), Ghh.raw_buffer());
            bprop_Gxh(t, x.raw_buffer(), dEdh_in.raw_buffer(), Gxh.raw_buffer(), Gxh.raw_buffer());
        }
        printf("Bprop Time: %f\n", CycleTimer::currentSeconds() - start_time);

        /* Gradient Descent */
        float learning_rate = 0.01f;
        grad_descent(learning_rate, Wxh.raw_buffer(), Gxh.raw_buffer(),Wxh.raw_buffer());
        grad_descent(learning_rate, Whh.raw_buffer(), Ghh.raw_buffer(),Whh.raw_buffer());
        grad_descent(learning_rate, Why.raw_buffer(), Ghy.raw_buffer(),Why.raw_buffer());
    }
    printf("Total: %f\n", CycleTimer::currentSeconds() - start_time);
}