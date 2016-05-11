#include <stdio.h>
#include "Halide.h"
#include "CycleTimer.h"
#include "halide_image_io.h"
#include "init_h0.h"
#include "transpose.h"
#include "fprop_h_t.h"
#include "fprop_y_t.h"
#include "loss.h"
#include "init_to_zero.h"
#include "bprop_dEdy_in.h"
#include "bprop_dEdh_in.h"
#include "bprop_Ghy.h"
#include "bprop_Ghh.h"
#include "bprop_Gxh.h"
#include "grad_descent.h"

using namespace Halide;

#define T (10)
#define NUM_EPOCH (10)
#define ALL_DIMS (4000)
#define NUM_INPUT (ALL_DIMS)
#define NUM_HIDDEN (ALL_DIMS)
#define NUM_OUTPUT (ALL_DIMS)
#define BATCH_SIZE (ALL_DIMS)
#define ROW_STRIDE (16)

//#define BREAKDOWN

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
        load_x(i, j, a) = cast<float>(img_x_t(j,i)) / 255.f - 0.5f;
    }
    Image<float> x = load_x.realize(BATCH_SIZE, NUM_INPUT, T + 1);

    /* Loading Image for Target */
    Func load_target;
    load_target(i,j,k) = 0.f;
    for (int a = 1; a <= T; a++){
        sprintf(name, "images/T_%d.png", a);
        Image<uint8_t> img_target_t = Tools::load_image(name);
        load_target(i, j, a) = cast<float>(img_target_t(j,i)) / 255.f - 0.5f;
    }
    Image<float> target = load_target.realize(BATCH_SIZE, NUM_OUTPUT, T + 1);

    /* Load h0 */
    Func load_h;
    load_h(i,j,k) = 0.f;
    sprintf(name, "images/h0.png");
    Image<uint8_t> img_h0 = Tools::load_image(name);
    load_h(i,j,0) = cast<float>(img_h0(j,i)) / 255.f - 0.5f;
    Image<float> h = load_h.realize(BATCH_SIZE, NUM_HIDDEN, T + 1);

    /* Load y */
    Image<float> y(BATCH_SIZE, NUM_OUTPUT, T+1);

    /* Load Wxh */
    Func load_Wxh;
    sprintf(name, "images/Wxh.png");
    Image<uint8_t> img_Wxh = Tools::load_image(name);
    load_Wxh(i,j) = cast<float>(img_Wxh(j,i)) / 255.f - 0.5f;
    Image<float> Wxh = load_Wxh.realize(NUM_INPUT, NUM_HIDDEN);

    /* Load Whh */
    Func load_Whh;
    sprintf(name, "images/Whh.png");
    Image<uint8_t> img_Whh = Tools::load_image(name);
    load_Whh(i,j) = cast<float>(img_Whh(j,i)) / 255.f - 0.5f;
    Image<float> Whh = load_Whh.realize(NUM_HIDDEN, NUM_HIDDEN);

    /* Load Why */
    Func load_Why;
    sprintf(name, "images/Why.png");
    Image<uint8_t> img_Why = Tools::load_image(name);
    load_Why(i,j) = cast<float>(img_Why(j,i)) / 255.f - 0.5f;
    Image<float> Why = load_Why.realize(NUM_HIDDEN, NUM_OUTPUT);

    /* Tranposed weights */
    Image<float> Whh_T(NUM_HIDDEN, NUM_HIDDEN);
    Image<float> Why_T(NUM_OUTPUT, NUM_INPUT);

    /* Training */
    double start_time = CycleTimer::currentSeconds();
    for (int epoch_num = 0; epoch_num < NUM_EPOCH; epoch_num++) {

        double epoch_time = CycleTimer::currentSeconds();

        /* Set up h_tm1 */
        Image<float> h_tm1(BATCH_SIZE, NUM_HIDDEN);
        init_h0(h.raw_buffer(), h_tm1.raw_buffer());

        /* Tranpose */
        transpose(Whh.raw_buffer(), Whh_T.raw_buffer());
        transpose(Why.raw_buffer(), Why_T.raw_buffer());

        Image<float> h_ptr[T+1];
        Image<float> y_ptr[T+1];
        h_ptr[0] = h_tm1;
        float total_loss = 0.f;

        #ifdef BREAKDOWN
            printf("Init_Fprop: %f\n", CycleTimer::currentSeconds() - epoch_time);
            epoch_time = CycleTimer::currentSeconds();
        #endif

        /* Forward propagation */
        for (int t = 1; t <= T; t++){
            /* Calculate h_t */
            Image<float> h_t(BATCH_SIZE, NUM_HIDDEN);
            fprop_h_t(t, x.raw_buffer(), h_tm1.raw_buffer(), Wxh.raw_buffer(), 
                Whh.raw_buffer(), h_t.raw_buffer());

            #ifdef BREAKDOWN
                printf("fprop-h_[%d]: %f\n", t, CycleTimer::currentSeconds() - epoch_time);
                epoch_time = CycleTimer::currentSeconds();
            #endif

            /* Calculate y_t */
            Image<float> y_t(BATCH_SIZE, NUM_OUTPUT);
            fprop_y_t(h_t.raw_buffer(), Why.raw_buffer(), y_t.raw_buffer());

            #ifdef BREAKDOWN
                printf("fprop-y_[%d]: %f\n", t, CycleTimer::currentSeconds() - epoch_time);
                epoch_time = CycleTimer::currentSeconds();
            #endif

            /* Save for next iteration */
            h_tm1 = h_t;
            h_ptr[t] = h_t;
            y_ptr[t] = y_t;

            /* Loss */
            Image<float> avg_loss(1,1);
            loss(t, y_t.raw_buffer(), target.raw_buffer(), avg_loss.raw_buffer());
            total_loss += avg_loss(0,0);

            #ifdef BREAKDOWN
                printf("loss_[%d]: %f\n", t, CycleTimer::currentSeconds() - epoch_time);
                epoch_time = CycleTimer::currentSeconds();
            #endif
        }

        #ifndef BREAKDOWN
            printf("loss: %f\n", total_loss / (T * BATCH_SIZE * NUM_INPUT));
        #endif

        /* Setup Gradients for Backprop */
        Image<float> Gxh(NUM_INPUT, NUM_HIDDEN);
        Image<float> Ghh(NUM_HIDDEN, NUM_HIDDEN);
        Image<float> Ghy(NUM_HIDDEN, NUM_OUTPUT);
        
        /* Initialize all things to 0 */
        Image<float> dEdh_in_tp1(BATCH_SIZE, NUM_HIDDEN);
        init_to_zero(dEdh_in_tp1.raw_buffer());
        init_to_zero(Gxh.raw_buffer());
        init_to_zero(Ghh.raw_buffer());
        init_to_zero(Ghy.raw_buffer());

        #ifdef BREAKDOWN
            printf("Init_Bprop: %f\n", CycleTimer::currentSeconds() - epoch_time);
            epoch_time = CycleTimer::currentSeconds();
        #endif

        /* Backward propagation */
        for (int t = T; t > 0; t--){
            Image<float> dEdy_in(BATCH_SIZE, NUM_OUTPUT);
            bprop_dEdy_in(t, y_ptr[t].raw_buffer(), target.raw_buffer(), dEdy_in.raw_buffer());
            
            #ifdef BREAKDOWN
                printf("dEdy_[%d]: %f\n", t, CycleTimer::currentSeconds() - epoch_time);
                epoch_time = CycleTimer::currentSeconds();
            #endif
            
            Image<float> dEdh_in(BATCH_SIZE, NUM_OUTPUT);
            bprop_dEdh_in(t, h_ptr[t].raw_buffer(), dEdh_in_tp1.raw_buffer(), dEdy_in.raw_buffer(), 
                Whh_T.raw_buffer(), Why_T.raw_buffer(), dEdh_in.raw_buffer());

            #ifdef BREAKDOWN
                printf("dEdh_[%d]: %f\n", t, CycleTimer::currentSeconds() - epoch_time);
                epoch_time = CycleTimer::currentSeconds();
            #endif

            /* Preserve the variable for next iteration */
            dEdh_in_tp1 = dEdh_in;
            /* Add up the gradient */
            bprop_Ghy(t, h_ptr[t].raw_buffer(), dEdy_in.raw_buffer(), Ghy.raw_buffer(), Ghy.raw_buffer());
            bprop_Ghh(t, h_ptr[t-1].raw_buffer(), dEdh_in.raw_buffer(), Ghh.raw_buffer(), Ghh.raw_buffer());
            bprop_Gxh(t, x.raw_buffer(), dEdh_in.raw_buffer(), Gxh.raw_buffer(), Gxh.raw_buffer());

            #ifdef BREAKDOWN
                printf("Grad_[%d]: %f\n", t, CycleTimer::currentSeconds() - epoch_time);
                epoch_time = CycleTimer::currentSeconds();
            #endif
        }

        /* Gradient Descent */
        float learning_rate = 0.01f;
        grad_descent(learning_rate, Wxh.raw_buffer(), Gxh.raw_buffer(), Wxh.raw_buffer());
        grad_descent(learning_rate, Whh.raw_buffer(), Ghh.raw_buffer(), Whh.raw_buffer());
        grad_descent(learning_rate, Why.raw_buffer(), Ghy.raw_buffer(), Why.raw_buffer());
        printf("Epoch_Time: %f\n", CycleTimer::currentSeconds() - epoch_time);
    }
    printf("Total: %f\n", CycleTimer::currentSeconds() - start_time);
}