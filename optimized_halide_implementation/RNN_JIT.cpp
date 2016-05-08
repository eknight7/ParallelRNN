#include "Halide.h"
#include <stdio.h>
#include "CycleTimer.h"
#include "halide_image_io.h"

/*
g++ RNN_CIFAR.cpp -g -I ~/Halide/include -I ~/Halide/tools -L ~/Halide/bin -lHalide `libpng-config --cflags --ldflags` -o rnn -std=c++11; 
DYLD_LIBRARY_PATH=~/Halide/bin ./rnn;
*/

using namespace Halide;

static int T = 10;
static int num_input = 1000;
static int num_hidden = 1000;
static int num_output = 1000;
static int batch_size = 1000;
#define ALL_DIMS 1000
#define ROW_STRIDE (16)

#define PARALLEL(func) \
    func \
        .tile(i, j, i_outer, j_outer, i_inner, j_inner, ALL_DIMS, ROW_STRIDE) \
        .fuse(i_outer, j_outer, tile_index) \
        .parallel(tile_index); \
    func \
        .tile(i_inner, j_inner, i_inner_outer, j_inner_outer, i_inner_inner, j_inner_inner, 16, 1) \
        .vectorize(i_inner_inner)

#define PARALLEL_MATRIX(func) \
    func \
        .tile(i, j, i_outer, j_outer, i_inner, j_inner, 16, 64) \
        .fuse(i_outer, j_outer, tile_index) \
        .parallel(tile_index) \
        .vectorize(i_inner)

int main(int argc, char **argv) {


    Var i, j, k;
    Image<float> x(num_input, batch_size, T+1);
    Image<float> h(num_hidden, batch_size, T+1);
    Image<float> y(num_output, batch_size, T+1);
    Image<float> target(num_output, batch_size, T+1);
    Image<float> Whh(num_hidden, num_hidden);
    Image<float> Wxh(num_hidden, num_input);
    Image<float> Why(num_output, num_hidden);
    char name[100];

    Func start_x, start_h, start_target, start_Wxh, start_Whh, start_Why;

    /* Load Input X */
    start_x(i,j,k) = 0.f;
    for (int a = 1; a <= T; a++){
        sprintf(name, "images/X_%d.png", a);
        Image<uint8_t> img_x_t = Tools::load_image(name);
        start_x(i, j, a) = cast<float>(img_x_t(i,j)) / 255.f - 0.5f;
    }
    start_x.realize(x);

    /* Load Target */
    start_target(i,j,k) = 0.f;
    for (int a = 1; a <= T; a++){
        sprintf(name, "images/T_%d.png", a);
        Image<uint8_t> img_target_t = Tools::load_image(name);
        start_target(i, j, a) = cast<float>(img_target_t(i,j)) / 255.f - 0.5f;
    }
    start_target.realize(target);

    /* Load h0 */
    start_h(i,j,k) = 0.f;
    sprintf(name, "images/h0.png");
    Image<uint8_t> img_h0 = Tools::load_image(name);
    start_h(i,j,0) = cast<float>(img_h0(i,j)) / 255.f - 0.5f;
    start_h.realize(h);

    /* Load Wxh */
    sprintf(name, "images/Wxh.png");
    Image<uint8_t> img_Wxh = Tools::load_image(name);
    start_Wxh(i,j) = cast<float>(img_Wxh(i,j)) / 255.f - 0.5f;
    start_Wxh.realize(Wxh);

    /* Load Whh */
    sprintf(name, "images/Whh.png");
    Image<uint8_t> img_Whh = Tools::load_image(name);
    start_Whh(i,j) = cast<float>(img_Whh(i,j)) / 255.f - 0.5f;
    start_Whh.realize(Whh);

    /* Load Why */
    sprintf(name, "images/Why.png");
    Image<uint8_t> img_Why = Tools::load_image(name);
    start_Why(i,j) = cast<float>(img_Why(i,j)) / 255.f - 0.5f;
    start_Why.realize(Why);

    /* Set the forward propagation iterator */
    RDom h_iter(0, num_hidden);
    RDom x_iter(0, num_input);
    RDom y_iter(0, num_output);
    RDom batch_iter(0, batch_size);

    double start_time = CycleTimer::currentSeconds();
    for (int e = 0; e < 10; e++){
        double epoch_time = CycleTimer::currentSeconds();
        /* Forward Prop */
        Func fprop_h, fprop_y; /* New defined functions */
        fprop_h(i,j,k) = h(i,j,k); /* Initialize */
        fprop_y(i,j,k) = 0.f; /* Initialize */

        /* Set up h0 */
        Func init_h, init_y;
        init_h(i,j) = h(i,j,0);
        init_y(i,j) = 0.f;

        Var i_outer, j_outer, i_inner, j_inner, tile_index;
        Var i_inner_outer, j_inner_outer, i_inner_inner, j_inner_inner;

        Image<float> h_tm1 = init_h.realize(num_hidden, batch_size);

        for (int t = 1; t <= T; t++){
            Func fprop_h_t, fprop_y_t, fprop_h_tm1; /* Only for that time instance */
            /* Compute Hidden Layer at Time T */
            fprop_h_t(i,j) = tanh(sum(x(x_iter,j,t) * Wxh(i,x_iter)) + sum(h_tm1(h_iter,j) * Whh(i,h_iter)));
            PARALLEL_MATRIX(fprop_h_t);
            Image<float> h_t = fprop_h_t.realize(num_hidden, batch_size);

            /* Compute Output Layer at Time T */
            fprop_y_t(i,j) = tanh(sum(h_t(h_iter,j) * Why(i,h_iter)));
            PARALLEL_MATRIX(fprop_y_t);
            Image<float> y_t = fprop_y_t.realize(num_output, batch_size);

            h_tm1 = h_t;
            fprop_h(i,j,t) = h_t(i,j);
            fprop_y(i,j,t) = y_t(i,j);
        }
        
        PARALLEL(fprop_h);
        PARALLEL(fprop_y);
        
        fprop_h.realize(h);
        fprop_y.realize(y);

        /* Calculate loss */
        Func sum_loss;
        RDom l(0, num_output, 0, batch_size, 1, T);
        sum_loss(i,j,k) = sum(((y(l.x,l.y,l.z) - target(l.x,l.y,l.z)) * (y(l.x,l.y,l.z) - target(l.x,l.y,l.z))));
        Image<float> loss = sum_loss.realize(1,1,1);

        printf("Total Loss: %f\n", loss(0,0,0) / (T * batch_size * num_input));
        
        Image<float> Gxh(num_hidden, num_input);
        Image<float> Ghh(num_hidden, num_hidden);
        Image<float> Ghy(num_output, num_hidden);

        /* Back prop */
        Func init_dEdh_in;
        init_dEdh_in(i,j) = 0.f;

        Image<float> dEdh_in_tp1 = init_dEdh_in.realize(num_hidden, batch_size);
        for (int t = T; t > 0; t--){
            /* Compute error with respect to (Wxh x_t + Whh h_t) */
            Func bprop_dEdy_in;
            bprop_dEdy_in(i,j) = 2 * (y(i,j,t) - target(i,j,t)) * (1 - (y(i,j,t) * y(i,j,t)));
            PARALLEL(bprop_dEdy_in);
            Image<float> dEdy_in = bprop_dEdy_in.realize(num_output, batch_size);

            Func bprop_dEdh_in;
            bprop_dEdh_in(i,j) = (sum(dEdy_in(y_iter,j) * Why(y_iter,i)) + sum(dEdh_in_tp1(h_iter,j) * Whh(h_iter,i))) * (1 - h(i, j, t) * h(i, j, t));
            PARALLEL_MATRIX(bprop_dEdh_in);
            Image<float> dEdh_in = bprop_dEdh_in.realize(num_hidden,batch_size);
            
            /* Preserve the variable for next iteration */
            dEdh_in_tp1 = dEdh_in;

            /* Add up the gradient */
            Func bprop_Gxh, bprop_Ghh, bprop_Ghy;
            bprop_Ghy(i,j) = Ghy(i,j) + sum(h(j,batch_iter,t) * dEdy_in(i,batch_iter));
            bprop_Ghh(i,j) = Ghh(i,j) + sum(h(j,batch_iter,t-1) * dEdh_in(i,batch_iter));
            bprop_Gxh(i,j) = Gxh(i,j) + sum(x(j,batch_iter,t) * dEdh_in(i,batch_iter));
            
            PARALLEL_MATRIX(bprop_Ghy);
            PARALLEL_MATRIX(bprop_Ghh);
            PARALLEL_MATRIX(bprop_Gxh);
            
            bprop_Gxh.realize(Gxh);
            bprop_Ghh.realize(Ghh);
            bprop_Ghy.realize(Ghy);
        }

        float learning_rate = 0.01f;
        Func grad_descent_Wxh, grad_descent_Whh, grad_descent_Why;
        grad_descent_Wxh(i,j) = Wxh(i,j) - learning_rate * Gxh(i,j);
        grad_descent_Whh(i,j) = Whh(i,j) - learning_rate * Ghh(i,j);
        grad_descent_Why(i,j) = Why(i,j) - learning_rate * Ghy(i,j);

        PARALLEL(grad_descent_Wxh);
        PARALLEL(grad_descent_Whh);
        PARALLEL(grad_descent_Why);

        grad_descent_Wxh.realize(Wxh);
        grad_descent_Whh.realize(Whh);
        grad_descent_Why.realize(Why);
        printf("[Epoch %d Time]: %f\n", e, CycleTimer::currentSeconds() - epoch_time);

    }
    printf("Time: %f\n", CycleTimer::currentSeconds() - start_time);
    return 0;
}