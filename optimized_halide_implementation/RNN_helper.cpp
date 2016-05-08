#include "Halide.h"
#include <stdio.h>
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

#define PARALLEL_MATRIX(func) \
    func \
        .tile(i, j, i_outer, j_outer, i_inner, j_inner, 16, 64) \
        .fuse(i_outer, j_outer, tile_index) \
        .parallel(tile_index) \
        .vectorize(i_inner)

int main(){
    /* Set the variables i,j,k */
    Var i, i_outer, i_inner, i_inner_outer, i_inner_inner;
    Var j, j_outer, j_inner, j_inner_outer, j_inner_inner;
    Var k, tile_index;
    /* Set the forward propagation iterator */
    RDom rx(0, NUM_INPUT);
    RDom rh(0, NUM_HIDDEN);
    RDom ry(0, NUM_OUTPUT);
    RDom rb(0, BATCH_SIZE);
    RDom rl(0, NUM_OUTPUT, 0, BATCH_SIZE, 1, T);
    /* Set Image Params and Params */
    Param<int32_t> t; /* Time Step */
    Param<float> learning_rate; /* Time Step */
    ImageParam x(type_of<float>(), 3);           /* 3D Input */
    ImageParam h(type_of<float>(), 3);           /* 3D Hidden */
    ImageParam h_t(type_of<float>(), 2);         /* Hidden Layer at Time t */
    ImageParam h_tm1(type_of<float>(), 2);       /* Hidden Layer at Time t-1 */
    ImageParam dEdh_in(type_of<float>(), 2);     /* Backprop error for dEdh before tanh at Time t */
    ImageParam dEdh_in_tp1(type_of<float>(), 2); /* Backprop error for dEdh before tanh at Time t+1 */
    ImageParam y(type_of<float>(), 3);           /* 3D Output */
    ImageParam y_t(type_of<float>(), 2);         /* Output Layer at Time t */
    ImageParam dEdy_in(type_of<float>(), 2);     /* Backprop error for dEdy before tanh at Time t */
    ImageParam target(type_of<float>(), 3);      /* 3D Target */
    ImageParam Wxh(type_of<float>(), 2);         /* Weight from Input to Hidden */
    ImageParam Whh(type_of<float>(), 2);         /* Recurrent Weight of Hidden */
    ImageParam Why(type_of<float>(), 2);         /* Weight from Hidden to Output */
    ImageParam Gxh(type_of<float>(), 2);         /* Grad from Input to Hidden */
    ImageParam Ghh(type_of<float>(), 2);         /* Recurrent Grad of Hidden */
    ImageParam Ghy(type_of<float>(), 2);         /* Grad from Hidden to Output */
    ImageParam weight(type_of<float>(), 2);       /* General Weight */
    ImageParam grad(type_of<float>(), 2);         /* General Grad */

    /* Setup h0 */
    Func init_h0;
    init_h0(i,j) = h(i,j,0);
    PARALLEL(init_h0);
    init_h0.compile_to_file("init_h0", {h});

    /* Forward propagation for H at time t */
    Func fprop_h_t;
    fprop_h_t(i,j) = tanh(sum(x(rx,j,t) * Wxh(i,rx)) + sum(h_tm1(rh,j) * Whh(i,rh)));
    PARALLEL_MATRIX(fprop_h_t);
    fprop_h_t.compile_to_file("fprop_h_t", {t, x, h_tm1, Wxh, Whh});

    /* Forward propagation for Y at time t */
    Func fprop_y_t;
    fprop_y_t(i,j) = tanh(sum(h_t(rh,j) * Why(i,rh)));
    PARALLEL_MATRIX(fprop_y_t);
    fprop_y_t.compile_to_file("fprop_y_t", {h_t, Why});

    /* Calculate Loss */
    Func loss;
    loss(i,j,k) = sum(((y(rl.x,rl.y,rl.z) - target(rl.x,rl.y,rl.z)) * (y(rl.x,rl.y,rl.z) - target(rl.x,rl.y,rl.z))));
    loss.compile_to_file("loss", {y, target});
    
    /* Setup dEdh_in_tp1 */
    Func init_dEdh_in_tp1;
    init_dEdh_in_tp1(i,j) = 0.f;
    PARALLEL(init_dEdh_in_tp1);
    init_dEdh_in_tp1.compile_to_file("init_dEdh_in_tp1",{});

    /* Compute error with respect to (Why h_t) */
    Func bprop_dEdy_in;
    bprop_dEdy_in(i,j) = 2 * (y(i,j,t) - target(i,j,t)) * (1 - (y(i,j,t) * y(i,j,t)));
    PARALLEL(bprop_dEdy_in);
    bprop_dEdy_in.compile_to_file("bprop_dEdy_in",{t,y,target});

    /* Compute error with respect to (Wxh x_t + Whh h_t) */
    Func bprop_dEdh_in;
    bprop_dEdh_in(i,j) = (sum(dEdy_in(ry,j) * Why(ry,i)) + sum(dEdh_in_tp1(rh,j) * Whh(rh,i)))*(1-h(i,j,t)*h(i,j,t));
    PARALLEL_MATRIX(bprop_dEdh_in);
    bprop_dEdh_in.compile_to_file("bprop_dEdh_in",{t, h, dEdh_in_tp1, dEdy_in, Whh, Why});

    /* Calculate Gradients Ghy */
    Func bprop_Ghy;
    bprop_Ghy(i,j) = Ghy(i,j) + sum(h(j,rb,t) * dEdy_in(i,rb));
    PARALLEL_MATRIX(bprop_Ghy);
    bprop_Ghy.compile_to_file("bprop_Ghy",{t, h, dEdy_in, Ghy});

    /* Calculate Gradients Ghh */
    Func bprop_Ghh; 
    bprop_Ghh(i,j) = Ghh(i,j) + sum(h(j,rb,t-1) * dEdh_in(i,rb));
    PARALLEL_MATRIX(bprop_Ghh);
    bprop_Ghh.compile_to_file("bprop_Ghh",{t,h,dEdh_in,Ghh});

    /* Calculate Gradients Gxh */
    Func bprop_Gxh;
    bprop_Gxh(i,j) = Gxh(i,j) + sum(x(j,rb,t) * dEdh_in(i,rb));
    PARALLEL_MATRIX(bprop_Gxh);
    bprop_Gxh.compile_to_file("bprop_Gxh",{t,x,dEdh_in,Gxh});

    /* Gradient Descent */
    Func grad_descent;
    grad_descent(i,j) = weight(i,j) - learning_rate * grad(i,j);
    PARALLEL(grad_descent);
    grad_descent.compile_to_file("grad_descent",{learning_rate,weight,grad});
    return 0;
}