
#include "Halide.h"
#include <stdio.h>
using namespace Halide;

#define T (10)
#define ALL_DIMS (4000)
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

// Modified from Ravi's code on
// https://github.com/ravi-teja-mullapudi/Halide/blob/master/apps/mat_mul/mat_mul.cpp
#define PARALLEL_MATRIX_RAVI(func, func_out, iter, i_stride, j_stride) \
    func_out(i,j) = func(i,j); \
    func.compute_at(func_out, i).vectorize(i); \
    func.update().reorder(i,j,iter).unroll(j).vectorize(i); \
    func_out \
        .tile(i, j, i_inner, j_inner, i_stride, j_stride) \
        .parallel(j) \
        .vectorize(i_inner) 

int main(){
    /* Set the variables i,j,k */
    Var i("i"), i_outer("i_outer"), i_inner("i_inner"), i_inner_outer("i_inner_outer"), i_inner_inner("i_inner_inner");
    Var j("j"), j_outer("j_outer"), j_inner("j_inner"), j_inner_outer("j_inner_outer"), j_inner_inner("j_inner_inner");
    Var k("k"), tile_index("tile_index");
    /* Set the forward propagation iterator */
    RDom rx(0, NUM_INPUT);
    RDom rh(0, NUM_HIDDEN);
    RDom ry(0, NUM_OUTPUT);
    RDom rb(0, BATCH_SIZE);
    RDom rl(0, BATCH_SIZE, 0, NUM_OUTPUT);
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
    ImageParam Whh_T(type_of<float>(), 2);         /* Recurrent Weight of Hidden */
    ImageParam Why_T(type_of<float>(), 2);         /* Weight from Hidden to Output */
    ImageParam Gxh(type_of<float>(), 2);         /* Grad from Input to Hidden */
    ImageParam Ghh(type_of<float>(), 2);         /* Recurrent Grad of Hidden */
    ImageParam Ghy(type_of<float>(), 2);         /* Grad from Hidden to Output */
    ImageParam weight(type_of<float>(), 2);       /* General Weight */
    ImageParam grad(type_of<float>(), 2);         /* General Grad */

    /* Setup h0 */
    Func init_h0("init_h0");
    init_h0(i,j) = h(i,j,0);
    PARALLEL(init_h0);
    init_h0.compile_to_file("init_h0", {h});

    /* Transpose */
    Func transpose("transpose");
    transpose(i,j) = weight(j,i);
    PARALLEL_MATRIX(transpose);
    transpose.compile_to_file("transpose",{weight});
    
    /* Forward propagation for H at time t */
    Func fprop_h_t("fprop_h_t");
    Func fprop_h_A("fprop_h_A"), fprop_h_A_out("fprop_h_A_out");
    Func fprop_h_B("fprop_h_B"), fprop_h_B_out("fprop_h_B_out");
    Func fprop_h_x_T("fprop_h_x_T"), fprop_h_h_T("fprop_h_h_T");
    fprop_h_A(i,j) = 0.0f;
    fprop_h_A(i,j) += x(i,rx,t) * Wxh(rx,j);
    PARALLEL_MATRIX_RAVI(fprop_h_A, fprop_h_A_out, rx, 16, 16);
    fprop_h_A_out.compute_root();
    fprop_h_B(i,j) = 0.0f;
    fprop_h_B(i,j) += h_tm1(i,rh) * Whh(rh,j);
    PARALLEL_MATRIX_RAVI(fprop_h_B, fprop_h_B_out, rh, 16, 16);
    fprop_h_B_out.compute_root();
    fprop_h_t(i,j) = tanh(fprop_h_A_out(i,j) + fprop_h_B_out(i,j));
    PARALLEL(fprop_h_t);
    fprop_h_t.compile_to_file("fprop_h_t", {t, x, h_tm1, Wxh, Whh});

    /* Forward propagation for Y at time t */
    Func fprop_y_t("fprop_y_t");
    Func fprop_y_A("fprop_y_A"), fprop_y_A_out("fprop_y_A_out");
    fprop_y_A(i,j) = 0.0f;
    fprop_y_A(i,j) += h_t(i,rh) * Why(rh,j);
    PARALLEL_MATRIX_RAVI(fprop_y_A, fprop_y_A_out, rh, 16, 16);
    fprop_y_A_out.compute_root();
    fprop_y_t(i,j) = tanh(fprop_y_A_out(i,j));
    PARALLEL(fprop_y_t);
    fprop_y_t.compile_to_file("fprop_y_t", {h_t, Why});

    /* Calculate Loss */
    Func loss("loss");
    loss(i,j) = sum(((y_t(rl.x,rl.y) - target(rl.x,rl.y,t)) * (y_t(rl.x,rl.y) - target(rl.x,rl.y,t))));
    loss.compile_to_file("loss", {t, y_t, target});
    
    /* Setup dEdh_in_tp1 */
    Func init_to_zero("init_to_zero");
    init_to_zero(i,j) = 0.f;
    PARALLEL(init_to_zero);
    init_to_zero.compile_to_file("init_to_zero",{});

    /* Compute error with respect to (Why h_t) */
    Func bprop_dEdy_in("bprop_dEdy_in");
    bprop_dEdy_in(i,j) = 2 * (y_t(i,j) - target(i,j,t)) * (1 - (y_t(i,j) * y_t(i,j)));
    PARALLEL(bprop_dEdy_in);
    bprop_dEdy_in.compile_to_file("bprop_dEdy_in",{t,y_t,target});

    /* Compute error with respect to (Wxh x_t + Whh h_t) */
    Func bprop_dEdh_in("bprop_dEdh_in");
    Func bprop_dEdh_in_A("bprop_dEdh_in_A"), bprop_dEdh_in_A_out("bprop_dEdh_in_A_out");
    Func bprop_dEdh_in_B("bprop_dEdh_in_B"), bprop_dEdh_in_B_out("bprop_dEdh_in_B_out");
    bprop_dEdh_in_A(i,j) = 0.0f;
    bprop_dEdh_in_A(i,j) += dEdy_in(i,ry) * Why_T(ry,j);
    PARALLEL_MATRIX_RAVI(bprop_dEdh_in_A, bprop_dEdh_in_A_out, ry, 16, 16);
    bprop_dEdh_in_A_out.compute_root();
    bprop_dEdh_in_B(i,j) = 0.0f;
    bprop_dEdh_in_B(i,j) += dEdh_in_tp1(i,rh) * Whh_T(rh,j);
    PARALLEL_MATRIX_RAVI(bprop_dEdh_in_B, bprop_dEdh_in_B_out, rh, 16, 16);
    bprop_dEdh_in_B_out.compute_root();
    bprop_dEdh_in(i,j) = (bprop_dEdh_in_A_out(i,j) + bprop_dEdh_in_B_out(i,j)) * (1 - h_t(i,j) * h_t(i,j));
    PARALLEL(bprop_dEdh_in);
    bprop_dEdh_in.compile_to_file("bprop_dEdh_in",{t, h_t, dEdh_in_tp1, dEdy_in, Whh_T, Why_T});

    /* Calculate Gradients Ghy */
    Func bprop_Ghy("bprop_Ghy");
    Func bprop_Ghy_A("bprop_Ghy_A"), bprop_Ghy_A_out("bprop_Ghy_A_out");
    bprop_Ghy_A(i,j) = 0.0f;
    bprop_Ghy_A(i,j) += h_t(rb,i) * dEdy_in(rb,j);
    PARALLEL_MATRIX_RAVI(bprop_Ghy_A, bprop_Ghy_A_out, rb, 16, 64);
    bprop_Ghy_A_out.compute_root();
    bprop_Ghy(i,j) = Ghy(i,j) + bprop_Ghy_A_out(i,j);
    PARALLEL(bprop_Ghy);
    bprop_Ghy.compile_to_file("bprop_Ghy",{t, h_t, dEdy_in, Ghy});

    /* Calculate Gradients Ghh */
    Func bprop_Ghh("bprop_Ghh"); 
    Func bprop_Ghh_A("bprop_Ghh_A"), bprop_Ghh_A_out("bprop_Ghh_A_out");
    bprop_Ghh_A(i,j) = 0.0f;
    bprop_Ghh_A(i,j) += h_tm1(rb,i) * dEdh_in(rb,j);
    PARALLEL_MATRIX_RAVI(bprop_Ghh_A, bprop_Ghh_A_out, rb, 16, 64);
    bprop_Ghh_A_out.compute_root();
    bprop_Ghh(i,j) = Ghh(i,j) + bprop_Ghh_A_out(i,j);
    PARALLEL(bprop_Ghh);
    bprop_Ghh.compile_to_file("bprop_Ghh",{t,h_tm1,dEdh_in,Ghh});

    /* Calculate Gradients Gxh */
    Func bprop_Gxh("bprop_Gxh"); 
    Func bprop_Gxh_A("bprop_Gxh_A"), bprop_Gxh_A_out("bprop_Gxh_A_out");
    bprop_Gxh_A(i,j) = 0.0f;
    bprop_Gxh_A(i,j) += x(rb,i,t) * dEdh_in(rb,j);
    PARALLEL_MATRIX_RAVI(bprop_Gxh_A, bprop_Gxh_A_out, rb, 16, 64);
    bprop_Gxh_A_out.compute_root();
    bprop_Gxh(i,j) = Gxh(i,j) + bprop_Gxh_A_out(i,j);
    PARALLEL(bprop_Gxh);
    bprop_Gxh.compile_to_file("bprop_Gxh",{t,x,dEdh_in,Gxh});

    /* Gradient Descent */
    Func grad_descent("grad_descent");
    grad_descent(i,j) = weight(i,j) - learning_rate * grad(i,j);
    PARALLEL(grad_descent);
    grad_descent.compile_to_file("grad_descent",{learning_rate,weight,grad});
    return 0;
}