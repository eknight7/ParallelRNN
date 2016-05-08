#include <stdio.h>

#include "Halide.h"
#include "CycleTimer.h"
#include "halide_image_io.h"

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char** argv) {

  // Number of input units
  int nin = 3;
  // Number of hidden units
  int nh = 4;
  // Number of output units
  int nout = 3;
  // Number of batches
  int nbatches = 2;
  // Time-steps
  int T = 3;

  // Initial values
  // h0: nbatches x nh
  double cpp_h0[2][4] = {
    {0.1, 0.1, 0.1, 0.1},
    {0.02, 0.02, 0.02, 0.02}
  };
  // x: T x nbatches x nin
  double cpp_x[3][2][3] = {
    {{3, 4, 5}, {6, 7, 8}},
    {{11, 12, 13}, {52, 53, 56}},
    {{17, 18, 19}, {32, 33, 34}}
  };
  // y: T x nbatches x nout
  double cpp_y[3][2][3] = {
    {{7, 8, 9}, {10, 11, 12}},
    {{15, 16, 17}, {56, 57, 60}},
    {{21, 22, 23}, {36, 37, 38}}
  };
  // W_xh: nin x nh
  double cpp_W_xh[3][4] = {
    {0.16, 0.17, 0.18, 0.19},
    {0.20, 0.21, 0.22, 0.23},
    {0.24, 0.25, 0.26, 0.27}
  };
  // W_hh: nh x nh
  double cpp_W_hh[4][4] = {
    {0.13, 0.14, 0.15, 0.16},
    {0.87, 0.88, 0.89, 0.90},
    {0.40, 0.41, 0.42, 0.43},
    {0.21, 0.22, 0.23, 0.24}
  };
  // W_hy: nh x nout
  double cpp_W_hy[4][3] = {
    {0.78, 0.52, 0.19},
    {0.41, 0.12, 0.64},
    {0.66, 0.50, 0.32},
    {0.02, 0.51, 0.46}
  };

  // Hidden layer state
  Image<double> h(T+1, nbatches, nh);
  // Output layer state
  Image<double> yhat(T+1, nbatches, nout);


  printf("[Loading to Halide] h0\n");
  Image<double> h0(nbatches, nh);
  for (int row = 0; row < nbatches; row++) {
    for (int col = 0; col < nh; col++) {
      h0(row, col) = cpp_h0[row][col];
      h(0, row, col) = cpp_h0[row][col];
      printf("%f\t", h0(row, col));
    }
    printf("\n");
  }

  printf("[Loading to Halide] x\n");
  Image<double> x(T, nbatches, nin);
  for (int t = 0; t < T; t++) {
    printf("t : %d\n", t);
    for (int row = 0; row < nbatches; row++) {
      printf("\t");
      for (int col = 0; col < nin; col++) {
        x(t, row, col) = cpp_x[t][row][col];
        printf("%f\t", x(t, row, col));
      }
      printf("\n");
    }
  }

  printf("[Loading to Halide] y\n");
  Image<double> y(T, nbatches, nout);
  for (int t = 0; t < T; t++) {
    printf("t : %d\n", t);
    for (int row = 0; row < nbatches; row++) {
      printf("\t");
      for (int col = 0; col < nout; col++) {
        y(t, row, col) = cpp_y[t][row][col];
        printf("%f\t", y(t, row, col));
      }
      printf("\n");
    }
  }
  
  printf("[Loading to Halide] W_xh\n");
  Image<double> W_xh(nin, nh);
  for (int row = 0; row < nin; row++) {
    for (int col = 0; col < nh; col++) {
      W_xh(row, col) = cpp_W_xh[row][col];
      printf("%f\t", W_xh(row, col));
    }
    printf("\n");
  }
  
  printf("[Loading to Halide] W_hh\n");
  Image<double> W_hh(nh, nh);
  for (int row = 0; row < nh; row++) {
    for (int col = 0; col < nh; col++) {
      W_hh(row, col) = cpp_W_hh[row][col];
      printf("%f\t", W_hh(row, col));
    }
    printf("\n");
  }

  printf("[Loading to Halide] W_hy\n");
  Image<double> W_hy(nh, nout);
  for (int row = 0; row < nh; row++) {
    for (int col = 0; col < nout; col++) {
      W_hy(row, col) = cpp_W_hy[row][col];
      printf("%f\t", W_hy(row, col));
    }
    printf("\n");
  }

  Var timestep, xcol, yrow;
  RDom batchIter(0, nbatches);

  RDom timeBatchIter(0, T, 0, nbatches);
  RDom ninIter(0, nin);
  RDom nhIter(0, nh);

  // Forward propagation
  double forwardPropStart = CycleTimer::currentSeconds();
  for (int t = 0; t < T; t++) {
    Func func_h;
    // Forward propagation: Update hidden state
    func_h(timestep, yrow, xcol) = h(timestep, yrow, xcol);
    func_h(t+1, yrow, xcol) = 
      tanh(sum(x(t, yrow, ninIter) * W_xh(ninIter, xcol)) 
        + sum(h(t, yrow, nhIter) * W_hh(nhIter, xcol)));
    func_h.realize(h);
  }
  // Forward propagation: Update output
  Func func_yhat;
  func_yhat(timestep, yrow, xcol) = tanh(sum(h(timestep, yrow, nhIter) * W_hy(nhIter, xcol)));
  func_yhat(0, yrow, xcol) = cast<double>(0);
  func_yhat.realize(yhat);
  double forwardPropEnd = CycleTimer::currentSeconds();
  printf("[Forward Propagation Running Time]:\t\t[%.3f] ms\n", 
                                    (forwardPropEnd - forwardPropStart) * 1000);

  printf("[Forward Propagation] h: T+1 x nbatches x nh: \n");
  for (int t = 0; t < T+1; t++) {
    printf("t : %d\n", t);
    for (int row = 0; row < nbatches; row++) {
      printf("\t");
      for (int col = 0; col < nh; col++) {
        printf("%f\t", h(t, row, col));
      }
      printf("\n");
    }
  }

  printf("[Forward Propagation] yhat: T+1 x nbatches x nout: \n");
  for (int t = 0; t < T+1; t++) {
    printf("t : %d\n", t);
    for (int row = 0; row < nbatches; row++) {
      printf("\t");
      for (int col = 0; col < nout; col++) {
        printf("%f\t", yhat(t, row, col));
      }
      printf("\n");
    }
  }

  // Backpropagation Through Time (BPTT)
  double bpttStart = CycleTimer::currentSeconds();
  // Find dE_hy (gradient of W_hy)
  Image<double> dE_hy(nh, nout);
  Func func_yhat_y_diff;
  func_yhat_y_diff(timestep, yrow, xcol) = 
              (yhat(timestep+1, yrow, xcol) - y(timestep, yrow, xcol)) *
              (1 - yhat(timestep+1, yrow, xcol) * yhat(timestep+1, yrow, xcol));

  Func func_dE_hy;
  func_dE_hy(yrow, xcol) = 
    sum(h(timeBatchIter.x+1, timeBatchIter.y, yrow) 
      * func_yhat_y_diff(timeBatchIter.x, timeBatchIter.y, xcol)); 
  func_dE_hy.realize(dE_hy);

  // Find dE_hh (gradient of W_hh) and dE_xh (gradient of W_xh)
  Image<double> dE_hh(nh, nh);
  Image<double> dE_xh(nin, nh);

  RDom noutIter(0, nout);
  Image<double> dhh(nbatches, nh);
  for (int t = T; t > 0; t--) {
    Func func_dhh;
    func_dhh(yrow, xcol) = (dhh(yrow, xcol) + 
      sum(func_yhat_y_diff(t-1, yrow, noutIter) * W_hy(xcol, noutIter)))
      * (1 - h(t, yrow, xcol) * h(t, yrow, xcol));
    func_dhh.realize(dhh);

    // Update gradient of W_hh
    Func func_dE_hh;
    func_dE_hh(yrow, xcol) = dE_hh(yrow, xcol) + 
                            sum(h(t-1, batchIter, yrow) * dhh(batchIter, xcol));
    func_dE_hh.realize(dE_hh);

    // Update gradient of W_xh
    Func func_dE_xh;
    func_dE_xh(yrow, xcol) = dE_xh(yrow, xcol) + 
                            sum(x(t-1, batchIter, yrow) * dhh(batchIter, xcol));        
    func_dE_xh.realize(dE_xh);
  }
  double bpttEnd = CycleTimer::currentSeconds();
  printf("[Backpropagation Through Time Running Time]:\t\t[%.3f] ms\n", 
                                    (bpttEnd - bpttStart) * 1000);

  printf("[Backpropagation Through Time] dE_hy: nh x nout: \n");
  for (int row = 0; row < nh; row++) {
    printf("\t");
    for (int col = 0; col < nout; col++) {
      printf("%f\t", dE_hy(row, col));
    }
    printf("\n");
  }

  printf("[Backpropagation Through Time] dE_hh: nh x nh: \n");
  for (int row = 0; row < nh; row++) {
    printf("\t");
    for (int col = 0; col < nh; col++) {
      printf("%f\t", dE_hh(row, col));
    }
    printf("\n");
  }

  printf("[Backpropagation Through Time] dE_xh: nin x nh: \n");
  for (int row = 0; row < nin; row++) {
    printf("\t");
    for (int col = 0; col < nh; col++) {
      printf("%f\t", dE_xh(row, col));
    }
    printf("\n");
  }

  return 0;
}







