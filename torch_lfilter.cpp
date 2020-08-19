#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

void lfilter_cpu_forward(at::Tensor x, at::Tensor y, at::Tensor b, at::Tensor a, int order, int num_timesteps){
    double* ap = a.data_ptr<double>();
    double* bp = b.data_ptr<double>();
    double* xp = x.data_ptr<double>();
    double* yp = y.data_ptr<double>();
    int Ns = at::size(x, 1);

    for (int s = 0; s<Ns; s++){
        yp[s] += bp[order-1] * xp[s];
        for (int n = 1; n < order; n++){
            for (int m = 0; m < n+1; m++){
                yp[n*Ns+s] += bp[order-1 - n + m] * xp[m*Ns+s];
            }
            for (int m = 0; m < n; m++){
                yp[n*Ns+s] -= ap[order-1-n+m] * yp[m*Ns+s];
            }
        }
        for (int n = order; n < num_timesteps; n++){
            for (int m = 0; m < order; m++){
                yp[n*Ns+s] += bp[m] * xp[(n - order + 1 + m)*Ns+s];
            }
            for (int m = 0; m < order-1; m++){
                yp[n*Ns+s] -= ap[m] * yp[(n - order + 1 + m)*Ns+s];
            }
        }
    }
}

void lfilter_cpu_backward(at::Tensor dL_dx, at::Tensor dL_dy, at::Tensor b, at::Tensor a, int order, int num_timesteps){
    double* ap = a.data_ptr<double>();
    double* bp = b.data_ptr<double>();
    double* dL_dxp = dL_dx.data_ptr<double>();
    double* dL_dyp = dL_dy.data_ptr<double>();
    int Ns = at::size(dL_dx, 1);

    for (int s = 0; s < Ns; s++){
        for (int n = num_timesteps-1; n>order-1; n--){
            for (int m = 0; m < order-1; m++){
                dL_dyp[(n-order+1+m)*Ns+s] -= ap[m]*dL_dyp[n*Ns+s];
            }
            for (int m = 0; m < order; m++){
                dL_dxp[(n-order+1+m)*Ns+s] += bp[m] * dL_dyp[n*Ns+s];
            }
        }
        for (int n = order-1; n > 0; n--){
            for (int m = 0; m<n; m++){
                dL_dyp[m*Ns+s] -= ap[order-1-n+m] * dL_dyp[n*Ns+s];
            }
            for (int m = 0; m<n+1; m++){
                dL_dxp[m*Ns+s] += bp[order-n-1+m] * dL_dyp[n*Ns+s];
            }
        }
        dL_dxp[s] += bp[order-1] * dL_dyp[s];
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_lfilter_cpu_forward",  &lfilter_cpu_forward,  "lfilter forward (CPU only)");
    m.def("_lfilter_cpu_backward", &lfilter_cpu_backward, "lfilter backward (CPU only)");
}
