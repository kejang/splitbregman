// For Split Breman TV reconstruction
// input is always positive, and output is
// max(s - thresh, 0) / s

extern "C"
__global__
void soft_thresh_real_ratio(
    float *x, const float thresh, const unsigned long long n
)
{
    const unsigned long long ind = threadIdx.x + blockIdx.x * blockDim.x;

    if (ind >= n) {
        return;
    }

    const float val = x[ind];

    const float val_thresh = val - thresh;
    
    if (val_thresh < 0) {
        x[ind] = 0;
    }
    else {
        x[ind] = val_thresh / val;
    }
}
