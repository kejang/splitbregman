
extern "C"
__global__
void soft_thresh_real(
    float *x, const float thresh, const unsigned long long n
)
{
    const unsigned long long ind = threadIdx.x + blockIdx.x * blockDim.x;

    if (ind >= n) {
    	return;
    }

    const float val = x[ind];
    const int is_pos = (val > 0) ? 1 : 0;

    const float abs_val_thresh = is_pos ? (val - thresh) : (-val - thresh);
    
    if (abs_val_thresh < 0) {
        x[ind] = 0;
    }
    else {
        x[ind] = is_pos ? abs_val_thresh : -abs_val_thresh;
    }
}
