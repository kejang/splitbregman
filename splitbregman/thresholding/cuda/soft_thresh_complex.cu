
extern "C"
__global__
void soft_thresh_complex(
    cuComplex *x, const float thresh, const unsigned long long n
)
{
    const unsigned long long ind = threadIdx.x + blockIdx.x * blockDim.x;

    if (ind >= n) {
    	return;
    }

    const cuComplex val = x[ind];
    const float abs_val = sqrtf(val.x*val.x + val.y*val.y);
    const float abs_val_thresh = abs_val - thresh;
    const float w = abs_val_thresh / abs_val;
    
    if (abs_val_thresh < 0) {
        x[ind].x = 0;
        x[ind].y = 0;
    }
    else {
        x[ind].x *= w;
        x[ind].y *= w;
    }
}
