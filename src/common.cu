#include <helper_cuda.h>
#include <helper_functions.h>

#include <cstdlib>

#include "common.h"

float randf() { return rand() / (RAND_MAX + 1.0f); }
void  __sync() { checkCudaErrors(cudaDeviceSynchronize()); }

int divUp(int a, int b) { return (a + b - 1) / b; }

dim3 blockConfig() { return dim3(8, 8, 8); }
dim3 gridConfig()
{
    return dim3(divUp(N, blockConfig().x),
                divUp(N, blockConfig().y),
                divUp(N, blockConfig().z));
}

constexpr float EPS        = 1e-8;
constexpr float scale      = 1.0f;
constexpr float brightness = 1.0f;

inline __host__ __device__ float transfer_func_opacity(float x)
{
    return x < 0.45 ? 0 : x < 0.55 ? 1 : 0;
}
inline __host__ __device__ vec3 transfer_func_color(float x, float opacity)
{
    return vec3(1, 0.95, 0.9) * scale * opacity * 10;
}
inline __host__ __device__ float simple_phong_shading(vec3 normal)
{
    vec3 light_dir = normalize(vec3(1, 1, 1));
    vec3 eye_dir(0, 0, 1);
    vec3 half_dir = normalize((light_dir + eye_dir) * 0.5);
    return 0.1f + fabs(dot(light_dir, normal)) +
           3.0f * powf(fabs(dot(half_dir, normal)), 60.0f);
}
__host__ __device__ float transfer_input(float x) { return x * 0.1 + 0.5; }
__global__ void _simple_scalar_vr(grid_cell<float> vol, frame_buffer fb)
{
    // c.f. http://http.developer.nvidia.com/GPUGems/gpugems_ch39.html
    KERNAL_CONFIG

    int width  = fb.getWidth();
    int height = fb.getHeight();

    if (i >= 0 && i < width && j >= 0 && j < height)
    {
        int nx = vol.get_nx();
        int ny = vol.get_ny();
        int nz = vol.get_nz();

        vec3  accum(0, 0, 0);
        float throughput = 1.0;

        for (int k = nz * 4 - 1; k >= 0; --k)
        {  // front-to-back rendering
            vec3 pos(
                (i + 0.5) / width * nx, (j + 0.5) / height * ny, (k + 0.5) / 4);
            float c =
                clampf(transfer_input(vol.interp(pos.x, pos.y, pos.z)), 0, 1);
            float opacity = transfer_func_opacity(c);
            vec3  estimated_normal =
                normalize(vec3(  // locally filtered
                    vol.interp(pos.x + 2, pos.y, pos.z) +
                        vol.interp(pos.x + 1, pos.y, pos.z) -
                        vol.interp(pos.x - 1, pos.y, pos.z) -
                        vol.interp(pos.x - 2, pos.y, pos.z) + EPS,
                    vol.interp(pos.x, pos.y + 2, pos.z) +
                        vol.interp(pos.x, pos.y + 1, pos.z) -
                        vol.interp(pos.x, pos.y - 1, pos.z) -
                        vol.interp(pos.x, pos.y - 2, pos.z) + EPS,
                    vol.interp(pos.x, pos.y, pos.z + 2) +
                        vol.interp(pos.x, pos.y, pos.z + 1) -
                        vol.interp(pos.x, pos.y, pos.z - 1) -
                        vol.interp(pos.x, pos.y, pos.z - 2) + EPS)) *
                -1;
            float shading = simple_phong_shading(estimated_normal);
            accum += (transfer_func_color(c, opacity) * shading) * throughput;
            throughput *= (1 - opacity);
            if (throughput < 1e-4f) break;
        }
        fb.get(i, j) = accum * 0.16 * 0.5 * brightness;
    }
}

void simple_scalar_vr(grid_cell<float> vol, frame_buffer fb)
{
    dim3 blockConfig2(16, 16, 1);
    dim3 gridConfig2(divUp(fb.getWidth(), blockConfig2.x), divUp(fb.getHeight(), blockConfig2.y), 1);
    _simple_scalar_vr<<<gridConfig2, blockConfig2>>>(vol, fb);
}
