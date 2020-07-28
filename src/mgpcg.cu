#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>

#include "common.h"
#include "mgpcg.h"

namespace pcg
{
__host__ __device__ bool is_fluid(
    CONST grid_cell<char>& f, int i, int j, int k, int nx, int ny, int nz) CONST
{
    return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz &&
           LIQUID == f.get(i, j, k);
}

__host__ __device__ bool is_solid(
    CONST grid_cell<char>& f, int i, int j, int k, int nx, int ny, int nz) CONST
{
    return i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz ||
           SOLID == f.get(i, j, k);
}

__host__ __device__ bool is_air(
    CONST grid_cell<char>& f, int i, int j, int k, int nx, int ny, int nz) CONST
{
    return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz &&
           AIR == f.get(i, j, k);
}

__host__ __device__ float neighbor_sum(CONST grid_cell<float>& Lxn,
                                       CONST grid_cell<float>& Lxp,
                                       CONST grid_cell<float>& Lyn,
                                       CONST grid_cell<float>& Lyp,
                                       CONST grid_cell<float>& Lzn,
                                       CONST grid_cell<float>& Lzp,
                                       CONST grid_cell<float>& x,
                                       CONST grid_cell<char>& f,
                                       int                    i,
                                       int                    j,
                                       int                    k,
                                       int                    nx,
                                       int                    ny,
                                       int                    nz) CONST
{
    int in = (i - 1 + nx) % nx;
    int ip = (i + 1) % nx;
    int jn = (j - 1 + ny) % ny;
    int jp = (j + 1) % ny;
    int kn = (k - 1 + nz) % nz;
    int kp = (k + 1) % nz;
    return x.get(in, j, k) * Lxn.get(i, j, k) +
           x.get(ip, j, k) * Lxp.get(i, j, k) +
           x.get(i, jn, k) * Lyn.get(i, j, k) +
           x.get(i, jp, k) * Lyp.get(i, j, k) +
           x.get(i, j, kn) * Lzn.get(i, j, k) +
           x.get(i, j, kp) * Lzp.get(i, j, k);
}

__global__ void init_L(CONST grid_cell<char> f,
                       grid_cell<float>      L_diag,
                       grid_cell<float>      L_diag_inv,
                       grid_cell<float>      Lxn,
                       grid_cell<float>      Lxp,
                       grid_cell<float>      Lyn,
                       grid_cell<float>      Lyp,
                       grid_cell<float>      Lzn,
                       grid_cell<float>      Lzp,
                       int                   nx,
                       int                   ny,
                       int                   nz)
{
    KERNAL_CONFIG

    // int nx = f.get_nx();
    // int ny = f.get_ny();
    // int nz = f.get_nz();

    if (i < nx && j < ny && k < nz)
    {
        if (LIQUID == f.get(i, j, k))
        {
            float s = 6.0f;
            s -= float(is_solid(f, i - 1, j, k, nx, ny, nz));
            s -= float(is_solid(f, i + 1, j, k, nx, ny, nz));
            s -= float(is_solid(f, i, j - 1, k, nx, ny, nz));
            s -= float(is_solid(f, i, j + 1, k, nx, ny, nz));
            s -= float(is_solid(f, i, j, k - 1, nx, ny, nz));
            s -= float(is_solid(f, i, j, k + 1, nx, ny, nz));
            L_diag.get(i, j, k)     = s;
            L_diag_inv.get(i, j, k) = 1.0f / s;
        }
        Lxn.get(i, j, k) = float(is_fluid(f, i - 1, j, k, nx, ny, nz));
        Lxp.get(i, j, k) = float(is_fluid(f, i + 1, j, k, nx, ny, nz));
        Lyn.get(i, j, k) = float(is_fluid(f, i, j - 1, k, nx, ny, nz));
        Lyp.get(i, j, k) = float(is_fluid(f, i, j + 1, k, nx, ny, nz));
        Lzn.get(i, j, k) = float(is_fluid(f, i, j, k - 1, nx, ny, nz));
        Lzp.get(i, j, k) = float(is_fluid(f, i, j, k + 1, nx, ny, nz));
    }
}

__host__ __device__ float get_Ldiag(
    CONST grid_cell<char> f, int i, int j, int k, int nx, int ny, int nz)
{
    float s = 6.0f;
    s -= float(is_solid(f, i - 1, j, k, nx, ny, nz));
    s -= float(is_solid(f, i + 1, j, k, nx, ny, nz));
    s -= float(is_solid(f, i, j - 1, k, nx, ny, nz));
    s -= float(is_solid(f, i, j + 1, k, nx, ny, nz));
    s -= float(is_solid(f, i, j, k - 1, nx, ny, nz));
    s -= float(is_solid(f, i, j, k + 1, nx, ny, nz));
    return s;
}

__host__ __device__ float get_Ldiaginv(
    CONST grid_cell<char> f, int i, int j, int k, int nx, int ny, int nz)
{
    float s = 6.0f;
    s -= float(is_solid(f, i - 1, j, k, nx, ny, nz));
    s -= float(is_solid(f, i + 1, j, k, nx, ny, nz));
    s -= float(is_solid(f, i, j - 1, k, nx, ny, nz));
    s -= float(is_solid(f, i, j + 1, k, nx, ny, nz));
    s -= float(is_solid(f, i, j, k - 1, nx, ny, nz));
    s -= float(is_solid(f, i, j, k + 1, nx, ny, nz));
    return 1.0f / s;
}

__host__ __device__ float get_Lxn(
    CONST grid_cell<char> f, int i, int j, int k, int nx, int ny, int nz)
{
    return float(is_fluid(f, i - 1, j, k, nx, ny, nz));
}

__host__ __device__ float get_Lxp(
    CONST grid_cell<char> f, int i, int j, int k, int nx, int ny, int nz)
{
    return float(is_fluid(f, i + 1, j, k, nx, ny, nz));
}

__host__ __device__ float get_Lyn(
    CONST grid_cell<char> f, int i, int j, int k, int nx, int ny, int nz)
{
    return float(is_fluid(f, i, j - 1, k, nx, ny, nz));
}

__host__ __device__ float get_Lyp(
    CONST grid_cell<char> f, int i, int j, int k, int nx, int ny, int nz)
{
    return float(is_fluid(f, i, j + 1, k, nx, ny, nz));
}

__host__ __device__ float get_Lzn(
    CONST grid_cell<char> f, int i, int j, int k, int nx, int ny, int nz)
{
    return float(is_fluid(f, i, j, k - 1, nx, ny, nz));
}

__host__ __device__ float get_Lzp(
    CONST grid_cell<char> f, int i, int j, int k, int nx, int ny, int nz)
{
    return float(is_fluid(f, i, j, k + 1, nx, ny, nz));
}

__host__ __device__ float neighbor_sum(CONST grid_cell<float>& x,
                                       CONST grid_cell<char>& f,
                                       int                    i,
                                       int                    j,
                                       int                    k,
                                       int                    nx,
                                       int                    ny,
                                       int                    nz) CONST
{
    int in = (i - 1 + nx) % nx;
    int ip = (i + 1) % nx;
    int jn = (j - 1 + ny) % ny;
    int jp = (j + 1) % ny;
    int kn = (k - 1 + nz) % nz;
    int kp = (k + 1) % nz;
    return x.get(in, j, k) * get_Lxn(f, i, j, k, nx, ny, nz) +
           x.get(ip, j, k) * get_Lxp(f, i, j, k, nx, ny, nz) +
           x.get(i, jn, k) * get_Lyn(f, i, j, k, nx, ny, nz) +
           x.get(i, jp, k) * get_Lyp(f, i, j, k, nx, ny, nz) +
           x.get(i, j, kn) * get_Lzn(f, i, j, k, nx, ny, nz) +
           x.get(i, j, kp) * get_Lzp(f, i, j, k, nx, ny, nz);
}

__global__ void smooth(grid_cell<float> z,
                       grid_cell<float> r,
                       grid_cell<char>  f,
                       CONST int        nx,
                       CONST int        ny,
                       CONST int        nz,
                       int              phase)
{
    KERNAL_CONFIG

    if (i < nx && j < ny && k < nz && (i + j + k) % 2 == phase &&
        LIQUID == f.get(i, j, k))
    {
        float rhs = r.get(i, j, k);
        rhs += neighbor_sum(z, f, i, j, k, nx, ny, nz);
        z.get(i, j, k) = rhs / get_Ldiag(f, i, j, k, nx, ny, nz);
    }
}

__global__ void regularize(
    grid_cell<float> data, grid_cell<char> f, int nx, int ny, int nz)
{
    KERNAL_CONFIG

    if (i < nx && j < ny && k < nz && LIQUID != f.get(i, j, k))
    {
        data.get(i, j, k) = 0.0f;
    }
}

__global__ void formPoisson_count_nonzero(CONST grid_cell<char> fluid_flag,
                                          CONST grid_cell<int> cell_index,
                                          int*                 count)
{
    KERNAL_CONFIG

    // build consistent ordering
    int nx = fluid_flag.get_nx();
    int ny = fluid_flag.get_ny();
    int nz = fluid_flag.get_nz();

    if (i < nx && j < ny && k < nz)
    {
        if (LIQUID == fluid_flag.get(i, j, k))
        {
            int cid = cell_index.get(i, j, k);

            int c = 1;

            c += is_fluid(fluid_flag, i, j, k - 1, nx, ny, nz);
            c += is_fluid(fluid_flag, i, j - 1, k, nx, ny, nz);
            c += is_fluid(fluid_flag, i - 1, j, k, nx, ny, nz);

            c += is_fluid(fluid_flag, i + 1, j, k, nx, ny, nz);
            c += is_fluid(fluid_flag, i, j + 1, k, nx, ny, nz);
            c += is_fluid(fluid_flag, i, j, k + 1, nx, ny, nz);

            count[cid] = c;
        }
    }
}

__global__ void formPoisson_build_matrix(CONST grid_cell<char> fluid_flag,
                                         CONST grid_cell<int> cell_index,
                                         CONST int*           I,
                                         int*                 J,
                                         float*               val)
{
    KERNAL_CONFIG

    // build consistent ordering
    int nx = fluid_flag.get_nx();
    int ny = fluid_flag.get_ny();
    int nz = fluid_flag.get_nz();

    if (i < nx && j < ny && k < nz)
    {
        if (LIQUID == fluid_flag.get(i, j, k))
        {
            int cid = cell_index.get(i, j, k);
            int NZ  = I[cid];

            if (is_fluid(fluid_flag, i, j, k - 1, nx, ny, nz))
            {
                J[NZ]   = cell_index.get(i, j, k - 1);
                val[NZ] = -1.0f;
                NZ++;
            }
            if (is_fluid(fluid_flag, i, j - 1, k, nx, ny, nz))
            {
                J[NZ]   = cell_index.get(i, j - 1, k);
                val[NZ] = -1.0f;
                NZ++;
            }
            if (is_fluid(fluid_flag, i - 1, j, k, nx, ny, nz))
            {
                J[NZ]   = cell_index.get(i - 1, j, k);
                val[NZ] = -1.0f;
                NZ++;
            }

            {
                J[NZ]   = cell_index.get(i, j, k);
                val[NZ] = get_Ldiag(fluid_flag, i, j, k, nx, ny, nz);
                NZ++;
            }

            if (is_fluid(fluid_flag, i + 1, j, k, nx, ny, nz))
            {
                J[NZ]   = cell_index.get(i + 1, j, k);
                val[NZ] = -1.0f;
                NZ++;
            }
            if (is_fluid(fluid_flag, i, j + 1, k, nx, ny, nz))
            {
                J[NZ]   = cell_index.get(i, j + 1, k);
                val[NZ] = -1.0f;
                NZ++;
            }
            if (is_fluid(fluid_flag, i, j, k + 1, nx, ny, nz))
            {
                J[NZ]   = cell_index.get(i, j, k + 1);
                val[NZ] = -1.0f;
                NZ++;
            }
        }
    }
}

__global__ void downsample_f(
    grid_cell<char> f_fine, grid_cell<char> f_coarse, int nx, int ny, int nz)
{
    KERNAL_CONFIG

    if (i < nx && j < ny && k < nz)
    {
        int i2 = i * 2;
        int j2 = j * 2;
        int k2 = k * 2;

        if (AIR == f_fine.get(i2, j2, k2) ||          //
            AIR == f_fine.get(i2 + 1, j2, k2) ||      //
            AIR == f_fine.get(i2, j2 + 1, k2) ||      //
            AIR == f_fine.get(i2 + 1, j2 + 1, k2) ||  //
            AIR == f_fine.get(i2, j2, k2 + 1) ||      //
            AIR == f_fine.get(i2 + 1, j2, k2 + 1) ||  //
            AIR == f_fine.get(i2, j2 + 1, k2 + 1) ||  //
            AIR == f_fine.get(i2 + 1, j2 + 1, k2 + 1))
        {
            f_coarse.get(i, j, k) = AIR;
        }
        else if (LIQUID == f_fine.get(i2, j2, k2) ||          //
                 LIQUID == f_fine.get(i2 + 1, j2, k2) ||      //
                 LIQUID == f_fine.get(i2, j2 + 1, k2) ||      //
                 LIQUID == f_fine.get(i2 + 1, j2 + 1, k2) ||  //
                 LIQUID == f_fine.get(i2, j2, k2 + 1) ||      //
                 LIQUID == f_fine.get(i2 + 1, j2, k2 + 1) ||  //
                 LIQUID == f_fine.get(i2, j2 + 1, k2 + 1) ||  //
                 LIQUID == f_fine.get(i2 + 1, j2 + 1, k2 + 1))
        {
            f_coarse.get(i, j, k) = LIQUID;
        }
        else
        {
            f_coarse.get(i, j, k) = SOLID;
        }
    }
}

__global__ void restrict_(grid_cell<float> r_fine,
                          grid_cell<char>  f_fine,
                          grid_cell<float> z_fine,
                          grid_cell<float> r_coarse,
                          CONST int        nx,
                          CONST int        ny,
                          CONST int        nz)
{
    KERNAL_CONFIG

    if (i < nx && j < ny && k < nz && LIQUID == f_fine.get(i, j, k))
    {
        float Az = get_Ldiag(f_fine, i, j, k, nx, ny, nz) * z_fine.get(i, j, k);
        Az -= neighbor_sum(z_fine, f_fine, i, j, k, nx, ny, nz);
        float res = r_fine.get(i, j, k) - Az;
        atomicAdd(&r_coarse.get(i / 2, j / 2, k / 2), res * 0.5f);
    }
}

__global__ void prolongate(grid_cell<float> z_fine,
                           grid_cell<float> z_coarse,
                           CONST int        nx,
                           CONST int        ny,
                           CONST int        nz)
{
    KERNAL_CONFIG

    if (i < nx && j < ny && k < nz)
    {
        z_fine.get(i, j, k) += z_coarse.get(i / 2, j / 2, k / 2);
    }
}

__global__ void calc_Ap_kernel(grid_cell<float> Ap,
                               grid_cell<float> p,
                               grid_cell<char>  f,
                               CONST int        nx,
                               CONST int        ny,
                               CONST int        nz)
{
    KERNAL_CONFIG

    if (i < nx && j < ny && k < nz && LIQUID == f.get(i, j, k))
    {
        float _Ap = get_Ldiag(f, i, j, k, nx, ny, nz) * p.get(i, j, k);
        _Ap -= neighbor_sum(p, f, i, j, k, nx, ny, nz);
        Ap.get(i, j, k) = _Ap;
    }
}

__global__ void calc_saxpy_kernel(grid_cell<float> x,
                                  grid_cell<float> y,
                                  grid_cell<char>  f,
                                  const float      a,
                                  CONST int        nx,
                                  CONST int        ny,
                                  CONST int        nz)
{
    KERNAL_CONFIG

    if (i < nx && j < ny && k < nz && LIQUID == f.get(i, j, k))
    {
        y.get(i, j, k) += a * x.get(i, j, k);
    }
}

__global__ void calc_sxpay_kernel(grid_cell<float> x,
                                  grid_cell<float> y,
                                  grid_cell<char>  f,
                                  const float      a,
                                  CONST int        nx,
                                  CONST int        ny,
                                  CONST int        nz)
{
    KERNAL_CONFIG

    if (i < nx && j < ny && k < nz && LIQUID == f.get(i, j, k))
    {
        y.get(i, j, k) = x.get(i, j, k) + a * y.get(i, j, k);
    }
}

class MGPCGSolver
{
protected:
    int  nx, ny, nz;
    int  max_iters;
    int  n_mg_levels;
    int  n_pre_and_pose_smoothing;
    int  n_bottom_smoothing;
    bool use_precon;

    std::vector<grid_cell<float>> __r;
    std::vector<grid_cell<float>> __z;
    std::vector<grid_cell<char>>  __f;

    cublasHandle_t   cublasHandle   = 0;
    cusparseHandle_t cusparseHandle = 0;

public:
    MGPCGSolver(int nx_, int ny_, int nz_) : nx(nx_), ny(ny_), nz(nz_)
    {
        max_iters = 100;
        // n_mg_levels              = 4;
        n_mg_levels = 5;  // reduce to RGBS preconditioning if =1

        // for low res grid
        // n_pre_and_pose_smoothing = 2;
        // n_bottom_smoothing       = 10;
        // for high res grid
        n_pre_and_pose_smoothing = 4;
        n_bottom_smoothing       = 30;

        use_precon = true;

        auto get_res = [this](int level) {
            return make_int3(
                nx / (1 << level), ny / (1 << level), nz / (1 << level));
        };

        __r.resize(n_mg_levels);
        __z.resize(n_mg_levels);
        __f.resize(n_mg_levels);

        // no level 0
        for (int l = 1; l < n_mg_levels; l++)
        {
            auto res = get_res(l);
            __r[l].init_gpu(res.x, res.y, res.z);
            __z[l].init_gpu(res.x, res.y, res.z);
            __f[l].init_gpu(res.x, res.y, res.z);
        }

        checkCudaErrors(cublasCreate(&cublasHandle));
        checkCudaErrors(cusparseCreate(&cusparseHandle));
    }

    ~MGPCGSolver()
    {
        // no level 0
        for (int l = 1; l < n_mg_levels; l++)
        {
            __r[l].free_gpu();
            __z[l].free_gpu();
            __f[l].free_gpu();
        }

        cublasDestroy(cublasHandle);
        cusparseDestroy(cusparseHandle);
    }

    void prepare_preconditioner(grid_cell<float>& r0_buffer,
                                grid_cell<float>& z0_buffer,
                                grid_cell<char>&  f0_buffer)
    {
        //__r[0].init_ref_gpu(nx, ny, nz, r0_buffer.get_ptr());
        //__z[0].init_ref_gpu(nx, ny, nz, z0_buffer.get_ptr());
        //__f[0].init_ref_gpu(nx, ny, nz, f0_buffer.get_ptr());
        __r[0] = r0_buffer.cast<grid_cell<float>>(nx, ny, nz);
        __z[0] = z0_buffer.cast<grid_cell<float>>(nx, ny, nz);
        __f[0] = f0_buffer.cast<grid_cell<char>>(nx, ny, nz);
    }

    void apply_preconditioner()
    {
        dim3 block(8, 8, 8);

        __z[0].clear_gpu();

        // pre smoothing
        for (int level = 0; level < n_mg_levels - 1; level++)
        {
            int  dim_x = nx / (1 << level);
            int  dim_y = ny / (1 << level);
            int  dim_z = nz / (1 << level);
            dim3 grid(divUp(dim_x, block.x),
                      divUp(dim_y, block.y),
                      divUp(dim_z, block.z));

            for (int i = 0; i < n_pre_and_pose_smoothing; i++)
            {
                for (int phase = 0; phase < 2; phase++)
                    smooth<<<grid, block>>>(__z[level],
                                            __r[level],
                                            __f[level],
                                            dim_x,
                                            dim_y,
                                            dim_z,
                                            phase);
            }

            __z[level + 1].clear_gpu();
            __r[level + 1].clear_gpu();

            restrict_<<<grid, block>>>(__r[level],
                                       __f[level],
                                       __z[level],
                                       __r[level + 1],
                                       dim_x,
                                       dim_y,
                                       dim_z);
        }

        // bottom smoothing
        {
            int  halfcount = n_bottom_smoothing / 2;
            int  level     = n_mg_levels - 1;
            int  dim_x     = nx / (1 << level);
            int  dim_y     = ny / (1 << level);
            int  dim_z     = nz / (1 << level);
            dim3 grid(divUp(dim_x, block.x),
                      divUp(dim_y, block.y),
                      divUp(dim_z, block.z));
            for (int order = 0; order < 2; order++)
            {
                for (int i = 0; i < halfcount; i++)
                {
                    for (int phase = 0; phase < 2; phase++)
                        smooth<<<grid, block>>>(__z[level],
                                                __r[level],
                                                __f[level],
                                                dim_x,
                                                dim_y,
                                                dim_z,
                                                (phase + order) % 2);
                }
            }
        }

        // post smoothing
        for (int level = n_mg_levels - 2; level >= 0; level--)
        {
            int  dim_x = nx / (1 << level);
            int  dim_y = ny / (1 << level);
            int  dim_z = nz / (1 << level);
            dim3 grid(divUp(dim_x, block.x),
                      divUp(dim_y, block.y),
                      divUp(dim_z, block.z));

            prolongate<<<grid, block>>>(
                __z[level], __z[level + 1], dim_x, dim_y, dim_z);

            for (int i = 0; i < n_pre_and_pose_smoothing; i++)
            {
                for (int phase = 0; phase < 2; phase++)
                    smooth<<<grid, block>>>(__z[level],
                                            __r[level],
                                            __f[level],
                                            dim_x,
                                            dim_y,
                                            dim_z,
                                            phase);
            }
        }
    }

    float calc_dot(grid_cell<float>& a, grid_cell<float>& b)
    {
        dim3 block(8, 8, 8);
        dim3 grid(divUp(nx, block.x), divUp(ny, block.y), divUp(nz, block.z));

        regularize<<<grid, block>>>(a, __f[0], nx, ny, nz);
        regularize<<<grid, block>>>(b, __f[0], nx, ny, nz);
        float dot;
        cublasSdot(
            cublasHandle, nx * ny * nz, a.get_ptr(), 1, b.get_ptr(), 1, &dot);
        return dot;
    }

    void calc_Ap(grid_cell<float>& Ap,
                 grid_cell<float>& p,
                 CONST int         nx,
                 CONST int         ny,
                 CONST int         nz)
    {
        dim3 block(8, 8, 8);
        dim3 grid(divUp(nx, block.x), divUp(ny, block.y), divUp(nz, block.z));

        regularize<<<grid, block>>>(p, __f[0], nx, ny, nz);
        calc_Ap_kernel<<<grid, block>>>(Ap, p, __f[0], nx, ny, nz);
        regularize<<<grid, block>>>(Ap, __f[0], nx, ny, nz);
    }

    void calc_saxpy(grid_cell<float>& x,
                    grid_cell<float>& y,
                    const float       a,
                    CONST int         nx,
                    CONST int         ny,
                    CONST int         nz)
    {
        dim3 block(8, 8, 8);
        dim3 grid(divUp(nx, block.x), divUp(ny, block.y), divUp(nz, block.z));

        regularize<<<grid, block>>>(x, __f[0], nx, ny, nz);
        regularize<<<grid, block>>>(y, __f[0], nx, ny, nz);
        calc_saxpy_kernel<<<grid, block>>>(x, y, __f[0], a, nx, ny, nz);
    }

    void calc_sxpay(grid_cell<float>& x,
                    grid_cell<float>& y,
                    const float       a,
                    CONST int         nx,
                    CONST int         ny,
                    CONST int         nz)
    {
        dim3 block(8, 8, 8);
        dim3 grid(divUp(nx, block.x), divUp(ny, block.y), divUp(nz, block.z));

        regularize<<<grid, block>>>(x, __f[0], nx, ny, nz);
        regularize<<<grid, block>>>(y, __f[0], nx, ny, nz);
        calc_sxpay_kernel<<<grid, block>>>(x, y, __f[0], a, nx, ny, nz);
    }

    void solve(grid_cell<float>& d_pressure,
               grid_cell<float>& d_rhs,
               grid_cell<float>& d_sdistance,
               grid_cell<char>&  d_fluid_flag,
               grid_cell<float>& d_temp_0,
               grid_cell<float>& d_temp_1,
               grid_cell<float>& d_temp_2,
               const int         nx,
               const int         ny,
               const int         nz)
    {
        dim3 block(8, 8, 8);
        dim3 grid(divUp(nx, block.x), divUp(ny, block.y), divUp(nz, block.z));

        float r0, r1, alpha, beta;
        float dot, nalpha;
        int   k;

        const int   max_iter = 1000;
        const float tol      = 1e-5f;
        int         precon   = 2;

        // Profiler _p1;

        prepare_preconditioner(d_rhs, d_temp_2, d_fluid_flag);

        auto& __x  = d_pressure;
        auto& __p  = d_temp_0;
        auto& __Ap = d_temp_1;

        __x.clear_gpu();

        if (precon == 2)
        {
            for (int level = 1; level < n_mg_levels; level++)
            {
                int  dim_x = nx / (1 << level);
                int  dim_y = ny / (1 << level);
                int  dim_z = nz / (1 << level);
                dim3 grid(divUp(dim_x, block.x),
                          divUp(dim_y, block.y),
                          divUp(dim_z, block.z));
                downsample_f<<<grid, block>>>(
                    __f[level - 1], __f[level], dim_x, dim_y, dim_z);
            }
        }

        //////////////////////////////////////////////////////////////////////////
        // in case the rhs is zero when all fluid is free falling
        {
            // r' * r
            dot = calc_dot(__r[0], __r[0]);
            if (dot <= tol * tol)
            {
                return;
            }
        }

        cusparseMatDescr_t descr = 0;
        checkCudaErrors(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

        printf("\tConvergence of conjugate gradient: \n");

        //////////////////////////////////////////////////////////////////////////
        if (precon == 0)
        {
            // z := r
            __z[0].copy_from_gpu(__r[0]);
        }
        else if (precon == 2)
        {
            apply_preconditioner();
        }
        else
        {
            printf("invalid precon\n");
            throw std::runtime_error("invalid precon");
        }

        // p := z
        __p.copy_from_gpu(__z[0]);

        // r' * z
        r1 = calc_dot(__r[0], __z[0]);

        k = 0;
        while (k++ < max_iter)
        {
            // A * p
            calc_Ap(__Ap, __p, nx, ny, nz);

            // p' * A * p
            dot = calc_dot(__p, __Ap);

            alpha = r1 / dot;

            // x + a * p
            calc_saxpy(__p, __x, alpha, nx, ny, nz);

            nalpha = -alpha;

            // r - a * A * p
            calc_saxpy(__Ap, __r[0], nalpha, nx, ny, nz);

            // r' * r
            dot = calc_dot(__r[0], __r[0]);

            if (dot <= tol * tol) break;

            if (precon == 0)
            {
                // z := r
                __z[0].copy_from_gpu(__r[0]);
            }
            else if (precon == 2)
            {
                apply_preconditioner();
            }
            else
            {
                printf("invalid precon\n");
                throw std::runtime_error("invalid precon");
            }

            r0 = r1;

            // r' * z
            r1 = calc_dot(__r[0], __z[0]);

            beta = r1 / r0;

            // z + b * p
            calc_sxpay(__z[0], __p, beta, nx, ny, nz);
        }

        __sync();
        printf("\titeration = %3d, residual = %e \n", k, sqrt(r1));
        //////////////////////////////////////////////////////////////////////////

        regularize<<<grid, block>>>(__x, __f[0], nx, ny, nz);
    }
};  // namespace pcg

void pcg_solve_poisson_gpu(grid_cell<float>& d_pressure,
                           grid_cell<float>& d_rhs,
                           grid_cell<float>& d_sdistance,
                           grid_cell<char>&  d_fluid_flag,
                           grid_cell<float>& d_temp_buffer_0,
                           grid_cell<float>& d_temp_buffer_1,
                           grid_cell<float>& d_temp_buffer_2,
                           const int         nx,
                           const int         ny,
                           const int         nz)
{
    static MGPCGSolver solver(nx, ny, nz);
    solver.solve(d_pressure,
                 d_rhs,
                 d_sdistance,
                 d_fluid_flag,
                 d_temp_buffer_0,
                 d_temp_buffer_1,
                 d_temp_buffer_2,
                 nx,
                 ny,
                 nz);
}

}  // namespace pcg
