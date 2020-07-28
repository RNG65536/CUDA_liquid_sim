#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>

#include "common.h"
#include "mgpcg.h"

//////////////////////////////////////////////////////////////////////////

namespace splash
{
//////////////////////////////////////////////////////////////////////////
grid_cell<float>    sdistance;
grid_cell<char>     fluid_flag;
grid_face_x<float>  ux;
grid_face_y<float>  uy;
grid_face_z<float>  uz;
grid_face_x<float>  ux_temp;
grid_face_y<float>  uy_temp;
grid_face_z<float>  uz_temp;
grid_cell<float>    pressure;
grid_cell<float>    rhs;
grid_cell<char>     vel_valid;
grid_cell<char>     temp_valid;

grid_cell<float> res;

grid_cell<char>     d_fluid_flag;
grid_cell<float>    d_pressure;
grid_cell<float>    d_rhs;

grid_face_x<float>  d_ux;
grid_face_y<float>  d_uy;
grid_face_z<float>  d_uz;
grid_face_x<float>  d_ux_temp;
grid_face_y<float>  d_uy_temp;
grid_face_z<float>  d_uz_temp;
grid_cell<float>    d_sdistance;

grid_cell<char>     d_vel_valid;
grid_cell<char>     d_temp_valid;

void init_memory() {
    sdistance       .init(N,N,N);      // signed distance function (levelset)
    fluid_flag      .init(N,N,N);

    ux              .init(N+1,N  ,N  );
    uy              .init(N  ,N+1,N  );
    uz              .init(N  ,N  ,N+1);
    ux_temp         .init(N+1,N  ,N  );
    uy_temp         .init(N  ,N+1,N  );
    uz_temp         .init(N  ,N  ,N+1);

    pressure        .init(N,N,N);
    rhs             .init(N, N, N);   // residual of Ax-b     
    vel_valid       .init(N+1, N+1, N+1);
    temp_valid      .init(N+1, N+1, N+1);

    res             .init(N, N, N);

    // ----------------- gpu -------------------
    size_t buffer_size = (N + 1) * (N + 1) * (N + 1);

    d_fluid_flag      .init_gpu(N,N,N);

    d_ux              .init_gpu(N+1,N  ,N  , buffer_size);
    d_uy              .init_gpu(N  ,N+1,N  , buffer_size);
    d_uz              .init_gpu(N  ,N  ,N+1, buffer_size);
    d_ux_temp         .init_gpu(N+1,N  ,N  , buffer_size);
    d_uy_temp         .init_gpu(N  ,N+1,N  , buffer_size);
    d_uz_temp         .init_gpu(N  ,N  ,N+1, buffer_size);
    d_sdistance       .init_gpu(N, N, N    , buffer_size);

    d_pressure.init_gpu(N, N, N);
    d_rhs.init_gpu(N, N, N);

    // ref
    void* buffer;

}

void free_memory() {
    sdistance       .free();
    fluid_flag      .free();
    ux              .free();
    uy              .free();
    uz              .free();
    ux_temp         .free();
    uy_temp         .free();
    uz_temp         .free();
    pressure        .free();
    rhs             .free();
    vel_valid       .free();
    temp_valid      .free();


    d_fluid_flag      .free_gpu();
    d_ux              .free_gpu();
    d_uy              .free_gpu();
    d_uz              .free_gpu();
    d_ux_temp         .free_gpu();
    d_uy_temp         .free_gpu();
    d_uz_temp         .free_gpu();
    d_pressure        .free_gpu();
    d_rhs             .free_gpu();

    d_sdistance       .free_gpu();
}

//////////////////////////////////////////////////////////////////////////
#define sdf_interp interp_cubic
#define vel_interp interp_cubic
__global__ void _advect(grid_cell<float> dst, grid_cell<float> src, 
    grid_face_x<float> ux, grid_face_y<float> uy, grid_face_z<float> uz, float time_step)
{
    KERNAL_CONFIG

    int nx = dst.get_nx();
    int ny = dst.get_ny();
    int nz = dst.get_nz();

    if ( /*i >= 0 &&*/ i < nx &&
         /*j >= 0 &&*/ j < ny &&
         /*k >= 0 &&*/ k < nz )
    {
        float this_x = i + 0.5f;
        float this_y = j + 0.5f;
        float this_z = k + 0.5f;

        float u = (ux.get(i, j, k) + ux.get(i + 1, j, k)) * 0.5f;
        float v = (uy.get(i, j, k) + uy.get(i, j + 1, k)) * 0.5f;
        float w = (uz.get(i, j, k) + uz.get(i, j, k + 1)) * 0.5f;
        float x = this_x - 0.5 * time_step * u;
        float y = this_y - 0.5 * time_step * v;
        float z = this_z - 0.5 * time_step * w;
        u = ux.sdf_interp( x, y, z );
        v = uy.sdf_interp( x, y, z );
        w = uz.sdf_interp( x, y, z );
        x = this_x - time_step * u;
        y = this_y - time_step * v;
        z = this_z - time_step * w;
        dst.get(i, j, k) = src.sdf_interp( x, y, z );//divide by cell count
    }
}
__global__ void _advect(grid_face_x<float> dst, grid_face_x<float> src, 
    grid_face_x<float> ux, grid_face_y<float> uy, grid_face_z<float> uz, float time_step)
{
    KERNAL_CONFIG

    int nx = dst.get_nx();
    int ny = dst.get_ny();
    int nz = dst.get_nz();

    if ( i >= 1 && i < nx-1 &&
         /*j >= 0 &&*/ j < ny &&
         /*k >= 0 &&*/ k < nz )
    {
        float this_x = i;
        float this_y = j + 0.5f;
        float this_z = k + 0.5f;

        float u =  ux.get(i,j,k);
        float v = (uy.get(i,j,k)+uy.get(i,j+1,k)+uy.get(i-1,j,k)+uy.get(i-1,j+1,k))*0.25f;
        float w = (uz.get(i,j,k)+uz.get(i,j,k+1)+uz.get(i-1,j,k)+uz.get(i-1,j,k+1))*0.25f;
        float x = this_x - 0.5 * time_step * u;
        float y = this_y - 0.5 * time_step * v;
        float z = this_z - 0.5 * time_step * w;
        u = ux.vel_interp( x, y, z );
        v = uy.vel_interp( x, y, z );
        w = uz.vel_interp( x, y, z );
        x = this_x - time_step * u;
        y = this_y - time_step * v;
        z = this_z - time_step * w;
        dst.get(i, j, k) = src.vel_interp( x, y, z );
    }
}
__global__ void _advect(grid_face_y<float> dst, grid_face_y<float> src, 
    grid_face_x<float> ux, grid_face_y<float> uy, grid_face_z<float> uz, float time_step)
{
    KERNAL_CONFIG

    int nx = dst.get_nx();
    int ny = dst.get_ny();
    int nz = dst.get_nz();

    if ( /*i >= 0 &&*/ i < nx &&
         j >= 1 && j < ny-1 &&
         /*k >= 0 &&*/ k < nz )
    {
        float this_x = i + 0.5f;
        float this_y = j;
        float this_z = k + 0.5f;

        float u = (ux.get(i,j,k)+ux.get(i+1,j,k)+ux.get(i,j-1,k)+ux.get(i+1,j-1,k))*0.25f;
        float v =  uy.get(i,j,k);
        float w = (uz.get(i,j,k)+uz.get(i,j,k+1)+uz.get(i,j-1,k)+uz.get(i,j-1,k+1))*0.25f;
        float x = this_x - 0.5 * time_step * u;
        float y = this_y - 0.5 * time_step * v;
        float z = this_z - 0.5 * time_step * w;
        u = ux.vel_interp( x, y, z );
        v = uy.vel_interp( x, y, z );
        w = uz.vel_interp( x, y, z );
        x = this_x - time_step * u;
        y = this_y - time_step * v;
        z = this_z - time_step * w;
        dst.get(i, j, k) = src.vel_interp( x, y, z );
    }
}
__global__ void _advect(grid_face_z<float> dst, grid_face_z<float> src, 
    grid_face_x<float> ux, grid_face_y<float> uy, grid_face_z<float> uz, float time_step)
{
    KERNAL_CONFIG

    int nx = dst.get_nx();
    int ny = dst.get_ny();
    int nz = dst.get_nz();

    if ( /*i >= 0 &&*/ i < nx &&
         /*j >= 0 &&*/ j < ny &&
         k >= 1 && k < nz-1 )
    {
        float this_x = i + 0.5f;
        float this_y = j + 0.5f;
        float this_z = k;

        float u = (ux.get(i,j,k)+ux.get(i+1,j,k)+ux.get(i,j,k-1)+ux.get(i+1,j,k-1))*0.25f;
        float v = (uy.get(i,j,k)+uy.get(i,j+1,k)+uy.get(i,j,k-1)+uy.get(i,j+1,k-1))*0.25f;
        float w =  uz.get(i,j,k);
        float x = this_x - 0.5 * time_step * u;
        float y = this_y - 0.5 * time_step * v;
        float z = this_z - 0.5 * time_step * w;
        u = ux.vel_interp( x, y, z );
        v = uy.vel_interp( x, y, z );
        w = uz.vel_interp( x, y, z );
        x = this_x - time_step * u;
        y = this_y - time_step * v;
        z = this_z - time_step * w;
        dst.get(i, j, k) = src.vel_interp( x, y, z );
    }
}
void advection_semilagrangian_gpu() { // RK2
    int   dim_x            = d_sdistance.get_nx();
    int   dim_y            = d_sdistance.get_ny();
    int   dim_z            = d_sdistance.get_nz();
    auto  d_sdistance_temp =
        d_ux_temp.cast_gpu<grid_cell<float>>(dim_x, dim_y, dim_z);

    _advect<<<gridConfig(), blockConfig()>>>(d_sdistance_temp, d_sdistance, d_ux, d_uy, d_uz, dt);
    d_sdistance.swap(d_sdistance_temp);

    _advect<<<gridConfig(), blockConfig()>>>(d_ux_temp, d_ux, d_ux, d_uy, d_uz, dt);
    _advect<<<gridConfig(), blockConfig()>>>(d_uy_temp, d_uy, d_ux, d_uy, d_uz, dt);
    _advect<<<gridConfig(), blockConfig()>>>(d_uz_temp, d_uz, d_ux, d_uy, d_uz, dt);
    d_ux.swap(d_ux_temp);
    d_uy.swap(d_uy_temp);
    d_uz.swap(d_uz_temp);
}

//////////////////////////////////////////////////////////////////////////
__global__ void _subtract_gradient(grid_face_x<float> ux, grid_cell<float> pressure,
    grid_cell<float> sdistance, grid_cell<char> fluid_flag) {
    KERNAL_CONFIG

    int nx = ux.get_nx();
    int ny = ux.get_ny();
    int nz = ux.get_nz();

    if ( i >= 1 && i < nx-1 &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        float pressure_p = pressure.get(i,j,k);
        float pressure_n = pressure.get(i-1,j,k);
#if SECOND_ORDER_BC
        if (sdistance.get(i, j, k) * sdistance.get(i - 1, j, k) < 0)
        {
            if (AIR == fluid_flag.get(i, j, k))
            {
                pressure_p = sdistance.get(i, j, k) /
                             f_min(1e-6f, sdistance.get(i - 1, j, k)) *
                             pressure.get(i - 1, j, k);
            }
            if (AIR == fluid_flag.get(i - 1, j, k))
            {
                pressure_n = sdistance.get(i - 1, j, k) /
                             f_min(1e-6f, sdistance.get(i, j, k)) *
                             pressure.get(i, j, k);
            }
        }
#endif
        ux.get(i, j, k) -= (pressure_p - pressure_n);
    }
}
__global__ void _subtract_gradient(grid_face_y<float> uy, grid_cell<float> pressure,
    grid_cell<float> sdistance, grid_cell<char> fluid_flag) {
    KERNAL_CONFIG

    int nx = uy.get_nx();
    int ny = uy.get_ny();
    int nz = uy.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 1 && j < ny-1 &&
         k >= 0 && k < nz )
    { 
        float pressure_p = pressure.get(i,j,k);
        float pressure_n = pressure.get(i,j-1,k);
#if SECOND_ORDER_BC
        if (sdistance.get(i, j, k) * sdistance.get(i, j - 1, k) < 0)
        {
            if (AIR == fluid_flag.get(i, j, k))
            {
                pressure_p = sdistance.get(i, j, k) /
                             f_min(1e-6f, sdistance.get(i, j - 1, k)) *
                             pressure.get(i, j - 1, k);
            }
            if (AIR == fluid_flag.get(i, j - 1, k))
            {
                pressure_n = sdistance.get(i, j - 1, k) /
                             f_min(1e-6f, sdistance.get(i, j, k)) *
                             pressure.get(i, j, k);
            }
        }
#endif
        uy.get(i, j, k) -= (pressure_p - pressure_n);
    }
}
__global__ void _subtract_gradient(grid_face_z<float> uz, grid_cell<float> pressure,
    grid_cell<float> sdistance, grid_cell<char> fluid_flag) {
    KERNAL_CONFIG

    int nx = uz.get_nx();
    int ny = uz.get_ny();
    int nz = uz.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 1 && k < nz-1 )
    {
        float pressure_p = pressure.get(i,j,k);
        float pressure_n = pressure.get(i,j,k-1);
#if SECOND_ORDER_BC
        if (sdistance.get(i, j, k) * sdistance.get(i, j, k - 1) < 0)
        {
            if (AIR == fluid_flag.get(i, j, k))
            {
                pressure_p = sdistance.get(i, j, k) /
                             f_min(1e-6f, sdistance.get(i, j, k - 1)) *
                             pressure.get(i, j, k - 1);
            }
            if (AIR == fluid_flag.get(i, j, k - 1))
            {
                pressure_n = sdistance.get(i, j, k - 1) /
                             f_min(1e-6f, sdistance.get(i, j, k)) *
                             pressure.get(i, j, k);
            }
        }
#endif
        uz.get(i, j, k) -= (pressure_p - pressure_n);
    }
}
void subtract_gradient_gpu() {
    _subtract_gradient<<<gridConfig(), blockConfig()>>>(d_ux, d_pressure, d_sdistance, d_fluid_flag);
    _subtract_gradient<<<gridConfig(), blockConfig()>>>(d_uy, d_pressure, d_sdistance, d_fluid_flag);
    _subtract_gradient<<<gridConfig(), blockConfig()>>>(d_uz, d_pressure, d_sdistance, d_fluid_flag);
}
}  // namespace splash

__global__ void _calc_rhs(grid_cell<char>    fluid_flag,
                          grid_cell<float>   rhs,
                          grid_face_x<float> ux,
                          grid_face_y<float> uy,
                          grid_face_z<float> uz)
{
    KERNAL_CONFIG

    int nx = rhs.get_nx();
    int ny = rhs.get_ny();
    int nz = rhs.get_nz();

    if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
    {
        if (LIQUID == fluid_flag.get(i, j, k))
        {
            rhs.get(i, j, k) =
                -  // negated divergence
                (ux.get(i + 1, j, k) - ux.get(i, j, k) + uy.get(i, j + 1, k) -
                 uy.get(i, j, k) + uz.get(i, j, k + 1) - uz.get(i, j, k));
        }
    }
}

void compute_rhs()
{
    splash::d_rhs.clear_gpu();
    _calc_rhs<<<gridConfig(), blockConfig()>>>(splash::d_fluid_flag,
                                               splash::d_rhs,
                                               splash::d_ux,
                                               splash::d_uy,
                                               splash::d_uz);
}

void pcg_solve()
{
    compute_rhs();

    pcg::pcg_solve_poisson_gpu(splash::d_pressure,
                               splash::d_rhs,
                               splash::d_sdistance,
                               splash::d_fluid_flag,
                               splash::d_ux_temp.cast_gpu<grid_cell<float>>(N, N, N),
                               splash::d_uy_temp.cast_gpu<grid_cell<float>>(N, N, N),
                               splash::d_uz_temp.cast_gpu<grid_cell<float>>(N, N, N),
                               splash::pressure.get_nx(),
                               splash::pressure.get_ny(),
                               splash::pressure.get_nx());
}

//////////////////////////////////////////////////////////////////////////
namespace splash
{

// the 2-norm is not a good measurement for high dimensional data
float inf_norm(grid_cell<float> &data) {
    float ret = 0;
    for (int n = 0; n < data.get_size(); n++) {
        if (LIQUID == fluid_flag.get(n))
        {
            ret = f_max(ret, fabs(data.get(n)));
        }
    }
    return ret;
}

float root_mean_square(grid_cell<float> &data) {
    float r0 = 0;
    for (int n = 0; n < res.get_size(); n++)
    {
        r0 += sq(res.get(n));
    }
    return sqrt(r0 / res.get_size());
}

// of -L (laplacian matrix)
float sparse_A_diag(int i, int j, int k) {
    //     return 6;

    //////////////////////////////////////////////////////////////////////////
    float diag = 6.0f;
    if (LIQUID != fluid_flag.get(i, j, k)) return diag;

    int nid[][3] =  { {i+1,j,k}, {i-1,j,k}, {i,j+1,k}, {i,j-1,k}, {i,j,k+1}, {i,j,k-1} };
    for (int m = 0; m < 6; m++) {
        int ni = nid[m][0];
        int nj = nid[m][1];
        int nk = nid[m][2];

        if (SOLID == fluid_flag.get(ni, nj, nk))
            diag -= 1.0f;
        else if (AIR ==
                 fluid_flag.get(ni, nj, nk))  // ghost fluid interface
            diag -= sdistance.get(ni, nj, nk) /
                    f_min(1e-6f, sdistance.get(i, j, k));
    }
    return diag;
}

// of -L (laplacian matrix)
float sparse_A_offdiag(int fi, int fj, int fk, int i, int j, int k) {
    //     return -1;

    //////////////////////////////////////////////////////////////////////////
    if (LIQUID == fluid_flag.get(fi, fj, fk) && 
        LIQUID == fluid_flag.get(i, j, k))
        return -1.0f;
    return 0.0f;
}

void calc_residual() // of L*x=div
{
    int nx = pressure.get_nx();
    int ny = pressure.get_ny();
    int nz = pressure.get_nz();

    res.clear();

    // compute residual
#pragma omp parallel for
    for(int i=0; i<nx; i++){
        for(int j=0; j<ny; j++){
            for(int k=0; k<nz; k++){
                if (LIQUID == fluid_flag.get(i, j, k))
                {
                    float _lap =
                           pressure.get(i+1,j,k) * -sparse_A_offdiag(i,j,k, i+1,j,k)
                        +  pressure.get(i-1,j,k) * -sparse_A_offdiag(i,j,k, i-1,j,k)
                        +  pressure.get(i,j+1,k) * -sparse_A_offdiag(i,j,k, i,j+1,k)
                        +  pressure.get(i,j-1,k) * -sparse_A_offdiag(i,j,k, i,j-1,k)
                        +  pressure.get(i,j,k+1) * -sparse_A_offdiag(i,j,k, i,j,k+1)
                        +  pressure.get(i,j,k-1) * -sparse_A_offdiag(i,j,k, i,j,k-1)
                        +  pressure.get(i,j,k)   * -sparse_A_diag(i,j,k);
                    float _div =
                        (ux.get(i + 1, j, k) - ux.get(i, j, k) +
                         uy.get(i, j + 1, k) - uy.get(i, j, k) +
                         uz.get(i, j, k + 1) - uz.get(i, j, k)
                        );
                    res.get(i, j, k) = _lap - _div;
                }
            }
        }
    }

    float r0 = root_mean_square(res);
    printf("\tresidual mean square = %e\n",r0);
    printf("\tresidual inf_norm |r|=%e\n", inf_norm(res));
}

__host__ __device__ float sign_of_sdf(float x) {
    return x<= 0 ? -1 : 1;
}
__global__ void _desingularize_signed_distance(grid_cell<float> sdistance) {
    KERNAL_CONFIG

    int nx = sdistance.get_nx();
    int ny = sdistance.get_ny();
    int nz = sdistance.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        float sdf = sdistance.get(i, j, k);
        sdistance.get(i, j, k) = sign_of_sdf(sdf) * f_max(1e-10f, fabs(sdf));
    }
}
void desingularize_signed_distance_gpu() {
    _desingularize_signed_distance<<<gridConfig(), blockConfig()>>>(d_sdistance);
}
__global__ void _update_fluid_flag(grid_cell<char> fluid_flag, grid_cell<float> sdistance) {
    KERNAL_CONFIG

    int nx = fluid_flag.get_nx();
    int ny = fluid_flag.get_ny();
    int nz = fluid_flag.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if (sdistance.get(i,j,k) <= 0)
            fluid_flag.get(i,j,k) = LIQUID;
        else
            fluid_flag.get(i,j,k) = AIR;

        if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1)
            fluid_flag.get(i,j,k) = SOLID;
    }
}
void update_fluid_flag_gpu() {
    _update_fluid_flag<<<gridConfig(), blockConfig()>>>(d_fluid_flag, d_sdistance);
}

__global__ void _addDistanceSphere(grid_cell<float> sdistance,
    grid_face_x<float> ux, grid_face_y<float> uy, grid_face_z<float> uz,
    float cx, float cy, float cz, float r) {
    KERNAL_CONFIG

    int nx = sdistance.get_nx();
    int ny = sdistance.get_ny();
    int nz = sdistance.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        float x = (i + 0.5) - cx;
        float y = (j + 0.5) - cy;
        float z = (k + 0.5) - cz;
        float oldDist = sdistance.get(i,j,k);
        float newDist = sqrtf(x*x + y*y + z*z) - r;
        sdistance.get(i,j,k) = f_min(newDist, oldDist);
        if (newDist < 0.5) {
            ux.get(i, j, k) = 0;
            ux.get(i+1, j, k) = 0;
            uy.get(i, j, k) = 0;
            uy.get(i, j+1, k) = 0;
            uz.get(i, j, k) = 0;
            uz.get(i, j, k+1) = 0;
        }
    }
}
__global__ void _add_gravity(grid_face_y<float> uy, grid_cell<char> fluid_flag) {
    KERNAL_CONFIG

    int nx = uy.get_nx();
    int ny = uy.get_ny();
    int nz = uy.get_nz();

    if ( i >= 0 && i < nx   &&
         j >= 1 && j < ny-1 &&
         k >= 0 && k < nz   ) {

        if (LIQUID == fluid_flag.get(i,j,k) || 
            LIQUID == fluid_flag.get(i,j-1,k))
        {
            uy.get(i,j,k) -= gravity * dt;
        }
    }
}
void add_source_gpu() {
    _add_gravity<<<gridConfig(), blockConfig()>>>(d_uy, d_fluid_flag);
    static int ii = 0;
    if (0 == ++ii % 10) {
        _addDistanceSphere<<<gridConfig(), blockConfig()>>>(d_sdistance, d_ux, d_uy, d_uz,
            (0.1 + 0.8 * randf()) * N, (0.5 + 0.4 * randf()) * N,
            (0.1 + 0.8 * randf()) * N, N * 0.08);
    }
}

// fast sweeping
void sort(double& a, double& b, double& c, double p, double q, double r) {
    //bubble sort ... [min, ..., max]
    a = p, b = q, c = r;
    if (a > b) swap(a, b);
    if (b > c) swap(b, c);
    if (a > b) swap(a, b);
}
void solve_distance_air(double p, double q, double r, float &o) {
    double a,b,c;
    sort(a,b,c,p,q,r);
    // scan from a to c, small to large
    double d=a+1; // solution to (x-a)^2 = 1, x > a, x < b
    if(d>b){
        d=(a+b+sqrt(2-sq(a-b)))/2; // solution to (x-a)^2 + (x-b)^2 = 1, x > a, x > b, x < c
        if(d>c){
            double w=2*(a+b+c);
            double delta = sqrt(w*w-12*(a*a+b*b+c*c-1));
            d=(w-delta)/6; // solution to (x-a)^2 + (x-b)^2 + (x-c)^2 = 1, x > a, x > b, x > c
            if(d<c){
                d=(w+delta)/6;
            }
        }
    }
    if(d<o)o=d;
}
void solve_distance_liquid(double p, double q, double r, float &o) {
    double a,b,c;
    sort(a,b,c,p,q,r);
    // scan from c to a, large to small
    double d=c-1; // solution to (x-c)^2 = 1, x < c, x > b
    if(d<b){
        d=(c+b-sqrt(2-sq(c-b)))/2; // solution to (x-c)^2 + (x-b)^2 = 1, x < c, x < b, x > a
        if(d<a){
            double w=2*(c+b+a);
            double delta = sqrt(w*w-12*(c*c+b*b+a*a-1));
            d=(w+delta)/6; // solution to (x-c)^2 + (x-b)^2 + (x-a)^2 = 1, x < c, x < b, x < a
            if(d>a){
                d=(w-delta)/6;
            }
        }
    }
    if(d>o)o=d;
}
void redistance(grid_cell<float>& phi, bool is_sweep_liquid) {
    int n = phi.get_size();
    int nx = phi.get_nx();
    int ny = phi.get_ny();
    int nz = phi.get_nz();

    const int num_dir = 8;
    int config[][6] = {
        {1   , nx,    1   , ny,    1   , nz },
        {nx-2, -1,    1   , ny,    1   , nz },
        {1   , nx,    ny-2, -1,    1   , nz },
        {nx-2, -1,    ny-2, -1,    1   , nz },
        {1   , nx,    1   , ny,    nz-2, -1 },
        {nx-2, -1,    1   , ny,    nz-2, -1 },
        {1   , nx,    ny-2, -1,    nz-2, -1 },
        {nx-2, -1,    ny-2, -1,    nz-2, -1 },
    };

    float large_distance = 1e6;

    for (int i = 0; i < n; ++i) {
        if (AIR == fluid_flag.get(i))
            phi.get(i) = large_distance;
    }
    for (int dir = 0; dir < num_dir; dir++) {
        int x_begin = config[dir][0];
        int x_end   = config[dir][1];
        int y_begin = config[dir][2];
        int y_end   = config[dir][3];
        int z_begin = config[dir][4];
        int z_end   = config[dir][5];
        int x_step = x_begin < x_end ? 1 : -1;
        int y_step = y_begin < y_end ? 1 : -1;
        int z_step = z_begin < z_end ? 1 : -1;
        for (int i = x_begin; i != x_end; i += x_step) {
            for (int j = y_begin; j != y_end; j += y_step) {
                for (int k = z_begin; k != z_end; k += z_step) {
                    if (AIR == fluid_flag.get(i, j, k))
                        solve_distance_air(
                        phi.get(i - x_step, j, k),
                        phi.get(i, j - y_step, k),
                        phi.get(i, j, k - z_step),
                        phi.get(i, j, k));
                }
            }
        }
    }

    if (is_sweep_liquid) {
        for (int i = 0; i < n; ++i) {
            if (LIQUID == fluid_flag.get(i))
                phi.get(i) = -large_distance;
        }
        for (int dir = 0; dir < num_dir; dir++) {
            int x_begin = config[dir][0];
            int x_end   = config[dir][1];
            int y_begin = config[dir][2];
            int y_end   = config[dir][3];
            int z_begin = config[dir][4];
            int z_end   = config[dir][5];
            int x_step = x_begin < x_end ? 1 : -1;
            int y_step = y_begin < y_end ? 1 : -1;
            int z_step = z_begin < z_end ? 1 : -1;
            for (int i = x_begin; i != x_end; i += x_step) {
                for (int j = y_begin; j != y_end; j += y_step) {
                    for (int k = z_begin; k != z_end; k += z_step) {
                        if (LIQUID == fluid_flag.get(i, j, k))
                            solve_distance_liquid(
                            phi.get(i - x_step, j, k),
                            phi.get(i, j - y_step, k),
                            phi.get(i, j, k - z_step),
                            phi.get(i, j, k));
                    }
                }
            }
        }
    }
}

// from christopher batty's code
//Apply several iterations of a very simple "Jacobi"-style propagation of valid velocity data in all directions
__global__ void _extrapolate(grid<float> data_grid, grid<float> temp_grid,
    grid_cell<char> valid, grid_cell<char> temp_valid) {
    KERNAL_CONFIG

    int nx = data_grid.get_nx();
    int ny = data_grid.get_ny();
    int nz = data_grid.get_nz();

    if ( i >= 1 && i < nx-1 &&
         j >= 1 && j < ny-1 &&
         k >= 1 && k < nz-1 )
    {
        float sum = 0;
        int count = 0;

        if(!temp_valid.get(i,j,k)) {

            if(temp_valid.get(i+1,j,k)) {    sum += data_grid.get(i+1,j,k);     ++count;    }
            if(temp_valid.get(i-1,j,k)) {    sum += data_grid.get(i-1,j,k);     ++count;    }
            if(temp_valid.get(i,j+1,k)) {    sum += data_grid.get(i,j+1,k);     ++count;    }
            if(temp_valid.get(i,j-1,k)) {    sum += data_grid.get(i,j-1,k);     ++count;    }
            if(temp_valid.get(i,j,k+1)) {    sum += data_grid.get(i,j,k+1);     ++count;    }
            if(temp_valid.get(i,j,k-1)) {    sum += data_grid.get(i,j,k-1);     ++count;    }

            if(temp_valid.get(i+1,j+1,k  )) {    sum += data_grid.get(i+1,j+1,k  );     ++count;    }
            if(temp_valid.get(i-1,j+1,k  )) {    sum += data_grid.get(i-1,j+1,k  );     ++count;    }
            if(temp_valid.get(i+1,j-1,k  )) {    sum += data_grid.get(i+1,j-1,k  );     ++count;    }
            if(temp_valid.get(i-1,j-1,k  )) {    sum += data_grid.get(i-1,j-1,k  );     ++count;    }
            if(temp_valid.get(i  ,j+1,k+1)) {    sum += data_grid.get(i  ,j+1,k+1);     ++count;    }
            if(temp_valid.get(i  ,j-1,k+1)) {    sum += data_grid.get(i  ,j-1,k+1);     ++count;    }
            if(temp_valid.get(i  ,j+1,k-1)) {    sum += data_grid.get(i  ,j+1,k-1);     ++count;    }
            if(temp_valid.get(i  ,j-1,k-1)) {    sum += data_grid.get(i  ,j-1,k-1);     ++count;    }
            if(temp_valid.get(i+1,j  ,k+1)) {    sum += data_grid.get(i+1,j  ,k+1);     ++count;    }
            if(temp_valid.get(i+1,j  ,k-1)) {    sum += data_grid.get(i+1,j  ,k-1);     ++count;    }
            if(temp_valid.get(i-1,j  ,k+1)) {    sum += data_grid.get(i-1,j  ,k+1);     ++count;    }
            if(temp_valid.get(i-1,j  ,k-1)) {    sum += data_grid.get(i-1,j  ,k-1);     ++count;    }

            if(temp_valid.get(i+1,j+1,k+1)) {    sum += data_grid.get(i+1,j+1,k+1);     ++count;    }
            if(temp_valid.get(i-1,j+1,k+1)) {    sum += data_grid.get(i-1,j+1,k+1);     ++count;    }
            if(temp_valid.get(i+1,j-1,k+1)) {    sum += data_grid.get(i+1,j-1,k+1);     ++count;    }
            if(temp_valid.get(i-1,j-1,k+1)) {    sum += data_grid.get(i-1,j-1,k+1);     ++count;    }
            if(temp_valid.get(i+1,j+1,k-1)) {    sum += data_grid.get(i+1,j+1,k-1);     ++count;    }
            if(temp_valid.get(i-1,j+1,k-1)) {    sum += data_grid.get(i-1,j+1,k-1);     ++count;    }
            if(temp_valid.get(i+1,j-1,k-1)) {    sum += data_grid.get(i+1,j-1,k-1);     ++count;    }
            if(temp_valid.get(i-1,j-1,k-1)) {    sum += data_grid.get(i-1,j-1,k-1);     ++count;    }

            if(count > 0) {
                temp_grid.get(i,j,k) = sum /(float)count;
                valid.get(i,j,k) = 1;
            }
        }
    }
}
void extrapolate(grid<float> data_grid, grid<float> temp_grid,
    grid_cell<char> valid, grid_cell<char> temp_valid) 
{
    for(int layers = 0; layers < MAXCFL; ++layers) { //propagation to several layers

        // note that these must match exactly in size
        temp_valid.copy_from_gpu(valid);
        temp_grid.copy_from_gpu(data_grid);
        _extrapolate<<<gridConfig(), blockConfig()>>>(data_grid, temp_grid, valid, temp_valid);
        data_grid.copy_from_gpu(temp_grid);

        __sync();
    }
}
__global__ void _set_valid_x(grid_face_x<float> ux, grid_face_y<float> uy, grid_face_z<float> uz, grid_cell<char> fluid_flag, grid_cell<char> vel_valid) {
    KERNAL_CONFIG

    int nx = ux.get_nx();
    int ny = ux.get_ny();
    int nz = ux.get_nz();

    if ( i >= 1 && i < nx-1 &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if(((  LIQUID == fluid_flag.get(i-1,j,k) 
            || LIQUID == fluid_flag.get(i,j,k)))
            && SOLID != fluid_flag.get(i-1,j,k)
            && SOLID != fluid_flag.get(i,j,k)) { 
                vel_valid.get(i,j,k) = 1;
        }
    }
}
__global__ void _set_valid_y(grid_face_x<float> ux, grid_face_y<float> uy, grid_face_z<float> uz, grid_cell<char> fluid_flag, grid_cell<char> vel_valid) {
    KERNAL_CONFIG

    int nx = uy.get_nx();
    int ny = uy.get_ny();
    int nz = uy.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 1 && j < ny-1 &&
         k >= 0 && k < nz )
    {
        if(((  LIQUID == fluid_flag.get(i,j-1,k)
            || LIQUID == fluid_flag.get(i,j,k)))
            && SOLID != fluid_flag.get(i,j-1,k)
            && SOLID != fluid_flag.get(i,j,k)) { 
                vel_valid.get(i,j,k) = 1;
        }
    }
}
__global__ void _set_valid_z(grid_face_x<float> ux, grid_face_y<float> uy, grid_face_z<float> uz, grid_cell<char> fluid_flag, grid_cell<char> vel_valid) {
    KERNAL_CONFIG

    int nx = uz.get_nx();
    int ny = uz.get_ny();
    int nz = uz.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 1 && k < nz-1 )
    {
        if(((  LIQUID == fluid_flag.get(i,j,k-1)
            || LIQUID == fluid_flag.get(i,j,k)))
            && SOLID != fluid_flag.get(i,j,k-1)
            && SOLID != fluid_flag.get(i,j,k)) {
                vel_valid.get(i,j,k) = 1;
        }
    }
}
void extrapolate_velocity_gpu() {
    int i, j, k;

    void* buffer;

    //u extrapolation
    {
        buffer = d_uy_temp.get_buffer(0);
        d_vel_valid.init_ref_gpu(N + 1, N + 1, N + 1, buffer);
        buffer = d_uy_temp.get_buffer(d_vel_valid.get_size());
        d_temp_valid.init_ref_gpu(N + 1, N + 1, N + 1, buffer);

        d_vel_valid.clear_gpu();
        _set_valid_x<<<gridConfig(), blockConfig()>>>(d_ux, d_uy, d_uz, d_fluid_flag, d_vel_valid);
        extrapolate(d_ux, d_ux_temp, d_vel_valid, d_temp_valid);
    }

    // v extrapolation
    {
        buffer = d_uz_temp.get_buffer(0);
        d_vel_valid.init_ref_gpu(N + 1, N + 1, N + 1, buffer);
        buffer = d_uz_temp.get_buffer(d_vel_valid.get_size());
        d_temp_valid.init_ref_gpu(N + 1, N + 1, N + 1, buffer);

        d_vel_valid.clear_gpu();
        _set_valid_y<<<gridConfig(), blockConfig()>>>(d_ux, d_uy, d_uz, d_fluid_flag, d_vel_valid);
        extrapolate(d_uy, d_uy_temp, d_vel_valid, d_temp_valid);
    }

    // w extrapolation
    {
        buffer = d_ux_temp.get_buffer(0);
        d_vel_valid.init_ref_gpu(N + 1, N + 1, N + 1, buffer);
        buffer = d_ux_temp.get_buffer(d_vel_valid.get_size());
        d_temp_valid.init_ref_gpu(N + 1, N + 1, N + 1, buffer);

        d_vel_valid.clear_gpu();
        _set_valid_z<<<gridConfig(), blockConfig()>>>(d_ux, d_uy, d_uz, d_fluid_flag, d_vel_valid);
        extrapolate(d_uz, d_uz_temp, d_vel_valid, d_temp_valid);
    }
}

__global__ void _wall_boundary_for_sdf(grid_cell<float> density, grid_cell<char> fluid_flag) {
    KERNAL_CONFIG

    int nx = density.get_nx();
    int ny = density.get_ny();
    int nz = density.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if (SOLID == fluid_flag.get(i, j, k)) {
            float sum = 0;
            int count = 0;

            bool not_x_0 = (i > 0)      ;
            bool not_x_1 = (i < nx - 1) ;
            bool not_y_0 = (j > 0)      ;
            bool not_y_1 = (j < ny - 1) ;
            bool not_z_0 = (k > 0)      ;
            bool not_z_1 = (k < nz - 1) ;

            if(not_x_1 && SOLID != fluid_flag.get(i+1,j,k)) {    sum += density.get(i+1,j,k);     ++count;    }
            if(not_x_0 && SOLID != fluid_flag.get(i-1,j,k)) {    sum += density.get(i-1,j,k);     ++count;    }
            if(not_y_1 && SOLID != fluid_flag.get(i,j+1,k)) {    sum += density.get(i,j+1,k);     ++count;    }
            if(not_y_0 && SOLID != fluid_flag.get(i,j-1,k)) {    sum += density.get(i,j-1,k);     ++count;    }
            if(not_z_1 && SOLID != fluid_flag.get(i,j,k+1)) {    sum += density.get(i,j,k+1);     ++count;    }
            if(not_z_0 && SOLID != fluid_flag.get(i,j,k-1)) {    sum += density.get(i,j,k-1);     ++count;    }

            if(not_x_1 && not_y_1 && SOLID != fluid_flag.get(i+1,j+1,k  )) {    sum += density.get(i+1,j+1,k  );     ++count;    }
            if(not_x_0 && not_y_1 && SOLID != fluid_flag.get(i-1,j+1,k  )) {    sum += density.get(i-1,j+1,k  );     ++count;    }
            if(not_x_1 && not_y_0 && SOLID != fluid_flag.get(i+1,j-1,k  )) {    sum += density.get(i+1,j-1,k  );     ++count;    }
            if(not_x_0 && not_y_0 && SOLID != fluid_flag.get(i-1,j-1,k  )) {    sum += density.get(i-1,j-1,k  );     ++count;    }
            if(not_y_1 && not_z_1 && SOLID != fluid_flag.get(i  ,j+1,k+1)) {    sum += density.get(i  ,j+1,k+1);     ++count;    }
            if(not_y_0 && not_z_1 && SOLID != fluid_flag.get(i  ,j-1,k+1)) {    sum += density.get(i  ,j-1,k+1);     ++count;    }
            if(not_y_1 && not_z_0 && SOLID != fluid_flag.get(i  ,j+1,k-1)) {    sum += density.get(i  ,j+1,k-1);     ++count;    }
            if(not_y_0 && not_z_0 && SOLID != fluid_flag.get(i  ,j-1,k-1)) {    sum += density.get(i  ,j-1,k-1);     ++count;    }
            if(not_z_1 && not_x_1 && SOLID != fluid_flag.get(i+1,j  ,k+1)) {    sum += density.get(i+1,j  ,k+1);     ++count;    }
            if(not_z_0 && not_x_1 && SOLID != fluid_flag.get(i+1,j  ,k-1)) {    sum += density.get(i+1,j  ,k-1);     ++count;    }
            if(not_z_1 && not_x_0 && SOLID != fluid_flag.get(i-1,j  ,k+1)) {    sum += density.get(i-1,j  ,k+1);     ++count;    }
            if(not_z_0 && not_x_0 && SOLID != fluid_flag.get(i-1,j  ,k-1)) {    sum += density.get(i-1,j  ,k-1);     ++count;    }

            if(not_x_1 && not_y_1 && not_z_1 && SOLID != fluid_flag.get(i+1,j+1,k+1)) {    sum += density.get(i+1,j+1,k+1);     ++count;    }
            if(not_x_0 && not_y_1 && not_z_1 && SOLID != fluid_flag.get(i-1,j+1,k+1)) {    sum += density.get(i-1,j+1,k+1);     ++count;    }
            if(not_x_1 && not_y_0 && not_z_1 && SOLID != fluid_flag.get(i+1,j-1,k+1)) {    sum += density.get(i+1,j-1,k+1);     ++count;    }
            if(not_x_0 && not_y_0 && not_z_1 && SOLID != fluid_flag.get(i-1,j-1,k+1)) {    sum += density.get(i-1,j-1,k+1);     ++count;    }
            if(not_x_1 && not_y_1 && not_z_0 && SOLID != fluid_flag.get(i+1,j+1,k-1)) {    sum += density.get(i+1,j+1,k-1);     ++count;    }
            if(not_x_0 && not_y_1 && not_z_0 && SOLID != fluid_flag.get(i-1,j+1,k-1)) {    sum += density.get(i-1,j+1,k-1);     ++count;    }
            if(not_x_1 && not_y_0 && not_z_0 && SOLID != fluid_flag.get(i+1,j-1,k-1)) {    sum += density.get(i+1,j-1,k-1);     ++count;    }
            if(not_x_0 && not_y_0 && not_z_0 && SOLID != fluid_flag.get(i-1,j-1,k-1)) {    sum += density.get(i-1,j-1,k-1);     ++count;    }

            if(count > 0) {
                density.get(i,j,k) = sum /(float)count;
            }
        }
    }
}
void wall_boundary_for_sdf_gpu(grid_cell<float> &density, grid_cell<char> &fluid_flag) {
    _wall_boundary_for_sdf<<<gridConfig(), blockConfig()>>>(density, fluid_flag);
}
__global__ void _solid_boundary_for_velocity(grid_face_x<float> ux, grid_face_y<float> uy, 
    grid_face_z<float> uz, grid_cell<char> fluid_flag) {
    KERNAL_CONFIG

    int nx = fluid_flag.get_nx();
    int ny = fluid_flag.get_ny();
    int nz = fluid_flag.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if (SOLID == fluid_flag.get(i, j, k)) {
            ux.get(i    , j    , k    ) = 0;
            ux.get(i + 1, j    , k    ) = 0;
            uy.get(i    , j    , k    ) = 0;
            uy.get(i    , j + 1, k    ) = 0;
            uz.get(i    , j    , k    ) = 0;
            uz.get(i    , j    , k + 1) = 0;
        }
    }
}
void solid_boundary_for_velocity_gpu(grid_face_x<float> ux, grid_face_y<float> uy, 
    grid_face_z<float> uz, grid_cell<char> fluid_flag) {
        //TODO: resolve conflict
    _solid_boundary_for_velocity<<<gridConfig(), blockConfig()>>>(ux, uy, uz, fluid_flag);
}
float computeCFL() { // not standard definition
    float CFL = 0;
    {
        int nx = ux.get_nx();
        int ny = ux.get_ny();
        int nz = ux.get_nz();

        float ux_max = 0;
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    if (LIQUID == fluid_flag.get(i - 1, j, k) || LIQUID == fluid_flag.get(i, j, k))
                    ux_max = f_max(ux_max, fabs(ux.get(i,j,k)));
                }
            }
        }
        CFL = f_max(CFL, ux_max * dt);
    }
    {
        int nx = uy.get_nx();
        int ny = uy.get_ny();
        int nz = uy.get_nz();

        float uy_max = 0;
        for (int i = 0; i < nx; i++) {
            for (int j = 1; j < ny - 1; j++) {
                for (int k = 0; k < nz; k++) {
                    if (LIQUID == fluid_flag.get(i, j - 1, k) || LIQUID == fluid_flag.get(i, j, k))
                    uy_max = f_max(uy_max, fabs(uy.get(i,j,k)));
                }
            }
        }
        CFL = f_max(CFL, uy_max * dt);
    }
    {
        int nx = uz.get_nx();
        int ny = uz.get_ny();
        int nz = uz.get_nz();

        float uz_max = 0;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 1; k < nz - 1; k++) {
                    if (LIQUID == fluid_flag.get(i, j, k - 1) || LIQUID == fluid_flag.get(i, j, k))
                    uz_max = f_max(uz_max, fabs(uz.get(i,j,k)));
                }
            }
        }
        CFL = f_max(CFL, uz_max * dt);
    }
    return CFL;
}

//////////////////////////////////////////////////////////////////////////
void init_field(){
    int nx = sdistance.get_nx();
    int ny = sdistance.get_ny();
    int nz = sdistance.get_nz();
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                float x = (i + 0.5) - 0.35 * nx;
                float y = (j + 0.5) - 0.65 * ny;
                float z = (k + 0.5) - 0.35 * nz;
                sdistance.get(i,j,k) = sqrtf(x*x + y*y + z*z) - nx* 0.08;
               
                // more fluid
                //sdistance.get(i,j,k) = f_min((j + 0.5) - ny * 0.1, sdistance.get(i,j,k));
                //sdistance.get(i,j,k) = f_min((j + 0.5) - ny * 0.5, sdistance.get(i,j,k));
                //sdistance.get(i,j,k) = f_min((i + 0.5) - nx * 0.5, sdistance.get(i,j,k));
            }
        }
    }

    sdistance.to_device(d_sdistance);
    d_fluid_flag.clear_gpu();
    desingularize_signed_distance_gpu();
    update_fluid_flag_gpu();
}
void run_step(){
    Profiler _p0;
    add_source_gpu();
    wall_boundary_for_sdf_gpu(d_sdistance, d_fluid_flag);
    desingularize_signed_distance_gpu();
    update_fluid_flag_gpu();
    __sync();
    printf(">> add source took %.5f seconds\n", _p0.get());

    Profiler _p1;
    solid_boundary_for_velocity_gpu(d_ux, d_uy, d_uz, d_fluid_flag);
        d_fluid_flag.to_host(fluid_flag);
        d_sdistance.to_host(sdistance);
        {
            pcg_solve();
            //d_pressure.to_host(pressure);
            //calc_residual();
        }
    subtract_gradient_gpu();
    __sync();
    printf(">> projection took %.5f seconds\n", _p1.get());

    Profiler _p2;
    extrapolate_velocity_gpu();
    solid_boundary_for_velocity_gpu(d_ux, d_uy, d_uz, d_fluid_flag);
    __sync();
    printf(">> extrapolation took %.5f seconds\n", _p2.get());

    Profiler _p3;
    advection_semilagrangian_gpu();
    __sync();
    printf(">> advection took %.5f seconds\n", _p3.get());

    Profiler _p4;
    desingularize_signed_distance_gpu();
    update_fluid_flag_gpu();
    static int interval = 0;
    if (0 == ++interval % 9) {
            d_fluid_flag.to_host(fluid_flag);
            d_sdistance.to_host(sdistance);
        redistance(sdistance, 0 == interval % 18);
            sdistance.to_device(d_sdistance);
            fluid_flag.to_device(d_fluid_flag);
        desingularize_signed_distance_gpu();
        update_fluid_flag_gpu();
    }
    __sync();
    printf(">> redistancing took %.5f seconds\n", _p4.get());

    d_ux.to_host(ux);
    d_uy.to_host(uy);
    d_uz.to_host(uz);
    printf(" < CFL = %f > \n", computeCFL());
}

void run_sim() {
    Profiler _p0;
    {
        static int idx=0;
        printf("\nFrame %d\n", idx);

        constexpr int res = 1024; //300;
        frame_buffer vrfb; vrfb.init(res, res); // volume rendering film size can be arbitrary
        frame_buffer d_vrfb; d_vrfb.init_gpu(res, res); // volume rendering film size can be arbitrary
        simple_scalar_vr(d_sdistance, d_vrfb);
        d_vrfb.to_host(vrfb);
        vrfb.flipud();
        char fn[256];
        sprintf_s(fn, "ppm/vr_ns_%07d.ppm", idx++);
        int total = vrfb.getTotal() * 3;
        float *vfb = reinterpret_cast<float*>(vrfb.ptr());
        FILE* fp = fopen(fn, "wb");
        fprintf(fp, "P6\n%d %d\n255\n", vrfb.getWidth(), vrfb.getHeight());
        unsigned char* cfb = new unsigned char[total];
        for (int n = 0; n < total; n++)
            cfb[n] = (unsigned char)(clampf(vfb[n], 0, 1) * 255);
        fwrite(cfb, 1, total, fp);
        fclose(fp);
        delete[] cfb;
        vrfb.free();
        d_vrfb.free_gpu();
        __sync();
    }
    printf(">> volume rendering took %.5f seconds\n", _p0.get());
    for(int n=0; n<3; n++) run_step();
}

int main(int argc, char **argv){
    system("mkdir ppm");
    system("mkdir vol");

    init_memory();
    init_field();
    while (1) run_sim();
    free_memory();

    return 0;
}
}  // namespace splash

int main(int argc, char** argv) { return splash::main(argc, argv); }