#pragma once

#define SECOND_ORDER_BC 0

namespace pcg
{
void pcg_solve_poisson_gpu(grid_cell<float>& d_pressure,
                           grid_cell<float>& d_rhs,
                           grid_cell<float>& d_sdistance,
                           grid_cell<char>&  d_fluid_flag,
                           grid_cell<float>& d_temp_buffer_0,
                           grid_cell<float>& d_temp_buffer_1,
                           grid_cell<float>& d_temp_buffer_2,
                           const int         nx,
                           const int         ny,
                           const int         nz);

}  // namespace pcg
