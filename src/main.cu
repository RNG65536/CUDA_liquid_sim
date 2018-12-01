#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define M_PI 3.14159265358979323846
float randf(){ return rand() / (RAND_MAX+1.0f); }
void __sync() { checkCudaErrors(cudaDeviceSynchronize()); }

#define N  256//128 //256 // 64// 400 //512 //
#define dt              0.01 * (N/5.0)
#define gravity         9.80 / (N/5.0)
#define MAXCFL          5

int divUp(int a, int b) { return (a + b - 1) / b; }
//dim3 blockConfig(4, 4, 4);
dim3 blockConfig(8, 8, 8);
dim3 gridConfig(divUp(N, blockConfig.x), divUp(N, blockConfig.y), divUp(N, blockConfig.z));

#define KERNAL_CONFIG \
int i = threadIdx.x + blockIdx.x * blockDim.x;  \
int j = threadIdx.y + blockIdx.y * blockDim.y;  \
int k = threadIdx.z + blockIdx.z * blockDim.z;  

class Profiler {
    double timeBegin;
    double stopwatch() {
        unsigned long long ticks, ticks_per_sec;
        QueryPerformanceFrequency( (LARGE_INTEGER *)&ticks_per_sec);
        QueryPerformanceCounter((LARGE_INTEGER *)&ticks);
        return ((double)ticks) / (double)ticks_per_sec;
    }
public:
    Profiler() { begin(); }
    void begin() { timeBegin = stopwatch(); }
    double get() {
        double timeGet = stopwatch();
        return timeGet - timeBegin;
    }
};

struct vec3{
    float x,y,z;
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float a) : x(a), y(a), z(a) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

__host__ __device__ vec3 cross(const vec3& a, const vec3& b){    return vec3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__host__ __device__ float dot(const vec3& a, const vec3& b){    return a.x*b.x+a.y*b.y+a.z*b.z;}
__host__ __device__ vec3 mult(const vec3& a, const vec3& b){    return vec3(a.x*b.x,a.y*b.y,a.z*b.z);}
__host__ __device__ vec3 div(const vec3& a, const vec3& b){    return vec3(a.x/b.x,a.y/b.y,a.z/b.z);}
__host__ __device__ vec3 operator*(const vec3& a, float b){    return vec3(a.x*b,a.y*b,a.z*b);}
__host__ __device__ vec3 operator+(const vec3& a, const vec3& b){    return vec3(a.x+b.x,a.y+b.y,a.z+b.z);}
__host__ __device__ vec3& operator+=(vec3& a, const vec3& b){    a.x+=b.x,a.y+=b.y,a.z+=b.z; return a;}
__host__ __device__ vec3 operator-(const vec3& a, const vec3& b){    return vec3(a.x-b.x,a.y-b.y,a.z-b.z);}
__host__ __device__ vec3 normalize(const vec3& a){    float _l = 1.0f/sqrtf(dot(a,a));    return a*_l;}
__host__ __device__ float f_min(float a, float b){ return a<b ? a : b; }
__host__ __device__ float f_max(float a, float b){ return a>b ? a : b; }
__host__ __device__ vec3 f_min(const vec3& a, const vec3& b){ return vec3(f_min(a.x,b.x),f_min(a.y,b.y),f_min(a.z,b.z)); }
__host__ __device__ vec3 f_max(const vec3& a, const vec3& b){ return vec3(f_max(a.x,b.x),f_max(a.y,b.y),f_max(a.z,b.z)); }
__host__ __device__ vec3 f_min(const vec3& a, float b){ return vec3(f_min(a.x,b),f_min(a.y,b),f_min(a.z,b)); }
__host__ __device__ vec3 f_max(const vec3& a, float b){ return vec3(f_max(a.x,b),f_max(a.y,b),f_max(a.z,b)); }
__host__ __device__ float sq(float x){ return x*x; }
__host__ __device__ float clampf(float x, float a, float b) { return x<a ? a : x>b ? b : x; }
__host__ __device__ int clampi(int x, int a, int b) { return x<a ? a : x>b ? b : x; }

template <class T>
void swap(T& a, T&b) {
    T t(b); b = a; a = t;
}
//////////////////////////////////////////////////////////////////////////
enum cell_type{
    AIR     = 0,
    LIQUID  = 1,
    SOLID   = 2,
};
__host__ __device__ int strict_floor(float x) {
    return int(x) - (x < 0);
}
__host__ __device__ void bary(float p, int dim, int& lower, float& offset){
    lower = strict_floor(p);
    offset = p - lower;
    if(lower<0    ){ lower = 0    ; offset = 0.0f; }
    if(lower>dim-2){ lower = dim-2; offset = 1.0f; }
}
__host__ __device__ bool bary_cubic(float p, int dim, int& lower, float& offset){
    lower = strict_floor(p);
    offset = p - lower;
    bool boundary = false;
    if(lower<2    ){ lower = 2    ; offset = 0.0f; boundary = true; }
    if(lower>dim-4){ lower = dim-4; offset = 1.0f; boundary = true; }
    return boundary;
}
template<typename T>
class grid{
private:
    T *data;
    int nx, ny, nz;
    int m_total, nxy;
    float corner_x, corner_y, corner_z;

    __host__ __device__ T lerp1(T a0, T a1, T x){//a_x
        return a0*(1-x)+a1*x;
    }
    __host__ __device__ T lerp2(T a00, T a10, //a_xy
        T a01, T a11, T x, T y  ){
            return lerp1(a00,a10,x)*(1-y)+lerp1(a01,a11,x)*y;
    }
    __host__ __device__ T lerp3(T a000, T a100, //a_xyz
        T a010, T a110, T a001, T a101, 
        T a011, T a111, T x, T y, T z  ){
            return lerp2(a000,a100,a010,a110,x,y)*(1-z)+lerp2(a001,a101,a011,a111,x,y)*z;
    }
    __host__ __device__ T lerp(int i0, int j0, int k0, T u, T v, T w){
        int i1 = i0 + 1;
        int j1 = j0 + 1;
        int k1 = k0 + 1;
        return lerp3(get(i0,j0,k0), get(i1,j0,k0), get(i0,j1,k0), get(i1,j1,k0),
            get(i0,j0,k1), get(i1,j0,k1), get(i0,j1,k1), get(i1,j1,k1), u, v, w);
    }

    __host__ __device__ T cubic1D(T t, T f[4])
    {
        T d1 = (f[2] - f[0]) * 0.5;
        T d2 = (f[3] - f[1]) * 0.5;
        T del = f[2] - f[1];
        if (d1 * del <= 0) d1 = 0;
        if (d2 * del <= 0) d1 = 0;
        T a0 = f[1];
        T a1 = d1;
        T a2 = 3 * del - 2 * d1 - d2;
        T a3 = d1 + d2 - 2 * del;
        return a3 * t*t*t + a2 * t*t + a1 * t + a0;
    }
    __host__ __device__ T cubic2D(T tx, T ty, T f[16])
    {
        T _f[4];
        _f[0] = cubic1D(tx, &f[ 0]);
        _f[1] = cubic1D(tx, &f[ 4]);
        _f[2] = cubic1D(tx, &f[ 8]);
        _f[3] = cubic1D(tx, &f[12]);
        return cubic1D(ty, _f);
    }
    __host__ __device__ T cubic3D(T tx, T ty, T tz, T f[64])
    {
        T _f[4];
        _f[0] = cubic2D(tx, ty, &f[ 0]);
        _f[1] = cubic2D(tx, ty, &f[16]);
        _f[2] = cubic2D(tx, ty, &f[32]);
        _f[3] = cubic2D(tx, ty, &f[48]);
        return cubic1D(tz, _f);
    }
    __host__ __device__ T cerp(int i0, int j0, int k0, T u, T v, T w)
    {
        int _i0 = i0, _i_ = i0 - 1, _i1 = i0 + 1, _i2 = i0 + 2;
        int _j0 = j0, _j_ = j0 - 1, _j1 = j0 + 1, _j2 = j0 + 2;
        int _k0 = k0, _k_ = k0 - 1, _k1 = k0 + 1, _k2 = k0 + 2;

        T tx = u;
        T ty = v;
        T tz = w;

        T f[64] = {
            get(_i_, _j_, _k_), get(_i0, _j_, _k_), get(_i1, _j_, _k_), get(_i2, _j_, _k_),
            get(_i_, _j0, _k_), get(_i0, _j0, _k_), get(_i1, _j0, _k_), get(_i2, _j0, _k_),
            get(_i_, _j1, _k_), get(_i0, _j1, _k_), get(_i1, _j1, _k_), get(_i2, _j1, _k_),
            get(_i_, _j2, _k_), get(_i0, _j2, _k_), get(_i1, _j2, _k_), get(_i2, _j2, _k_),
            get(_i_, _j_, _k0), get(_i0, _j_, _k0), get(_i1, _j_, _k0), get(_i2, _j_, _k0),
            get(_i_, _j0, _k0), get(_i0, _j0, _k0), get(_i1, _j0, _k0), get(_i2, _j0, _k0),
            get(_i_, _j1, _k0), get(_i0, _j1, _k0), get(_i1, _j1, _k0), get(_i2, _j1, _k0),
            get(_i_, _j2, _k0), get(_i0, _j2, _k0), get(_i1, _j2, _k0), get(_i2, _j2, _k0),
            get(_i_, _j_, _k1), get(_i0, _j_, _k1), get(_i1, _j_, _k1), get(_i2, _j_, _k1),
            get(_i_, _j0, _k1), get(_i0, _j0, _k1), get(_i1, _j0, _k1), get(_i2, _j0, _k1),
            get(_i_, _j1, _k1), get(_i0, _j1, _k1), get(_i1, _j1, _k1), get(_i2, _j1, _k1),
            get(_i_, _j2, _k1), get(_i0, _j2, _k1), get(_i1, _j2, _k1), get(_i2, _j2, _k1),
            get(_i_, _j_, _k2), get(_i0, _j_, _k2), get(_i1, _j_, _k2), get(_i2, _j_, _k2),
            get(_i_, _j0, _k2), get(_i0, _j0, _k2), get(_i1, _j0, _k2), get(_i2, _j0, _k2),
            get(_i_, _j1, _k2), get(_i0, _j1, _k2), get(_i1, _j1, _k2), get(_i2, _j1, _k2),
            get(_i_, _j2, _k2), get(_i0, _j2, _k2), get(_i1, _j2, _k2), get(_i2, _j2, _k2),
        };

        return cubic3D(tx, ty, tz, f);
    }
    __host__ __device__ inline int ix(int i, int j, int k) { return i + j * nx + k * nxy; }

public:
    __host__ __device__ grid(): data(NULL) {}
    __host__ __device__ grid(T cx, T cy, T cz): data(NULL), corner_x(cx), corner_y(cy), corner_z(cz) {}
    void init(int x, int y, int z) {
        if (data) return;
        nx = x, ny = y, nz = z;
        m_total = nx*ny*nz;
        nxy = nx*ny;
        data = new T[m_total];
        clear();
    }
    void init_gpu(int x, int y, int z) {
        if (data) return;
        nx = x, ny = y, nz = z;
        m_total = nx*ny*nz;
        nxy = nx*ny;
        checkCudaErrors(cudaMalloc((void**)&data, sizeof(T) * m_total));
        clear_gpu();
    }
    void free() {
        if (data) {
            delete [] data;
            data = NULL;
        }
    }
    void free_gpu() {
        checkCudaErrors(cudaFree(data));
        data = NULL;
    }
    void to_host(grid& h_data) {
        if(nx!=h_data.nx||ny!=h_data.ny||nz!=h_data.nz){
            printf("to_host:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(h_data.get_ptr(), data, sizeof(T) * m_total, cudaMemcpyDeviceToHost));
    }
    void to_device(grid& d_data) {
        if(nx!=d_data.nx||ny!=d_data.ny||nz!=d_data.nz){
            printf("to_device:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(d_data.get_ptr(), data, sizeof(T) * m_total, cudaMemcpyHostToDevice));
    }

    __host__ __device__ inline T& get(int i, int j, int k){ return data[ix(i, j, k)]; }
    __host__ __device__ inline T& get(int n){ return data[n]; }
    __host__ __device__ inline T *get_ptr(void){ return data; }
    __host__ __device__ inline int get_nx(){ return nx; }
    __host__ __device__ inline int get_ny(){ return ny; }
    __host__ __device__ inline int get_nz(){ return nz; }
    __host__ __device__ inline int get_size(){ return m_total; }
    void clear(){
        for (int n = 0; n < m_total; n++) data[n] = T(0);
    }
    void clear_gpu(){
        checkCudaErrors(cudaMemset(data, 0, sizeof(T) * m_total));
    }
    void swap(grid& a){
        if(nx!=a.nx||ny!=a.ny||nz!=a.nz){
            printf("not swapped!\n");
            return;
        }
        std::swap(data, a.data);
    }
    void copy_from(grid& a) { // it seems that the class is a friend of itself, so the private members can be freely accessed
        if(nx!=a.nx||ny!=a.ny||nz!=a.nz){
            printf("copy_from:: not copied!\n");
            return;
        }
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    this->get(i,j,k) = a.get(i,j,k);
                }
            }
        }
    }
    void copy_from_gpu(grid& a) { // it seems that the class is a friend of itself, so the private members can be freely accessed
        if(nx!=a.nx||ny!=a.ny||nz!=a.nz){
            printf("copy_from:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(data, a.data, sizeof(T) * m_total, cudaMemcpyDeviceToDevice));
    }

    __host__ __device__ T interp(T x, T y, T z){
        x -= corner_x;
        y -= corner_y;
        z -= corner_z;
        int x0;
        int y0;
        int z0;
        T u;
        T v;
        T w;
        bary(x, nx, x0, u);
        bary(y, ny, y0, v);
        bary(z, nz, z0, w);
        return lerp(x0, y0, z0, u, v, w);
    }
    __host__ __device__ T interp_cubic(T x, T y, T z){
        x -= corner_x;
        y -= corner_y;
        z -= corner_z;
        int x0;
        int y0;
        int z0;
        T u;
        T v;
        T w;
        bool bx = bary_cubic(x, nx, x0, u);
        bool by = bary_cubic(y, ny, y0, v);
        bool bz = bary_cubic(z, nz, z0, w);
        if (bx || by || bz) { // the cubic interpolation must be off the boundary for now
            bary(x, nx, x0, u);
            bary(y, ny, y0, v);
            bary(z, nz, z0, w);
            return lerp(x0, y0, z0, u, v, w);
        } else {
            return cerp(x0, y0, z0, u, v, w);
        }
    }
};
template<typename T> struct grid_cell  :public grid<T> { __host__ __device__ grid_cell  () : grid(0.5, 0.5, 0.5) {} };
template<typename T> struct grid_face_x:public grid<T> { __host__ __device__ grid_face_x() : grid(0.0, 0.5, 0.5) {} };
template<typename T> struct grid_face_y:public grid<T> { __host__ __device__ grid_face_y() : grid(0.5, 0.0, 0.5) {} };
template<typename T> struct grid_face_z:public grid<T> { __host__ __device__ grid_face_z() : grid(0.5, 0.5, 0.0) {} };

//////////////////////////////////////////////////////////////////////////
//raycasting in the -z direction;
#define EPS 1e-8
#define scale       1.0f
#define brightness  1.0f
class frame_buffer
{
    vec3 *m_buf;
    int m_width, m_height;
    int m_total;
    float m_factor;
    __host__ __device__ inline int ix(int i, int j) const { return i+j*m_width; }
    void swap_line(vec3 *a, vec3 *b, int total) {
        for (int n = 0; n < total; n++) {
            swap(a[n], b[n]);
        }
    }
public:
    __host__ __device__ frame_buffer(): m_buf(NULL) {}
    void free() { if (m_buf) delete [] m_buf; }
    void init(int _w, int _h){
        if (m_buf) return;
        m_width = _w;
        m_height = _h;
        m_total = m_width * m_height;
        m_buf = new vec3[m_total];
    }
    void free_gpu() { if (m_buf) checkCudaErrors(cudaFree(m_buf)); }
    void init_gpu(int _w, int _h){
        if (m_buf) return;
        m_width = _w;
        m_height = _h;
        m_total = m_width * m_height;
        checkCudaErrors(cudaMalloc((void**)&m_buf, sizeof(vec3) * m_total));
    }
    void to_host(frame_buffer& h_data) {
        if(m_width!=h_data.m_width||m_height!=h_data.m_height){
            printf("to_host:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(h_data.ptr(), m_buf, sizeof(vec3) * m_total, cudaMemcpyDeviceToHost));
    }
    void to_device(frame_buffer& d_data) {
        if(m_width!=d_data.m_width||m_height!=d_data.m_height){
            printf("to_device:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(d_data.ptr(), m_buf, sizeof(vec3) * m_total, cudaMemcpyHostToDevice));
    }
    void flipud(){
        for(int j=0; j<m_height/2; j++){
            swap_line(&m_buf[ix(0,j)], &m_buf[ix(0,m_height-1-j)], m_width);
        }
    }
    __host__ __device__ vec3& get(int i, int j){
        return m_buf[ix(i,j)];
    }
    __host__ __device__ const vec3& get(int i, int j) const {
        return m_buf[ix(i,j)];
    }
    __host__ __device__ vec3 *ptr() { return m_buf; }
    __host__ __device__ int getWidth() const { return m_width; }
    __host__ __device__ int getHeight() const { return m_height; }
    __host__ __device__ int getTotal() const { return m_total; }
};
__host__ __device__ float transfer_func_opacity(float x) {
    return x < 0.45 ? 0 
        : x < 0.55 ? 1
        :            0;
}
__host__ __device__ vec3 transfer_func_color(float x, float opacity) {
    return vec3(1, 0.95, 0.9) * scale * opacity * 10;
}
__host__ __device__ float simple_phong_shading(vec3 normal) {
    vec3 light_dir = normalize(vec3(1, 1, 1));
    vec3 eye_dir(0, 0, 1);
    vec3 half_dir = normalize((light_dir + eye_dir) * 0.5);
    return 0.1f + fabs(dot(light_dir, normal)) + 3.0f * powf(fabs(dot(half_dir, normal)), 60.0f);
}
__host__ __device__ float transfer_input(float x) {
    return x * 0.1 + 0.5;
}
__global__ void _simple_scalar_vr(grid_cell<float> vol, frame_buffer fb) {
    // c.f. http://http.developer.nvidia.com/GPUGems/gpugems_ch39.html
    KERNAL_CONFIG

    int width = fb.getWidth();
    int height = fb.getHeight();

    if ( i >= 0 && i < width &&
         j >= 0 && j < height )
    {
        int nx = vol.get_nx();
        int ny = vol.get_ny();
        int nz = vol.get_nz();

        vec3 accum(0, 0, 0);
        float throughput = 1.0;

        for (int k=nz * 2-1; k>=0; --k) { // front-to-back rendering
            vec3 pos((i + 0.5) / width * nx, (j + 0.5) / height * ny, (k + 0.5) / 2);
            float c = clampf(transfer_input(vol.interp(pos.x, pos.y, pos.z)), 0, 1);
            float opacity = transfer_func_opacity(c);
            vec3 estimated_normal = normalize(vec3( //locally filtered
                vol.interp(pos.x + 2, pos.y, pos.z) + vol.interp(pos.x + 1, pos.y, pos.z) - vol.interp(pos.x - 1, pos.y, pos.z) - vol.interp(pos.x - 2, pos.y, pos.z) + EPS,
                vol.interp(pos.x, pos.y + 2, pos.z) + vol.interp(pos.x, pos.y + 1, pos.z) - vol.interp(pos.x, pos.y - 1, pos.z) - vol.interp(pos.x, pos.y - 2, pos.z) + EPS,
                vol.interp(pos.x, pos.y, pos.z + 2) + vol.interp(pos.x, pos.y, pos.z + 1) - vol.interp(pos.x, pos.y, pos.z - 1) - vol.interp(pos.x, pos.y, pos.z - 2) + EPS
                )) * -1;
            float shading = simple_phong_shading(estimated_normal);
            accum += (transfer_func_color(c, opacity) * shading)
                * throughput;
            throughput *= (1 - opacity);
            if (throughput < 1e-4f) break;
        }
        fb.get(i, j) = accum * 0.16 * 0.5 * brightness;
    }
}
#undef EPS
#undef scale
#undef brightness

//////////////////////////////////////////////////////////////////////////
grid_cell<float>    sdistance;
grid_cell<float>    sdistance_temp;
grid_cell<char>     fluid_flag;
grid_face_x<float>  ux;
grid_face_y<float>  uy;
grid_face_z<float>  uz;
grid_face_x<float>  ux_temp;
grid_face_y<float>  uy_temp;
grid_face_z<float>  uz_temp;
grid_cell<float>    pressure;
grid_cell<int>      cell_index;
grid_cell<float>    poisson_0;
grid_cell<float>    poisson_1;
grid_cell<float>    poisson_2;
grid_cell<float>    poisson_3;
grid_cell<float>    rhs;
grid_cell<char>     vel_valid;
grid_cell<char>     temp_valid;

grid_cell<float> preconditioner;
grid_cell<float> m;
grid_cell<float> r;
grid_cell<float> z;
grid_cell<float> s;
grid_cell<float> res;

grid_cell<float>    d_sdistance;
grid_cell<float>    d_sdistance_temp;
grid_cell<char>     d_fluid_flag;
grid_face_x<float>  d_ux;
grid_face_y<float>  d_uy;
grid_face_z<float>  d_uz;
grid_face_x<float>  d_ux_temp;
grid_face_y<float>  d_uy_temp;
grid_face_z<float>  d_uz_temp;
grid_cell<float>    d_pressure;
grid_cell<int>      d_cell_index;
grid_cell<float>    d_poisson_0;
grid_cell<float>    d_poisson_1;
grid_cell<float>    d_poisson_2;
grid_cell<float>    d_poisson_3;
grid_cell<float>    d_rhs;
grid_cell<char>     d_vel_valid;
grid_cell<char>     d_temp_valid;
grid_cell<float>    d_temp_a;
grid_cell<float>    d_temp_b;

void init_memory() {
    sdistance       .init(N,N,N);      // signed distance function (levelset)
    sdistance_temp  .init(N,N,N); // used for both advection and extrapolation
    fluid_flag      .init(N,N,N);
    ux              .init(N+1,N  ,N  );
    uy              .init(N  ,N+1,N  );
    uz              .init(N  ,N  ,N+1);
    ux_temp         .init(N+1,N  ,N  );
    uy_temp         .init(N  ,N+1,N  );
    uz_temp         .init(N  ,N  ,N+1);
    pressure        .init(N,N,N);
    cell_index      .init(N, N, N);
    poisson_0       .init(N, N, N);     // poisson matrix (diag)    
    poisson_1       .init(N, N, N);     // poisson matrix (plus_x)
    poisson_2       .init(N, N, N);     // poisson matrix (plus_y)
    poisson_3       .init(N, N, N);     // poisson matrix (plus_z)
    rhs             .init(N, N, N);   // residual of Ax-b     
    vel_valid       .init(N+1, N+1, N+1);
    temp_valid      .init(N+1, N+1, N+1);

    preconditioner  .init(N, N, N);  // MIC(0) preconditioner 
    m               .init(N, N, N);   //
    r               .init(N, N, N);   // residual of Ax-b     
    z               .init(N, N, N);   // 
    s               .init(N, N, N);   //
//     res             = preconditioner;
    res             .init(N, N, N);

    d_sdistance       .init_gpu(N,N,N);
    d_sdistance_temp  .init_gpu(N,N,N);
    d_fluid_flag      .init_gpu(N,N,N);
    d_ux              .init_gpu(N+1,N  ,N  );
    d_uy              .init_gpu(N  ,N+1,N  );
    d_uz              .init_gpu(N  ,N  ,N+1);
    d_ux_temp         .init_gpu(N+1,N  ,N  );
    d_uy_temp         .init_gpu(N  ,N+1,N  );
    d_uz_temp         .init_gpu(N  ,N  ,N+1);
    d_pressure        .init_gpu(N,N,N);
    d_cell_index      .init_gpu(N, N, N);
    d_poisson_0       .init_gpu(N, N, N);
    d_poisson_1       .init_gpu(N, N, N);
    d_poisson_2       .init_gpu(N, N, N);
    d_poisson_3       .init_gpu(N, N, N);
    d_rhs             .init_gpu(N, N, N);
    d_vel_valid       .init_gpu(N+1, N+1, N+1);
    d_temp_valid      .init_gpu(N+1, N+1, N+1);
    d_temp_a       .init_gpu(N, N, N);
    d_temp_b       .init_gpu(N, N, N);
}

void free_memory() {
    sdistance       .free();
    sdistance_temp  .free();
    fluid_flag      .free();
    ux              .free();
    uy              .free();
    uz              .free();
    ux_temp         .free();
    uy_temp         .free();
    uz_temp         .free();
    pressure        .free();
    cell_index      .free();
    poisson_0       .free();
    poisson_1       .free();
    poisson_2       .free();
    poisson_3       .free();
    rhs             .free();
    vel_valid       .free();
    temp_valid      .free();

    preconditioner  .free();
    m               .free();
    r               .free();
    z               .free();
    s               .free();

    d_sdistance       .free_gpu();
    d_sdistance_temp  .free_gpu();
    d_fluid_flag      .free_gpu();
    d_ux              .free_gpu();
    d_uy              .free_gpu();
    d_uz              .free_gpu();
    d_ux_temp         .free_gpu();
    d_uy_temp         .free_gpu();
    d_uz_temp         .free_gpu();
    d_pressure        .free_gpu();
    d_cell_index      .free_gpu();
    d_poisson_0       .free_gpu();
    d_poisson_1       .free_gpu();
    d_poisson_2       .free_gpu();
    d_poisson_3       .free_gpu();
    d_rhs             .free_gpu();
    d_vel_valid       .free_gpu();
    d_temp_valid      .free_gpu();
    d_temp_a       .free_gpu();
    d_temp_b       .free_gpu();
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

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
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
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
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

    if ( i >= 0 && i < nx &&
         j >= 1 && j < ny-1 &&
         k >= 0 && k < nz )
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

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
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
    _advect<<<gridConfig, blockConfig>>>(d_sdistance_temp, d_sdistance, d_ux, d_uy, d_uz, dt);
    d_sdistance.swap(d_sdistance_temp);

    _advect<<<gridConfig, blockConfig>>>(d_ux_temp, d_ux, d_ux, d_uy, d_uz, dt);
    _advect<<<gridConfig, blockConfig>>>(d_uy_temp, d_uy, d_ux, d_uy, d_uz, dt);
    _advect<<<gridConfig, blockConfig>>>(d_uz_temp, d_uz, d_ux, d_uy, d_uz, dt);
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
        if (sdistance.get(i,j,k) * sdistance.get(i-1,j,k) < 0) {
            if (AIR == fluid_flag.get(i,j,k)) {
                pressure_p = sdistance.get(i,j,k) / f_min(1e-6f, sdistance.get(i-1,j,k)) * pressure.get(i-1,j,k);
            }
            if (AIR == fluid_flag.get(i-1,j,k)) {
                pressure_n = sdistance.get(i-1,j,k) / f_min(1e-6f, sdistance.get(i,j,k)) * pressure.get(i,j,k);
            }
        }
        ux.get(i,j,k) -= (pressure_p - pressure_n);
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
        if (sdistance.get(i,j,k) * sdistance.get(i,j-1,k) < 0) {
            if (AIR == fluid_flag.get(i,j,k)) {
                pressure_p = sdistance.get(i,j,k) / f_min(1e-6f, sdistance.get(i,j-1,k)) * pressure.get(i,j-1,k);
            }
            if (AIR == fluid_flag.get(i,j-1,k)) {
                pressure_n = sdistance.get(i,j-1,k) / f_min(1e-6f, sdistance.get(i,j,k)) * pressure.get(i,j,k);
            }
        }
        uy.get(i,j,k) -= (pressure_p - pressure_n);
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
        if (sdistance.get(i,j,k) * sdistance.get(i,j,k-1) < 0) {
            if (AIR == fluid_flag.get(i,j,k)) {
                pressure_p = sdistance.get(i,j,k) / f_min(1e-6f, sdistance.get(i,j,k-1)) * pressure.get(i,j,k-1);
            }
            if (AIR == fluid_flag.get(i,j,k-1)) {
                pressure_n = sdistance.get(i,j,k-1) / f_min(1e-6f, sdistance.get(i,j,k)) * pressure.get(i,j,k);
            }
        }
        uz.get(i,j,k) -= (pressure_p - pressure_n);
    }
}
void subtract_gradient_gpu() {
    _subtract_gradient<<<gridConfig, blockConfig>>>(d_ux, d_pressure, d_sdistance, d_fluid_flag);
    _subtract_gradient<<<gridConfig, blockConfig>>>(d_uy, d_pressure, d_sdistance, d_fluid_flag);
    _subtract_gradient<<<gridConfig, blockConfig>>>(d_uz, d_pressure, d_sdistance, d_fluid_flag);
}
__global__ void _form_poisson(grid_cell<float> poisson_0, grid_cell<float> poisson_1,
    grid_cell<float> poisson_2, grid_cell<float> poisson_3,
    grid_cell<char> fluid_flag, grid_cell<float> sdistance)
{
    KERNAL_CONFIG

    int nx = poisson_0.get_nx();
    int ny = poisson_0.get_ny();
    int nz = poisson_0.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if( LIQUID == fluid_flag.get(i, j, k) ) {
            float diag = 6.0;
            int q[][3] = {
                {i-1,j,k},
                {i+1,j,k}, 
                {i,j-1,k}, 
                {i,j+1,k}, 
                {i,j,k-1},
                {i,j,k+1}
            };
            for( int m=0; m<6; m++ ) {
                int qi = q[m][0];
                int qj = q[m][1];
                int qk = q[m][2];
                if( SOLID == fluid_flag.get(qi, qj, qk) ) {
                    diag -= 1.0;
                }
                else if( AIR == fluid_flag.get(qi, qj, qk) ) {
                    diag -= sdistance.get(qi, qj, qk) / f_min(1.0e-6, sdistance.get(i, j, k));
                }
            }
            poisson_0.get(i,j,k) = diag;
        }

        if (LIQUID == fluid_flag.get(i,j,k)) {
            if(LIQUID == fluid_flag.get(i+1,j,k))
                poisson_1.get(i,j,k)=-1;
            if(LIQUID == fluid_flag.get(i,j+1,k))
                poisson_2.get(i,j,k)=-1;
            if(LIQUID == fluid_flag.get(i,j,k+1))
                poisson_3.get(i,j,k)=-1;
        }
    }
}
__global__ void _calc_rhs(grid_cell<char> fluid_flag, grid_cell<float> rhs,
    grid_face_x<float> ux, grid_face_y<float> uy, grid_face_z<float> uz) {
    KERNAL_CONFIG

    int nx = rhs.get_nx();
    int ny = rhs.get_ny();
    int nz = rhs.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if (LIQUID == fluid_flag.get(i,j,k)) {
            rhs.get(i,j,k) = -  //negated divergence
                (ux.get(i+1,j,k)-ux.get(i,j,k)
                +uy.get(i,j+1,k)-uy.get(i,j,k)
                +uz.get(i,j,k+1)-uz.get(i,j,k));
        }
    }
}
void buildCellOrder_serial(grid_cell<char> fluid_flag, grid_cell<int> cell_index) { // build consistent ordering
    int nx = fluid_flag.get_nx();
    int ny = fluid_flag.get_ny();
    int nz = fluid_flag.get_nz();

    int cellIndex = 0;
    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                if(LIQUID == fluid_flag.get(i,j,k)){ //assume no fluid cell on bounding box
                    cell_index.get(i, j, k) = cellIndex;
                    cellIndex++;
                } else {
                    cell_index.get(i, j, k) = -1;
                }
            }
        }
    }
}
void formPoisson_serial(int *I, int *J, float *val, int &NN, int &NZ,
    grid_cell<float> poisson_0, grid_cell<float> poisson_1, grid_cell<float> poisson_2, grid_cell<float> poisson_3) {

    int nx = pressure.get_nx();
    int ny = pressure.get_ny();
    int nz = pressure.get_nz();

    NN = 0; // num of rows
    NZ = 0; // num of non-zeros

    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                if(LIQUID == fluid_flag.get(i,j,k)){ //assume no fluid cell on bounding box
                    NN++;

                    I[cell_index.get(i, j, k)] = NZ;

                    if (poisson_3.get(i, j, k-1)) {
                        J[NZ] = cell_index.get(i, j, k-1);
                        val[NZ] = poisson_3.get(i, j, k-1);
                        NZ++;
                    }
                    if (poisson_2.get(i, j-1, k)) {
                        J[NZ] = cell_index.get(i, j-1, k);
                        val[NZ] = poisson_2.get(i, j-1, k);
                        NZ++;
                    }
                    if (poisson_1.get(i-1, j, k)) {
                        J[NZ] = cell_index.get(i-1, j, k);
                        val[NZ] = poisson_1.get(i-1, j, k);
                        NZ++;
                    }

                    if (poisson_0.get(i, j, k)) {
                        J[NZ] = cell_index.get(i, j, k);
                        val[NZ] = poisson_0.get(i, j, k);
                        NZ++;
                    }

                    if (poisson_1.get(i, j, k)) {
                        J[NZ] = cell_index.get(i+1, j, k);
                        val[NZ] = poisson_1.get(i, j, k);
                        NZ++;
                    }
                    if (poisson_2.get(i, j, k)) {
                        J[NZ] = cell_index.get(i, j+1, k);
                        val[NZ] = poisson_2.get(i, j, k);
                        NZ++;
                    }
                    if (poisson_3.get(i, j, k)) {
                        J[NZ] = cell_index.get(i, j, k+1);
                        val[NZ] = poisson_3.get(i, j, k);
                        NZ++;
                    }
                }
            }
        }
    }
    I[NN] = NZ;
}
__global__ void _init_rhs_and_x(float *new_rhs, float *x, grid_cell<char> fluid_flag, grid_cell<int> cell_index, grid_cell<float> rhs) {
    KERNAL_CONFIG

    int nx = rhs.get_nx();
    int ny = rhs.get_ny();
    int nz = rhs.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if(LIQUID == fluid_flag.get(i,j,k)){ //assume no fluid cell on bounding box
            new_rhs[cell_index.get(i,j,k)] = rhs.get(i,j,k);
            x[cell_index.get(i,j,k)] = 0;
        }
    }
}
__global__ void _apply_result(float *x, grid_cell<float> pressure, grid_cell<char> fluid_flag, grid_cell<int> cell_index) {
    KERNAL_CONFIG

    int nx = pressure.get_nx();
    int ny = pressure.get_ny();
    int nz = pressure.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if(LIQUID == fluid_flag.get(i,j,k)){ //assume no fluid cell on bounding box
            pressure.get(i,j,k) = x[cell_index.get(i,j,k)];
        }
        else {
            pressure.get(i,j,k) = 0;
        }
    }
}
struct sparse_matrix_csr {}; // TODO
__global__ void _toSparse(float *x, grid_cell<float> d_x, grid_cell<char> fluid_flag,
                            grid_cell<int> cell_index) {
    KERNAL_CONFIG

    int nx = d_x.get_nx();
    int ny = d_x.get_ny();
    int nz = d_x.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if(LIQUID == fluid_flag.get(i,j,k)){ //assume no fluid cell on bounding box
            x[cell_index.get(i,j,k)] = d_x.get(i,j,k);
        }
    }
}
__global__ void _fromSparse(float *x, grid_cell<float> d_x, grid_cell<char> fluid_flag,
                            grid_cell<int> cell_index) {
    KERNAL_CONFIG

    int nx = d_x.get_nx();
    int ny = d_x.get_ny();
    int nz = d_x.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if(LIQUID == fluid_flag.get(i,j,k)){ //assume no fluid cell on bounding box
            d_x.get(i,j,k) = x[cell_index.get(i,j,k)];
        }
        else {
            d_x.get(i,j,k) = 0;
        }
    }
}

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implements a conjugate gradient solver on GPU
 * using CUBLAS and CUSPARSE
 *
 */

void pcg_solve_poisson_gpu_ilu0()
{
    /* Laplacian matrix in CSR format */
    const int max_iter = 1000;
    int k, _M = 0, _N = 0, _nz = 0, *I = NULL, *J = NULL;
    int *d_col, *d_row;
    const float tol = 1e-5f;
    float r0, r1, alpha, beta;
    float *d_val, *d_x;
    float *d_z;
    float *d_r, *d_p, *d_omega, *d_y;
    float *val = NULL;
    float *d_valsILU0;
    float *valsILU0;
    float rsum, diff, err = 0.0;
    float qaerr1, qaerr2 = 0.0;
    float dot, numerator, denominator, nalpha;
    const float floatone = 1.0;
    const float floatzero = 0.0;

    int ntotal =
             pressure.get_nx()*
             pressure.get_ny()*
             pressure.get_nz();

    __sync();
    Profiler _p0;
    buildCellOrder_serial(fluid_flag, cell_index);
    cell_index.to_device(d_cell_index);
    __sync();
    printf("\t build cell order (serial) took %.5f seconds\n", _p0.get());

    // allocate to the maximum
    I   = new int  [ntotal + 1]; // row expressed as a range of the non-zeros
    J   = new int  [ntotal * 7]; // column indices in matrix of the non-zeros
    val = new float[ntotal * 7];

    d_rhs.clear_gpu();
    _calc_rhs<<<gridConfig, blockConfig>>>(d_fluid_flag, d_rhs, d_ux, d_uy, d_uz);

    d_poisson_0.clear_gpu();
    d_poisson_1.clear_gpu();
    d_poisson_2.clear_gpu();
    d_poisson_3.clear_gpu();
    _form_poisson<<<gridConfig, blockConfig>>>(d_poisson_0, d_poisson_1,
        d_poisson_2, d_poisson_3, d_fluid_flag, d_sdistance);

    __sync();
    Profiler _p1;
    d_poisson_0.to_host(poisson_0);
    d_poisson_1.to_host(poisson_1);
    d_poisson_2.to_host(poisson_2);
    d_poisson_3.to_host(poisson_3);
    formPoisson_serial(I, J, val, _N, _nz, poisson_0, poisson_1, poisson_2, poisson_3);
    __sync();
    printf("\t form poisson (serial) took %.5f seconds\n", _p0.get());

//     checkCudaErrors(cudaMalloc((void **)&d_x, _N*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_r, _N*sizeof(float)));
    d_x = (float *) d_rhs.get_ptr();          // reusing memory
    d_r = (float *) d_pressure.get_ptr();     // reusing memory

    // by using cell_index, d_r and d_x do not have to be of maximum size
    _init_rhs_and_x<<<gridConfig, blockConfig>>>(d_r, d_x, d_fluid_flag, d_cell_index, d_rhs);

    //////////////////////////////////////////////////////////////////////////
    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    checkCudaErrors(cublasCreate(&cublasHandle));

    // in case the rhs is zero when all fluid is free falling
    {
        cublasSdot(cublasHandle, _N, d_r, 1, d_r, 1, &dot); // r' * r
        if (dot <= tol*tol) {
            delete [] I;
            delete [] J;
            delete [] val;
            cublasDestroy(cublasHandle);
            return;
        }
    }

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    cusparseMatDescr_t descr = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descr));

    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    checkCudaErrors(cudaMalloc((void **)&d_row, (_N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_col, _nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, _nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_valsILU0, _nz*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_y, _N*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_p, _N*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_omega, _N*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_z, (_N)*sizeof(float)));
    d_y     = (float *) d_poisson_0.get_ptr();  // reusing memory
    d_p     = (float *) d_poisson_1.get_ptr();  // reusing memory
    d_omega = (float *) d_poisson_2.get_ptr();  // reusing memory
    d_z     = (float *) d_poisson_3.get_ptr();  // reusing memory

    cudaMemcpy(d_row, I, (_N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, J, _nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, _nz*sizeof(float), cudaMemcpyHostToDevice);

    printf("\tConvergence of conjugate gradient using incomplete LU preconditioning: \n");

    /* create the analysis info object for the A matrix */
    cusparseSolveAnalysisInfo_t infoA = 0;
    checkCudaErrors(cusparseCreateSolveAnalysisInfo(&infoA));

    /* Perform the analysis for the Non-Transpose case */
    checkCudaErrors(cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             _N, _nz, descr, d_val, d_row, d_col, infoA));

    /* Copy A data to ILU0 vals as input*/
    cudaMemcpy(d_valsILU0, d_val, _nz*sizeof(float), cudaMemcpyDeviceToDevice);

    /* generate the Incomplete LU factor H for the matrix A using cudsparseScsrilu0 */ // in place??
    checkCudaErrors(cusparseScsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            _N, descr, d_valsILU0, d_row, d_col, infoA));

    /* Create info objects for the ILU0 preconditioner */
    cusparseSolveAnalysisInfo_t info_u;
    cusparseCreateSolveAnalysisInfo(&info_u);

    cusparseMatDescr_t descrL = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descrL));
    cusparseSetMatType(descrL,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrL,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseMatDescr_t descrU = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descrU));
    cusparseSetMatType(descrU,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrU,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);

    checkCudaErrors(cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            _N, _nz, descrU, d_val, d_row, d_col, info_u));
    
    //////////////////////////////////////////////////////////////////////////
//     Profiler _s1;
    // Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
    checkCudaErrors(cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, _N, &floatone, descrL,
                                            d_valsILU0, d_row, d_col, infoA, d_r, d_y)); // L * y = r
//     printf("\t solve phase 1 took %f seconds\n", _s1.get());
//     Profiler _s2;
    // Back Substitution
    checkCudaErrors(cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, _N, &floatone, descrU,
                                            d_valsILU0, d_row, d_col, info_u, d_y, d_z)); // U * z = y, (z = inv(M) * r)
//     printf("\t solve phase 2 took %f seconds\n", _s1.get());

    cublasScopy(cublasHandle, _N, d_z, 1, d_p, 1); // p := z
    cublasSdot(cublasHandle, _N, d_r, 1, d_z, 1, &r1); // r' * z

    k = 0;
    while (k++ < max_iter)
    {
//         Profiler _s0;
//         cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
//             _N, _N, _nz, &floatone, descrU, d_val, d_row, d_col, d_p, &floatzero, d_omega); // A * p
        cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
            _N, _N, _nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega); // A * p
//         printf("\t A * p took %f seconds\n", _s0.get());
        cublasSdot(cublasHandle, _N, d_p, 1, d_omega, 1, &dot); // p' * A * p
        alpha = r1 / dot;
        cublasSaxpy(cublasHandle, _N, &alpha, d_p, 1, d_x, 1); // x + a * p
        nalpha = -alpha;
        cublasSaxpy(cublasHandle, _N, &nalpha, d_omega, 1, d_r, 1); // r - a * A * p

        cublasSdot(cublasHandle, _N, d_r, 1, d_r, 1, &dot); // r' * r
        if (dot <= tol*tol) break;

        // Forward Solve, we can re-use infoA since the sparsity pattern of A matches that of L
        checkCudaErrors(cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, _N, &floatone, descrL,
                                                d_valsILU0, d_row, d_col, infoA, d_r, d_y)); // L * y = r
        // Back Substitution
        checkCudaErrors(cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, _N, &floatone, descrU,
                                                d_valsILU0, d_row, d_col, info_u, d_y, d_z)); // U * z = y, (z = inv(M) * r)

        r0 = r1;
        cublasSdot(cublasHandle, _N, d_r, 1, d_z, 1, &r1); // r' * z
        beta = r1/r0;
        cublasSscal(cublasHandle, _N, &beta, d_p, 1); // b * p
        cublasSaxpy(cublasHandle, _N, &floatone, d_z, 1, d_p, 1) ; // z + b * p
    }

    printf("\titeration = %3d, residual = %e \n", k, sqrt(r1));
    //////////////////////////////////////////////////////////////////////////

    _apply_result<<<gridConfig, blockConfig>>>(d_x, d_pressure, d_fluid_flag, d_cell_index);

    //////////////////////////////////////////////////////////////////////////
    /* Destroy parameters */
    cusparseDestroySolveAnalysisInfo(infoA);
    cusparseDestroySolveAnalysisInfo(info_u);

    /* Destroy contexts */
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    /* Free device memory */
    free(I);
    free(J);
    free(val);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_valsILU0);
//     cudaFree(d_x);
//     cudaFree(d_r);
//     cudaFree(d_y);
//     cudaFree(d_p);
//     cudaFree(d_omega);
//     cudaFree(d_z);
}

void pcg_solve_poisson_gpu_ic0()
{
    /* Laplacian matrix in CSR format */
    const int max_iter = 1000;
    int k, _M = 0, _N = 0, _nz = 0, *I = NULL, *J = NULL;
    int *d_col, *d_row;
    const float tol = 1e-5f;
    float r0, r1, alpha, beta;
    float *d_val, *d_x;
    float *d_z;
    float *d_r, *d_p, *d_omega, *d_y;
    float *val = NULL;
    float *d_valsIC0;
    float rsum, diff, err = 0.0;
    float qaerr1, qaerr2 = 0.0;
    float dot, numerator, denominator, nalpha;
    const float floatone = 1.0;
    const float floatzero = 0.0;

    int ntotal =
             pressure.get_nx()*
             pressure.get_ny()*
             pressure.get_nz();

    __sync();
    Profiler _p0;
    buildCellOrder_serial(fluid_flag, cell_index);
    cell_index.to_device(d_cell_index);
    __sync();
    printf("\t build cell order (serial) took %.5f seconds\n", _p0.get());

    // allocate to the maximum
    I   = new int  [ntotal + 1]; // row expressed as a range of the non-zeros
    J   = new int  [ntotal * 7]; // column indices in matrix of the non-zeros
    val = new float[ntotal * 7];

    d_rhs.clear_gpu();
    _calc_rhs<<<gridConfig, blockConfig>>>(d_fluid_flag, d_rhs, d_ux, d_uy, d_uz);

    d_poisson_0.clear_gpu();
    d_poisson_1.clear_gpu();
    d_poisson_2.clear_gpu();
    d_poisson_3.clear_gpu();
    _form_poisson<<<gridConfig, blockConfig>>>(d_poisson_0, d_poisson_1,
        d_poisson_2, d_poisson_3, d_fluid_flag, d_sdistance);

    __sync();
    Profiler _p1;
    d_poisson_0.to_host(poisson_0);
    d_poisson_1.to_host(poisson_1);
    d_poisson_2.to_host(poisson_2);
    d_poisson_3.to_host(poisson_3);
    formPoisson_serial(I, J, val, _N, _nz, poisson_0, poisson_1, poisson_2, poisson_3);
    __sync();
    printf("\t form poisson (serial) took %.5f seconds\n", _p0.get());

//     checkCudaErrors(cudaMalloc((void **)&d_x, _N*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_r, _N*sizeof(float)));
    d_x = (float *) d_rhs.get_ptr();          // reusing memory
    d_r = (float *) d_pressure.get_ptr();     // reusing memory

    // by using cell_index, d_r and d_x do not have to be of maximum size
    _init_rhs_and_x<<<gridConfig, blockConfig>>>(d_r, d_x, d_fluid_flag, d_cell_index, d_rhs);

    //////////////////////////////////////////////////////////////////////////
    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    checkCudaErrors(cublasCreate(&cublasHandle));

    // in case the rhs is zero when all fluid is free falling
    {
        cublasSdot(cublasHandle, _N, d_r, 1, d_r, 1, &dot); // r' * r
        if (dot <= tol*tol) {
            delete [] I;
            delete [] J;
            delete [] val;
            cublasDestroy(cublasHandle);
            return;
        }
    }

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    checkCudaErrors(cudaMalloc((void **)&d_row, (_N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_col, _nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, _nz*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_y, _N*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_p, _N*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_omega, _N*sizeof(float)));
//     checkCudaErrors(cudaMalloc((void **)&d_z, (_N)*sizeof(float)));
    d_y     = (float *) d_poisson_0.get_ptr();  // reusing memory
    d_p     = (float *) d_poisson_1.get_ptr();  // reusing memory
    d_omega = (float *) d_poisson_2.get_ptr();  // reusing memory
    d_z     = (float *) d_poisson_3.get_ptr();  // reusing memory

    cudaMemcpy(d_row, I, (_N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, J, _nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, _nz*sizeof(float), cudaMemcpyHostToDevice);

    printf("\tConvergence of conjugate gradient using incomplete Cholesky preconditioning: \n");

    int *d_lower_col;
    int *d_lower_row;
    float *d_lower_val;
    int lower_nz;
    {
        int *lower_I;         // because lower triangular times vector is too slow
        int *lower_J;         // because lower triangular times vector is too slow
        float *lower_val;     // because lower triangular times vector is too slow
        // to lower triangle
        int nlower = (_N + _nz) / 2;
        lower_I = new int[_N + 1];
        lower_J = new int[nlower];
        lower_val = new float[nlower];

        int k = 0;
        for (int i = 0; i < _N; i++) {
            lower_I[i] = k;
            for (int j = I[i]; j < I[i+1]; j++) {
                if (!(J[j] > i)) {
                    lower_J[k] = J[j];
                    lower_val[k] = val[j];
                    k++;
                }
            }
        }
        lower_I[_N] = nlower;
        lower_nz = nlower;

        checkCudaErrors(cudaMalloc((void **)&d_lower_row, (_N+1)*sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_lower_col, lower_nz*sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_lower_val, lower_nz*sizeof(float)));
        cudaMemcpy(d_lower_row, lower_I, (_N+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lower_col, lower_J, lower_nz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lower_val, lower_val, lower_nz*sizeof(float), cudaMemcpyHostToDevice);
        free(lower_I);
        free(lower_J);
        free(lower_val);
//         swap(I, lower_I);
//         swap(J, lower_J);
//         swap(val, lower_val);
//         delete [] new_I;
//         delete [] new_J;
//         delete [] new_val;
//         full_I = new_I;
//         full_J = new_J;
//         full_val = new_val;
//         full_nz = _nz;
    }
//     checkCudaErrors(cudaMalloc((void **)&d_valsIC0, _nz*sizeof(float)));
    cusparseMatDescr_t descrLower = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descrLower));
    cusparseSetMatType(descrLower, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatFillMode(descrLower, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrLower, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(descrLower, CUSPARSE_INDEX_BASE_ZERO);

    /* create the analysis info object for the A matrix */
    cusparseSolveAnalysisInfo_t infoA = 0;
    checkCudaErrors(cusparseCreateSolveAnalysisInfo(&infoA));

    /* Perform the analysis for the Non-Transpose case */
    checkCudaErrors(cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             _N, lower_nz, descrLower, d_lower_val, d_lower_row, d_lower_col, infoA));

    /* Copy A data to IC0 vals as input*/
//     cudaMemcpy(d_valsIC0, d_lower_val, lower_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    d_valsIC0 = d_lower_val; // use in place

    /* generate the Incomplete Cholesky factor H for the matrix A using cudsparseScsric0 */ // in place??
    checkCudaErrors(cusparseScsric0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            _N, descrLower, d_valsIC0, d_lower_row, d_lower_col, infoA));

    cusparseDestroySolveAnalysisInfo(infoA);
//     cudaFree(d_lower_val);
    cusparseDestroyMatDescr(descrLower);

    //////////////////////////////////////////////////////////////////////////
    // describe the lower triangular matrix of the Cholesky factor
    cusparseMatDescr_t descrL = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descrL));
    cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
    cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);

    // create the analysis info object for the lower Cholesky factor
    cusparseSolveAnalysisInfo_t infoL = 0;
    checkCudaErrors(cusparseCreateSolveAnalysisInfo(&infoL));
    checkCudaErrors(cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        _N, lower_nz, descrL, d_valsIC0, d_lower_row, d_lower_col, infoL));

    // create the analysis info object for the upper Cholesky factor
    cusparseSolveAnalysisInfo_t infoU = 0;
    checkCudaErrors(cusparseCreateSolveAnalysisInfo(&infoU));
    checkCudaErrors(cusparseScsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
        _N, lower_nz, descrL, d_valsIC0, d_lower_row, d_lower_col, infoU));

    //////////////////////////////////////////////////////////////////////////
    cusparseMatDescr_t descr = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

//     Profiler _s1;
    checkCudaErrors(cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, _N, &floatone, descrL,
                                            d_valsIC0, d_lower_row, d_lower_col, infoL, d_r, d_y)); // L * y = r
//     printf("\t solve phase 1 took %f seconds\n", _s1.get());
//     Profiler _s2;
    checkCudaErrors(cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, _N, &floatone, descrL,
                                            d_valsIC0, d_lower_row, d_lower_col, infoU, d_y, d_z)); // L' * z = y, (z = inv(M) * r)
//     printf("\t solve phase 2 took %f seconds\n", _s1.get());

    cublasScopy(cublasHandle, _N, d_z, 1, d_p, 1); // p := z
    cublasSdot(cublasHandle, _N, d_r, 1, d_z, 1, &r1); // r' * z

    k = 0;
    while (k++ < max_iter)
    {
//         Profiler _s0;
//         cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, // too slow
//             _N, _N, _nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega); // A * p
        cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
            _N, _N, _nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega); // A * p
//         printf("\t A * p took %f seconds\n", _s0.get());
        cublasSdot(cublasHandle, _N, d_p, 1, d_omega, 1, &dot); // p' * A * p
        alpha = r1 / dot;
        cublasSaxpy(cublasHandle, _N, &alpha, d_p, 1, d_x, 1); // x + a * p
        nalpha = -alpha;
        cublasSaxpy(cublasHandle, _N, &nalpha, d_omega, 1, d_r, 1); // r - a * A * p

        cublasSdot(cublasHandle, _N, d_r, 1, d_r, 1, &dot); // r' * r
        if (dot <= tol*tol) break;

        checkCudaErrors(cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, _N, &floatone, descrL,
                                                d_valsIC0, d_lower_row, d_lower_col, infoL, d_r, d_y)); // L * y = r
        checkCudaErrors(cusparseScsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, _N, &floatone, descrL,
                                                d_valsIC0, d_lower_row, d_lower_col, infoU, d_y, d_z)); // L' * z = y, (z = inv(M) * r)

        r0 = r1;
        cublasSdot(cublasHandle, _N, d_r, 1, d_z, 1, &r1); // r' * z
        beta = r1/r0;
        cublasSscal(cublasHandle, _N, &beta, d_p, 1); // b * p
        cublasSaxpy(cublasHandle, _N, &floatone, d_z, 1, d_p, 1) ; // z + b * p
    }

    printf("\titeration = %3d, residual = %e \n", k, sqrt(r1));
    //////////////////////////////////////////////////////////////////////////

    _apply_result<<<gridConfig, blockConfig>>>(d_x, d_pressure, d_fluid_flag, d_cell_index);

    //////////////////////////////////////////////////////////////////////////
    /* Destroy parameters */
    cusparseDestroySolveAnalysisInfo(infoL);
    cusparseDestroySolveAnalysisInfo(infoU);
    cusparseDestroyMatDescr(descrL);

    /* Destroy contexts */
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    /* Free device memory */
    free(I);
    free(J);
    free(val);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_valsIC0);
    cudaFree(d_lower_col);
    cudaFree(d_lower_row);
//     cudaFree(d_x);
//     cudaFree(d_r);
//     cudaFree(d_y);
//     cudaFree(d_p);
//     cudaFree(d_omega);
//     cudaFree(d_z);
}

__global__ void _RBGS(grid_cell<float> pressure, grid_cell<float> rhs, grid_cell<float> poisson_0, grid_cell<float> poisson_1,
                        grid_cell<float> poisson_2, grid_cell<float> poisson_3,
                        grid_cell<char> fluid_flag, int red_or_black) {
    KERNAL_CONFIG

    int nx = pressure.get_nx();
    int ny = pressure.get_ny();
    int nz = pressure.get_nz();

    if ( i >= 0 && i < nx &&
         j >= 0 && j < ny &&
         k >= 0 && k < nz )
    {
        if ((i + j + k) % 2 == red_or_black && LIQUID == fluid_flag.get(i, j, k)) {
            float p_x0 = pressure.get(i-1, j, k) * poisson_1.get(i-1, j  , k  );
            float p_x1 = pressure.get(i+1, j, k) * poisson_1.get(i  , j  , k  );
            float p_y0 = pressure.get(i, j-1, k) * poisson_2.get(i  , j-1, k  );
            float p_y1 = pressure.get(i, j+1, k) * poisson_2.get(i  , j  , k  );
            float p_z0 = pressure.get(i, j, k-1) * poisson_3.get(i  , j  , k-1);
            float p_z1 = pressure.get(i, j, k+1) * poisson_3.get(i  , j  , k  );
            float diag = poisson_0.get(i, j, k);
            pressure.get(i, j, k) = (rhs.get(i, j, k) - (p_x0 + p_x1 + p_y0 + p_y1 + p_z0 + p_z1)) / diag;
        }
    }
}

void pcg_solve_poisson_gpu_debug() {
    /* Laplacian matrix in CSR format */
    const int max_iter = 1000;
    int k, _M = 0, _N = 0, _nz = 0, *I = NULL, *J = NULL;
    int *d_col, *d_row;
    const float tol = 1e-5f;
    float r0, r1, alpha, beta;
    float *d_val, *d_x;
    float *d_z;
    float *d_r, *d_p, *d_omega, *d_y;
    float *val = NULL;
    float rsum, diff, err = 0.0;
    float qaerr1, qaerr2 = 0.0;
    float dot, numerator, denominator, nalpha;
    const float floatone = 1.0;
    const float floatzero = 0.0;

    int ntotal =
        pressure.get_nx()*
        pressure.get_ny()*
        pressure.get_nz();

    d_rhs.clear_gpu();
    _calc_rhs<<<gridConfig, blockConfig>>>(d_fluid_flag, d_rhs, d_ux, d_uy, d_uz);

    d_poisson_0.clear_gpu();
    d_poisson_1.clear_gpu();
    d_poisson_2.clear_gpu();
    d_poisson_3.clear_gpu();
    _form_poisson<<<gridConfig, blockConfig>>>(d_poisson_0, d_poisson_1,
        d_poisson_2, d_poisson_3, d_fluid_flag, d_sdistance);

    d_pressure.clear_gpu();
    //////////////////////////////////////////////////////////////////////////
    {
        for (int k = 0; k < 200; k++) {
            _RBGS<<<gridConfig, blockConfig>>>(d_pressure, d_rhs, d_poisson_0, d_poisson_1,
                d_poisson_2, d_poisson_3, d_fluid_flag, 0);
            _RBGS<<<gridConfig, blockConfig>>>(d_pressure, d_rhs, d_poisson_0, d_poisson_1,
                d_poisson_2, d_poisson_3, d_fluid_flag, 1);
        }
    }
}

void pcg_solve_poisson_gpu()
{
    /* Laplacian matrix in CSR format */
    const int max_iter = 1000;
    int k, _M = 0, _N = 0, _nz = 0, *I = NULL, *J = NULL;
    int *d_col, *d_row;
    const float tol = 1e-5f;
    float r0, r1, alpha, beta;
    float *d_val, *d_x;
    float *d_z;
    float *d_r, *d_p, *d_omega, *d_y;
    float *val = NULL;
    float dot, numerator, denominator, nalpha;
    const float floatone = 1.0;
    const float floatzero = 0.0;
    bool no_precon = false;

    int ntotal =
        pressure.get_nx()*
        pressure.get_ny()*
        pressure.get_nz();

    __sync();
    Profiler _p0;
    buildCellOrder_serial(fluid_flag, cell_index);
    cell_index.to_device(d_cell_index);
    __sync();
    printf("\t build cell order (serial) took %.5f seconds\n", _p0.get());

    // allocate to the maximum
    I   = new int  [ntotal + 1]; // row expressed as a range of the non-zeros
    J   = new int  [ntotal * 7]; // column indices in matrix of the non-zeros
    val = new float[ntotal * 7];

    d_rhs.clear_gpu();
    _calc_rhs<<<gridConfig, blockConfig>>>(d_fluid_flag, d_rhs, d_ux, d_uy, d_uz);

    d_poisson_0.clear_gpu();
    d_poisson_1.clear_gpu();
    d_poisson_2.clear_gpu();
    d_poisson_3.clear_gpu();
    _form_poisson<<<gridConfig, blockConfig>>>(d_poisson_0, d_poisson_1,
        d_poisson_2, d_poisson_3, d_fluid_flag, d_sdistance);

    __sync();
    Profiler _p1;
    d_poisson_0.to_host(poisson_0);
    d_poisson_1.to_host(poisson_1);
    d_poisson_2.to_host(poisson_2);
    d_poisson_3.to_host(poisson_3);
    formPoisson_serial(I, J, val, _N, _nz, poisson_0, poisson_1, poisson_2, poisson_3);
    __sync();
    printf("\t form poisson (serial) took %.5f seconds\n", _p0.get());

    d_x = (float *) d_rhs.get_ptr();          // reusing memory   (_N floats)
    d_r = (float *) d_pressure.get_ptr();     // reusing memory   (_N floats)

    // by using cell_index, d_r and d_x do not have to be of maximum size
    _init_rhs_and_x<<<gridConfig, blockConfig>>>(d_r, d_x, d_fluid_flag, d_cell_index, d_rhs);

    //////////////////////////////////////////////////////////////////////////
    cublasHandle_t cublasHandle = 0;
    checkCudaErrors(cublasCreate(&cublasHandle));

    // in case the rhs is zero when all fluid is free falling
    {
        cublasSdot(cublasHandle, _N, d_r, 1, d_r, 1, &dot); // r' * r
        if (dot <= tol*tol) {
            delete [] I;
            delete [] J;
            delete [] val;
            cublasDestroy(cublasHandle);
            return;
        }
    }

    cusparseHandle_t cusparseHandle = 0;
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    cusparseMatDescr_t descr = 0;
    checkCudaErrors(cusparseCreateMatDescr(&descr));
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    checkCudaErrors(cudaMalloc((void **)&d_row, (_N+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_col, _nz*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_val, _nz*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_y, _N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, _N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_omega, _N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_z, (_N)*sizeof(float)));

    cudaMemcpy(d_row, I, (_N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, J, _nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, _nz*sizeof(float), cudaMemcpyHostToDevice);

    printf("\tConvergence of conjugate gradient: \n");

    //////////////////////////////////////////////////////////////////////////
    if (no_precon) {
        cublasScopy(cublasHandle, _N, d_r, 1, d_z, 1); // z := r
    } else {
        d_temp_a.clear_gpu(); // for precon, must reset
        _fromSparse<<<gridConfig, blockConfig>>>(d_r, d_temp_b, d_fluid_flag, d_cell_index);
        for (int k = 0; k < 10; k++) {
            _RBGS<<<gridConfig, blockConfig>>>(d_temp_a, d_temp_b, d_poisson_0, d_poisson_1,
                                                d_poisson_2, d_poisson_3, d_fluid_flag, 0);
            _RBGS<<<gridConfig, blockConfig>>>(d_temp_a, d_temp_b, d_poisson_0, d_poisson_1,
                                                d_poisson_2, d_poisson_3, d_fluid_flag, 1);
        }
        _toSparse<<<gridConfig, blockConfig>>>(d_z, d_temp_a, d_fluid_flag, d_cell_index);
    }
    cublasScopy(cublasHandle, _N, d_z, 1, d_p, 1); // p := z
    cublasSdot(cublasHandle, _N, d_r, 1, d_z, 1, &r1); // r' * z

    k = 0;
    while (k++ < max_iter)
    {
        //         Profiler _s0;
        cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
            _N, _N, _nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega); // A * p
        //         printf("\t A * p took %f seconds\n", _s0.get());
        cublasSdot(cublasHandle, _N, d_p, 1, d_omega, 1, &dot); // p' * A * p
        alpha = r1 / dot;
        cublasSaxpy(cublasHandle, _N, &alpha, d_p, 1, d_x, 1); // x + a * p
        nalpha = -alpha;
        cublasSaxpy(cublasHandle, _N, &nalpha, d_omega, 1, d_r, 1); // r - a * A * p

        cublasSdot(cublasHandle, _N, d_r, 1, d_r, 1, &dot); // r' * r
        if (dot <= tol*tol) break;

        if (no_precon) {
            cublasScopy(cublasHandle, _N, d_r, 1, d_z, 1); // z := r
        } else {
            d_temp_a.clear_gpu(); // for precon, must reset
            _fromSparse<<<gridConfig, blockConfig>>>(d_r, d_temp_b, d_fluid_flag, d_cell_index);
            for (int k = 0; k < 10; k++) {
                _RBGS<<<gridConfig, blockConfig>>>(d_temp_a, d_temp_b, d_poisson_0, d_poisson_1,
                    d_poisson_2, d_poisson_3, d_fluid_flag, 0);
                _RBGS<<<gridConfig, blockConfig>>>(d_temp_a, d_temp_b, d_poisson_0, d_poisson_1,
                    d_poisson_2, d_poisson_3, d_fluid_flag, 1);
            }
            _toSparse<<<gridConfig, blockConfig>>>(d_z, d_temp_a, d_fluid_flag, d_cell_index);
        }

        r0 = r1;
        cublasSdot(cublasHandle, _N, d_r, 1, d_z, 1, &r1); // r' * z
        beta = r1/r0;
        cublasSscal(cublasHandle, _N, &beta, d_p, 1); // b * p
        cublasSaxpy(cublasHandle, _N, &floatone, d_z, 1, d_p, 1) ; // z + b * p
    }

    printf("\titeration = %3d, residual = %e \n", k, sqrt(r1));
    //////////////////////////////////////////////////////////////////////////

    _apply_result<<<gridConfig, blockConfig>>>(d_x, d_pressure, d_fluid_flag, d_cell_index);

    //////////////////////////////////////////////////////////////////////////
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);

    free(I);
    free(J);
    free(val);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_y);
    cudaFree(d_p);
    cudaFree(d_omega);
    cudaFree(d_z);
}

double dot(grid_cell<float>& x, grid_cell<float>& y) {
    double ans = 0.0;
    for( int i=0; i<N; i++ ) {
        for( int j=0; j<N; j++ ) {
            for( int k=0; k<N; k++ ) {
                if (LIQUID == fluid_flag.get(i,j,k))
                {
                    ans += x.get(i,j,k) * y.get(i,j,k);
                }
            }
        }
    }
    return ans;
}

void increment(grid_cell<float>& data, double scale, grid_cell<float>& a) {
    for( int i=0; i<N; i++ ) {
        for( int j=0; j<N; j++ ) {
            for( int k=0; k<N; k++ ) {
                if (LIQUID == fluid_flag.get(i,j,k)) 
                {
                    data.get(i,j,k) += scale * a.get(i,j,k);
                }
            }
        }
    }
}

void scale_and_increment(grid_cell<float>& data, double scale, grid_cell<float>& a) {
    for( int i=0; i<N; i++ ) {
        for( int j=0; j<N; j++ ) {
            for( int k=0; k<N; k++ ) {
                if (LIQUID == fluid_flag.get(i,j,k)) 
                {
                    data.get(i,j,k) = scale * data.get(i,j,k) + a.get(i,j,k);
                }
            }
        }
    }
}

void apply_poisson(grid_cell<float> &x, grid_cell<float> &y)//matrix vector product (pressure poisson)
{
    int nx = y.get_nx();
    int ny = y.get_ny();
    int nz = y.get_nz();

    y.clear();
    for(int i=0; i<nx; ++i){
        for(int j=0; j<ny; ++j) {
            for(int k=0; k<nz; ++k) {
                if(LIQUID == fluid_flag.get(i,j,k)){ //assume no fluid cell on bounding box
                    y.get(i,j,k)=
                        poisson_0.get(i  ,j  ,k) * x.get(i,j,k) 
                        + poisson_1.get(i-1,j  ,k) * x.get(i-1,j,k)
                        + poisson_1.get(i  ,j  ,k) * x.get(i+1,j,k)
                        + poisson_2.get(i  ,j-1,k) * x.get(i,j-1,k)
                        + poisson_2.get(i  ,j  ,k) * x.get(i,j+1,k)
                        + poisson_3.get(i  ,j,k-1) * x.get(i,j,k-1)
                        + poisson_3.get(i  ,j,k  ) * x.get(i,j,k+1)
                        ;
                }
            }
        }
    }
}

void form_preconditioner()
{
    int nx = preconditioner.get_nx();
    int ny = preconditioner.get_ny();
    int nz = preconditioner.get_nz();

    const double mic_parameter=0.97;
    const double a=0.25;

    preconditioner.clear();
    for(int i=0; i<nx; ++i){
        for(int j=0; j<ny; ++j) {
            for(int k=0; k<nz; ++k) {
                if(LIQUID == fluid_flag.get(i,j,k)){
                    double d = poisson_0.get(i,j,k) 
                        - sq( poisson_1.get(i-1,j,k) * preconditioner.get(i-1,j,k) ) //p_xx^2  <first index is first argument of poisson(), second index is last argument of poisson()>
                        - sq( poisson_2.get(i,j-1,k) * preconditioner.get(i,j-1,k) ) //p_yy^2
                        - sq( poisson_3.get(i,j,k-1) * preconditioner.get(i,j,k-1) ) //p_zz^2
                        - mic_parameter*
                        (poisson_1.get(i-1,j,k)*(poisson_2.get(i-1,j,k)+poisson_3.get(i-1,j,k))*sq(preconditioner.get(i-1,j,k))
                        +poisson_2.get(i,j-1,k)*(poisson_1.get(i,j-1,k)+poisson_3.get(i,j-1,k))*sq(preconditioner.get(i,j-1,k)) 
                        +poisson_3.get(i,j,k-1)*(poisson_1.get(i,j,k-1)+poisson_2.get(i,j,k-1))*sq(preconditioner.get(i,j,k-1)) 
                        );
                    //          preconditioner(i,j,k)=1/sqrt(d+1e-6);
                    //          preconditioner(i,j,k)=0.1;
                    if(d < poisson_0.get(i,j,k)*a){
                        d = poisson_0.get(i,j,k);
                    }
                    preconditioner.get(i,j,k)=1.0/sqrt(d);
                }
            }
        }
    }
}

void apply_preconditioner(grid_cell<float> &x, grid_cell<float> &y, grid_cell<float> &m)
{
    int nx = preconditioner.get_nx();
    int ny = preconditioner.get_ny();
    int nz = preconditioner.get_nz();

    int i, j, k;
    m.clear();

    // solve L*m=x
    for(int i=0; i<nx; ++i){
        for(int j=0; j<ny; ++j) {
            for(int k=0; k<nz; ++k) {
                if(LIQUID == fluid_flag.get(i,j,k)){
                    double d = x.get(i,j,k) 
                        - poisson_1.get(i-1,j,k)*preconditioner.get(i-1,j,k)*m.get(i-1,j,k)
                        - poisson_2.get(i,j-1,k)*preconditioner.get(i,j-1,k)*m.get(i,j-1,k)
                        - poisson_3.get(i,j,k-1)*preconditioner.get(i,j,k-1)*m.get(i,j,k-1);
                    m.get(i,j,k) = preconditioner.get(i,j,k) * d;
                }
            }
        }
    }

    // solve L'*y=m
    y.clear();
    for(int i=nx-1; i>=0; --i){
        for(int j=ny-1; j>=0; --j) {
            for(int k=nz-1; k>=0; --k) {
                if(LIQUID == fluid_flag.get(i,j,k)){
                    double d = m.get(i,j,k) 
                        - poisson_1.get(i,j,k)*preconditioner.get(i,j,k)*y.get(i+1,j,k)
                        - poisson_2.get(i,j,k)*preconditioner.get(i,j,k)*y.get(i,j+1,k)
                        - poisson_3.get(i,j,k)*preconditioner.get(i,j,k)*y.get(i,j,k+1);
                    y.get(i,j,k) = preconditioner.get(i,j,k) * d;
                }
            }
        }
    }
}

// the 2-norm is not a good measurement for high dimensional data
float inf_norm(grid_cell<float> &data) {
    float ret = 0;
    for (int n = 0; n < data.get_size(); n++) {
        if (LIQUID == fluid_flag.get(n)) {
            ret = f_max(ret, fabs(data.get(n)));
        }
    }
    return ret;
}

void solve_pressure(int maxits, double tolerance)
{
    printf("\t|r|=%g\n",inf_norm(r));
    int its;
    double tol=tolerance*inf_norm(r);
    //    double tol=tolerance;
    pressure.clear();
    if(inf_norm(r)==0)
        return;

    apply_preconditioner(r, z, m);
    s.copy_from(z);
    double rho=dot(z,r);
    if(rho==0)
        return;

    for(its=0; its<maxits; ++its){
        printf("\rPCG: iter %d", its);
        apply_poisson(s, z);

        double alpha=rho/dot(s,z);
        //         if(dot(s,z) == 0) {
        //             system("pause");
        //             alpha = 1;
        //         }
        //         printf("alpha = %f = %f / %f\n", alpha, rho, dot(s,z));
        //         if(alpha != alpha) alpha = 1;
        //         printf("alpha = %f\n", alpha);

        increment(pressure, alpha, s);
        increment(r, -alpha, z);
        if(inf_norm(r)<=tol){
            printf("\r\tpressure converged to %g < %g in %d iterations\n", inf_norm(r), tol, its);
            return;
        }
        apply_preconditioner(r, z, m);
        double rhonew=dot(z,r);

        double beta=rhonew/rho;
        //         if (rho == 0) {
        //             system("pause");
        //             beta = 1;
        //         }
        //         printf("beta = %f = %f / %f\n", beta, rhonew, rho);
        //         if (beta != beta) beta = 1;
        //         printf("beta = %f\n", beta);

        scale_and_increment(s, beta, z);
        rho=rhonew;
    }
    printf("\r\tDidn't converge in pressure solve (its=%d, tol=%g, |r|=%g)\n", its, tol, inf_norm(r));
}

float root_mean_square(grid_cell<float> &data) {
    float r0 = 0;
    for(int n=0; n<res.get_size(); n++){
        r0 += sq(res.get(n));
    }
    return sqrt(r0 / res.get_size());
}

// of -L (laplacian matrix)
float sparse_A_diag(int i, int j, int k) {
    //     return 6;

    //////////////////////////////////////////////////////////////////////////
    float diag = 6.0f;
    if (LIQUID != fluid_flag.get(i,j,k)) return diag;

    int nid[][3] =  { {i+1,j,k}, {i-1,j,k}, {i,j+1,k}, {i,j-1,k}, {i,j,k+1}, {i,j,k-1} };
    for (int m = 0; m < 6; m++) {
        int ni = nid[m][0];
        int nj = nid[m][1];
        int nk = nid[m][2];

        if (SOLID == fluid_flag.get(ni, nj, nk))
            diag -= 1.0f;
        else if (AIR == fluid_flag.get(ni,nj,nk)) // ghost fluid interface
            diag -= sdistance.get(ni,nj,nk) / f_min(1e-6f, sdistance.get(i,j,k));
    }
    return diag;
}

// of -L (laplacian matrix)
float sparse_A_offdiag(int fi, int fj, int fk, int i, int j, int k) {
    //     return -1;

    //////////////////////////////////////////////////////////////////////////
    if (LIQUID == fluid_flag.get(fi,fj,fk) && 
        LIQUID == fluid_flag.get(i,j,k))
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
                if (LIQUID == fluid_flag.get(i,j,k)) {
                    float _lap =
                           pressure.get(i+1,j,k) * -sparse_A_offdiag(i,j,k, i+1,j,k)
                        +  pressure.get(i-1,j,k) * -sparse_A_offdiag(i,j,k, i-1,j,k)
                        +  pressure.get(i,j+1,k) * -sparse_A_offdiag(i,j,k, i,j+1,k)
                        +  pressure.get(i,j-1,k) * -sparse_A_offdiag(i,j,k, i,j-1,k)
                        +  pressure.get(i,j,k+1) * -sparse_A_offdiag(i,j,k, i,j,k+1)
                        +  pressure.get(i,j,k-1) * -sparse_A_offdiag(i,j,k, i,j,k-1)
                        +  pressure.get(i,j,k)   * -sparse_A_diag(i,j,k);
                    float _div = (
                         ux.get(i+1,j,k)-ux.get(i,j,k)
                        +uy.get(i,j+1,k)-uy.get(i,j,k)
                        +uz.get(i,j,k+1)-uz.get(i,j,k)
                        );
                    res.get(i,j,k) = _lap - _div;
                }
            }
        }
    }

    float r0 = root_mean_square(res);
    printf("\tresidual mean square = %e\n",r0);
    printf("\tresidual inf_norm |r|=%e\n",inf_norm(res));
}

void pcg_solve_poisson() {
    d_fluid_flag.to_host(fluid_flag);
    d_sdistance.to_host(sdistance);
    d_ux.to_host(ux);
    d_uy.to_host(uy);
    d_uz.to_host(uz);

    int nx = pressure.get_nx();
    int ny = pressure.get_ny();
    int nz = pressure.get_nz();

    pressure.clear();
    r.clear(); // rhs

    // rhs
#pragma omp parallel for
    for(int i=0; i<nx; i++){
        for(int j=0; j<ny; j++){
            for(int k=0; k<nz; k++){

                if (LIQUID == fluid_flag.get(i,j,k)) {
                    r.get(i,j,k) = -  //negated divergence
                        (ux.get(i+1,j,k)-ux.get(i,j,k)
                        +uy.get(i,j+1,k)-uy.get(i,j,k)
                        +uz.get(i,j,k+1)-uz.get(i,j,k)
                        )
                        //                     + 0.15 // volume source
                        ;
                }
            }
        }
    }

    d_poisson_0.clear_gpu();
    d_poisson_1.clear_gpu();
    d_poisson_2.clear_gpu();
    d_poisson_3.clear_gpu();
    _form_poisson<<<gridConfig, blockConfig>>>(d_poisson_0, d_poisson_1,
        d_poisson_2, d_poisson_3, d_fluid_flag, d_sdistance);
    d_poisson_0.to_host(poisson_0);
    d_poisson_1.to_host(poisson_1);
    d_poisson_2.to_host(poisson_2);
    d_poisson_3.to_host(poisson_3);

    form_preconditioner();
    //     solve_pressure(200, 1e-5);
    solve_pressure(50, 1e-7);

    pressure.to_device(d_pressure);
}
//////////////////////////////////////////////////////////////////////////
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
    _desingularize_signed_distance<<<gridConfig, blockConfig>>>(d_sdistance);
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
    _update_fluid_flag<<<gridConfig, blockConfig>>>(d_fluid_flag, d_sdistance);
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
    _add_gravity<<<gridConfig, blockConfig>>>(d_uy, d_fluid_flag);
    static int ii = 0;
    if (0 == ++ii % 10) {
        _addDistanceSphere<<<gridConfig, blockConfig>>>(d_sdistance, d_ux, d_uy, d_uz,
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
        _extrapolate<<<gridConfig, blockConfig>>>(data_grid, temp_grid, valid, temp_valid);
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

    {   //u extrapolation
        d_vel_valid.clear_gpu();
        _set_valid_x<<<gridConfig, blockConfig>>>(d_ux, d_uy, d_uz, d_fluid_flag, d_vel_valid);
        extrapolate(d_ux, d_ux_temp, d_vel_valid, d_temp_valid);
    }

    {   //v extrapolation
        d_vel_valid.clear_gpu();
        _set_valid_y<<<gridConfig, blockConfig>>>(d_ux, d_uy, d_uz, d_fluid_flag, d_vel_valid);
        extrapolate(d_uy, d_uy_temp, d_vel_valid, d_temp_valid);
    }

    {   //w extrapolation
        d_vel_valid.clear_gpu();
        _set_valid_z<<<gridConfig, blockConfig>>>(d_ux, d_uy, d_uz, d_fluid_flag, d_vel_valid);
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
    _wall_boundary_for_sdf<<<gridConfig, blockConfig>>>(density, fluid_flag);
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
    _solid_boundary_for_velocity<<<gridConfig, blockConfig>>>(ux, uy, uz, fluid_flag);
}
float computeCFL() { // not standard definition
    float CFL = 0;
    {
        int nx = ux.get_nx();
        int ny = ux.get_ny();
        int nz = ux.get_nz();

        float ux_max = 0;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
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
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
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
                for (int k = 0; k < nz; k++) {
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
//                 sdistance.get(i,j,k) = f_min((j + 0.5) - ny * 0.1, sdistance.get(i,j,k));
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
            pcg_solve_poisson_gpu_ilu0();
//             pcg_solve_poisson_gpu();
//             pcg_solve_poisson_gpu_debug();
            //     pcg_solve_poisson();
            d_pressure.to_host(pressure);
            calc_residual();
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

        frame_buffer vrfb; vrfb.init(300, 300); // volume rendering film size can be arbitrary
        frame_buffer d_vrfb; d_vrfb.init_gpu(300, 300); // volume rendering film size can be arbitrary
        dim3 blockConfig2(16, 16, 1);
        dim3 gridConfig2(divUp(300, blockConfig2.x), divUp(300, blockConfig2.y), 1);
        _simple_scalar_vr<<<gridConfig2, blockConfig2>>>(d_sdistance, d_vrfb);
        d_vrfb.to_host(vrfb);
        vrfb.flipud();
        char fn[256];
        sprintf_s(fn, "ppm/vr_ns_%07d.ppm", idx++);
        FILE *fp = fopen(fn, "wb");
        fprintf(fp, "P6\n%d %d\n255\n", vrfb.getWidth(), vrfb.getHeight());
        int total = vrfb.getTotal() * 3;
        float *vfb = reinterpret_cast<float*>(vrfb.ptr());
        unsigned char *cfb = new unsigned char[total];
        for (int n = 0; n < total; n++) cfb[n] = (unsigned char)(clampf(vfb[n], 0, 1) * 255);
        fwrite(cfb, 1, total, fp);
        fclose(fp);
        delete [] cfb;
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