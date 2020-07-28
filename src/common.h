#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define CONST

constexpr float M_PI = 3.14159265358979323846;
float           randf();
void            __sync();

constexpr bool matrixfree = true;

// constexpr int N = 512;
constexpr int N = 256;

constexpr float dt      = 0.01 * (N / 5.0);
constexpr float gravity = 9.80 / (N / 5.0);
constexpr float MAXCFL = 10;

int divUp(int a, int b);

dim3 blockConfig();
dim3 gridConfig();

#define KERNAL_CONFIG                              \
    int i = threadIdx.x + blockIdx.x * blockDim.x; \
    int j = threadIdx.y + blockIdx.y * blockDim.y; \
    int k = threadIdx.z + blockIdx.z * blockDim.z;

class Profiler
{
    double timeBegin;
    double stopwatch()
    {
        unsigned long long ticks, ticks_per_sec;
        QueryPerformanceFrequency((LARGE_INTEGER*)&ticks_per_sec);
        QueryPerformanceCounter((LARGE_INTEGER*)&ticks);
        return ((double)ticks) / (double)ticks_per_sec;
    }

public:
    Profiler() { begin(); }
    void   begin() { timeBegin = stopwatch(); }
    double get()
    {
        double timeGet = stopwatch();
        return timeGet - timeBegin;
    }
};

struct vec3
{
    float               x, y, z;
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float a) : x(a), y(a), z(a) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

inline __host__ __device__ vec3 cross(const vec3& a, const vec3& b)
{
    return vec3(
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline __host__ __device__ float dot(const vec3& a, const vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ vec3 mult(const vec3& a, const vec3& b)
{
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ vec3 div(const vec3& a, const vec3& b)
{
    return vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ vec3 operator*(const vec3& a, float b)
{
    return vec3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ vec3 operator+(const vec3& a, const vec3& b)
{
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ vec3& operator+=(vec3& a, const vec3& b)
{
    a.x += b.x, a.y += b.y, a.z += b.z;
    return a;
}
inline __host__ __device__ vec3 operator-(const vec3& a, const vec3& b)
{
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ vec3 normalize(const vec3& a)
{
    float _l = 1.0f / sqrtf(dot(a, a));
    return a * _l;
}
inline __host__ __device__ float f_min(float a, float b)
{
    return a < b ? a : b;
}
inline __host__ __device__ float f_max(float a, float b)
{
    return a > b ? a : b;
}
inline __host__ __device__ vec3 f_min(const vec3& a, const vec3& b)
{
    return vec3(f_min(a.x, b.x), f_min(a.y, b.y), f_min(a.z, b.z));
}
inline __host__ __device__ vec3 f_max(const vec3& a, const vec3& b)
{
    return vec3(f_max(a.x, b.x), f_max(a.y, b.y), f_max(a.z, b.z));
}
inline __host__ __device__ vec3 f_min(const vec3& a, float b)
{
    return vec3(f_min(a.x, b), f_min(a.y, b), f_min(a.z, b));
}
inline __host__ __device__ vec3 f_max(const vec3& a, float b)
{
    return vec3(f_max(a.x, b), f_max(a.y, b), f_max(a.z, b));
}
inline __host__ __device__ float sq(float x) { return x * x; }
inline __host__ __device__ float clampf(float x, float a, float b)
{
    return x < a ? a : x > b ? b : x;
}
inline __host__ __device__ int clampi(int x, int a, int b)
{
    return x < a ? a : x > b ? b : x;
}

template <class T>
void swap(T& a, T& b)
{
    T t(b);
    b = a;
    a = t;
}
//////////////////////////////////////////////////////////////////////////
enum cell_type
{
    AIR    = 0,
    LIQUID = 1,
    SOLID  = 2,
};

inline __host__ __device__ int strict_floor(float x)
{
    return int(x) - (x < 0);
}
inline __host__ __device__ void bary(float  p,
                                     int    dim,
                                     int&   lower,
                                     float& offset)
{
    lower  = strict_floor(p);
    offset = p - lower;
    if (lower < 0)
    {
        lower  = 0;
        offset = 0.0f;
    }
    if (lower > dim - 2)
    {
        lower  = dim - 2;
        offset = 1.0f;
    }
}
inline __host__ __device__ bool bary_cubic(float  p,
                                           int    dim,
                                           int&   lower,
                                           float& offset)
{
    lower         = strict_floor(p);
    offset        = p - lower;
    bool boundary = false;
    if (lower < 2)
    {
        lower    = 2;
        offset   = 0.0f;
        boundary = true;
    }
    if (lower > dim - 4)
    {
        lower    = dim - 4;
        offset   = 1.0f;
        boundary = true;
    }
    return boundary;
}
template <typename T>
class grid
{
protected:
    T*    data;
    int   nx, ny, nz;
    int   m_total, nxy;
    float corner_x, corner_y, corner_z;
    int   capacity;
    grid* base;

protected:
    __host__ __device__ T lerp1(T a0, T a1, T x)
    {  // a_x
        return a0 * (1 - x) + a1 * x;
    }
    __host__ __device__ T lerp2(T a00,
                                T a10,  // a_xy
                                T a01,
                                T a11,
                                T x,
                                T y)
    {
        return lerp1(a00, a10, x) * (1 - y) + lerp1(a01, a11, x) * y;
    }
    __host__ __device__ T lerp3(T a000,
                                T a100,  // a_xyz
                                T a010,
                                T a110,
                                T a001,
                                T a101,
                                T a011,
                                T a111,
                                T x,
                                T y,
                                T z)
    {
        return lerp2(a000, a100, a010, a110, x, y) * (1 - z) +
               lerp2(a001, a101, a011, a111, x, y) * z;
    }
    __host__ __device__ T lerp(int i0, int j0, int k0, T u, T v, T w)
    {
        int i1 = i0 + 1;
        int j1 = j0 + 1;
        int k1 = k0 + 1;
        return lerp3(get(i0, j0, k0),
                     get(i1, j0, k0),
                     get(i0, j1, k0),
                     get(i1, j1, k0),
                     get(i0, j0, k1),
                     get(i1, j0, k1),
                     get(i0, j1, k1),
                     get(i1, j1, k1),
                     u,
                     v,
                     w);
    }

    __host__ __device__ T cubic1D(T t, T f[4])
    {
        T d1  = (f[2] - f[0]) * 0.5;
        T d2  = (f[3] - f[1]) * 0.5;
        T del = f[2] - f[1];
        if (d1 * del <= 0) d1 = 0;
        if (d2 * del <= 0) d1 = 0;
        T a0 = f[1];
        T a1 = d1;
        T a2 = 3 * del - 2 * d1 - d2;
        T a3 = d1 + d2 - 2 * del;
        return a3 * t * t * t + a2 * t * t + a1 * t + a0;
    }
    __host__ __device__ T cubic2D(T tx, T ty, T f[16])
    {
        T _f[4];
        _f[0] = cubic1D(tx, &f[0]);
        _f[1] = cubic1D(tx, &f[4]);
        _f[2] = cubic1D(tx, &f[8]);
        _f[3] = cubic1D(tx, &f[12]);
        return cubic1D(ty, _f);
    }
    __host__ __device__ T cubic3D(T tx, T ty, T tz, T f[64])
    {
        T _f[4];
        _f[0] = cubic2D(tx, ty, &f[0]);
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
            get(_i_, _j_, _k_), get(_i0, _j_, _k_), get(_i1, _j_, _k_),
            get(_i2, _j_, _k_), get(_i_, _j0, _k_), get(_i0, _j0, _k_),
            get(_i1, _j0, _k_), get(_i2, _j0, _k_), get(_i_, _j1, _k_),
            get(_i0, _j1, _k_), get(_i1, _j1, _k_), get(_i2, _j1, _k_),
            get(_i_, _j2, _k_), get(_i0, _j2, _k_), get(_i1, _j2, _k_),
            get(_i2, _j2, _k_), get(_i_, _j_, _k0), get(_i0, _j_, _k0),
            get(_i1, _j_, _k0), get(_i2, _j_, _k0), get(_i_, _j0, _k0),
            get(_i0, _j0, _k0), get(_i1, _j0, _k0), get(_i2, _j0, _k0),
            get(_i_, _j1, _k0), get(_i0, _j1, _k0), get(_i1, _j1, _k0),
            get(_i2, _j1, _k0), get(_i_, _j2, _k0), get(_i0, _j2, _k0),
            get(_i1, _j2, _k0), get(_i2, _j2, _k0), get(_i_, _j_, _k1),
            get(_i0, _j_, _k1), get(_i1, _j_, _k1), get(_i2, _j_, _k1),
            get(_i_, _j0, _k1), get(_i0, _j0, _k1), get(_i1, _j0, _k1),
            get(_i2, _j0, _k1), get(_i_, _j1, _k1), get(_i0, _j1, _k1),
            get(_i1, _j1, _k1), get(_i2, _j1, _k1), get(_i_, _j2, _k1),
            get(_i0, _j2, _k1), get(_i1, _j2, _k1), get(_i2, _j2, _k1),
            get(_i_, _j_, _k2), get(_i0, _j_, _k2), get(_i1, _j_, _k2),
            get(_i2, _j_, _k2), get(_i_, _j0, _k2), get(_i0, _j0, _k2),
            get(_i1, _j0, _k2), get(_i2, _j0, _k2), get(_i_, _j1, _k2),
            get(_i0, _j1, _k2), get(_i1, _j1, _k2), get(_i2, _j1, _k2),
            get(_i_, _j2, _k2), get(_i0, _j2, _k2), get(_i1, _j2, _k2),
            get(_i2, _j2, _k2),
        };

        return cubic3D(tx, ty, tz, f);
    }
    __host__ __device__ inline int ix(int i, int j, int k)
    {
        return i + j * nx + k * nxy;
    }

public:
    __host__ __device__ grid() : data(NULL) {}
    __host__ __device__ grid(T cx, T cy, T cz)
        : data(NULL), corner_x(cx), corner_y(cy), corner_z(cz)
    {
    }
    void init(int x, int y, int z, size_t size = 0)
    {
        if (data) return;
        nx = x, ny = y, nz = z;
        m_total = nx * ny * nz;
        nxy     = nx * ny;
        if (size == 0)
        {
            data     = new T[m_total];
            capacity = m_total;
            base     = nullptr;
        }
        else
        {
            data     = new T[size];
            capacity = size;
            base     = nullptr;
        }
        clear();
    }
    void init_gpu(int x, int y, int z, size_t size = 0)
    {
        if (data) return;
        nx = x, ny = y, nz = z;
        m_total = nx * ny * nz;
        nxy     = nx * ny;
        if (size == 0)
        {
            checkCudaErrors(cudaMalloc((void**)&data, sizeof(T) * m_total));
            capacity = m_total;
            base     = nullptr;
        }
        else
        {
            checkCudaErrors(cudaMalloc((void**)&data, sizeof(T) * size));
            capacity = size;
            base     = nullptr;
        }
        clear_gpu();
    }
    void init_ref(int x, int y, int z, void* buffer)
    {
        // if (data) return;
        nx = x, ny = y, nz = z;
        m_total  = nx * ny * nz;
        nxy      = nx * ny;
        data     = reinterpret_cast<T*>(buffer);
        capacity = 0;
        base     = nullptr;
        // clear();
    }
    void init_ref_gpu(int x, int y, int z, void* buffer)
    {
        // if (data) return;
        nx = x, ny = y, nz = z;
        m_total  = nx * ny * nz;
        nxy      = nx * ny;
        data     = reinterpret_cast<T*>(buffer);
        capacity = 0;
        base     = nullptr;
        // clear_gpu();
    }
    void free()
    {
        if (data)
        {
            delete[] data;
            data = NULL;
        }
    }
    void free_gpu()
    {
        checkCudaErrors(cudaFree(data));
        data = NULL;
    }
    void to_host(grid& h_data)
    {
        if (nx != h_data.nx || ny != h_data.ny || nz != h_data.nz)
        {
            printf("to_host:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(h_data.get_ptr(),
                                   data,
                                   sizeof(T) * m_total,
                                   cudaMemcpyDeviceToHost));
    }
    void to_device(grid& d_data)
    {
        if (nx != d_data.nx || ny != d_data.ny || nz != d_data.nz)
        {
            printf("to_device:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(d_data.get_ptr(),
                                   data,
                                   sizeof(T) * m_total,
                                   cudaMemcpyHostToDevice));
    }

    __host__ __device__ inline T& get(int i, int j, int k)
    {
        return data[ix(i, j, k)];
    }
    __host__ __device__ inline T& get(int n) { return data[n]; }
    __host__ __device__ inline T* get_ptr(void) { return data; }
    __host__ __device__ inline void* get_buffer(size_t offset)
    {
        return reinterpret_cast<uint8_t*>(data) + offset;
    }
    __host__ __device__ inline int get_nx() { return nx; }
    __host__ __device__ inline int get_ny() { return ny; }
    __host__ __device__ inline int get_nz() { return nz; }
    __host__ __device__ inline int get_size() { return m_total; }
    void                           clear()
    {
        for (int n = 0; n < m_total; n++) data[n] = T(0);
    }
    void clear_gpu()
    {
        checkCudaErrors(cudaMemset(data, 0, sizeof(T) * m_total));
    }

    template <class T>
    T cast(int x, int y, int z)
    {
        T ret;
        ret.init_ref(x, y, z, data);
        ret.capacity = capacity;
        ret.base     = this;
        return ret;
    }

    template <class T>
    T cast_gpu(int x, int y, int z)
    {
        T ret;
        ret.init_ref_gpu(x, y, z, data);
        ret.capacity = capacity;
        ret.base     = this;
        return ret;
    }

    void swap(grid& a)
    {
        // if (nx != a.nx || ny != a.ny || nz != a.nz)
        if (capacity != a.capacity)
        {
            printf("not swapped!\n");
            throw std::runtime_error("not swapped!");
            return;
        }
        std::swap(data, a.data);

        if (this->base)
        {
            this->base->data = this->data;
        }
        if (a.base)
        {
            a.base->data = a.data;
        }
    }
    void copy_from(grid& a)
    {  // it seems that the class is a friend of itself, so the private members
       // can be freely accessed
        if (nx != a.nx || ny != a.ny || nz != a.nz)
        {
            printf("copy_from:: not copied!\n");
            return;
        }
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int k = 0; k < nz; k++)
                {
                    this->get(i, j, k) = a.get(i, j, k);
                }
            }
        }
    }
    void copy_from_gpu(grid& a)
    {  // it seems that the class is a friend of itself, so the private members
       // can be freely accessed
        if (nx != a.nx || ny != a.ny || nz != a.nz)
        {
            printf("copy_from:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(
            data, a.data, sizeof(T) * m_total, cudaMemcpyDeviceToDevice));
    }

    __host__ __device__ T interp(T x, T y, T z)
    {
        x -= corner_x;
        y -= corner_y;
        z -= corner_z;
        int x0;
        int y0;
        int z0;
        T   u;
        T   v;
        T   w;
        bary(x, nx, x0, u);
        bary(y, ny, y0, v);
        bary(z, nz, z0, w);
        return lerp(x0, y0, z0, u, v, w);
    }
    __host__ __device__ T interp_cubic(T x, T y, T z)
    {
        x -= corner_x;
        y -= corner_y;
        z -= corner_z;
        int  x0;
        int  y0;
        int  z0;
        T    u;
        T    v;
        T    w;
        bool bx = bary_cubic(x, nx, x0, u);
        bool by = bary_cubic(y, ny, y0, v);
        bool bz = bary_cubic(z, nz, z0, w);
        if (bx || by || bz)
        {  // the cubic interpolation must be off the boundary for now
            bary(x, nx, x0, u);
            bary(y, ny, y0, v);
            bary(z, nz, z0, w);
            return lerp(x0, y0, z0, u, v, w);
        }
        else
        {
            return cerp(x0, y0, z0, u, v, w);
        }
    }
};
template <typename T>
struct grid_cell : public grid<T>
{
    __host__ __device__ grid_cell() : grid(0.5, 0.5, 0.5) {}
};
template <typename T>
struct grid_face_x : public grid<T>
{
    __host__ __device__ grid_face_x() : grid(0.0, 0.5, 0.5) {}
};
template <typename T>
struct grid_face_y : public grid<T>
{
    __host__ __device__ grid_face_y() : grid(0.5, 0.0, 0.5) {}
};
template <typename T>
struct grid_face_z : public grid<T>
{
    __host__ __device__ grid_face_z() : grid(0.5, 0.5, 0.0) {}
};

class frame_buffer
{
    vec3*                          m_buf;
    int                            m_width, m_height;
    int                            m_total;
    float                          m_factor;
    __host__ __device__ inline int ix(int i, int j) const
    {
        return i + j * m_width;
    }
    void swap_line(vec3* a, vec3* b, int total)
    {
        for (int n = 0; n < total; n++)
        {
            swap(a[n], b[n]);
        }
    }

public:
    __host__ __device__ frame_buffer() : m_buf(NULL) {}
    void                free()
    {
        if (m_buf) delete[] m_buf;
    }
    void init(int _w, int _h)
    {
        if (m_buf) return;
        m_width  = _w;
        m_height = _h;
        m_total  = m_width * m_height;
        m_buf    = new vec3[m_total];
    }
    void free_gpu()
    {
        if (m_buf) checkCudaErrors(cudaFree(m_buf));
    }
    void init_gpu(int _w, int _h)
    {
        if (m_buf) return;
        m_width  = _w;
        m_height = _h;
        m_total  = m_width * m_height;
        checkCudaErrors(cudaMalloc((void**)&m_buf, sizeof(vec3) * m_total));
    }
    void to_host(frame_buffer& h_data)
    {
        if (m_width != h_data.m_width || m_height != h_data.m_height)
        {
            printf("to_host:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(h_data.ptr(),
                                   m_buf,
                                   sizeof(vec3) * m_total,
                                   cudaMemcpyDeviceToHost));
    }
    void to_device(frame_buffer& d_data)
    {
        if (m_width != d_data.m_width || m_height != d_data.m_height)
        {
            printf("to_device:: not copied!\n");
            return;
        }
        checkCudaErrors(cudaMemcpy(d_data.ptr(),
                                   m_buf,
                                   sizeof(vec3) * m_total,
                                   cudaMemcpyHostToDevice));
    }
    void flipud()
    {
        for (int j = 0; j < m_height / 2; j++)
        {
            swap_line(
                &m_buf[ix(0, j)], &m_buf[ix(0, m_height - 1 - j)], m_width);
        }
    }
    __host__ __device__ vec3& get(int i, int j) { return m_buf[ix(i, j)]; }
    __host__ __device__ const vec3& get(int i, int j) const
    {
        return m_buf[ix(i, j)];
    }
    __host__ __device__ vec3* ptr() { return m_buf; }
    __host__ __device__ int   getWidth() const { return m_width; }
    __host__ __device__ int   getHeight() const { return m_height; }
    __host__ __device__ int   getTotal() const { return m_total; }
};

// raycasting in the -z direction;
void simple_scalar_vr(grid_cell<float> vol, frame_buffer fb);
