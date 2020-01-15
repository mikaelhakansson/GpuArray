#pragma once

#include <cuda.h>

// Kernel that expands nested operator[] in expr
template <typename T, typename E>
__global__ void expression_kernel(const E expr, const int N, T* data)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        data[idx] = expr[idx];
    }
}

// CRTP base class. Must be used for Array class and all expression objects.
template <typename T, typename E>
class ArrayExpression
{
 public:
    __device__ T operator[](const size_t i) const { return static_cast<const E&>(*this)[i]; }
    size_t size() const { return static_cast<const E&>(*this).size(); }
};

// Forward declarations (due to circular dependencies)
template <typename T>
class View;

template <typename T>
class Slice;


// GpuArray class
template <typename T>
class Array : public ArrayExpression<T, Array<T>>  // CRTP inheritance
{
 public:
    using value_type = T;
    using handle_type = View<T>;

    // Default constructor
    Array() : m_data(nullptr), m_size(0) {}

    // Construct array of n elements
    explicit Array(const size_t n) : m_size(n)
    {
        cudaMalloc(&m_data, m_size * sizeof(T));
    }

    // Copy-constructor
    explicit Array(const Array& v) : Array(v.m_size)
    {
        cudaMemcpy(m_data, v.m_data, m_size * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    // Construct from ArrayExpression, forcing evaluation of nested operator[] in expr
    template <typename E>
    Array(const E& expr) : Array(expr.size())
    {
        dim3 tpb(256, 1);
        dim3 bpg(m_size / tpb.x, 1);
        expression_kernel<<<bpg, tpb>>>(expr, expr.size(), m_data);
    }

    // Destructor
    ~Array()
    {
        if (m_data)
        {
            cudaFree(m_data);
        }
    }

    // Assignment operator
    template <typename E>
    Array& operator=(const E& expr)
    {
        if (m_size != expr.size())
        {
            if (m_data)
            {
                cudaFree(m_data);
            }

            cudaMalloc(&m_data, expr.size() * sizeof(T));
            m_size = expr.size();
        }

        dim3 tpb(256, 1);
        dim3 bpg(m_size / tpb.x, 1);
        expression_kernel<<<bpg, tpb>>>(expr, expr.size(), m_data);

        return *this;
    }

    // Create a slice
    Slice<T> slice(const int begin, const int end_ = -1, const int stride = 1)
    {
        const int end = (end_ == -1) ? m_size : std::min(end_, m_size);
        return Slice<T>(View<T>(*this), begin, end, stride);
    }

    size_t size() const { return m_size; }
    T* data() const {return m_data;}

private:
    T* m_data;
    int m_size;
};


// View of GpuArray. Copyable without copying underlying memory.
template <typename T>
class View : public ArrayExpression<T, View<T>>
{
 public:
    using value_type = T;
    using handle_type = View;

    __device__ T operator[](const size_t i) const { return m_data[i]; }

    View(T* data, const size_t size) : m_data(data), m_size(size) {}
    View(const Array<T>& arr) : View(arr.data(), arr.size()) {}
    View(const View& s) : m_data(s.m_data), m_size(s.m_size) {}

    size_t size() const { return m_size; }

private:
    T* m_data;
    size_t m_size;
};

// Slice expression class
template <typename T>
class Slice : public ArrayExpression<T, Slice<T>>
{
 public:
    using value_type = T;
    using handle_type = Slice;

    __device__ T operator[](const size_t i) const
    {
        const int idx = m_begin + i*m_stride;
        return m_view[idx];
    }

    Slice(const View<T>& view, const int begin, const int end, const int stride) :
        m_view(view), m_begin(begin), m_end(end), m_stride(stride) {}

    Slice(const Slice& s) : Slice(s.m_view, s.m_begin, s.m_end, s.m_stride) {}

    size_t size() const { return (m_end - m_begin + m_stride - 1) / m_stride; }

private:
    View<T> m_view;

    const int m_begin;
    const int m_end;
    const int m_stride;
};

// Scalar class. Represents scalars as expression objects.
template <typename T>
class Scalar : public ArrayExpression<T, Scalar<T>>
{
 public:
    using value_type = T;
    using handle_type = Scalar;

    __device__ T operator[](const size_t i) const { return m_data; }

    Scalar(const T value) : m_data(value) {}
    Scalar(const Scalar& s) : m_data(s.m_data) {}

    size_t size() const { return 1; }

private:
    T m_data;
};

// Generic unary operator expression class
template <typename E, template <typename> typename F>
class ArrayOpUnary : public ArrayExpression<typename E::value_type, ArrayOpUnary<E, F>>
{
    const typename E::handle_type m_expr;

 public:
    using value_type = typename E::value_type;
    using handle_type = ArrayOpUnary;

    ArrayOpUnary(const ArrayOpUnary& op) : m_expr(op.m_expr) {}
    ArrayOpUnary(const E& expr) : m_expr(expr) {}

    __device__ value_type operator[](const size_t i) const
    {
        return F<typename E::value_type>::op(this->m_expr[i]);
    }

    size_t size() const { return this->m_expr.size(); }
};

// Sin operation
template <typename T>
struct Sin
{
    __device__ static inline T op(const T a) { return sin(a); }
};

// Function that returns a sin expression object
template <typename E>
inline ArrayOpUnary<E, Sin> sin(const ArrayExpression<typename E::value_type, E>& expr)
{
    return ArrayOpUnary<E, Sin>(static_cast<const E&>(expr));
}

// Generic binary operator expression class
template <typename E1, typename E2, template <typename> typename F>
class ArrayOpBinary : public ArrayExpression<typename E1::value_type, ArrayOpBinary<E1, E2, F>>
{
    const typename E1::handle_type m_expr_left;
    const typename E2::handle_type m_expr_right;

 public:
    using value_type = typename E1::value_type;
    using handle_type = ArrayOpBinary;

    ArrayOpBinary(const ArrayOpBinary& op) :
        m_expr_left(op.m_expr_left), m_expr_right(op.m_expr_right)
    {}

    ArrayOpBinary(const E1& expr_left, const E2& expr_right) :
        m_expr_left(expr_left), m_expr_right(expr_right)
    {}

    __device__ value_type operator[](const size_t i) const
    {
        return F<typename E1::value_type>::op(this->m_expr_left[i], this->m_expr_right[i]);
    }

    size_t size() const
    {
        return this->m_expr_left.size() == 1 ? this->m_expr_right.size() : this->m_expr_left.size();
    }
};

// Addition operation
template <typename T>
struct Add
{
    __device__ static inline T op(const T a, const T b) { return a + b; }
};

// Add: array + array
template <typename E1, typename E2>
inline ArrayOpBinary<E1, E2, Add> operator+(
    const ArrayExpression<typename E1::value_type, E1>& a,
    const ArrayExpression<typename E2::value_type, E2>& b)
{
    return ArrayOpBinary<E1, E2, Add>(static_cast<const E1&>(a),
                                    static_cast<const E2&>(b));
}

// Add: scalar + array
template <typename E>
inline ArrayOpBinary<Scalar<typename E::value_type>, E, Add> operator+(
    const Scalar<typename E::value_type>& a,
    const ArrayExpression<typename E::value_type, E>& b)
{
    return ArrayOpBinary<Scalar<typename E::value_type>, E, Add>(a, static_cast<const E&>(b));
}

// Add: array + scalar
template <typename E>
inline ArrayOpBinary<Scalar<typename E::value_type>, E, Add> operator+(
    const ArrayExpression<typename E::value_type, E>& a,
    const Scalar<typename E::value_type>& b)
{
    return b + a;
}

// Multiplication operation
template <typename T>
struct Mul
{
    __device__ static inline T op(const T a, const T b) { return a * b; }
};

// Multiply: array * array
template <typename E1, typename E2>
inline ArrayOpBinary<E1, E2, Mul> operator*(
    const ArrayExpression<typename E1::value_type, E1>& a,
    const ArrayExpression<typename E2::value_type, E2>& b)
{
    return ArrayOpBinary<E1, E2, Mul>(static_cast<const E1&>(a),
                                    static_cast<const E2&>(b));
}

// Multiply: scalar * array
template <typename E>
inline ArrayOpBinary<Scalar<typename E::value_type>, E, Mul> operator*(
    const Scalar<typename E::value_type>& a,
    const ArrayExpression<typename E::value_type, E>& b)
{
    return ArrayOpBinary<Scalar<typename E::value_type>, E, Mul>(a, static_cast<const E&>(b));
}

// Multiply: array * scalar
template <typename E>
inline ArrayOpBinary<Scalar<typename E::value_type>, E, Mul> operator*(
    const ArrayExpression<typename E::value_type, E>& a,
    const Scalar<typename E::value_type>& b)
{
    return b * a;
}

