#include <vector>

#include "CudaTimer.h"
#include "GpuArray.h"

// Helper kernel to set data in array
template <typename T>
__global__ void fill_incrementing_kernel(T* data, const T val, const int N)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        data[idx] = val + idx;
    }
}

// Fill array with incrementing values (just a test helper)
template <typename T>
void fill_incrementing(Array<T>& arr, const T val)
{
    dim3 tpb(256, 1);
    dim3 bpg(arr.size() / tpb.x, 1);
    fill_incrementing_kernel<<<bpg, tpb>>>(arr.data(), val, arr.size());
}

template <typename T>
void print(const Array<T>& arr, const int n = 32)
{
    std::vector<T> result(n);
    cudaMemcpy(result.data(), arr.data(), n * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << std::endl;    
    for (const auto& e : result)
    {
        std::cout << e << " ";
    }
    std::cout << std::endl;    
}

void test(const std::string& msg)
{
    const size_t N = 1024 * 1024 * 50;

    Array<float> a0(N);
    Array<float> a1(N);
    Array<float> a2(N);
    Array<float> a3(N);
    Array<float> a4(N);

    fill_incrementing(a0, 1.0f); // 1, 2, 3, 4, 5, 6, ...
    fill_incrementing(a1, 2.0f); // 2, 3, 4, 5, 6, 7, ...
    fill_incrementing(a2, 3.0f); // 3, 4, 5, 6, 7, 9, ...
    fill_incrementing(a3, 4.0f); // ...
    fill_incrementing(a4, 5.0f); // ...

    Array<float> result;

    CudaTimer t;
    t.start();
    // Addition and multiplication
    result = a3 * a0 + 2.0f * a1 + a2 + 1.0f + a4;
    t.stop();
    t.print(msg);

    print(result);

    // Slicing and unary ops
    result = sin(a0.slice(0, -1, 2) + a0.slice(1, -1, 2)) * 0.5f;

    print(result);
    std::cout << "\nOriginal --> Sliced array size: " << a0.size() << " --> " << result.size()
              << " (should be " << (a0.size()/2) << ")" << std::endl;
}

int main(int argc, char const* argv[])
{
    test("Expression Templates:");
    return 0;
}
