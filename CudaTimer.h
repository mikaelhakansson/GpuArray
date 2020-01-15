#pragma once

#include <iostream>
#include <cuda.h>

class CudaTimer
{
public:
    CudaTimer()
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_stop);
    }

    void start()
    {
        cudaEventRecord(m_start);
    }

    void stop()
    {
        cudaEventRecord(m_stop);
    }

    float ms()
    {
        cudaEventSynchronize(m_stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, m_start, m_stop);
        return ms;       
    }

    void print(const std::string& msg = "")
    {
        const float millisec = ms();
        std::cout << msg << " " << millisec << " ms" << std::endl;
    }

    ~CudaTimer()
    {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
    }

private:
    cudaEvent_t m_start;
    cudaEvent_t m_stop;
};
