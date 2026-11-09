
#pragma once
// FaCT-GS-style per-Gaussian bucket scheduling; no depth key.
namespace {
__global__ void fastCountBuckets(int N, const uint2* ranges, uint32_t* counts) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) counts[i]=(ranges[i].y-ranges[i].x+31)/32;
}
__global__ void fastMapBuckets(int N, const uint32_t* offsets, uint32_t* map) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    uint32_t begin=i?offsets[i-1]:0;
    for(uint32_t k=begin;k<offsets[i];++k) map[k]=i;
}
}
