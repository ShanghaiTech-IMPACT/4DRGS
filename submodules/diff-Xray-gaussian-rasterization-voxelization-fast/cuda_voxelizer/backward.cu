/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov3D2(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

__global__ void computeCov3DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float sVoxel_x, float sVoxel_y, float sVoxel_z,
	const float center_x, float center_y, float center_z,
	const float* dL_dconic3D,
	float3* dL_dmean3D,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float dVoxel_x = sVoxel_x / (float)nVoxel_x;
	float dVoxel_y = sVoxel_y / (float)nVoxel_y;
	float dVoxel_z = sVoxel_z / (float)nVoxel_z;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;
	float3 mean = means[idx];

	float dL_dconic_a = dL_dconic3D[idx * 6 + 0];
	float dL_dconic_b = dL_dconic3D[idx * 6 + 1];
	float dL_dconic_c = dL_dconic3D[idx * 6 + 2];
	float dL_dconic_d = dL_dconic3D[idx * 6 + 3];
	float dL_dconic_e = dL_dconic3D[idx * 6 + 4];
	float dL_dconic_f = dL_dconic3D[idx * 6 + 5];

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	glm::mat3 M = glm::mat3(
		1.0f / dVoxel_x, 0.0f, 0.0f,
		0.0f, 1.0f  / dVoxel_y, 0.0f,
		0.0f, 0.0f, 1.0f / dVoxel_z);
	glm::mat3 cov = glm::transpose(M) * glm::transpose(Vrk) * M;

	float hata = cov[0][0];
	float hatb = cov[0][1];
	float hatc = cov[0][2];
	float hatd = cov[1][1];
	float hate = cov[1][2];
	float hatf = cov[2][2];
	float denom = hata * hatd * hatf + 2 * hatb * hatc * hate - hata * hate * hate - hatf * hatb * hatb - hatd * hatc * hatc;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	float dL_dhata = 0, dL_dhatb = 0, dL_dhatc = 0, dL_dhatd = 0, dL_dhate = 0, dL_dhatf = 0;

	if (denom2inv != 0)
	{
		float denom_da = hatd * hatf - hate * hate;
		float denom_db = 2 * hatc * hate - 2 * hatf * hatb;
		float denom_dc = 2 * hatb * hate - 2 * hatd * hatc;
		float denom_dd = hata * hatf - hatc * hatc;
		float denom_de = 2 * hatb * hatc - 2 * hata * hate;
		float denom_df = hata * hatd - hatb * hatb;

		float ce_bf = hatc * hate - hatb * hatf;
		float be_cd = hatb * hate - hatc * hatd;
		float bc_ae = hatb * hatc - hata * hate;

		dL_dhata = denom2inv * (-denom_da*denom_da*dL_dconic_a - ce_bf*denom_da*dL_dconic_b - be_cd*denom_da*dL_dconic_c + (hatf*denom-denom_dd*denom_da)*dL_dconic_d + (-hate*denom-bc_ae*denom_da)*dL_dconic_e + (hatd*denom-denom_df*denom_da)*dL_dconic_f);
		dL_dhatb = denom2inv * (-denom_da*denom_db*dL_dconic_a + (-hatf*denom-ce_bf*denom_db)*dL_dconic_b + (hate*denom-be_cd*denom_db)*dL_dconic_c - denom_dd*denom_db*dL_dconic_d + (hatc*denom-bc_ae*denom_db)*dL_dconic_e + (-2*hatb*denom-denom_df*denom_db)*dL_dconic_f);
		dL_dhatc = denom2inv * (-denom_da*denom_dc*dL_dconic_a + (hate*denom-ce_bf*denom_dc)*dL_dconic_b + (-hatd*denom-be_cd*denom_dc)*dL_dconic_c + (-2*hatc*denom-denom_dd*denom_dc)*dL_dconic_d + (hatb*denom-bc_ae*denom_dc)*dL_dconic_e - denom_df*denom_dc*dL_dconic_f);
		dL_dhatd = denom2inv * ((hatf*denom-denom_da*denom_dd)*dL_dconic_a - ce_bf*denom_dd*dL_dconic_b +(-hatc*denom-be_cd*denom_dd)*dL_dconic_c - denom_dd*denom_dd*dL_dconic_d - bc_ae*denom_dd*dL_dconic_e + (hata*denom-denom_df*denom_dd)*dL_dconic_f);
		dL_dhate = denom2inv * ((-2*hate*denom-denom_da*denom_de)*dL_dconic_a + (hatc*denom-ce_bf*denom_de)*dL_dconic_b + (hatb*denom-be_cd*denom_de)*dL_dconic_c - denom_dd*denom_de*dL_dconic_d + (-hata*denom-bc_ae*denom_de)*dL_dconic_e + -denom_df*denom_de*dL_dconic_f);
		dL_dhatf = denom2inv * ((hatd*denom-denom_da*denom_df)*dL_dconic_a + (-hatb*denom-ce_bf*denom_df)*dL_dconic_b - be_cd*denom_df*dL_dconic_c + (hata*denom-denom_dd*denom_df)*dL_dconic_d - bc_ae*denom_df*dL_dconic_e - denom_df*denom_df*dL_dconic_f);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry
		// dL_da
		dL_dcov[6 * idx + 0] += M[0][0]*M[0][0]*dL_dhata + M[0][0]*M[1][0]*dL_dhatb + M[0][0]*M[2][0]*dL_dhatc + M[1][0]*M[1][0]*dL_dhatd + M[1][0]*M[2][0]*dL_dhate + M[2][0]*M[2][0]*dL_dhatf;
		// dL_dd
		dL_dcov[6 * idx + 3] += M[0][1]*M[0][1]*dL_dhata + M[0][1]*M[1][1]*dL_dhatb + M[0][1]*M[2][1]*dL_dhatc + M[1][1]*M[1][1]*dL_dhatd + M[1][1]*M[2][1]*dL_dhate + M[2][1]*M[2][1]*dL_dhatf;
		// dL_df
		dL_dcov[6 * idx + 5] += M[0][2]*M[0][2]*dL_dhata + M[0][2]*M[1][2]*dL_dhatb + M[0][2]*M[2][2]*dL_dhatc + M[1][2]*M[1][2]*dL_dhatd + M[1][2]*M[2][2]*dL_dhate + M[2][2]*M[2][2]*dL_dhatf;

		// dL_db
		dL_dcov[6 * idx + 1] += 2*M[0][0]*M[0][1]*dL_dhata + (M[0][1]*M[1][0]+M[0][0]*M[1][1])*dL_dhatb + (M[0][1]*M[2][0]+M[0][0]*M[2][1])*dL_dhatc + 2*M[1][0]*M[1][1]*dL_dhatd + (M[1][1]*M[2][0]+M[1][0]*M[2][1])*dL_dhate + 2*M[2][0]*M[2][1]*dL_dhatf;
		// dL_dc
		dL_dcov[6 * idx + 2] += 2*M[0][0]*M[0][2]*dL_dhata + (M[0][2]*M[1][0]+M[0][0]*M[1][2])*dL_dhatb + (M[0][2]*M[2][0]+M[0][0]*M[2][2])*dL_dhatc + 2*M[1][0]*M[1][2]*dL_dhatd + (M[1][2]*M[2][0]+M[1][0]*M[2][2])*dL_dhate + 2*M[2][0]*M[2][2]*dL_dhatf;
		// dL_de
		dL_dcov[6 * idx + 4] += 2*M[0][1]*M[0][2]*dL_dhata + (M[0][2]*M[1][1]+M[0][1]*M[1][2])*dL_dhatb + (M[0][2]*M[2][1]+M[0][1]*M[2][2])*dL_dhatc + 2*M[1][1]*M[1][2]*dL_dhatd + (M[1][2]*M[2][1]+M[1][1]*M[2][2])*dL_dhate + 2*M[2][1]*M[2][2]*dL_dhatf;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}
}


template<int C>
__global__ void preprocessCUDA(
	int P,
	const float3* means,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float3* dL_dmean3D_norm,
	glm::vec3* dL_dmeans,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	glm::vec3 dL_dmean;
	dL_dmean.x = dL_dmean3D_norm[idx].x;
	dL_dmean.y = dL_dmean3D_norm[idx].y;
	dL_dmean.z = dL_dmean3D_norm[idx].z;

	// printf("dL_dmean.x: %f\n", dL_dmean.x);
	// printf("dL_dmean.y: %f\n", dL_dmean.y);
	// printf("dL_dmean.z: %f\n", dL_dmean.z);

	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D2(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);

}


template <uint32_t C>
__global__ void fastBackward3D(
    const uint2* ranges, const uint32_t* point_list,
    int nx, int ny, int nz, float sx, float sy, float sz,
    const float3* means, const float* conics,
    const float* grad_volume, const float* grad_others,
    float3* grad_mean, float* grad_conic, float* grad_density,
    const uint32_t* offsets, const uint32_t* bucket_map, uint32_t total_buckets)
{
    auto warp=cg::tiled_partition<32>(cg::this_thread_block());
    uint32_t bid=blockIdx.x*(blockDim.x/32)+warp.meta_group_rank();
    if(bid>=total_buckets) return;
    uint32_t tile=bucket_map[bid]; uint2 range=ranges[tile];
    uint32_t base=tile?offsets[tile-1]:0;
    uint32_t index=range.x+(bid-base)*32+warp.thread_rank();
    bool valid=index<range.y; int gid=valid?point_list[index]:0;
    float3 xyz=valid?means[gid]:make_float3(0,0,0);
    float co[7]={0}; if(valid) for(int i=0;i<7;++i) co[i]=conics[gid*7+i];
    int tx=(nx+BLOCK3D_X-1)/BLOCK3D_X, ty=(ny+BLOCK3D_Y-1)/BLOCK3D_Y;
    int ox=(tile%tx)*BLOCK3D_X, oy=((tile/tx)%ty)*BLOCK3D_Y, oz=(tile/(tx*ty))*BLOCK3D_Z;
    float acc[10]={0}; int active=0,x=0,y=0,z=0; float grad=0,geo=0;
    for(int it=0;it<BLOCK3D_SIZE+31;++it){
        active=warp.shfl_up(active,1); x=warp.shfl_up(x,1); y=warp.shfl_up(y,1); z=warp.shfl_up(z,1);
        grad=warp.shfl_up(grad,1); geo=warp.shfl_up(geo,1);
        if(warp.thread_rank()==0){
            x=ox+it%BLOCK3D_X; y=oy+(it/BLOCK3D_X)%BLOCK3D_Y; z=oz+it/(BLOCK3D_X*BLOCK3D_Y);
            active=it<BLOCK3D_SIZE && x<nx && y<ny && z<nz; grad=0; geo=0;
            if(active){
                int vi=x*ny*nz+y*nz+z; // Preserve 4DRGS XYZ output order.
                for(int c=0;c<C;++c) grad+=grad_volume[c*nx*ny*nz+vi];
                #if VOXEL_AXUTILITY_GEO_BACKWARD
                geo=grad_others[GEO_OFFSET*nx*ny*nz+vi];
                #endif
            }
        }
        if(!active || !valid) continue;
        float dx=xyz.x-(x+0.5f),dy=xyz.y-(y+0.5f),dz=xyz.z-(z+0.5f);
        float power=-0.5f*(co[0]*dx*dx+co[3]*dy*dy+co[5]*dz*dz)-co[1]*dx*dy-co[2]*dx*dz-co[4]*dy*dz;
        if(power>0) continue;
        float G=expf(power); if(co[6]*G<0.000001f) continue;
        float dlG=co[6]*grad+geo, gx=G*dx,gy=G*dy,gz=G*dz;
        // Match legacy's existing mean-gradient convention.
        acc[0]+=dlG*(-co[0]*gx-co[1]*gy-co[2]*gz)*(sx/nx);
        acc[1]+=dlG*(-co[3]*gy-co[1]*gx-co[4]*gz)*(sy/ny);
        acc[2]+=dlG*(-co[5]*gz-co[2]*gx-co[4]*gy)*(sz/nz);
        acc[3]+=-0.5f*gx*dx*dlG; acc[4]+=-gx*dy*dlG; acc[5]+=-gx*dz*dlG;
        acc[6]+=-0.5f*gy*dy*dlG; acc[7]+=-gy*dz*dlG; acc[8]+=-0.5f*gz*dz*dlG;
        acc[9]+=G*grad;
    }
    if(valid){
        atomicAdd(&grad_mean[gid].x,acc[0]); atomicAdd(&grad_mean[gid].y,acc[1]); atomicAdd(&grad_mean[gid].z,acc[2]);
        for(int i=0;i<6;++i) atomicAdd(grad_conic+gid*6+i,acc[3+i]);
        atomicAdd(grad_density+gid,acc[9]);
    }
}

void BACKWARD::preprocess(
	int P,
	const float3* means3D,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float sVoxel_x, float sVoxel_y, float sVoxel_z,
	const float center_x, float center_y, float center_z,
	const float3* dL_dmean3D_norm,
	const float* dL_dconic3D,
	glm::vec3* dL_dmean3D,
	float* dL_dcov3D,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	computeCov3DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		nVoxel_x, nVoxel_y, nVoxel_z,
		sVoxel_x, sVoxel_y, sVoxel_z,
		center_x, center_y, center_z,
		dL_dconic3D,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		(float3*)means3D,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		(float3*)dL_dmean3D_norm,
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		dL_dscale,
		dL_drot);

}

void BACKWARD::render(
    const uint32_t* bucket_offsets, const uint32_t* bucket_map, uint32_t total_buckets,
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const int nVoxel_x, int nVoxel_y, int nVoxel_z,
	const float sVoxel_x, float sVoxel_y, float sVoxel_z,
	const float3* means3D_norm,
	const float* conic_opacity,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dothers,
	float3* dL_dmean3D_norm,
	float* dL_dconic3D,
	float* dL_dopacity
	)
{
    if(total_buckets) fastBackward3D<NUM_CHANNELS><<<(total_buckets+3)/4,128>>>(
        ranges,point_list,nVoxel_x,nVoxel_y,nVoxel_z,sVoxel_x,sVoxel_y,sVoxel_z,
        means3D_norm,conic_opacity,dL_dpixels,dL_dothers,dL_dmean3D_norm,
        dL_dconic3D,dL_dopacity,bucket_offsets,bucket_map,total_buckets);
}
