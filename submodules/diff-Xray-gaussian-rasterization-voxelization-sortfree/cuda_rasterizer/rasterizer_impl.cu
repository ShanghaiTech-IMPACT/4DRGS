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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.etas, P, 128);
	return geom;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, 
	const int width, int height,
	const float* means3D,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	const int mode,
	float* out_color,
	float* out_others,
	int* radii,
	bool debug)
{	
	(void)binningBuffer;
	(void)imageBuffer;

	const float focal_y = height / (2.0f * tan_fovy);  // tan_fovy=1 when parallel  focal_y = height * (DSD / proj_phy_h)
	const float focal_x = width / (2.0f * tan_fovx); // tan_fovx=1 when parallel   focal_x = weight * (DSD / proj_phy_w)

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

    if (NUM_CHANNELS != 1)
    {
        throw std::runtime_error("This version of Diff Gaussian Splatting only handle opacity, not RGB");
    }

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, 
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		cov3D_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.cov3D,
		geomState.conic_opacity,
		geomState.etas,
		tile_grid,
		prefiltered,
		mode
	), debug)

	// Accumulate every Gaussian directly into its covered pixels. X-ray line
	// integrals are additive, so neither tile binning nor depth sorting is needed.
	CHECK_CUDA(FORWARD::render(
		P,
		tile_grid,
		width, height,
		radii,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.etas,
		out_color,
		out_others), debug)

	return P;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int R, 
	const int width, int height,
	const float* means3D,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_deta,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dscale,
	float* dL_drot,
	const int mode,
	bool debug)
{
	(void)R;
	(void)binning_buffer;
	(void)img_buffer;
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);  // tan_fovy=1 when parallel  focal_y = height * (DSD / proj_phy_h)
	const float focal_x = width / (2.0f * tan_fovx); // tan_fovx=1 when parallel   focal_x = weight * (DSD / proj_phy_w)

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	CHECK_CUDA(BACKWARD::render(
		P,
		tile_grid,
		width, height,
		radii,
		geomState.means2D,
		geomState.conic_opacity,
		geomState.etas,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_deta), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, 
		(float3*)means3D,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_deta,
		(glm::vec3*)dL_dmean3D,
		dL_dcov3D,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		mode), debug)

}
