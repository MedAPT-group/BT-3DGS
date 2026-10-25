#ifndef R2SPLAT_OPS_H
#define R2SPLAT_OPS_H

#include <torch/extension.h>
#include <tuple>

namespace r2splat {

std::tuple<torch::Tensor, torch::Tensor> parameter_transform_fwd_tensor(
    const torch::Tensor&, const torch::Tensor&, double, double);
std::tuple<torch::Tensor, torch::Tensor> parameter_transform_bwd_tensor(
    const torch::Tensor&, const torch::Tensor&, double, double,
    const torch::Tensor&);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_tensor(
    const torch::Tensor&, const torch::Tensor&, uint32_t, uint32_t, uint32_t);
torch::Tensor isect_offset_encode_tensor(
    const torch::Tensor&, uint32_t, uint32_t, uint32_t);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_3d_tensor(
    const torch::Tensor&, const torch::Tensor&, uint32_t, uint32_t, uint32_t, uint32_t);
torch::Tensor isect_offset_encode_3d_tensor(
    const torch::Tensor&, uint32_t, uint32_t, uint32_t);

std::tuple<torch::Tensor, torch::Tensor> conics_mu_fwd_tensor(
    const torch::Tensor&, double);
torch::Tensor conics_mu_bwd_tensor(
    const torch::Tensor&, double, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_image_projection_fwd_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, uint32_t, uint32_t, float, float);
std::tuple<torch::Tensor, torch::Tensor> world_to_image_projection_bwd_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, uint32_t, uint32_t, const torch::Tensor&,
    const torch::Tensor&);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_volume_mapping_fwd_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, uint32_t, uint32_t, uint32_t,
    double, double);
std::tuple<torch::Tensor, torch::Tensor> world_to_volume_mapping_bwd_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, uint32_t, uint32_t, uint32_t, double,
    const torch::Tensor&, const torch::Tensor&);

torch::Tensor rasterize_to_pixels_fwd_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    uint32_t, uint32_t, uint32_t);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, uint32_t, uint32_t, uint32_t, bool);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_per_gaussian_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, uint32_t, uint32_t, uint32_t,
    bool);

torch::Tensor voxelize_to_voxels_fwd_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, uint32_t, uint32_t, uint32_t,
    uint32_t);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
voxelize_to_voxels_bwd_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, uint32_t,
    uint32_t, uint32_t, uint32_t);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
voxelize_to_voxels_bwd_per_gaussian_tensor(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, uint32_t, uint32_t, uint32_t, uint32_t);

}  // namespace r2splat

#endif  // R2SPLAT_OPS_H
