#include <torch/extension.h>
#include "../include/Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Gaussian CT splatting CUDA extension";

    m.def("parameter_transform_fwd", &r2splat::parameter_transform_fwd_tensor);
    m.def("parameter_transform_bwd", &r2splat::parameter_transform_bwd_tensor);

    m.def("isect_tiles", &r2splat::isect_tiles_tensor);
    m.def("isect_offset_encode", &r2splat::isect_offset_encode_tensor);
    m.def("isect_tiles_3d", &r2splat::isect_tiles_3d_tensor);
    m.def("isect_offset_encode_3d", &r2splat::isect_offset_encode_3d_tensor);

    m.def("conics_mu_fwd", &r2splat::conics_mu_fwd_tensor);
    m.def("conics_mu_bwd", &r2splat::conics_mu_bwd_tensor);
    m.def("world_to_image_projection_fwd", &r2splat::world_to_image_projection_fwd_tensor);
    m.def("world_to_image_projection_bwd", &r2splat::world_to_image_projection_bwd_tensor);
    m.def("world_to_volume_mapping_fwd", &r2splat::world_to_volume_mapping_fwd_tensor);
    m.def("world_to_volume_mapping_bwd", &r2splat::world_to_volume_mapping_bwd_tensor);

    m.def("rasterize_to_pixels_fwd", &r2splat::rasterize_to_pixels_fwd_tensor);
    m.def("rasterize_to_pixels_bwd", &r2splat::rasterize_to_pixels_bwd_tensor);
    m.def("rasterize_to_pixels_bwd_per_gaussian", &r2splat::rasterize_to_pixels_bwd_per_gaussian_tensor);
    m.def("voxelize_to_voxels_fwd", &r2splat::voxelize_to_voxels_fwd_tensor);
    m.def("voxelize_to_voxels_bwd", &r2splat::voxelize_to_voxels_bwd_tensor);
    m.def("voxelize_to_voxels_bwd_per_gaussian", &r2splat::voxelize_to_voxels_bwd_per_gaussian_tensor);
}
