from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type, Dict, List, Tuple

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.model_components.losses import orientation_loss, pred_normal_loss
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler, UniformSampler, ProposalNetworkSampler
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from nerfstudio.utils import colormaps


@dataclass
class NerfactoModelWithBackgroundConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: NerfactoModelWithBackground)

    num_samples_outside: int = 48
    """ Sample numbers outside sphere.   """

    far_plane_bg: float = 100
    """sample inversely from far to 1000 and points and forward the bg model"""

    use_background: bool = False


class NerfactoModelWithBackground(NerfactoModel):
    config: NerfactoModelWithBackgroundConfig

    def populate_modules(self):

        super(NerfactoModelWithBackground, self).populate_modules()
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=None,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        # Collider
        self.collider = AABBBoxCollider(self.scene_box, near_plane=0.05)
        if self.config.use_background:
            self.field_background = TCNNNerfactoField(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            )
            self.density_fns_background = []
            num_prop_nets = self.config.num_proposal_iterations
            # Build the proposal network(s)
            self.proposal_networks_background = torch.nn.ModuleList()

            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=self.scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks_background.append(network)
            self.density_fns_background.extend([network.density_fn for network in self.proposal_networks])
            self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

            update_schedule = lambda step: np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

            # Change proposal network initial sampler if uniform
            initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
            if self.config.proposal_initial_sampler == "uniform":
                initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

            self.proposal_sampler_background = ProposalNetworkSampler(
                num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
                num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
                num_proposal_network_iterations=self.config.num_proposal_iterations,
                single_jitter=self.config.use_single_jitter,
                update_sched=update_schedule,
                initial_sampler=initial_sampler,
            )

        else:
            self.background = torch.nn.Parameter(torch.tensor(0, dtype=torch.float, device="cuda"))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        if self.config.use_background:
            param_groups["background_fields"] = list(self.field_background.parameters())
        else:
            param_groups["background_fields"] = (self.background,)
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples: RaySamples
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights, transmittance = ray_samples.get_weights_and_transmittance(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        bg_transmittance = transmittance[:, -1, :]

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "foreground": rgb.detach().clone()
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # background side
        if self.config.use_background:
            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
            depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
            accumulation_bg = self.renderer_accumulation(weights=weights_bg)

            # merge background color to forgound color
            rgb = rgb + bg_transmittance * rgb_bg

            bg_outputs = {
                "bg_rgb": rgb_bg,
                "bg_accumulation": accumulation_bg,
                "bg_depth": depth_bg,
                "bg_transmittance": bg_transmittance
            }
            outputs.update(bg_outputs)
            outputs["rgb"] = rgb

        return outputs

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        torch.clamp_(rgb, min=0.0, max=1.0)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i
        # background
        if self.config.use_background:
            images_dict["foreground"] = outputs["foreground"]
            images_dict["bg_rgb"] = outputs["bg_rgb"]
            images_dict["bg_accumulation"] = outputs["bg_accumulation"]

        return metrics_dict, images_dict
