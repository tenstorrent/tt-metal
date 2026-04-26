# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import timm
import numpy as np
from typing import Dict, Any
import copy
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class TransfuserBackbone(nn.Module):
    """Multi-scale Fusion Transformer for image + LiDAR feature fusion."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        try:
            self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)
        except Exception as e:
            print(f"Failed to load image encoder with error: {e}")
            self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)

        in_channels = 1  # For lidar
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Simple fusion: concatenate features
        self.fusion = nn.Conv2d(512 + 512, 512, kernel_size=1)

    def forward(self, image, lidar):
        image_features = self.image_encoder(image)
        lidar_features = self.lidar_encoder(lidar)
        # Assume last feature is 512 channels
        fused = torch.cat([image_features[-1], lidar_features], dim=1)
        fused = self.fusion(fused)
        return fused


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(self, num_agents: int, d_ffn: int, d_model: int):
        super(AgentHead, self).__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, 6),  # x, y, heading, w, h, class
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        agent_states = self._mlp_states(agent_queries)
        return {"agent_states": agent_states}


class TrajectoryHead(nn.Module):
    """Trajectory prediction head with simplified diffusion."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super(TrajectoryHead, self).__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        # Simplified: no diffusion, just predict trajectory
        self._mlp_traj = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * 3),  # x, y, heading per pose
        )

    def forward(self, ego_query, agents_query, bev_feature, bev_spatial_shape, status_encoding, targets=None, global_img=None) -> Dict[str, torch.Tensor]:
        # Simplified forward
        traj_pred = self._mlp_traj(ego_query)
        traj_pred = traj_pred.view(-1, self._num_poses, 3)
        return {"trajectory": traj_pred}


class V2TransfuserModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._backbone = TransfuserBackbone(config)
        self._tf_decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model=config.tf_d_model, nhead=config.tf_num_head, dim_feedforward=config.tf_d_ffn, dropout=config.tf_dropout, batch_first=True),
            num_layers=config.tf_num_layers
        )
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )
        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )
        # Queries
        self.ego_query = nn.Parameter(torch.randn(1, 1, config.tf_d_model))
        self.agent_queries = nn.Parameter(torch.randn(1, config.num_bounding_boxes, config.tf_d_model))

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        image = features["camera_feature"]
        lidar = features["lidar_feature"]
        bev_feature = self._backbone(image, lidar)
        # Flatten BEV for transformer
        b, c, h, w = bev_feature.shape
        bev_flat = bev_feature.flatten(2).permute(0, 2, 1)  # b, h*w, c
        # Dummy memory for decoder
        memory = bev_flat
        ego_out = self._tf_decoder(self.ego_query.repeat(b, 1, 1), memory)
        agent_out = self._tf_decoder(self.agent_queries.repeat(b, 1, 1), memory)
        agent_pred = self._agent_head(agent_out)
        traj_pred = self._trajectory_head(ego_out[:, 0:1], agent_out, bev_feature, (h, w), None)
        return {**agent_pred, **traj_pred}


# Config classes
class TrajectorySampling:
    num_poses = 8

class TransfuserConfig:
    image_architecture = "resnet34"
    lidar_resolution_width = 256
    lidar_resolution_height = 256
    tf_d_model = 256
    tf_num_head = 8
    tf_d_ffn = 512
    tf_dropout = 0.1
    tf_num_layers = 2
    num_bounding_boxes = 10
    trajectory_sampling = TrajectorySampling()


def create_model():
    config = TransfuserConfig()
    model = V2TransfuserModel(config)
    return model