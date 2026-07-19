"""
Standalone model definition for the DR-grading desktop app.

This module intentionally contains NO side effects (no downloads, no training,
no dataset loading). It is a clean extraction of the model classes from
`fusion.py` so the Qt application can import the architecture and load a
trained checkpoint without triggering the notebook's training pipeline.

Source of truth: DRConceptFusionModel in fusion.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TinyBackbone(nn.Module):
    """Small backbone used only for synthetic smoke-test checkpoints."""

    output_dim = 64

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)


def build_backbone(name, pretrained, freeze_backbone):
    if name == "tiny":
        backbone = TinyBackbone()
        feat_dim = backbone.output_dim
    elif name == "densenet121":
        if hasattr(models, "DenseNet121_Weights"):
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            base = models.densenet121(weights=weights)
        else:
            base = models.densenet121(pretrained=pretrained)

        class DenseNetFeatures(nn.Module):
            def __init__(self, features):
                super().__init__()
                self.features = features

            def forward(self, x):
                feature_map = self.features(x)
                feature_map = F.relu(feature_map, inplace=False)
                return F.adaptive_avg_pool2d(feature_map, 1).flatten(1)

        backbone = DenseNetFeatures(base.features)
        feat_dim = 1024
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    return backbone, feat_dim


class DRConceptFusionModel(nn.Module):
    """All variants predict grade only from the final concept bottleneck."""

    VALID_MODES = {
        "strict_cbm",
        "gated_concept_fusion",
        "residual_concept_fusion",
    }

    def __init__(
        self,
        n_concepts,
        n_quant_features,
        n_classes=5,
        mode="strict_cbm",
        backbone_name="densenet121",
        pretrained=True,
        freeze_backbone=False,
        dropout_rate=0.3,
        residual_scale=0.5,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}")
        if n_quant_features <= 0:
            raise ValueError("n_quant_features must be positive")

        self.mode = mode
        self.n_concepts = n_concepts
        self.residual_scale = float(residual_scale)
        self.backbone, image_dim = build_backbone(
            backbone_name, pretrained, freeze_backbone
        )
        self.image_concept_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(image_dim, n_concepts),
        )
        self.task_head = nn.Sequential(
            nn.Linear(n_concepts, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, n_classes),
        )

        quant_hidden = max(32, min(128, n_quant_features * 2))
        if mode != "strict_cbm":
            self.quant_encoder = nn.Sequential(
                nn.LayerNorm(n_quant_features),
                nn.Linear(n_quant_features, quant_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate / 2),
            )

        if mode == "gated_concept_fusion":
            self.feature_concept_head = nn.Linear(quant_hidden, n_concepts)
            self.gate_net = nn.Sequential(
                nn.Linear(2 * n_concepts + quant_hidden, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(64, n_concepts),
                nn.Sigmoid(),
            )
        elif mode == "residual_concept_fusion":
            self.residual_head = nn.Sequential(
                nn.Linear(quant_hidden, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, n_concepts),
                nn.Tanh(),
            )

    def forward(self, images, quant_features):
        image_features = self.backbone(images)
        image_concept_logits = self.image_concept_head(image_features)
        image_concept_probs = torch.sigmoid(image_concept_logits)

        output = {
            "image_concept_logits": image_concept_logits,
            "image_concept_probs": image_concept_probs,
            "feature_concept_logits": None,
            "feature_concept_probs": None,
            "gate_values": None,
            "residual_values": None,
        }

        if self.mode == "strict_cbm":
            fused_logits = image_concept_logits
            fused_probs = image_concept_probs
        elif self.mode == "gated_concept_fusion":
            q_embed = self.quant_encoder(quant_features)
            feature_logits = self.feature_concept_head(q_embed)
            feature_probs = torch.sigmoid(feature_logits)
            gate = self.gate_net(
                torch.cat([image_concept_probs, feature_probs, q_embed], dim=1)
            )
            fused_probs = gate * image_concept_probs + (1.0 - gate) * feature_probs
            fused_probs = fused_probs.clamp(1e-5, 1.0 - 1e-5)
            fused_logits = torch.logit(fused_probs)
            output.update(
                feature_concept_logits=feature_logits,
                feature_concept_probs=feature_probs,
                gate_values=gate,
            )
        else:
            q_embed = self.quant_encoder(quant_features)
            residual = self.residual_head(q_embed)
            fused_logits = image_concept_logits + self.residual_scale * residual
            fused_probs = torch.sigmoid(fused_logits)
            output["residual_values"] = residual

        output["concept_logits"] = fused_logits
        output["concept_probs"] = fused_probs
        output["task_logits"] = self.task_head(fused_probs)
        return output
