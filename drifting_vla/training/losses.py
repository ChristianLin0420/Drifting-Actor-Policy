"""
Drifting Loss Functions
=======================

This module implements the loss functions for training Drifting-VLA models:

1. DriftingLoss: Basic drifting loss in action space
   L = E[||x - sg(x + V(x))||^2]
   
2. FeatureSpaceLoss: Multi-scale drifting loss in learned feature space
   L = sum_j E[||phi_j(x) - sg(phi_j(x) + V(phi_j(x)))||^2]

The key insight is that the stop-gradient (sg) on the target means the
model learns to generate samples that minimize the distance to their
"drifted" positions, effectively evolving the generated distribution
toward the data distribution.
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from drifting_vla.training.drifting_field import compute_drifting_field


@dataclass
class DriftingLossOutput:
    """
    Output container for drifting loss computation.
    
    Attributes:
        loss: The computed loss value (scalar). With normalize_drift=True,
            this is always ≈ 1.0 by design — NOT the convergence metric.
        drift_norm: E[||V_normalized||^2] (constant ≈ D with normalization).
        raw_drift_norm: E[||V_raw||^2] BEFORE normalization.
            THIS is the true convergence metric — decreases as model learns.
        lambda_V: Normalization factor sqrt(E[||V||^2]/D).
            Also a convergence metric — approaches 0 at equilibrium.
        per_temp_losses: Optional dict of losses per temperature.
        drift_field: The computed drifting field V (for visualization).
    """
    loss: torch.Tensor
    drift_norm: torch.Tensor
    raw_drift_norm: Optional[torch.Tensor] = None
    lambda_V: Optional[torch.Tensor] = None
    per_temp_losses: Optional[dict[str, torch.Tensor]] = None
    drift_field: Optional[torch.Tensor] = None


class DriftingLoss(nn.Module):
    """
    Drifting loss for action-space training.
    
    Computes: L = E[||x - sg(x + V_{p,q}(x))||^2]
    
    where:
    - x: Generated actions from the model
    - V_{p,q}(x): Drifting field pointing toward data distribution
    - sg: Stop-gradient operator
    
    The loss encourages the model to generate samples that, when drifted,
    stay in place (i.e., are already at the target distribution).
    
    Args:
        temperatures: List of temperature values for multi-scale kernels.
            Default: [0.02, 0.05, 0.2]
        normalize_features: Apply feature normalization for scale invariance.
            Default: True
        normalize_drift: Apply drift normalization for stable training.
            Default: True
        reduction: How to reduce the loss ('mean', 'sum', 'none').
            Default: 'mean'
    
    Example:
        >>> loss_fn = DriftingLoss(temperatures=[0.02, 0.05, 0.2])
        >>> actions = model(noise, images, language)  # [B, T, D]
        >>> actions_flat = actions.view(B, -1)  # Flatten action sequence
        >>> output = loss_fn(actions_flat, expert_actions_flat, actions_flat.detach())
        >>> output.loss.backward()
    """
    
    def __init__(
        self,
        temperatures: list[float] = [0.02, 0.05, 0.2],
        normalize_features: bool = True,
        normalize_drift: bool = True,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.temperatures = temperatures
        self.normalize_features = normalize_features
        self.normalize_drift = normalize_drift
        self.reduction = reduction
    
    def forward(
        self,
        x: torch.Tensor,
        y_pos: torch.Tensor,
        y_neg: torch.Tensor,
    ) -> DriftingLossOutput:
        """
        Compute the drifting loss.
        
        Args:
            x: Generated samples from the model.
                Shape: [N, D] where N is batch size, D is flattened action dim
            y_pos: Positive samples from data distribution (expert actions).
                Shape: [N_pos, D]
            y_neg: Negative samples from generated distribution.
                Shape: [N_neg, D]. Often this is x.detach() or samples from
                a replay buffer of recently generated actions.
        
        Returns:
            DriftingLossOutput containing:
                - loss: Scalar loss value
                - drift_norm: ||V||^2 for monitoring equilibrium
        """
        # Compute drifting field with raw norm tracking
        V, raw_drift_norm, lambda_V = compute_drifting_field(
            x=x,
            y_pos=y_pos,
            y_neg=y_neg,
            temperatures=self.temperatures,
            normalize_features=self.normalize_features,
            normalize_drift=self.normalize_drift,
            return_raw_norm=True,
        )
        
        # Target: x + V (with stop-gradient)
        x_target = (x + V).detach()
        
        # Compute MSE loss
        # NOTE: With normalize_drift=True, this is ≈ 1.0 by design.
        # The true convergence signal is raw_drift_norm and lambda_V.
        loss_per_sample = ((x - x_target) ** 2).mean(dim=1)  # [N]
        
        if self.reduction == 'mean':
            loss = loss_per_sample.mean()
        elif self.reduction == 'sum':
            loss = loss_per_sample.sum()
        else:  # 'none'
            loss = loss_per_sample
        
        # Post-normalization drift norm (constant ≈ D)
        drift_norm = (V ** 2).sum(dim=1).mean()
        
        # Per-temperature losses (for visualization)
        per_temp_losses = {}
        try:
            for tau in self.temperatures:
                _, raw_norm_tau, _ = compute_drifting_field(
                    x=x, y_pos=y_pos, y_neg=y_neg,
                    temperatures=[tau],
                    normalize_features=self.normalize_features,
                    normalize_drift=self.normalize_drift,
                    return_raw_norm=True,
                )
                per_temp_losses[f'tau_{tau}'] = raw_norm_tau
        except Exception:
            per_temp_losses = None
        
        return DriftingLossOutput(
            loss=loss,
            drift_norm=drift_norm,
            raw_drift_norm=raw_drift_norm,
            lambda_V=lambda_V,
            per_temp_losses=per_temp_losses,
            drift_field=V.detach(),
        )


class FeatureSpaceLoss(nn.Module):
    """
    Multi-scale feature-space drifting loss.
    
    Instead of computing drifting loss directly in action space, this loss
    operates on learned feature representations at multiple scales. This
    provides richer gradient information similar to perceptual losses in
    image generation.
    
    The loss computes drifting in feature space:
    L = sum_j E[||phi_j(a) - sg(phi_j(a) + V(phi_j(a)))||^2]
    
    where phi_j represents features at different temporal scales (global,
    local patches of 2-steps, 4-steps, etc.)
    
    Args:
        feature_encoder: Neural network that extracts multi-scale features
            from action sequences. Should implement extract_multi_scale().
        feature_scales: List of scale names to use for loss computation.
            Default: ['global', 'patch_2', 'patch_4']
        temperatures: Temperature values for drifting field computation.
            Default: [0.02, 0.05, 0.2]
        scale_weights: Optional weights for each scale. If None, uniform.
    
    Example:
        >>> encoder = ActionFeatureEncoder(...)
        >>> loss_fn = FeatureSpaceLoss(encoder, ['global', 'patch_2'])
        >>> actions = model(noise, images, language)  # [B, T, D]
        >>> output = loss_fn(actions, expert_actions, actions.detach())
    """
    
    def __init__(
        self,
        feature_encoder: nn.Module,
        feature_scales: list[str] = ['global', 'patch_2', 'patch_4'],
        temperatures: list[float] = [0.02, 0.05, 0.2],
        scale_weights: Optional[dict[str, float]] = None,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.feature_scales = feature_scales
        self.temperatures = temperatures
        self.scale_weights = scale_weights or {s: 1.0 for s in feature_scales}
        
        # Per-scale drifting loss
        self.drifting_loss = DriftingLoss(
            temperatures=temperatures,
            normalize_features=True,
            normalize_drift=True,
        )
    
    def forward(
        self,
        actions: torch.Tensor,
        actions_pos: torch.Tensor,
        actions_neg: torch.Tensor,
    ) -> DriftingLossOutput:
        """
        Compute multi-scale feature-space drifting loss.
        
        Args:
            actions: Generated action sequences from the model.
                Shape: [B, T, D_a] where T is horizon, D_a is action dim
            actions_pos: Positive (expert) action sequences.
                Shape: [N_pos, T, D_a]
            actions_neg: Negative (generated) action sequences.
                Shape: [N_neg, T, D_a]
        
        Returns:
            DriftingLossOutput with total loss summed across scales
        """
        # Extract multi-scale features
        features = self.feature_encoder.extract_multi_scale(actions)
        features_pos = self.feature_encoder.extract_multi_scale(actions_pos)
        features_neg = self.feature_encoder.extract_multi_scale(actions_neg)
        
        total_loss = torch.tensor(0.0, device=actions.device)
        total_drift_norm = torch.tensor(0.0, device=actions.device)
        per_scale_losses = {}
        
        for scale in self.feature_scales:
            if scale not in features:
                continue
            
            # Get features at this scale
            feat = features[scale]  # [B, D_feat]
            feat_pos = features_pos[scale]  # [N_pos, D_feat]
            feat_neg = features_neg[scale]  # [N_neg, D_feat]
            
            # Compute drifting loss in feature space
            output = self.drifting_loss(feat, feat_pos, feat_neg)
            
            # Weighted sum
            weight = self.scale_weights.get(scale, 1.0)
            total_loss = total_loss + weight * output.loss
            total_drift_norm = total_drift_norm + output.drift_norm
            per_scale_losses[f'loss_{scale}'] = output.loss.detach()
        
        # Average drift norm across scales
        avg_drift_norm = total_drift_norm / len(self.feature_scales)
        
        return DriftingLossOutput(
            loss=total_loss,
            drift_norm=avg_drift_norm,
            per_temp_losses=per_scale_losses,
        )


class CombinedDriftingLoss(nn.Module):
    """
    Combined action-space and feature-space drifting loss.
    
    This loss combines direct action-space drifting with multi-scale
    feature-space drifting for the best of both worlds:
    
    L = lambda_action * L_action + lambda_feature * L_feature
    
    Args:
        feature_encoder: Optional feature encoder for feature-space loss.
            If None, only action-space loss is used.
        action_weight: Weight for action-space loss. Default: 1.0
        feature_weight: Weight for feature-space loss. Default: 1.0
        temperatures: Temperature values for drifting field.
        feature_scales: Scales to use for feature-space loss.
    
    Example:
        >>> encoder = ActionFeatureEncoder(...)
        >>> loss_fn = CombinedDriftingLoss(encoder, action_weight=1.0, feature_weight=0.5)
        >>> output = loss_fn(actions, expert_actions, actions.detach())
    """
    
    def __init__(
        self,
        feature_encoder: Optional[nn.Module] = None,
        action_weight: float = 1.0,
        feature_weight: float = 1.0,
        temperatures: list[float] = [0.02, 0.05, 0.2],
        feature_scales: list[str] = ['global', 'patch_2', 'patch_4'],
    ):
        super().__init__()
        self.action_weight = action_weight
        self.feature_weight = feature_weight
        self.feature_encoder = feature_encoder
        
        # Action-space loss
        self.action_loss = DriftingLoss(temperatures=temperatures)
        
        # Feature-space loss (if encoder provided)
        if feature_encoder is not None:
            self.feature_loss = FeatureSpaceLoss(
                feature_encoder=feature_encoder,
                feature_scales=feature_scales,
                temperatures=temperatures,
            )
        else:
            self.feature_loss = None
    
    def forward(
        self,
        actions: torch.Tensor,
        actions_pos: torch.Tensor,
        actions_neg: torch.Tensor,
    ) -> DriftingLossOutput:
        """
        Compute combined drifting loss.
        
        Args:
            actions: Generated action sequences [B, T, D_a]
            actions_pos: Expert action sequences [N_pos, T, D_a]
            actions_neg: Negative action sequences [N_neg, T, D_a]
        
        Returns:
            DriftingLossOutput with combined loss
        """
        B, T, D_a = actions.shape
        
        # Flatten actions for action-space loss
        actions_flat = actions.view(B, -1)
        actions_pos_flat = actions_pos.view(actions_pos.shape[0], -1)
        actions_neg_flat = actions_neg.view(actions_neg.shape[0], -1)
        
        # Action-space loss
        action_output = self.action_loss(actions_flat, actions_pos_flat, actions_neg_flat)
        total_loss = self.action_weight * action_output.loss
        
        losses_dict = {
            'loss_action': action_output.loss.detach(),
            'drift_norm_action': action_output.drift_norm.detach(),
        }
        
        # Feature-space loss
        if self.feature_loss is not None:
            feature_output = self.feature_loss(actions, actions_pos, actions_neg)
            total_loss = total_loss + self.feature_weight * feature_output.loss
            
            losses_dict['loss_feature'] = feature_output.loss.detach()
            losses_dict['drift_norm_feature'] = feature_output.drift_norm.detach()
            
            if feature_output.per_temp_losses:
                losses_dict.update(feature_output.per_temp_losses)
        
        return DriftingLossOutput(
            loss=total_loss,
            drift_norm=action_output.drift_norm,
            per_temp_losses=losses_dict,
        )


