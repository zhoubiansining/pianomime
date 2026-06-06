from typing import Dict, List, Type, Union

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.utils import get_device


class LayerNormMlpExtractor(MlpExtractor):
    """MlpExtractor variant that inserts LayerNorm after each hidden linear layer."""

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        nn.Module.__init__(self)
        device = get_device(device)

        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])
            vf_layers_dims = net_arch.get("vf", [])
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        self.policy_net, self.latent_dim_pi = self._build_branch(
            feature_dim,
            pi_layers_dims,
            activation_fn,
            layer_norm_eps,
            device,
        )
        self.value_net, self.latent_dim_vf = self._build_branch(
            feature_dim,
            vf_layers_dims,
            activation_fn,
            layer_norm_eps,
            device,
        )

    @staticmethod
    def _build_branch(
        input_dim: int,
        layer_dims: List[int],
        activation_fn: Type[nn.Module],
        layer_norm_eps: float,
        device: Union[th.device, str],
    ) -> tuple[nn.Sequential, int]:
        modules: List[nn.Module] = []
        last_dim = input_dim
        for layer_dim in layer_dims:
            modules.append(nn.Linear(last_dim, layer_dim))
            modules.append(nn.LayerNorm(layer_dim, eps=layer_norm_eps))
            modules.append(activation_fn())
            last_dim = layer_dim
        return nn.Sequential(*modules).to(device), last_dim


class LayerNormActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with LayerNorm in the actor and critic MLP towers."""

    def __init__(self, *args, layer_norm_eps: float = 1e-5, **kwargs):
        self.layer_norm_eps = layer_norm_eps
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = LayerNormMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            layer_norm_eps=self.layer_norm_eps,
        )
