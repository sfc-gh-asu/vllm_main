# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            logger.debug("Loading weights on %s ...", load_device)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            # By default, the postprocessed weights will be pinned to CPU pinned memory.
            # But in case of weight offloading, we will check if weight_offloading is enabled and moe_allgather_only is not enabled;
            # weight_offloading is enabled by setting "weight_offloading" to True in additional_config.
            # For single GPU, moe_allgather_only is not enabled by default; 
            # For multi-GPU, moe_allgather_only is enabled by setting "moe_allgather_only" to True in additional_config.
            # If both conditions are met (H2D is enabled), we will not pin the postprocessed weights to CPU pinned memory, 
            # because we will pin the postprocessed weights in the weight offloading manager later in GPU Model Runner, 
            # so that the unpinned CPU memory for postprocessed weights will be freed after weight offloader initialization.
            additional_config = getattr(vllm_config, "additional_config", None)
            not_pin_postprocessed_weights_to_cpu = bool(isinstance(additional_config, dict) and additional_config.get("weight_offloading", False) and not additional_config.get("moe_allgather_only", False))
            process_weights_after_loading(model, model_config, target_device, not_pin_postprocessed_weights_to_cpu)
            # process_weights_after_loading(model, model_config, target_device)
        return model.eval()
