"""
Configuration Management
========================

Utilities for loading and managing configurations using Hydra/OmegaConf.
"""

import os
from pathlib import Path
from typing import Any, Optional, Union
from dataclasses import dataclass, field, asdict
import logging

try:
    from omegaconf import OmegaConf, DictConfig
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False

import yaml

logger = logging.getLogger(__name__)


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[dict] = None,
) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file.
        overrides: Optional dict of overrides to apply.
    
    Returns:
        Configuration dictionary.
    
    Example:
        >>> config = load_config('configs/training/baseline.yaml')
        >>> config = load_config('configs/model/dit_l2.yaml', {'hidden_dim': 1024})
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    # Apply overrides
    if overrides:
        config = _deep_update(config, overrides)
    
    # Convert to OmegaConf if available
    if HAS_OMEGACONF:
        config = OmegaConf.create(config)
    
    logger.info(f"Loaded config from {config_path}")
    
    return config


def save_config(config: Union[dict, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dict or OmegaConf object.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict if needed
    if HAS_OMEGACONF and isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    elif hasattr(config, '__dataclass_fields__'):
        config = asdict(config)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved config to {path}")


def _deep_update(base: dict, updates: dict) -> dict:
    """
    Recursively update a dict with another dict.
    
    Args:
        base: Base dictionary.
        updates: Updates to apply.
    
    Returns:
        Updated dictionary.
    """
    result = base.copy()
    
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    
    return result


def config_to_dict(config: Any) -> dict:
    """
    Convert config object to dictionary.
    
    Args:
        config: Config object (dict, OmegaConf, dataclass, etc.)
    
    Returns:
        Dictionary representation.
    """
    if isinstance(config, dict):
        return config
    elif HAS_OMEGACONF and isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    elif hasattr(config, '__dataclass_fields__'):
        return asdict(config)
    else:
        return dict(config)


def merge_configs(*configs: dict) -> dict:
    """
    Merge multiple config dicts, later ones override earlier.
    
    Args:
        *configs: Config dictionaries to merge.
    
    Returns:
        Merged configuration.
    """
    result = {}
    for config in configs:
        if config:
            result = _deep_update(result, config)
    return result


def get_config_value(config: dict, key: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.
    
    Args:
        config: Configuration dictionary.
        key: Key path (e.g., 'model.hidden_dim').
        default: Default value if key not found.
    
    Returns:
        Config value or default.
    
    Example:
        >>> config = {'model': {'hidden_dim': 1024}}
        >>> get_config_value(config, 'model.hidden_dim')  # Returns 1024
        >>> get_config_value(config, 'model.layers', 12)  # Returns 12
    """
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def set_config_value(config: dict, key: str, value: Any) -> dict:
    """
    Set a nested config value using dot notation.
    
    Args:
        config: Configuration dictionary (modified in place).
        key: Key path (e.g., 'model.hidden_dim').
        value: Value to set.
    
    Returns:
        Modified config.
    """
    keys = key.split('.')
    current = config
    
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    
    current[keys[-1]] = value
    return config


