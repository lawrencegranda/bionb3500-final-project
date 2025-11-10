"""Common helper functions for scripts."""

import argparse
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

import yaml


@dataclass
class PathsConfig:
    """Configuration for path-related settings."""

    metrics_dir: Path
    plots_dir: Path
    dataset_path: Path
    target_words_path: Path
    sense_map_path: Path
    corpora_paths: List[Path]

    @staticmethod
    def from_dict(data: dict) -> "PathsConfig":
        """Create PathsConfig from a dict (YAML section)."""
        corpora_paths = [Path(path) for path in (data.get("corpora_paths") or [])]
        return PathsConfig(
            metrics_dir=Path(data["metrics_dir"]),
            plots_dir=Path(data["plots_dir"]),
            dataset_path=Path(data["dataset_path"]),
            target_words_path=Path(data["target_words_path"]),
            sense_map_path=Path(data["sense_map_path"]),
            corpora_paths=corpora_paths,
        )


@dataclass
class CorporaConfig:
    """Configuration for corpora and dataset settings."""

    wn_key_type: str
    max_sentences: int

    @staticmethod
    def from_dict(data: dict) -> "CorporaConfig":
        """Create CorporaConfig from a dict (YAML section)."""
        wn_key_type = str(data["wn_key_type"])
        assert wn_key_type in [
            "wn16_key",
            "wn21_key",
            "wn30_key",
        ], "Invalid WordNet key type"
        max_sentences = int(data["max_sentences"])
        assert max_sentences > 0, "Max sentences must be positive"
        return CorporaConfig(wn_key_type=wn_key_type, max_sentences=max_sentences)


@dataclass
class ModelConfig:
    """Configuration for model-related settings."""

    model_names: List[str]
    random_state: int
    clustering_layers: Dict[str, List[int]]

    @staticmethod
    def from_dict(data: dict) -> "ModelConfig":
        """Create ModelConfig from a dict (YAML section)."""
        model_names = list(data["model_names"])
        assert len(model_names) > 0, "Model names must be non-empty"
        clustering_layers = {k: list(v) for k, v in data["clustering_layers"].items()}
        assert len(clustering_layers) > 0, "Clustering layers must be non-empty"

        return ModelConfig(
            model_names=model_names,
            random_state=int(data["random_state"]),
            clustering_layers=clustering_layers,
        )


@dataclass
class ConfigType:
    """Top-level configuration for the script, containing subconfigs."""

    paths: PathsConfig
    corpora: CorporaConfig
    model: ModelConfig

    @staticmethod
    def from_yaml(config_path: Path) -> "ConfigType":
        """Load the configuration from a YAML file and split into subconfigs."""
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        # The YAML can be flat, so just pass the full dict to each "from_dict"
        paths_config = PathsConfig.from_dict(data)
        corpora_config = CorporaConfig.from_dict(data)
        model_config = ModelConfig.from_dict(data)

        return ConfigType(
            paths=paths_config,
            corpora=corpora_config,
            model=model_config,
        )


@dataclass
class Args:
    """Arguments for the script."""

    config: ConfigType
    model: Optional[str] = None


def add_data_config_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add the standard data-config-path argument to an argument parser.
    """
    parser.add_argument(
        "-d",
        "--data-config-path",
        type=Path,
        required=True,
        help="Path to the YAML data configuration file.",
    )


def add_model_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add the standard model argument to an argument parser.
    """
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model identifier",
    )


def load_config_from_parser_args(args: argparse.Namespace) -> ConfigType:
    """
    Given argparse args with a data_config_path, return a ConfigType object with all configuration.
    """
    return ConfigType.from_yaml(args.data_config_path)


def get_args(description: str, get_model_arg: bool = True) -> Args:
    """
    Create an Args object with the standard data-config-path argument.
    """
    parser = argparse.ArgumentParser(description=description)
    add_data_config_argument(parser)
    if get_model_arg:
        add_model_argument(parser)

    args = parser.parse_args()
    config = ConfigType.from_yaml(args.data_config_path)
    model = args.model if get_model_arg else None
    return Args(config, model)


__all__ = [
    "Args",
    "ConfigType",
    "PathsConfig",
    "CorporaConfig",
    "ModelConfig",
    "get_args",
]
