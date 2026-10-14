"""
Register WaterOVS instance segmentation datasets for FC-CLIP.

The image and annotation files stay in the MARIS repository. Set
WATEROVS_DATASET_ROOT to override the default data root.
"""

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Mapping, Sequence

from detectron2.data.datasets.coco import register_coco_instances


_DEFAULT_MARIS_ROOT = Path("/data/gjf/code/MARIS")
_EXPECTED_CATEGORY_COUNTS = {
    "openvocab_waterovs_train": 84,
    "openvocab_waterovs_val": 115,
}


def _get_maris_root() -> Path:
    return Path(os.getenv("MARIS_REPO_ROOT", str(_DEFAULT_MARIS_ROOT))).expanduser()


def _get_waterovs_root() -> Path:
    default_root = _get_maris_root() / "datasets" / "WaterOVS"
    return Path(os.getenv("WATEROVS_DATASET_ROOT", str(default_root))).expanduser()


def _load_maris_openseg_classes() -> ModuleType:
    classes_path = _get_maris_root() / "maris" / "data" / "datasets" / "openseg_classes.py"
    if not classes_path.is_file():
        raise FileNotFoundError(
            f"WaterOVS class definitions not found at {classes_path}. "
            "Set MARIS_REPO_ROOT to the MARIS repository root."
        )

    spec = importlib.util.spec_from_file_location("maris_openseg_classes", classes_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load MARIS class definitions from {classes_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _format_category_id(category_id: Any) -> str:
    return f"{int(category_id):03d}"


def _validate_categories(
    dataset_name: str,
    categories: Sequence[Mapping[str, Any]],
) -> None:
    expected_count = _EXPECTED_CATEGORY_COUNTS[dataset_name]
    if len(categories) != expected_count:
        raise ValueError(
            f"{dataset_name} expected {expected_count} categories, got {len(categories)}"
        )

    category_ids = [_format_category_id(category["id"]) for category in categories]
    if len(set(category_ids)) != len(category_ids):
        raise ValueError(f"{dataset_name} contains duplicate category ids")


def _get_waterovs_instances_meta(
    dataset_name: str,
    categories: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    _validate_categories(dataset_name, categories)

    category_ids = [_format_category_id(category["id"]) for category in categories]
    thing_dataset_id_to_contiguous_id = {
        category_id: index for index, category_id in enumerate(category_ids)
    }

    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": [str(category["name"]) for category in categories],
        "thing_colors": [list(category["color"]) for category in categories],
    }


def register_all_waterovs_instance() -> None:
    waterovs_root = _get_waterovs_root()
    openseg_classes = _load_maris_openseg_classes()

    predefined_splits = {
        "openvocab_waterovs_train": (
            "train",
            "annotations/instances_train.json",
            openseg_classes.get_waterovs_categories_with_prompt_eng_train(),
        ),
        "openvocab_waterovs_val": (
            "val",
            "annotations/instances_val.json",
            openseg_classes.get_waterovs_categories_with_prompt_eng_val(),
        ),
    }

    for dataset_name, (image_dir, json_file, categories) in predefined_splits.items():
        register_coco_instances(
            dataset_name,
            _get_waterovs_instances_meta(dataset_name, categories),
            str(waterovs_root / json_file),
            str(waterovs_root / image_dir),
        )


register_all_waterovs_instance()
