"""
Full test suite for `grounding_dino_prelabelling.py`.

Covers:
- _get_image_files
- generate_gd_prelabelling (normal and edge cases)
- device='auto' fallback
- corrupted/unreadable images
"""

import pytest
import numpy as np
import cv2
import sys
import os
from pathlib import Path
from src.pipeline.prelabelling.grounding_dino_prelabelling import (
    generate_gd_prelabelling,
    _get_image_files
)
# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DummyModel:
    def predict_with_classes(self, *args, **kwargs):
        class Output:
            xyxy = [[10, 10, 40, 40]]
            confidence = [0.6]
            class_id = [0]
        return Output()


@pytest.fixture
def patch_model(monkeypatch):
    monkeypatch.setattr("groundingdino.util.inference.Model", lambda *a, **k: DummyModel())


@pytest.fixture
def tmp_dirs_with_images(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    dummy = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(raw / "img.jpg"), dummy)
    (raw / "bad.png").write_bytes(b"\x00\x11")
    return raw, out


def test_get_image_files_valid(tmp_dirs_with_images):
    raw, _ = tmp_dirs_with_images
    files = _get_image_files(raw)
    assert len(files) == 2
    assert all(f.suffix.lower() in {".jpg", ".jpeg", ".png"} for f in files)


def test_generate_gd_prelabelling_json_output(patch_model, tmp_dirs_with_images):
    raw, out = tmp_dirs_with_images
    config = {"torch_device": "cpu", "dino_box_threshold": 0.3, "dino_text_threshold": 0.25}
    generate_gd_prelabelling(raw, out, config, Path("fake.pt"), Path("fake.py"))
    assert any(out.glob("*.json"))


def test_device_auto_triggers_detect(monkeypatch, patch_model, tmp_dirs_with_images):
    raw, out = tmp_dirs_with_images
    monkeypatch.setattr("src.pipeline.prelabelling.grounding_dino_prelabelling.detect_device", lambda: "cpu")
    config = {"torch_device": "auto", "dino_box_threshold": 0.3, "dino_text_threshold": 0.25}
    generate_gd_prelabelling(raw, out, config, Path("fake.pt"), Path("fake.py"))
    assert any(out.glob("*.json"))


def test_skips_unreadable_image(patch_model, tmp_dirs_with_images):
    raw, out = tmp_dirs_with_images
    config = {"torch_device": "cpu", "dino_box_threshold": 0.3, "dino_text_threshold": 0.25}
    generate_gd_prelabelling(raw, out, config, Path("fake.pt"), Path("fake.py"))
    # only 1 valid image should succeed
    assert len(list(out.glob("*.json"))) == 1
