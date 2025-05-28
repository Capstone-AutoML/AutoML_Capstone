"""
Full test suite for `matching.py`.

Covers:
- compute_iou
- normalize_class
- match_predictions
- match_and_filter (including edge cases)
"""

import pytest
import json
from pathlib import Path
from src.pipeline.prelabelling.matching import (
    compute_iou,
    normalize_class,
    match_predictions,
    match_and_filter
)


def test_compute_iou_overlap():
    assert round(compute_iou([0, 0, 2, 2], [1, 1, 3, 3]), 2) == 0.14


def test_compute_iou_no_overlap():
    assert compute_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_normalize_class_strips_suffix():
    assert normalize_class("FireBSI") == "fire"
    assert normalize_class("person ") == "person"


def test_match_predictions_positive():
    yolo = [{"bbox": [0, 0, 2, 2], "class": "fire"}]
    dino = [{"bbox": [1, 1, 3, 3], "class": "fire"}]
    assert match_predictions(yolo, dino, iou_thresh=0.1) == [True]


def test_match_predictions_negative_iou():
    yolo = [{"bbox": [0, 0, 1, 1], "class": "fire"}]
    dino = [{"bbox": [5, 5, 6, 6], "class": "fire"}]
    assert match_predictions(yolo, dino, iou_thresh=0.1) == [False]


@pytest.fixture
def match_dirs(tmp_path):
    yolo = tmp_path / "yolo"; dino = tmp_path / "dino"
    labeled = tmp_path / "labeled"; pending = tmp_path / "pending"
    for d in [yolo, dino]: d.mkdir()
    return yolo, dino, labeled, pending


def make_pair(yolo_file, dino_file, yolo_data, dino_data):
    yolo_file.write_text(json.dumps(yolo_data))
    dino_file.write_text(json.dumps(dino_data))


def test_match_and_filter_matched_goes_labeled(match_dirs):
    yolo, dino, labeled, pending = match_dirs
    f = "test.json"
    make_pair(yolo / f, dino / f,
              {"predictions": [{"bbox": [0, 0, 2, 2], "confidence": 0.9, "class": "fire"}]},
              {"predictions": [{"bbox": [0, 0, 2, 2], "confidence": 0.9, "class": "fire"}]})
    match_and_filter(yolo, dino, labeled, pending, {
        "iou_threshold": 0.5, "low_conf_threshold": 0.3,
        "mid_conf_threshold": 0.6, "dino_false_negative_threshold": 0.5
    })
    assert (labeled / f).exists()


def test_match_and_filter_low_conf_goes_pending(match_dirs):
    yolo, dino, labeled, pending = match_dirs
    f = "low.json"
    make_pair(yolo / f, dino / f,
              {"predictions": [{"bbox": [0, 0, 1, 1], "confidence": 0.2, "class": "fire"}]},
              {"predictions": []})
    match_and_filter(yolo, dino, labeled, pending, {
        "iou_threshold": 0.5, "low_conf_threshold": 0.3,
        "mid_conf_threshold": 0.6, "dino_false_negative_threshold": 0.5
    })
    assert (pending / f).exists()


def test_dino_false_negative_goes_pending(match_dirs):
    yolo, dino, labeled, pending = match_dirs
    f = "fn.json"
    make_pair(yolo / f, dino / f,
              {"predictions": []},
              {"predictions": [{"bbox": [0, 0, 2, 2], "confidence": 0.8, "class": "fire"}]})
    match_and_filter(yolo, dino, labeled, pending, {
        "iou_threshold": 0.5, "low_conf_threshold": 0.3,
        "mid_conf_threshold": 0.6, "dino_false_negative_threshold": 0.5
    })
    assert (pending / f).exists()


def test_match_and_filter_invalid_yolo_json_skipped(match_dirs):
    yolo, dino, labeled, pending = match_dirs
    f = "bad.json"
    yolo_file = yolo / f
    dino_file = dino / f
    yolo_file.write_text("{ invalid json }")
    dino_file.write_text(json.dumps({"predictions": []}))
    match_and_filter(yolo, dino, labeled, pending, {
        "iou_threshold": 0.5, "low_conf_threshold": 0.3,
        "mid_conf_threshold": 0.6, "dino_false_negative_threshold": 0.5
    })
    assert not (labeled / f).exists()
    assert not (pending / f).exists()
