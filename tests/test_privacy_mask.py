import numpy as np

from surgical_pipeline.privacy_blur import PrivacyMasker, feather_mask
from surgical_pipeline.schemas import Detection


def test_dilation_feather_and_blur_change_only_mask_region() -> None:
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[15:35, 15:35] = 255
    mask = np.zeros((50, 50), dtype=np.uint8); mask[20:30, 20:30] = 1
    alpha = feather_mask(mask, dilation_px=3, feather_px=3)
    assert alpha[18, 18] > 0
    output = PrivacyMasker(blur_kernel=11, dilation_px=3, feather_px=3).apply(frame, [Detection(0, "health_personel", 0.9, 1, mask, 0, 0.0)], 0)
    assert np.array_equal(output[0, 0], frame[0, 0])
    assert not np.array_equal(output[19, 19], frame[19, 19])


def test_tracker_mask_persists_only_for_configured_short_window() -> None:
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    frame[8:22, 8:22] = 220
    mask = np.zeros((30, 30), dtype=np.uint8); mask[10:20, 10:20] = 1
    masker = PrivacyMasker(blur_kernel=9, persistence_frames=2)
    masker.apply(frame, [Detection(0, "health_personel", 0.9, 4, mask, 0, 0)], 0)
    assert not np.array_equal(masker.apply(frame, [], 1), frame)
    assert np.array_equal(masker.apply(frame, [], 3), frame)
