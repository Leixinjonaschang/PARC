"""P3 integration: Normalizer shape checks report the shapes (CPU).

Their assert messages called Logger.print (which returns None) on strings formatted with {:d},
which raises TypeError for a torch.Size, or with str.shape (AttributeError), so a shape mismatch
surfaced as an unrelated error.
"""
import pytest
import torch

import parc.motion_tracker.learning.normalizer as normalizer


def test_init_mean_shape_mismatch():
    with pytest.raises(AssertionError, match=r"init mean shape mismatch, expecting torch.Size\(\[3\]\), "
                                             r"but got torch.Size\(\[4\]\)"):
        normalizer.Normalizer(torch.Size([3]), "cpu", init_mean=torch.zeros(4))


def test_init_std_shape_mismatch():
    with pytest.raises(AssertionError, match="init std shape mismatch"):
        normalizer.Normalizer(torch.Size([3]), "cpu", init_mean=torch.zeros(3), init_std=torch.ones(2))


def test_set_mean_std_shape_mismatch():
    norm = normalizer.Normalizer(torch.Size([3]), "cpu", init_mean=torch.zeros(3), init_std=torch.ones(3))
    with pytest.raises(AssertionError, match=r"expecting size torch.Size\(\[3\]\), but got torch.Size\(\[3\]\) "
                                             r"and torch.Size\(\[5\]\)"):
        norm.set_mean_std(torch.zeros(3), torch.ones(5))
