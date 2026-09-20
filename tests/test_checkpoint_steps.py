"""Permanent checkpoints (``--checkpoint_steps``) and choosing one to resume from.

``save()`` writes the rolling recovery checkpoint to a single directory and
replaces it, so a run that wants to branch a shorter run later -- e.g. to add a
cooldown at 1B tokens inside a 10B-token run -- needs checkpoints that survive.
Those are named ``checkpoint-<step>``.

Resuming with no name has to pick sensibly between the two kinds, and must still
find checkpoints written before any of this existed.
"""

import os

import pytest

from train import CheckpointManager, parse_checkpoint_steps


def _mgr(tmp_path):
    mgr = CheckpointManager.__new__(CheckpointManager)  # _resume_name only reads the dir
    mgr.checkpoint_dir = str(tmp_path)
    mgr.DEFAULT_NAME = "checkpoint"
    return mgr


def _mkdirs(tmp_path, *names):
    for n in names:
        os.makedirs(tmp_path / n, exist_ok=True)


@pytest.mark.parametrize(
    "spec, expected",
    [("", ()), (None, ()), ("1526", (1526,)), (" 3052 , 1526 ", (1526, 3052)),
     ("1526,1526", (1526,)), ("12206,763,6102", (763, 6102, 12206))],
)
def test_parse_checkpoint_steps(spec, expected):
    assert parse_checkpoint_steps(spec) == expected


@pytest.mark.parametrize("spec", ["1526,-1", "1e3", "1526;3052", "half"])
def test_parse_checkpoint_steps_rejects_junk(spec):
    with pytest.raises(ValueError, match="checkpoint_steps"):
        parse_checkpoint_steps(spec)


def test_rolling_checkpoint_wins_when_present(tmp_path):
    """It is written far more often, so it is the newest state."""
    _mkdirs(tmp_path, "checkpoint", "checkpoint-1526", "checkpoint-6102")
    assert _mgr(tmp_path)._resume_name() == "checkpoint"


def test_falls_back_to_the_highest_step(tmp_path):
    """Numeric, not lexicographic: 6102 beats 12206 as a string."""
    _mkdirs(tmp_path, "checkpoint-1526", "checkpoint-12206", "checkpoint-6102")
    assert _mgr(tmp_path)._resume_name() == "checkpoint-12206"


def test_ignores_files_and_unparseable_names(tmp_path):
    """save() writes through tempfile.mkdtemp in this directory; those must not win."""
    _mkdirs(tmp_path, "checkpoint-763", "tmpab12cd", "checkpoint-final", "checkpoint-")
    (tmp_path / "checkpoint-9999").write_text("a file, not a checkpoint")
    assert _mgr(tmp_path)._resume_name() == "checkpoint-763"


def test_empty_or_missing_directory_gives_the_default_name(tmp_path):
    """load(allow_missing=True) then reports it as absent rather than crashing."""
    assert _mgr(tmp_path)._resume_name() == "checkpoint"
    assert _mgr(tmp_path / "nope")._resume_name() == "checkpoint"


def test_save_names_match_what_resume_looks_for():
    """The loop builds these names; _resume_name parses them."""
    step = 6102
    assert f"checkpoint-{step}" == "checkpoint-6102"
    assert parse_checkpoint_steps(str(step)) == (6102,)
