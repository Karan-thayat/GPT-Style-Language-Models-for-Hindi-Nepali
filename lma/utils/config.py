"""Loading pipeline configuration with profile overrides.

Shared by every stage script so that a profile means exactly the same thing to
the corpus builder, the tokenizer trainer, and the token counter. Keeping one
implementation matters: if `build_corpus.py` applied a profile's split ratios but
`train_tokenizer.py` did not, the two stages would silently disagree about which
documents are held out.
"""

from __future__ import annotations

from pathlib import Path

import yaml

#: Top-level sections a profile may override key-by-key.
OVERRIDABLE_SECTIONS = ("splits", "cleaning", "tokenizer")


def load_pipeline_config(path: Path, profile: str) -> dict:
    """Load a language's pipeline YAML and merge in one profile's overrides.

    Profile keys that name a top-level section (see :data:`OVERRIDABLE_SECTIONS`)
    are merged into that section, so a profile states only what differs. Remaining
    profile keys become ``config["profile"]``.

    Args:
        path: Path to a `pipeline.yaml`.
        profile: Profile name, e.g. ``"sampled"`` or ``"full"``.

    Returns:
        The resolved configuration, with a ``"profile"`` entry recording the
        profile name and its non-section settings.

    Raises:
        SystemExit: If the named profile does not exist, listing what does.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    profiles = config.pop("profiles", {})
    if profile not in profiles:
        raise SystemExit(f"Profile {profile!r} not found in {path}; available: {sorted(profiles)}")

    overrides = dict(profiles[profile])
    for section in OVERRIDABLE_SECTIONS:
        section_override = overrides.pop(section, None)
        if section_override:
            config[section] = {**config.get(section, {}), **section_override}

    config["profile"] = {"name": profile, **overrides}
    return config
