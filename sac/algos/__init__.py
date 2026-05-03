from .sac import SAC
from .skill_discovery_sac import SkillDiscoverySAC


def DIAYN(*args, **kwargs):
    """DIAYN-style objective on the shared skill-discovery SAC implementation."""

    kwargs.setdefault('skill_objective', 'diayn')
    return SkillDiscoverySAC(*args, **kwargs)


def DADS(*args, **kwargs):
    """DADS-style objective on the shared skill-discovery SAC implementation."""

    kwargs.setdefault('skill_objective', 'dads')
    return SkillDiscoverySAC(*args, **kwargs)


def LSD(*args, **kwargs):
    """LSD-style displacement objective on the shared skill-discovery SAC implementation."""

    kwargs.setdefault('skill_objective', 'lsd_delta')
    return SkillDiscoverySAC(*args, **kwargs)


__all__ = ['SAC', 'SkillDiscoverySAC', 'DIAYN', 'DADS', 'LSD']
