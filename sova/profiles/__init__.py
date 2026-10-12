from .fire import PROFILE as FIRE
from .flood import PROFILE as FLOOD
from .heat import PROFILE as HEAT

__all__ = ["FIRE", "FLOOD", "HEAT", "PROFILES"]

PROFILES = {p.key: p for p in (FIRE, FLOOD, HEAT)}
