"""
Trinity Vector Implementation
Scaffold + operational code
"""
import math
import cmath
from typing import Dict, Tuple

class TrinityVector:
    def __init__(self, existence: float, goodness: float, truth: float):
        self.existence = max(0, min(1, existence))
        self.goodness = max(0, min(1, goodness))
        self.truth = max(0, min(1, truth))

    def to_dict(self) -> Dict[str, float]:
        return {"existence":self.existence, "goodness":self.goodness, "truth":self.truth}

    def to_tuple(self) -> Tuple[float,float,float]:
        return (self.existence, self.goodness, self.truth)

    def to_complex(self) -> complex:
        return complex(self.existence*self.truth, self.goodness)

    @classmethod
    def from_complex(cls, c: complex):
        e = min(1, abs(c.real))
        g = min(1, c.imag if isinstance(c.imag,float) else 1)
        t = min(1, abs(c.imag))
        return cls(e,g,t)

    def calculate_modal_status(self):
        coh = self.goodness / (self.existence*self.truth+1e-6)
        if self.truth>0.9 and coh>0.9:
            return ("necessary", coh)
        if self.truth>0.5:
            return ("actual", coh)
        if self.truth>0.1:
            return ("possible", coh)
        return ("impossible", coh)
