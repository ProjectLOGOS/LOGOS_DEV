"""
Trinity Vector Processing - UIP Step 1 Component
================================================

Advanced Trinity vector processing with fractal orbital analysis.
Consolidates Trinity vector mathematics with complex orbital predictions.

Adapted from:
- V2_Possible_Gap_Fillers/fractal orbital predictor/trinity_vector.py
- V2_Possible_Gap_Fillers/fractal orbital predictor/class_fractal_orbital_predictor.py
"""

import cmath
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from protocols.shared.system_imports import *


@dataclass
class OrbitalProperties:
    """Properties of Trinity vector in fractal orbital space"""

    depth: int
    in_mandelbrot_set: bool
    orbital_type: str  # "convergent", "periodic", "chaotic", "divergent"
    escape_velocity: Optional[float]
    period: Optional[int]
    attracting_cycle: Optional[List[complex]]
    fractal_dimension: float


@dataclass
class TrinityVectorAnalysis:
    """Comprehensive Trinity vector analysis results"""

    trinity_values: Tuple[float, float, float]  # (E, G, T)
    complex_representation: complex
    modal_status: Tuple[str, float]  # (status, coherence)
    orbital_properties: OrbitalProperties
    geometric_properties: Dict[str, float]
    theological_interpretation: Dict[str, Any]
    convergence_metrics: Dict[str, float]


class TrinityVector:
    """Enhanced Trinity vector with fractal orbital analysis"""

    def __init__(self, existence: float, goodness: float, truth: float):
        """Initialize Trinity vector with normalization"""
        self.existence = max(0.0, min(1.0, existence))
        self.goodness = max(0.0, min(1.0, goodness))
        self.truth = max(0.0, min(1.0, truth))

        # Derived properties
        self._complex_cache = None
        self._orbital_cache = None
        self._modal_cache = None

    @property
    def e(self) -> float:
        """Existence dimension accessor"""
        return self.existence

    @property
    def g(self) -> float:
        """Goodness dimension accessor"""
        return self.goodness

    @property
    def t(self) -> float:
        """Truth dimension accessor"""
        return self.truth

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation"""
        return {
            "existence": self.existence,
            "goodness": self.goodness,
            "truth": self.truth,
        }

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple representation"""
        return (self.existence, self.goodness, self.truth)

    def to_complex(self) -> complex:
        """Convert to complex number representation"""
        if self._complex_cache is None:
            # Use E*T for real part, G for imaginary part
            self._complex_cache = complex(self.existence * self.truth, self.goodness)
        return self._complex_cache

    @classmethod
    def from_complex(cls, c: complex) -> "TrinityVector":
        """Create Trinity vector from complex number"""
        # Reconstruct E, G, T from complex representation
        magnitude = abs(c)

        # Extract components with proper normalization
        e_component = min(1.0, abs(c.real) / max(1e-6, magnitude))
        g_component = min(1.0, abs(c.imag))

        # Reconstruct T from real part and E
        if e_component > 1e-6:
            t_component = min(1.0, abs(c.real) / e_component)
        else:
            t_component = min(1.0, abs(c.real))

        return cls(e_component, g_component, t_component)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "TrinityVector":
        """Create Trinity vector from dictionary"""
        return cls(
            data.get("existence", 0.0),
            data.get("goodness", 0.0),
            data.get("truth", 0.0),
        )

    def calculate_modal_status(self) -> Tuple[str, float]:
        """Calculate modal logic status with coherence measure"""
        if self._modal_cache is not None:
            return self._modal_cache

        # Calculate coherence based on dimensional harmony
        coherence = self._calculate_coherence()

        # Determine modal status based on Truth and coherence
        if self.truth > 0.9 and coherence > 0.8:
            status = "necessary"
        elif self.truth > 0.7 and coherence > 0.6:
            status = "actual"
        elif self.truth > 0.3:
            status = "possible"
        elif self.truth > 0.1:
            status = "contingent"
        else:
            status = "impossible"

        self._modal_cache = (status, coherence)
        return self._modal_cache

    def _calculate_coherence(self) -> float:
        """Calculate dimensional coherence measure"""
        # Coherence based on balance and mutual reinforcement
        balance = 1.0 - np.std([self.existence, self.goodness, self.truth])

        # Mutual reinforcement (geometric mean vs arithmetic mean ratio)
        geometric_mean = (self.existence * self.goodness * self.truth) ** (1 / 3)
        arithmetic_mean = (self.existence + self.goodness + self.truth) / 3

        if arithmetic_mean > 1e-6:
            reinforcement = geometric_mean / arithmetic_mean
        else:
            reinforcement = 0.0

        # Combined coherence measure
        coherence = (balance + reinforcement) / 2.0
        return max(0.0, min(1.0, coherence))

    def calculate_geometric_properties(self) -> Dict[str, float]:
        """Calculate geometric properties in Trinity space"""
        return {
            "magnitude": math.sqrt(
                self.existence**2 + self.goodness**2 + self.truth**2
            ),
            "manhattan_distance": self.existence + self.goodness + self.truth,
            "max_distance": max(self.existence, self.goodness, self.truth),
            "min_distance": min(self.existence, self.goodness, self.truth),
            "volume": self.existence * self.goodness * self.truth,
            "surface_area": 2
            * (
                self.existence * self.goodness
                + self.goodness * self.truth
                + self.truth * self.existence
            ),
            "centroid_distance": abs(
                1 / 3 - (self.existence + self.goodness + self.truth) / 3
            ),
        }


class FractalOrbitalAnalyzer:
    """Fractal orbital analysis for Trinity vectors in complex plane"""

    def __init__(self, max_iterations: int = 100, escape_radius: float = 2.0):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius
        self.logger = logging.getLogger(__name__)

    def analyze_orbital_properties(
        self, trinity_vector: TrinityVector
    ) -> OrbitalProperties:
        """Analyze fractal orbital properties of Trinity vector"""
        c = trinity_vector.to_complex()

        # Mandelbrot iteration analysis
        z = complex(0, 0)
        trajectory = [z]

        for i in range(self.max_iterations):
            z = z * z + c
            trajectory.append(z)

            # Check for escape
            if abs(z) > self.escape_radius:
                return self._analyze_divergent_orbit(trajectory, i + 1)

        # Orbit didn't escape - analyze convergent/periodic behavior
        return self._analyze_bounded_orbit(trajectory, c)

    def _analyze_divergent_orbit(
        self, trajectory: List[complex], escape_iteration: int
    ) -> OrbitalProperties:
        """Analyze divergent orbital behavior"""

        # Calculate escape velocity
        if len(trajectory) >= 2:
            escape_velocity = abs(trajectory[-1] - trajectory[-2])
        else:
            escape_velocity = abs(trajectory[-1])

        # Estimate fractal dimension using escape time
        fractal_dim = 2.0 - (escape_iteration / self.max_iterations)

        return OrbitalProperties(
            depth=escape_iteration,
            in_mandelbrot_set=False,
            orbital_type="divergent",
            escape_velocity=escape_velocity,
            period=None,
            attracting_cycle=None,
            fractal_dimension=max(1.0, fractal_dim),
        )

    def _analyze_bounded_orbit(
        self, trajectory: List[complex], c: complex
    ) -> OrbitalProperties:
        """Analyze bounded orbital behavior for convergent/periodic orbits"""

        # Look for periodic behavior in the latter part of trajectory
        stable_trajectory = trajectory[-50:]  # Analyze last 50 points

        period = self._detect_period(stable_trajectory)

        if period:
            # Periodic orbit
            cycle_start = len(stable_trajectory) - period
            attracting_cycle = stable_trajectory[cycle_start : cycle_start + period]
            orbital_type = "periodic"
        else:
            # Check for convergent behavior
            if len(stable_trajectory) >= 10:
                recent_variation = np.std([abs(z) for z in stable_trajectory[-10:]])
                if recent_variation < 0.01:
                    orbital_type = "convergent"
                    attracting_cycle = [stable_trajectory[-1]]  # Fixed point
                else:
                    orbital_type = "chaotic"
                    attracting_cycle = None
            else:
                orbital_type = "unknown"
                attracting_cycle = None

        # Estimate fractal dimension for bounded orbits
        fractal_dim = self._estimate_bounded_fractal_dimension(stable_trajectory)

        return OrbitalProperties(
            depth=self.max_iterations,
            in_mandelbrot_set=True,
            orbital_type=orbital_type,
            escape_velocity=None,
            period=period,
            attracting_cycle=attracting_cycle,
            fractal_dimension=fractal_dim,
        )

    def _detect_period(
        self, trajectory: List[complex], tolerance: float = 1e-6
    ) -> Optional[int]:
        """Detect periodic behavior in trajectory"""
        n = len(trajectory)

        # Check for periods from 1 to n//3
        for period in range(1, min(n // 3, 20) + 1):
            is_periodic = True

            # Check if trajectory repeats with this period
            for i in range(period, min(n, 3 * period)):
                if abs(trajectory[i] - trajectory[i - period]) > tolerance:
                    is_periodic = False
                    break

            if is_periodic:
                return period

        return None

    def _estimate_bounded_fractal_dimension(self, trajectory: List[complex]) -> float:
        """Estimate fractal dimension for bounded orbits using box counting"""
        if len(trajectory) < 10:
            return 1.0

        # Simple box counting approximation
        # Convert trajectory to coordinate pairs
        coords = [(z.real, z.imag) for z in trajectory]

        # Calculate range
        x_coords, y_coords = zip(*coords)
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        if x_range < 1e-10 or y_range < 1e-10:
            return 1.0  # Essentially 1D

        # Rough fractal dimension estimate
        complexity = len(set([(round(x, 3), round(y, 3)) for x, y in coords]))
        normalized_complexity = complexity / len(coords)

        # Map to fractal dimension between 1 and 2
        fractal_dim = 1.0 + normalized_complexity
        return min(2.0, max(1.0, fractal_dim))


class TrinityVectorProcessor:
    """Main Trinity vector processing engine"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize analyzers
        max_iterations = self.config.get("fractal_max_iterations", 100)
        escape_radius = self.config.get("fractal_escape_radius", 2.0)
        self.orbital_analyzer = FractalOrbitalAnalyzer(max_iterations, escape_radius)

        self.logger.info("Trinity vector processor initialized")

    def process_trinity_vector(
        self, existence: float, goodness: float, truth: float
    ) -> TrinityVectorAnalysis:
        """
        Comprehensive Trinity vector processing and analysis

        Args:
            existence, goodness, truth: Trinity dimensions [0,1]

        Returns:
            TrinityVectorAnalysis: Complete analysis results
        """
        try:
            # Create Trinity vector
            trinity_vector = TrinityVector(existence, goodness, truth)

            # Calculate modal status
            modal_status = trinity_vector.calculate_modal_status()

            # Analyze orbital properties
            orbital_props = self.orbital_analyzer.analyze_orbital_properties(
                trinity_vector
            )

            # Calculate geometric properties
            geometric_props = trinity_vector.calculate_geometric_properties()

            # Generate theological interpretation
            theological_interp = self._generate_theological_interpretation(
                trinity_vector, modal_status, orbital_props
            )

            # Calculate convergence metrics
            convergence_metrics = self._calculate_convergence_metrics(
                trinity_vector, orbital_props
            )

            analysis = TrinityVectorAnalysis(
                trinity_values=(existence, goodness, truth),
                complex_representation=trinity_vector.to_complex(),
                modal_status=modal_status,
                orbital_properties=orbital_props,
                geometric_properties=geometric_props,
                theological_interpretation=theological_interp,
                convergence_metrics=convergence_metrics,
            )

            self.logger.debug(
                f"Trinity vector analysis completed: {modal_status[0]} status"
            )
            return analysis

        except Exception as e:
            self.logger.error(f"Trinity vector processing failed: {e}")
            raise

    def _generate_theological_interpretation(
        self,
        trinity_vector: TrinityVector,
        modal_status: Tuple[str, float],
        orbital_props: OrbitalProperties,
    ) -> Dict[str, Any]:
        """Generate theological interpretation of Trinity vector analysis"""

        interpretation = {
            "divine_attributes": {
                "existence_dominance": trinity_vector.existence > 0.7,
                "goodness_dominance": trinity_vector.goodness > 0.7,
                "truth_dominance": trinity_vector.truth > 0.7,
                "balanced_perfection": all(v > 0.6 for v in trinity_vector.to_tuple()),
            },
            "modal_theological_status": {
                "necessity_indication": modal_status[0] == "necessary",
                "coherence_level": modal_status[1],
                "divine_consistency": modal_status[1] > 0.8,
            },
            "fractal_theological_properties": {
                "self_similarity": orbital_props.orbital_type
                in ["convergent", "periodic"],
                "infinite_depth": orbital_props.in_mandelbrot_set,
                "harmonic_resonance": orbital_props.period is not None,
                "transcendent_complexity": orbital_props.fractal_dimension > 1.5,
            },
        }

        return interpretation

    def _calculate_convergence_metrics(
        self, trinity_vector: TrinityVector, orbital_props: OrbitalProperties
    ) -> Dict[str, float]:
        """Calculate convergence and stability metrics"""

        # Base stability from Trinity balance
        balance_stability = 1.0 - np.std(trinity_vector.to_tuple())

        # Orbital stability
        if orbital_props.orbital_type == "convergent":
            orbital_stability = 1.0
        elif orbital_props.orbital_type == "periodic":
            # More stable with shorter periods
            period_stability = 1.0 / (1.0 + (orbital_props.period or 10))
            orbital_stability = 0.8 + 0.2 * period_stability
        elif orbital_props.orbital_type == "chaotic":
            orbital_stability = 0.3
        else:  # divergent
            orbital_stability = 0.1

        # Complex plane stability
        c = trinity_vector.to_complex()
        complex_stability = max(0.0, 1.0 - abs(c) / 2.0)  # More stable closer to origin

        return {
            "balance_stability": balance_stability,
            "orbital_stability": orbital_stability,
            "complex_stability": complex_stability,
            "overall_convergence": (
                balance_stability + orbital_stability + complex_stability
            )
            / 3.0,
            "fractal_complexity": orbital_props.fractal_dimension,
        }


# Global processor instance
trinity_vector_processor = TrinityVectorProcessor()


__all__ = [
    "TrinityVector",
    "OrbitalProperties",
    "TrinityVectorAnalysis",
    "FractalOrbitalAnalyzer",
    "TrinityVectorProcessor",
    "trinity_vector_processor",
]
