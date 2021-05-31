from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import NewType, Protocol, Callable, Tuple

import numpy as np
import numpy.typing as npt


@dataclass
class JuliaCalculator:
    p: np.poly1d
    n: int

    x_bounds: Tuple[float, float]
    y_bounds: Tuple[float, float]

    shape: Tuple[int, int]

    @property
    def x_width(self) -> float:
        return self.x_bounds[1] - self.x_bounds[0]
    
    @property
    def y_width(self) -> float:
        return self.y_bounds[1] - self.y_bounds[0]

    def escape_radius(self) -> float:
        return julia_poly_radius(self.p)

    def z_from_shape_coord(self, i: complex, j: complex) -> complex:
        zx = self.x_bounds[0] + (self.x_width * np.real(j) / (self.shape[1] - 1))
        zy = self.y_bounds[0] + (self.y_width * (1 - (np.real(i) / (self.shape[0] - 1))))
        return zx + 1j * zy

    def calc(self) -> np.ndarray: 
        myR: float = self.escape_radius()
        z: np.ndarray = np.fromfunction(np.vectorize(self.z_from_shape_coord), self.shape, dtype=complex)

        # TODO figure out mask
        # vectorize


        return z    



# radius formula
# https://math.stackexchange.com/questions/3839683/the-escape-radius-of-a-polynomial-and-its-filled-julia-set/3839776#3839776
def julia_poly_radius(p: np.poly1d) -> float:
    if not p.coeffs:
        return np.Infinity
    return (1 + sum(abs(c) for c in p.coeffs[1:])) / abs(p.coeffs[0])


jc = JuliaCalculator(np.poly1d([1]), 1, (-2, 2), (-3, 3), (5, 5))
print(jc.x_width)
print(jc.calc())



