# pyright: reportMissingImports=false, reportMissingModuleSource=false
import numpy as np
import ufl

EPSILON_1D     = 0.01
EPSILON_SPHERE = 0.1
R_SPHERE       = 0.5


def get_problem(name: str) -> tuple:
    """Return (f_factory, g, u_exact) for the named problem.

    f_factory(msh) -> UFL expression for the RHS
    g(x)           -> numpy array (Dirichlet BC / exact solution at boundary)
    u_exact(x)     -> numpy array (exact solution for error metrics)
    """
    if name == "1d":
        return _problem_1d()
    if name == "sphere":
        return _problem_sphere()
    raise ValueError(f"Unknown problem '{name}'. Available: '1d', 'sphere'.")


def _problem_1d(epsilon: float = EPSILON_1D):
    def f_factory(msh):
        x = ufl.SpatialCoordinate(msh)
        t = ufl.tanh(x[2] / epsilon)
        return (2.0 / epsilon**2) * t * (1.0 - t**2)

    def g(x: np.ndarray) -> np.ndarray:
        return np.tanh(x[2] / epsilon)

    return f_factory, g, g


def _problem_sphere(R: float = R_SPHERE, epsilon: float = EPSILON_SPHERE):
    def f_factory(msh):
        x = ufl.SpatialCoordinate(msh)
        r = ufl.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + 1e-16)
        s = R - r
        in_layer = ufl.And(ufl.ge(s, -epsilon), ufl.le(s, epsilon))
        return ufl.conditional(
            in_layer,
            ufl.pi / (2.0 * epsilon**2) * ufl.sin(ufl.pi * s / epsilon)
            + (1.0 / (epsilon * r)) * (1.0 + ufl.cos(ufl.pi * s / epsilon)),
            ufl.as_ufl(0.0),
        )

    def g(x: np.ndarray) -> np.ndarray:
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        s = R - r
        return np.where(
            s >= epsilon, 1.0,
            np.where(
                s <= -epsilon, 0.0,
                (s + epsilon) / (2.0 * epsilon)
                + np.sin(np.pi * s / epsilon) / (2.0 * np.pi),
            ),
        )

    return f_factory, g, g
