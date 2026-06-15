# pyright: reportMissingImports=false, reportMissingModuleSource=false
import numpy as np
import ufl

EPSILON_1D     = 0.1 
EPSILON_SPHERE = 0.01
R_SPHERE       = 0.5
EPSILON_PLAN   = 0.01
# Tokamak wall-adaptation problem parameters (distances in mm)
R_IN_WALL  = 200.0                          # inner cylindrical wall radius
R_OUT_WALL = 800.0                          # outer cylindrical wall radius
A_WALL     = 200.0                          # cosh amplitude / length scale
RC_WALL    = (R_IN_WALL + R_OUT_WALL) / 2  # radial midpoint = 500 mm
# TCV tokamak torus parameters (distances in mm)
R0_TOK      = 640.0   # major radius of the torus axis
Z0_TOK      = 400.0   # vertical centre of the cross-section
Rc_TOK      = 580  # poloidal R-coordinate of the shell centre (= R0_TOK → centred on axis)
Zc_TOK      = 400.0   # poloidal Z-coordinate of the shell centre (= Z0_TOK → centred on axis)
r_shell_TOK = 70    # radius of the spherical shell
EPSILON_TOK = 10.0    # transition-layer half-width


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
    if name == "plan":
        return _problem_plan()
    if name == "tok-sphere":
        return _problem_tok_sphere()
    if name == "tok-wall":
        return _problem_tok_wall()
    raise ValueError(f"Unknown problem '{name}'. Available: '1d', 'sphere', 'plan', 'tok-sphere', 'tok-wall'.")


def _problem_1d(epsilon: float = EPSILON_1D):
    def f_factory(msh):
        x = ufl.SpatialCoordinate(msh)
        t = ufl.tanh(x[0] / epsilon)
        return (2.0 / epsilon**2) * t * (1.0 - t**2)

    def g(x: np.ndarray) -> np.ndarray:
        return np.tanh(x[0] / epsilon)

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


def _problem_tok_sphere(
    R0: float = R0_TOK,
    Z0: float = Z0_TOK,
    Rc: float = Rc_TOK,
    Zc: float = Zc_TOK,
    r_shell: float = r_shell_TOK,
    epsilon: float = EPSILON_TOK,
):
    """Smoothed-Heaviside spherical shell inside a tokamak domain.

    The shell is a sphere of radius ``r_shell`` centred at the poloidal
    point ``(Rc, Zc)`` (in mm, cylindrical coordinates).  Setting
    ``Rc = R0, Zc = Z0`` centres the shell on the magnetic axis.

    Exact solution:  u = H_epsilon(s),  s = r_shell - d
    where d = sqrt((sqrt(x²+y²) - Rc)² + (z - Zc)²) is the poloidal
    distance from the shell centre, and H_epsilon is the cosine-smoothed
    Heaviside:

        H_ε(s) = 0                                          s ≤ -ε
               = (s+ε)/(2ε) + sin(πs/ε)/(2π)              |s| ≤ ε
               = 1                                          s ≥  ε

    RHS  f = -Δu  via the chain rule:
        Δu = H_ε''(s)|∇s|² + H_ε'(s) Δs,   |∇s|=1,  Δs = -Δd
        Δd = 1/d + (R_xy - Rc) / (R_xy · d)   [cylindrical Laplacian of d]
    """
    def f_factory(msh):
        x   = ufl.SpatialCoordinate(msh)
        Rxy = ufl.sqrt(x[0]**2 + x[1]**2 + 1e-16)
        d   = ufl.sqrt((Rxy - Rc)**2 + (x[2] - Zc)**2 + 1e-16)
        s   = r_shell - d

        in_layer = ufl.And(ufl.ge(s, -epsilon), ufl.le(s, epsilon))

        lap_d = 1.0 / d + (Rxy - Rc) / (Rxy * d)

        return ufl.conditional(
            in_layer,
            ufl.pi / (2.0 * epsilon**2) * ufl.sin(ufl.pi * s / epsilon)
            + (1.0 / (2.0 * epsilon)) * (1.0 + ufl.cos(ufl.pi * s / epsilon)) * lap_d,
            ufl.as_ufl(0.0),
        )

    def g(x: np.ndarray) -> np.ndarray:
        Rxy = np.sqrt(x[0]**2 + x[1]**2)
        d   = np.sqrt((Rxy - Rc)**2 + (x[2] - Zc)**2)
        s   = r_shell - d
        return np.where(
            s >=  epsilon, 1.0,
            np.where(
                s <= -epsilon, 0.0,
                (s + epsilon) / (2.0 * epsilon)
                + np.sin(np.pi * s / epsilon) / (2.0 * np.pi),
            ),
        )

    return f_factory, g, g

def _problem_tok_wall(
    Rc: float = RC_WALL,
    a:  float = A_WALL,
):
    """Axisymmetric cosh wall-adaptation problem inside a tokamak domain.

    Exact solution (cylindrical coordinates, no Z or phi dependence):
        u(x,y,z) = cosh(xi^4) / a,     xi = (R_xy - Rc) / a
    where R_xy = sqrt(x^2 + y^2) is the cylindrical radius.

    The solution is minimal (= 1/a) on the mid-surface R_xy = Rc and grows
    symmetrically toward both walls, making it a direct test of anisotropic
    refinement near the inner (R=200 mm) and outer (R=800 mm) cylindrical walls.

    RHS  f = -Delta u  in 3D, derived analytically via the cylindrical Laplacian
    for an axisymmetric function (no Z, no phi dependence):

        Delta u = d2u/dR2 + (1/R) * du/dR

        du/dR   =  4 * xi^3 * sinh(xi^4) / a^2
        d2u/dR2 = (12 * xi^2 * sinh(xi^4) + 16 * xi^6 * cosh(xi^4)) / a^3

        f = -d2u/dR2 - (1/R_xy) * du/dR

    No UFL conditional is needed: cosh/sinh(xi^4) are smooth everywhere and
    FFCX compiles the expression without quadrature explosion.
    """
    def f_factory(msh):
        x   = ufl.SpatialCoordinate(msh)
        Rxy = ufl.sqrt(x[0]**2 + x[1]**2 + 1e-16)  # regularised at R=0
        xi  = (Rxy - Rc) / a
        xi4 = xi**4

        d2u_dR2    = (12.0 * xi**2 * ufl.sinh(xi4)
                      + 16.0 * xi**6 * ufl.cosh(xi4)) / a**3
        inv_R_du_dR = 4.0 * xi**3 * ufl.sinh(xi4) / (a**2 * Rxy)

        return -(d2u_dR2 + inv_R_du_dR)

    def g(x: np.ndarray) -> np.ndarray:
        Rxy = np.sqrt(x[0]**2 + x[1]**2)
        xi  = (Rxy - Rc) / a
        return np.cosh(xi**4) / a

    return f_factory, g, g
def get_grad_exact(name: str):
    """Return exact gradient callable for the named problem.

    Signature: ``grad_u(x)`` where ``x`` has shape ``(3, n_dofs)``
    (FEniCSx vector-interpolation convention) and the return has
    shape ``(3, n_dofs)``.  Suitable for ``fem.Function.interpolate``
    on a ``("Lagrange", degree, (3,))`` vector space.
    """
    if name == "1d":
        return _grad_1d()
    if name == "sphere":
        return _grad_sphere()
    if name == "plan":
        return _grad_plan()
    if name == "tok-sphere":
        return _grad_tok_sphere()
    if name == "tok-wall":
        return _grad_tok_wall()
    raise ValueError(f"Unknown problem '{name}'.")


def _grad_1d(epsilon: float = EPSILON_1D):
    def grad_u(x):  # x: (3, n) → (3, n)
        t = np.tanh(x[0] / epsilon)
        out = np.zeros_like(x)
        out[0] = (1.0 - t**2) / epsilon
        return out
    return grad_u


def _grad_sphere(R: float = R_SPHERE, epsilon: float = EPSILON_SPHERE):
    def grad_u(x):  # x: (3, n) → (3, n)
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        s = R - r
        h_prime = np.where(
            np.abs(s) < epsilon,
            (1.0 + np.cos(np.pi * s / epsilon)) / (2.0 * epsilon),
            0.0,
        )
        r_safe = np.where(r < 1e-14, 1.0, r)
        out = np.zeros_like(x)
        out[0] = h_prime * (-x[0] / r_safe)
        out[1] = h_prime * (-x[1] / r_safe)
        out[2] = h_prime * (-x[2] / r_safe)
        return out
    return grad_u


def _grad_tok_sphere(
    Rc: float = Rc_TOK,
    Zc: float = Zc_TOK,
    r_shell: float = r_shell_TOK,
    epsilon: float = EPSILON_TOK,
):
    def grad_u(x):  # x: (3, n) → (3, n)
        Rxy = np.sqrt(x[0]**2 + x[1]**2)
        d = np.sqrt((Rxy - Rc)**2 + (x[2] - Zc)**2)
        s = r_shell - d
        h_prime = np.where(
            np.abs(s) < epsilon,
            (1.0 + np.cos(np.pi * s / epsilon)) / (2.0 * epsilon),
            0.0,
        )
        d_safe = np.where(d < 1e-14, 1.0, d)
        Rxy_safe = np.where(Rxy < 1e-14, 1.0, Rxy)
        out = np.zeros_like(x)
        out[0] = h_prime * (-(Rxy - Rc) / d_safe * x[0] / Rxy_safe)
        out[1] = h_prime * (-(Rxy - Rc) / d_safe * x[1] / Rxy_safe)
        out[2] = h_prime * (-(x[2] - Zc) / d_safe)
        return out
    return grad_u


def _grad_tok_wall(Rc: float = RC_WALL, a: float = A_WALL):
    def grad_u(x):  # x: (3, n) → (3, n)
        Rxy = np.sqrt(x[0]**2 + x[1]**2)
        xi = (Rxy - Rc) / a
        du_dR = 4.0 * xi**3 * np.sinh(xi**4) / a**2
        Rxy_safe = np.where(Rxy < 1e-14, 1.0, Rxy)
        out = np.zeros_like(x)
        out[0] = du_dR * x[0] / Rxy_safe
        out[1] = du_dR * x[1] / Rxy_safe
        return out
    return grad_u


def _grad_plan(epsilon: float = EPSILON_PLAN):
    sqrt2 = np.sqrt(2.0)
    def grad_u(x):  # x: (3, n) → (3, n)
        d = (x[0] + x[1]) / (epsilon * sqrt2)
        t = np.tanh(d)
        coeff = (1.0 - t**2) / (epsilon * sqrt2)
        out = np.zeros_like(x)
        out[0] = coeff
        out[1] = coeff
        return out
    return grad_u


def _problem_plan(epsilon: float = EPSILON_PLAN):
    """Tanh boundary layer on the diagonal plane x + y = 0 inside [-1,1]^3.

    Exact solution:  u(x,y,z) = tanh((x + y) / (epsilon * sqrt(2)))
    The gradient is directed along (1,1,0)/sqrt(2), so the anisotropy
    direction is diagonal in the x-y plane and uniform in z.

    Derivation:
      d = (x + y) / (epsilon * sqrt(2))
      Delta u = -2 tanh(d)(1 - tanh^2(d)) / epsilon^2
      f = -Delta u = (2/epsilon^2) * tanh(d) * (1 - tanh^2(d))
    """
    sqrt2 = np.sqrt(2.0)

    def f_factory(msh):
        x = ufl.SpatialCoordinate(msh)
        d = (x[0] + x[1]) / (epsilon * ufl.sqrt(2.0))
        t = ufl.tanh(d)
        return (2.0 / epsilon**2) * t * (1.0 - t**2)

    def g(x: np.ndarray) -> np.ndarray:
        d = (x[0] + x[1]) / (epsilon * sqrt2)
        return np.tanh(d)

    return f_factory, g, g
