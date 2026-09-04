"""
JAX Automatic Differentiation Optimizer for E(r) profiles.

Parameterizes E(r) and optimizes for maximum inter-branch connectivity
subject to physical constraints.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
from typing import Callable, Tuple, Optional
from config import JAX_LEARNING_RATE, JAX_MAX_ITER, JAX_TOL


def parameterize_E(r_grid: jnp.ndarray, params: jnp.ndarray, 
                   basis: str = 'spline') -> jnp.ndarray:
    """
    Reconstruct E(r) from optimization parameters.
    
    Parameters
    ----------
    r_grid : jnp.ndarray, shape (nr,)
        Radial coordinates.
    params : jnp.ndarray
        Optimizable parameters.
    basis : str
        'spline' -> cubic spline coefficients
        'fourier' -> Fourier series
        'nn' -> simple neural network
    
    Returns
    -------
    jnp.ndarray
        E(r) evaluated on r_grid.
    """
    if basis == 'spline':
        # Simple linear interpolation for now; upgrade to cubic later
        # params = values at knot points
        return jnp.interp(r_grid, jnp.linspace(r_grid.min(), r_grid.max(), len(params)), params)
    elif basis == 'fourier':
        # E(r) = Σ a_n sin(n π r / r_max)
        n = jnp.arange(1, len(params) + 1)
        modes = jnp.sin(n[:, None] * jnp.pi * r_grid[None, :] / r_grid.max())
        return jnp.dot(params, modes)
    else:
        raise ValueError(f"Unknown basis: {basis}")


def physical_constraints(E_profile: jnp.ndarray, 
                         M_profile: jnp.ndarray) -> jnp.ndarray:
    constraints = jnp.array([
        jnp.sum(jnp.maximum(0.0, -M_profile)),                    # M > 0
        jnp.sum(jnp.maximum(0.0, jnp.abs(E_profile) - 1.0)),      # |E| < 1
        jnp.sum(jnp.asarray(jnp.gradient(E_profile)) ** 2)        # Smoothness  ← fixed
    ])
    return constraints


def make_objective(M_func: Callable, r_grid: jnp.ndarray, 
                   t_grid: jnp.ndarray, 
                   target_connectivity: float = 0.5):
    def objective(params: jnp.ndarray) -> jnp.ndarray:   # ← was float
        E_profile = parameterize_E(r_grid, params, basis='spline')
        M_profile = jax.vmap(M_func)(r_grid)
        
        connectivity = jnp.mean(jnp.abs(E_profile))
        constraints = physical_constraints(E_profile, M_profile)
        
        loss = -connectivity + 10.0 * jnp.sum(constraints)
        return loss
    
    return objective


def optimize_E_profile(M_func: Callable, 
                       r_grid: jnp.ndarray,
                       t_grid: jnp.ndarray,
                       n_params: int = 16,
                       n_iter: int = JAX_MAX_ITER,
                       lr: float = JAX_LEARNING_RATE) -> Tuple[jnp.ndarray, list]:
    """
    Optimize the E(r) profile using JAX + Optax.
    
    Parameters
    ----------
    M_func : callable
        Mass function M(r).
    r_grid, t_grid : jnp.ndarray
        Spacetime grids.
    n_params : int
        Number of parameters for E(r) representation.
    n_iter : int
        Maximum optimization iterations.
    lr : float
        Learning rate.
    
    Returns
    -------
    params_opt : jnp.ndarray
        Optimized parameters.
    history : list
        Loss values during optimization.
    """
    # Initialize parameters (small random values near zero)
    key = jax.random.PRNGKey(42)
    params = jax.random.normal(key, (n_params,)) * 0.1
    
    objective = make_objective(M_func, r_grid, t_grid)
    grad_fn = jit(value_and_grad(objective))
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    
    history = []
    for step in range(n_iter):
        loss, grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        history.append(float(loss))
        
        if step % 100 == 0:
            print(f"Step {step:5d} | Loss = {loss:.6f}")
        
        if step > 10 and abs(history[-1] - history[-10]) < JAX_TOL:
            print(f"Converged at step {step}")
            break
    
    return jnp.asarray(params), history


if __name__ == "__main__":
    print("[optimize_jax.py] JAX optimizer module loaded.")
    print("  JAX devices:", jax.devices())