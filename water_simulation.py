"""
Water Simulation Engine
=======================

This module implements a simple two‑dimensional water surface simulation using
finite difference methods. The simulation solves a discretised form of the 2D
wave equation with an optional damping term to model energy loss over time.

The implementation draws inspiration from publicly available discussions on
numerically solving the wave equation. In particular, the engine uses three
numpy arrays (`u_prev`, `u`, and `u_next`) to store the previous,
current and next states of the surface. The update rule computes the next
height value at each grid cell using the heights of its four neighbours and
itself, scaled by the wave speed and timestep. A damping term reduces
oscillation amplitude in a physically plausible way【247236719887788†L95-L141】.

Example usage::

    from water_simulation import WaterSimulation
    sim = WaterSimulation(width=150, height=150, c=1.0, damping=0.01)
    # Give the centre of the grid an initial disturbance
    sim.disturb(sim.height // 2, sim.width // 2, magnitude=1.0)
    # Run 200 frames of the simulation and visualise
    for frame in sim.run(frames=200):
        # Each `frame` is a 2D numpy array of floats representing water height.
        # Insert your rendering code here (e.g., using matplotlib.imshow).
        pass

The `run` method yields successive states of the grid, which can be plotted or
exported to video. See the `__main__` block at the bottom of this file for a
simple example using matplotlib to animate the simulation.
"""

from __future__ import annotations

import numpy as np
from typing import Iterator, Tuple


class WaterSimulation:
    """Simulate 2D wave propagation on a rectangular grid.

    Parameters
    ----------
    width : int
        Number of grid cells in the horizontal direction.
    height : int
        Number of grid cells in the vertical direction.
    c : float, optional
        Wave propagation speed. Higher values lead to faster wave travel.
    dt : float, optional
        Discrete timestep used for integration. Must satisfy the Courant
        condition ``c*dt <= 1/sqrt(2)`` for stability when `dx = dy = 1`.
    damping : float, optional
        Non‑negative damping coefficient. A value of 0 corresponds to an
        undamped system. Small positive values progressively reduce the
        amplitude of oscillations.

    Notes
    -----
    The simulation uses fixed boundary conditions: the edges of the grid are
    clamped to zero height at all times. Disturbances injected into the
    interior reflect off the boundaries and eventually dissipate via damping.
    """

    def __init__(self,
                 width: int,
                 height: int,
                 c: float = 1.0,
                 dt: float = 0.15,
                 damping: float = 0.0) -> None:
        if width < 3 or height < 3:
            raise ValueError("Grid must be at least 3×3 to compute neighbours.")
        if c <= 0:
            raise ValueError("Wave speed c must be positive.")
        if dt <= 0:
            raise ValueError("Timestep dt must be positive.")
        if damping < 0:
            raise ValueError("Damping coefficient must be non‑negative.")

        # Courant–Friedrichs–Lewy (CFL) condition for stability when dx=dy=1.
        # See the reference for discrete 2D wave equation stability【247236719887788†L95-L141】.
        cfl = c * dt / np.sqrt(2)
        if cfl > 1:
            raise ValueError(
                f"Unstable parameters: c*dt/sqrt(2) = {cfl:.3f} > 1. "
                "Decrease dt or c to satisfy the CFL condition.")

        self.width = width
        self.height = height
        self.c = c
        self.dt = dt
        self.damping = damping

        # Internal state arrays: previous (u_prev), current (u), next (u_next).
        # All arrays include an extra border (ghost cells) for boundary
        # conditions. The simulation only updates interior cells (1:-1,1:-1).
        self.u_prev = np.zeros((height, width), dtype=np.float64)
        self.u = np.zeros((height, width), dtype=np.float64)
        self.u_next = np.zeros((height, width), dtype=np.float64)

        # Precompute constants for the update equation to avoid recomputation
        # inside the loop. The discrete Laplacian coefficient is (c*dt)^2.
        self.coeff = (c * dt) ** 2

    def disturb(self, y: int, x: int, magnitude: float = 1.0, radius: int = 1) -> None:
        """Apply a disturbance to the grid at (y, x).

        Parameters
        ----------
        y, x : int
            Coordinates of the disturbance centre. Coordinates are in
            array‑index order (row, column).
        magnitude : float, optional
            Peak height of the disturbance.
        radius : int, optional
            Radius (in grid cells) over which to distribute the disturbance.

        Notes
        -----
        A circular disturbance is created by setting the current state `u`
        within the specified radius to `magnitude`. The previous state
        `u_prev` remains unchanged, producing an initial velocity and thus
        waves propagating outward.
        """
        if radius < 0:
            raise ValueError("Radius must be non‑negative.")
        # Generate a mask for a disc centred at (y,x)
        y_indices, x_indices = np.ogrid[:self.height, :self.width]
        mask = (y_indices - y) ** 2 + (x_indices - x) ** 2 <= radius ** 2
        self.u[mask] += magnitude

    def step(self) -> np.ndarray:
        """Advance the simulation by one time step.

        Returns
        -------
        np.ndarray
            The updated grid (current state) after the step. The returned
            array is the same object as `self.u` for convenience.
        """
        # Alias local variables for speed inside the loop.
        u_prev = self.u_prev
        u = self.u
        u_next = self.u_next
        coeff = self.coeff
        damp = self.damping

        # Compute the next state for interior points only.
        # Discrete 2D wave equation (five‑point stencil) with damping:
        # u_next[i,j] = 2*u[i,j] - u_prev[i,j] + coeff * (
        #     u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]
        # ) - damping * (u[i,j] - u_prev[i,j])
        # We use array slicing for vectorised computation.
        # Compute Laplacian of u
        laplacian = (
            u[0:-2, 1:-1] + u[2:, 1:-1] +
            u[1:-1, 0:-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1]
        )
        u_next[1:-1, 1:-1] = (
            2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + coeff * laplacian
            - damp * (u[1:-1, 1:-1] - u_prev[1:-1, 1:-1])
        )

        # Enforce fixed boundary conditions: edges remain at zero height.
        u_next[0, :] = 0.0
        u_next[-1, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0

        # Rotate buffers: the new state becomes current, current becomes previous.
        self.u_prev, self.u, self.u_next = self.u, self.u_next, self.u_prev

        return self.u

    def run(self, frames: int) -> Iterator[np.ndarray]:
        """Generate successive frames of the simulation.

        Parameters
        ----------
        frames : int
            Number of time steps to simulate.

        Yields
        ------
        np.ndarray
            A 2D array representing the water height at the current time step.
        """
        for _ in range(frames):
            yield self.step()


def _demo_simulation() -> None:
    """Run a demonstration of the water simulation using matplotlib.

    This function creates a WaterSimulation instance, adds an initial
    disturbance, and animates the resulting wave propagation using
    `matplotlib.animation.FuncAnimation`. It serves as a quick visual check
    that the simulation behaves reasonably but is not required for use as a
    library.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    # Set up the simulation
    width, height = 100, 100
    sim = WaterSimulation(width, height, c=1.0, dt=0.15, damping=0.01)
    sim.disturb(height // 2, width // 2, magnitude=1.0, radius=2)

    fig, ax = plt.subplots()
    im = ax.imshow(sim.u, cmap='Blues', vmin=-1, vmax=1, animated=True)
    ax.set_title("Water Simulation Demo")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_index: int) -> list:
        # Advance the simulation and update the image
        frame = sim.step()
        im.set_data(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=200, interval=30, blit=True, repeat=False
    )
    plt.show()


if __name__ == '__main__':
    _demo_simulation()
