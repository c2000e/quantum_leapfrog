import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time


# Initial shapes.

class Gaussian:

    def __init__(self, a, b, c, k):
        self.a = a
        self.b = b
        self.c = c
        self.k = k

    def r_at(self, x):
        return self.a * np.exp(-(0.5 * (x - self.b) / self.c) ** 2) \
                * np.cos(self.k * x)

    def i_at(self, x):
        return self.a * np.exp(-(0.5 * (x - self.b) / self.c) ** 2) \
                * np.sin(self.k * x)

    def at(self, x):
        return np.transpose(np.array([self.r_at(x), self.i_at(x)]))


# Potential functions.

class ZeroPotential:

    def __init__(self):
        pass

    def at(self, x):
        return 0 * x 

class SHOPotential:

    def __init__(self, x, k):
        self.x = x
        self.k = k

    def at(self, x):
        return self.k * (x - self.x) ** 2

class WallPotential:

    def __init__(self, x, a):
        self.x = x
        self.a = a

    def at(self, x):
        return np.where(x < self.x, 0, self.a)


# Staggered leapfrog solver.

class LeapfrogSolver:
    
    def __init__(self, x_i, x_f, M, t_f, N, F, V, e = 0.1):

        # The number of grid points between x_i and x_f.
        self.M = M
        
        # The spacing between grid points.
        self.dx = (x_f - x_i) / M

        # The number of time steps between 0 and t_f.
        self.N = N

        # The length of a time step.
        self.dt = t_f / N

        # This is a term used frequently during integration.
        self.dtdx2 = self.dt / (2 * (self.dx ** 2))

        # X.
        self.X = np.array([x_i + m * self.dx for m in range(self.M + 1)])

        # Psi.
        self.Y = np.zeros((N + 1, M + 1, 2))
        self.Y[0] = F(self.X)
        
        # Probability.
        self.P = np.zeros((N + 1, M + 1))

        # Expectation of X.
        self.U = np.zeros(N + 1)

        # Potential.
        self.V = V(self.X)

        # Back-integrate the imaginary component of Psi half a time step.        
        d2R = self.Y[0, 2:, 0] - 2 * self.Y[0, 1:-1, 0] \
                + self.Y[0, 0:-2, 0]

        d2R = np.concatenate(([0], d2R, [0]))

        I = self.Y[0, :, 1] - self.dtdx2 * d2R + self.dt * \
                np.multiply(self.V, self.Y[0, :, 0])

        self.Y[0, :, 1] = I

        # Normalize the initial state.
        self.P[0] = self.norm_squared(0)
        total_p = np.sum(self.P[0])

        self.Y[0] = self.Y[0] / np.sqrt(total_p)
        self.P[0] = self.P[0] / total_p

        # Calculate the initial expected value of X.
        self.U[0] = self.expected_x(0)

        # Set the acceptable range for the total probability.
        self.min_p = 1 - e
        self.max_p = 1 + e

        # Stores the last calculated time step, in case of failure.
        self.MAX_N = N

    # Calculate I(x, t + dt/2).
    def next_i(self, n):
        
        d2R = self.Y[n, 2:, 0] - 2 * self.Y[n, 1:-1, 0] \
                + self.Y[n, 0:-2, 0]

        d2R = np.concatenate(([0], d2R, [0]))

        I = self.Y[n, :, 1] + self.dtdx2 * d2R - self.dt * \
                np.multiply(self.V, self.Y[n, :, 0])

        return I

    # Calculate R(x, t + dt).
    def next_r(self, n, I):

        d2I = I[2:] - 2 * I[1:-1] + I[0:-2]

        d2I = np.concatenate(([0], d2I, [0]))

        R = self.Y[n, :, 0] - self.dtdx2 * d2I + self.dt * \
                np.multiply(self.V, I)

        return R

    # Calculate |Psi|^2.
    def norm_squared(self, n):
        P = np.sum(self.Y[n] ** 2, axis=1) 
        return P

    # Calculate <X>.
    def expected_x(self, n):
        U = np.sum(np.multiply(self.X, self.P[n]))
        return U

    # Move forward one step in time.
    def step(self, n):

        I = self.next_i(n)
        R = self.next_r(n, I)

        self.Y[n + 1] = np.stack((R, I), axis = -1)
        
        self.P[n + 1] = self.norm_squared(n + 1)

        self.U[n + 1] = self.expected_x(n + 1)

    # Propagate Psi from t=0 to t=t_f.
    def integrate(self):

        prev_time = time.time()

        print("Propagating over " + str(self.N) + " time steps...")
        for n in range(self.N):

            self.step(n)

            # Ensure the solution is still valid.
            total_p = np.sum(self.P[n + 1])
            if total_p < self.min_p:
                print("Total probability below acceptable range at n=" \
                        + str(n))
                print("Total probability: " + str(total_p))
                print("Minimum acceptable: " + str(self.min_p))
                self.MAX_N = n
            if total_p > self.max_p:
                print("Total probability above acceptable range at n=" \
                        + str(n))
                print("Total probability: " + str(total_p))
                print("Maximum acceptable: " + str(self.max_p))
                self.MAX_N = n

            # Give a progress report.
            if (n % 1000 == 0) and (n > 0):

                curr_time = time.time()
                elap_time = curr_time - prev_time
                prev_time = curr_time
                    
                print("Completed " + str(n) + " steps. Elapsed time: " \
                        + str(elap_time))

        return self.N


# Wavefunction animation.

class Animator:

    def __init__(self, solver):
        self.X = solver.X
        self.R = solver.Y[:, :, 0]
        self.I = solver.Y[:, :, 1]
        self.P = solver.P

        self.V = solver.V

        self.T = np.array([n * solver.dt for n in range(solver.N + 1)])
        self.U = solver.U

        self.fig, ((self.r_ax, self.p_ax), (self.i_ax, self.u_ax)) \
                = plt.subplots(2, 2)

        self.fig.tight_layout(pad = 1.0)

        self.r_ax.set_title("real")
        self.i_ax.set_title("imaginary")
        self.p_ax.set_title("probability")
        self.u_ax.set_title("expected x")

        max_p = np.amax(self.P)

        self.r_ax.set_ylim(np.amin(self.R), np.amax(self.R))
        self.i_ax.set_ylim(np.amin(self.I), np.amax(self.I))
        self.p_ax.set_ylim(0, max_p)
        self.u_ax.set_xlim(0, self.T[-1])
        self.u_ax.set_ylim(np.amin(self.U), np.amax(self.U))

        min_v = np.amin(self.V)
        max_v = np.amax(self.V)

        vp_scale = max_p / (max_v - min_v)

        self.r_graph, = self.r_ax.plot(self.X, self.R[0])
        self.i_graph, = self.i_ax.plot(self.X, self.I[0])
        
        self.p_graph, = self.p_ax.plot(self.X, self.P[0])
        self.vp_graph, = self.p_ax.plot(self.X, vp_scale \
                * (self.V - min_v))

        self.u_graph, = self.u_ax.plot(self.T[:1], self.U[:1])

    def step(self, n):
        self.r_graph.set_ydata(self.R[n])
        self.i_graph.set_ydata(self.I[n])
        self.p_graph.set_ydata(self.P[n])
        self.u_graph.set_data(self.T[:n], self.U[:n])

    def animate(self):
        return FuncAnimation(self.fig, self.step, frames = len(self.T),\
                interval = 1)


if __name__ == "__main__":

    # The initial shape of the wave function.
    F = Gaussian(1, 0, 0.3, 100)

    #V = ZeroPotential() # A free particle.
    #V = SHOPotential(0.5, 4000) # An SHO-like potential.
    V = WallPotential(1, 4000) # A potential barrier.

    solver = LeapfrogSolver(-2, 3, 1000, 0.1, 10000, F.at, V.at)
    solver.integrate()

    animator = Animator(solver)
    animation = animator.animate()

    plt.show()

