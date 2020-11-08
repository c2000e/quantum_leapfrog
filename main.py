from leapfrog import *

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
    
#    animation.save('quantum_barrier.mp4')

    plt.show()
