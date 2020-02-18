import numpy as np

class optimization_input():
    """This class optimizes 1D linear dynamic systems. The class requires the mass, damping, and stiffness matrices of
    the classical equations of motion. The matrices need to be written in the direct form (the mass matrix needs to be
    diagonal and contain only the masses of the individual degrees of freedom DOFs). Besides the matrices, also the
    power spectral density PSD function of the ground motion and its corresponding frequency range need to be provided
    to the class."""

    def __init__(self, Mass, Damping, Stiffness, Spectrum, omega_range):
        """Here the matrices are collected.
        The spectrum represents the ground motion and is formulated as a PSD.
        omega_range is a numpy array that contains all the frequency step values of the spectrum in rad/s.
        For example: omega_range = ([0.0, 0.1, 0.2, ..., 100.0, 100.1])"""
        self.M = Mass
        self.C = Damping
        self.K = Stiffness
        self.spectrum = Spectrum
        self.omega_range = omega_range

    def incrementK(self, step_size, dof1, dof2):
        """This function increments the stiffness matrix for a spring that acts between dof1 and dof2.
        The step_size controls the amount of stiffness that is added to the spring.
        dof1 and dof2 mark the dofs which are connected by the spring, which is supposed to be incremented."""
        self.K[dof1 - 1, dof1 - 1] += step_size
        self.K[dof1 - 1, dof2 - 1] += -step_size
        self.K[dof2 - 1, dof1 - 1] += -step_size
        self.K[dof2 - 1, dof2 - 1] += step_size

    def incrementC(self, step_size, dof1, dof2):
        """Same function as above for the damping matrix"""
        self.C[dof1 - 1, dof1 - 1] += step_size
        self.C[dof1 - 1, dof2 - 1] += -step_size
        self.C[dof2 - 1, dof1 - 1] += -step_size
        self.C[dof2 - 1, dof2 - 1] += step_size

    def VarianceOfResponse(self):
        """This functions calculates the variance of the response for all DOFs. It uses the matrices of the system,
        computes the transmission matrix H, and integrates it together with the PSD of the spectrum.
        In principle, the the PSD of the response can be calculated with PSD_reponse = abs(H)^2 * PSD_spectrum.
        Here the PSD_spectrum represents a vector that contains the diagonal of the mass matrix (or M*I, with I being
        the identity vector) and multiplies it with the spectrum defined by the user (self.spectrum)."""
        H = []
        for i in range(len(self.omega_range)):
            """Caclulation of the Transmission matrix H"""
            H.append(np.linalg.inv((-self.omega_range[i] ** 2 * self.M
                                    - 1j * self.omega_range[i] * self.C
                                    + self.K)))
        """squared absolute of the transmission matrix H multiplied with the diagonal of the mass matrix M (or M*I)"""
        HabsVec = [abs(matrix**2).dot(np.transpose(np.diagonal(self.M))) for matrix in H]
        """Response of all DOFs as PSD"""
        RespPSD = [HabsVec[wincr] * self.spectrum[wincr] for wincr in range(len(self.spectrum))]
        """The variance of the response can be obtained with the integral of the response PSD. 
        integral(PSD_response)"""
        variance = (np.trapz(RespPSD, self.omega_range, axis=0))
        return variance

    def optimizationK(self, step_size, dof1, dof2, controlDOF):
        """Optimization of the system by minimizing the variance"""
        """First, the initial variance of the system is computed for the matrices given by the user"""
        Variance = []
        """The sign variable keeps track of the direction the algorithm is stepping"""
        sign = []
        sign.append(0)
        Variance.append(self.VarianceOfResponse())

        """The matrix K is incremented by one step with size = step_size. for the spring that connects dof1 and dof2"""
        sign.append(1)
        self.incrementK(step_size, dof1, dof2)
        Variance.append(self.VarianceOfResponse())

        """Loop that iteratively steps down the optimal direction of increasing or reducing K"""
        while len(sign) < 4 or np.sum(sign[-4:]) != 0:
            """First condition demands a minimum of 4 steps. Second condition sums up the direction of the last 4 steps
            if the sum equals 0, the minimum value for the variance has been found."""
            if Variance[-2][int(controlDOF - 1)] > Variance[-1][int(controlDOF - 1)]:
                """If the last variance is smaller than the variance before, keep going in that direction."""
                self.incrementK(step_size * sign[-1], dof1, dof2)
                Variance.append(self.VarianceOfResponse())
                sign.append(sign[-1])
            elif Variance[-2][int(controlDOF - 1)] < Variance[-1][int(controlDOF - 1)]:
                """If the last variance is bigger than the one before, turn around and go the other way."""
                self.incrementK(step_size * sign[-1] * -1, dof1, dof2)
                Variance.append(self.VarianceOfResponse())
                sign.append(sign[-1] * -1)
        return Variance

    def optimizationC(self, step_size, dof1, dof2, controlDOF):
        """Same as for the stiffness matrix"""
        Variance = []
        sign = []
        sign.append(0)
        Variance.append(self.VarianceOfResponse())

        sign.append(1)
        self.incrementC(step_size, dof1, dof2)
        Variance.append(self.VarianceOfResponse())

        while len(sign) < 4 or np.sum(sign[-4:]) != 0:
            if Variance[-2][int(controlDOF - 1)] > Variance[-1][int(controlDOF - 1)]:
                self.incrementC(step_size * sign[-1], dof1, dof2)
                Variance.append(self.VarianceOfResponse())
                sign.append(sign[-1])
            elif Variance[-2][int(controlDOF - 1)] < Variance[-1][int(controlDOF - 1)]:
                self.incrementC(step_size * sign[-1] * -1, dof1, dof2)
                Variance.append(self.VarianceOfResponse())
                sign.append(sign[-1] * -1)
        return Variance