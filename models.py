import numpy as np
import matplotlib.pyplot as plt
import unittest


class LocalVol:
    @staticmethod
    def term_structure_skew(t):
        tau0 = 0.1
        # tau0 = 100.
        gamma = 0.5
        alpha0 = -1.5
        # alpha0 = 0
        t_floor = np.clip(t, a_min=tau0, a_max=None)
        return alpha0 * (tau0 / t_floor) ** gamma

    @staticmethod
    def local_volatility(t, s, add_derivatives=False):
        sigma0 = 0.2
        sigma_min = 0.001
        sigma_max = 5.
        beta = 0.0
        alpha_value = LocalVol.term_structure_skew(t=t)
        unbounded = sigma0 + (alpha_value + beta * (s - 1)) * (s - 1)
        bounded = np.clip(unbounded, a_min=sigma_min, a_max=sigma_max)
        if not add_derivatives:
            return bounded
        unbounded_derivative = alpha_value + 2 * beta * (s - 1)
        bounded_derivative = ((unbounded < sigma_max) & (unbounded > sigma_min)) * unbounded_derivative
        return bounded + s * bounded_derivative

    @staticmethod
    def local_volatility_zero():
        return LocalVol.local_volatility(t=0, s=1)

    @staticmethod
    def skew_terminal(t):
        dt = np.diff(t)
        t_start = t[:-1]
        alpha_t = LocalVol.term_structure_skew(t=t_start)
        integrand_t = 0.5 * (t[:-1] + t[1:]) * alpha_t * dt
        integrated_t = np.cumsum(integrand_t)
        integrated_scaled = integrated_t / t[1:] ** 2
        result = np.insert(integrated_scaled, 0, integrated_scaled[0])
        return result

    @staticmethod
    def ssr_terminal(t):
        skew_t = LocalVol.skew_terminal(t=t)
        dt = np.diff(t)
        integrand_t = 0.5 * (skew_t[:-1] + skew_t[1:]) * dt
        integrated_t = np.cumsum(integrand_t)
        integrated_scaled = integrated_t / t[1:] / skew_t[1:]
        result = np.insert(integrated_scaled, 0, integrated_scaled[0])
        return 1 + result


class StochasticVol:
    nu = 3.1
    theta = 0.139
    kx = 8.59
    ky = 0.47
    rho_xy = 0
    rho_sx = -0.54
    rho_sy = -0.623

    @staticmethod
    def alpha_theta():
        return (
            (1 - StochasticVol.theta) ** 2
            + StochasticVol.theta ** 2
            + 2 * StochasticVol.rho_xy * (1 - StochasticVol.theta) * StochasticVol.theta
        ) ** - 0.5

    @staticmethod
    def omega():
        return 2 * StochasticVol.nu

    @staticmethod
    def x_tt(x_s, y_s):
        return StochasticVol.alpha_theta() * (
            (1 - StochasticVol.theta) * x_s + StochasticVol.theta * y_s
        )

    @staticmethod
    def chi_tt(t):
        return StochasticVol.alpha_theta() * (
            (1 - StochasticVol.theta) ** 2 * (1 - np.exp(-2 * StochasticVol.kx * t)) / (2 * StochasticVol.kx)
            + StochasticVol.theta ** 2 * (1 - np.exp(-2 * StochasticVol.ky * t)) / (2 * StochasticVol.ky)
            + 2 * StochasticVol.rho_xy * (1 - StochasticVol.theta) * StochasticVol.theta * (
                    1 - np.exp(-(StochasticVol.kx + StochasticVol.ky) * t)
            ) / (StochasticVol.kx + StochasticVol.ky)
        )

    @staticmethod
    def tangent_x_factor(t):
        return 0.5 * StochasticVol.omega() * StochasticVol.alpha_theta() * (
            (1 - StochasticVol.theta) * np.exp(-StochasticVol.kx * t)
        )

    @staticmethod
    def tangent_y_factor(t):
        return 0.5 * StochasticVol.omega() * StochasticVol.alpha_theta() * (
            StochasticVol.theta * np.exp(-StochasticVol.ky * t)
        )

    @staticmethod
    def instantaneous_variance_swap_volatility():
        return 0.21

    @staticmethod
    def local_volatility_zero():
        return StochasticVol.instantaneous_variance_swap_volatility()

    @staticmethod
    def skew_terminal(t):
        t = np.clip(t, a_min=1e-6, a_max=None)
        return StochasticVol.nu * StochasticVol.alpha_theta() * (
            (1 - StochasticVol.theta) * StochasticVol.rho_sx * (
                (StochasticVol.kx * t - 1 + np.exp(-StochasticVol.kx * t)) / (StochasticVol.kx * t) ** 2
            )
            + StochasticVol.theta * StochasticVol.rho_sy * (
                (StochasticVol.ky * t - 1 + np.exp(-StochasticVol.ky * t)) / (StochasticVol.ky * t) ** 2
            )
        )

    @staticmethod
    def ssr_terminal(t):
        t = np.clip(t, a_min=1e-6, a_max=None)
        return (
            (1 - StochasticVol.theta) * StochasticVol.rho_sx * (
                (1 - np.exp(-StochasticVol.kx * t)) / (StochasticVol.kx * t)
            )
            + StochasticVol.theta * StochasticVol.rho_sy * (
                    (1 - np.exp(-StochasticVol.ky * t)) / (StochasticVol.ky * t)
            )
        ) / (
            (1 - StochasticVol.theta) * StochasticVol.rho_sx * (
                (StochasticVol.kx * t - 1 + np.exp(-StochasticVol.kx * t)) / (StochasticVol.kx * t) ** 2
            )
            + StochasticVol.theta * StochasticVol.rho_sy * (
                (StochasticVol.ky * t - 1 + np.exp(-StochasticVol.ky * t)) / (StochasticVol.ky * t) ** 2
            )
        )


class StochasticLocalVol(LocalVol, StochasticVol):
    @staticmethod
    def skew_terminal(t):
        raise ValueError("skew_terminal should not be used from StochasticLocalVol")

    @staticmethod
    def ssr_terminal(t):
        raise ValueError("ssr_terminal should not be used from StochasticLocalVol")


def main():
    lv = LocalVol()
    print(lv)
    print(lv.local_volatility(t=1, s=1))
    print(lv.local_volatility(t=2, s=2))
    print(lv.local_volatility(t=0.2, s=0.5))
    print(lv.skew_terminal(t=np.array([0, 0.5, 2])))
    print(lv.skew_terminal(t=np.linspace(0, 1, 20)))
    print(lv.ssr_terminal(t=np.linspace(0, 1, 20)))
    print(lv.local_volatility_zero())

    sv = StochasticVol()
    print(sv)
    print(sv.skew_terminal(t=np.array([0, 0.5, 2])))
    print(sv.ssr_terminal(t=np.linspace(0, 1, 20)))
    print(sv.local_volatility_zero())

    slv = StochasticLocalVol()
    print(slv)
    print(slv.nu)
    print(slv.local_volatility(t=1, s=1))
    print(slv.local_volatility(t=2, s=2))
    print(slv.local_volatility(t=0.2, s=0.5))
    print(slv.local_volatility_zero())
    print()

    plt.figure("SkewTerminal")
    t_range = np.linspace(0.0, 1.0, 1000)
    plt.plot(t_range, LocalVol().skew_terminal(t=t_range), label='LocalVol')
    plt.plot(t_range, StochasticVol().skew_terminal(t=t_range), label='StochasticVol')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Skew")
    plt.title("Terminal Skew")
    plt.minorticks_on()
    plt.grid(visible=True)

    plt.figure("SsrTerminal")
    t_range = np.linspace(0.0, 1.0, 1000)
    plt.plot(t_range, LocalVol().ssr_terminal(t=t_range), label='LocalVol')
    plt.plot(t_range, StochasticVol().ssr_terminal(t=t_range), label='StochasticVol')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SkewStickinessRatio")
    plt.title("Terminal SSR")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.show()


class TestModels(unittest.TestCase):
    def test_lv_stuff(self):
        lv = LocalVol()
        self.assertFalse(hasattr(lv, 'nu'))
        self.assertTrue(isinstance(lv.local_volatility(t=1, s=1), float))
        self.assertTrue(isinstance(lv.local_volatility(t=2, s=2), float))
        self.assertTrue(isinstance(lv.local_volatility(t=0.2, s=0.5), float))
        self.assertTrue(isinstance(lv.skew_terminal(t=np.array([0, 0.5, 2])), np.ndarray))
        self.assertTrue(isinstance(lv.ssr_terminal(t=np.array([0, 0.5, 2])), np.ndarray))
        self.assertTrue(isinstance(lv.local_volatility_zero(), float))

    def test_sv_stuff(self):
        sv = StochasticVol()
        self.assertTrue(hasattr(sv, 'nu'))
        self.assertFalse(hasattr(sv, 'local_volatility'))
        self.assertTrue(isinstance(sv.skew_terminal(t=np.array([0, 0.5, 2])), np.ndarray))
        self.assertTrue(isinstance(sv.ssr_terminal(t=np.array([0, 0.5, 2])), np.ndarray))
        self.assertTrue(isinstance(sv.instantaneous_variance_swap_volatility(), float))
        self.assertTrue(isinstance(sv.local_volatility_zero(), float))

    def test_slv_stuff(self):
        slv = StochasticLocalVol()
        self.assertTrue(isinstance(slv.nu, float))
        self.assertTrue(isinstance(slv.local_volatility(t=1, s=1), float))
        self.assertTrue(isinstance(slv.local_volatility(t=2, s=2), float))
        self.assertTrue(isinstance(slv.local_volatility(t=0.2, s=0.5), float))
        self.assertRaises(ValueError, slv.skew_terminal, t=[0, 0.5, 2])
        self.assertRaises(ValueError, slv.ssr_terminal, t=[0, 0.5, 2])
        self.assertTrue(isinstance(slv.local_volatility_zero(), float))


if __name__ == '__main__':
    main()
