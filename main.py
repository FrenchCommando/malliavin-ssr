import numpy as np
from matplotlib import pyplot as plt

from models import StochasticVol, LocalVol
from montecarlo import montecarlo_local_vol, montecarlo_stochastic_vol, diffusion_all


def compare_local_vol():
    t_range = np.concatenate((np.linspace(0, 0.02, 100 + 1), np.linspace(0.02, 1, 100 + 1)))
    out = montecarlo_local_vol(**dict(
        n_mc=100000,
        ts=t_range,
    ))

    skew_mc = out['atmskew']
    ssr_mc = out['atmssr']

    plt.figure("SkewTerminal - Local Vol")
    plt.plot(t_range, LocalVol().skew_terminal(t=t_range), label='Skew-LocalVol-Formula')
    plt.plot(t_range[1:], skew_mc, label='Skew-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Skew")
    plt.title("Skew - Local Vol")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("SsrTerminal - Local Vol")
    plt.plot(t_range, LocalVol().ssr_terminal(t=t_range), label='SSR-LocalVol-Formula')
    plt.plot(t_range[1:], ssr_mc, label='SSR-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SkewStickinessRatio")
    plt.title("SSR - Local Vol")
    plt.minorticks_on()
    plt.grid(visible=True)


def compare_stochastic_vol():
    t_range = np.concatenate((np.linspace(0, 0.02, 100 + 1), np.linspace(0.02, 1, 100 + 1)))
    out = montecarlo_stochastic_vol(**dict(
        n_mc=100000,
        ts=t_range
    ))
    skew_mc = out['atmskew']
    ssr_mc = out['atmssr']

    plt.figure("SkewTerminal - Stochastic Vol")
    plt.plot(t_range, StochasticVol().skew_terminal(t=t_range), label='Skew-StochasticVol-Formula')
    plt.plot(t_range[1:], skew_mc, label='Skew-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Skew")
    plt.title("Skew - Stochastic Vol")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("SsrTerminal - Stochastic Vol")
    plt.plot(t_range, StochasticVol().ssr_terminal(t=t_range), label='SSR-StochasticVol-Formula')
    plt.plot(t_range[1:], ssr_mc, label='SSR-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SkewStickinessRatio")
    plt.title("SSR - Stochastic Vol")
    plt.minorticks_on()
    plt.grid(visible=True)


def main():
    # compare_local_vol()
    # compare_stochastic_vol()
    diffusion_all()
    plt.show()


if __name__ == '__main__':
    main()
