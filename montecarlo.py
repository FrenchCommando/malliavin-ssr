import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt

from models import LocalVol, StochasticVol, StochasticLocalVol
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks.analytical import vega, delta

epsilon_skew = 1e-8
epsilon_ssr = 1e-6


def safe_implied_volatility(price, s, k, t):
    if np.isnan(price):
        return np.nan
    return implied_volatility(price=price, S=s, K=k, r=0, t=t, flag='c')


def safe_vega(sigma, s, k, t):
    if np.isnan(sigma):
        return np.nan
    return vega(sigma=sigma, S=s, K=k, r=0, t=t, flag='c') * 100


def price_options(spots_for_price, forwards_for_price, strikes, t_values):
    atm_call = np.mean(np.clip(spots_for_price.T - strikes, a_min=0, a_max=None), axis=0)
    option_vol = np.array([safe_implied_volatility(
        price=price_value, s=forward_value, k=strike_value, t=t_value,
    ) for price_value, forward_value, strike_value, t_value
        in zip(atm_call[1:], forwards_for_price[1:], strikes[1:], t_values[1:])
    ])
    return option_vol


def vega_options(vols, forwards_for_vega, strikes, t_values):
    option_vega = np.array([safe_vega(
        sigma=sigma_value, s=forward_value, k=strike_value, t=t_value,
    ) for sigma_value, forward_value, strike_value, t_value
        in zip(vols, forwards_for_vega[1:], strikes[1:], t_values[1:])
    ])
    return option_vega


def montecarlo_local_vol(
    n_mc, ts,
):
    nt = len(ts) - 1
    dts = np.diff(ts)

    def diffuse_spots(spot_values, normal_values):
        for i in range(nt):
            volatility = np.sqrt(dts[i]) * LocalVol.local_volatility(t=ts[i], s=spot_values[i, :])
            spot_values[i + 1, :] = spot_values[i, :] * np.exp(
                - 0.5 * volatility * volatility + volatility * normal_values[i, :]
            )

    def diffuse_spots_tangent(spot_tangent_values, normal_values):
        for i in range(nt):
            sqrt_dt = np.sqrt(dts[i])

            volatility = sqrt_dt * LocalVol.local_volatility(t=ts[i], s=spot_tangent_values[i, 0, :])
            spot_tangent_values[i + 1, 0, :] = spot_tangent_values[i, 0, :] * np.exp(
                - 0.5 * volatility * volatility + volatility * normal_values[i, :]
            )

            volatility_with_derivatives = sqrt_dt * LocalVol.local_volatility(
                t=ts[i], s=spot_tangent_values[i, 0, :], add_derivatives=True,
            )
            spot_tangent_values[i + 1, 1, :] = spot_tangent_values[i, 1, :] * np.exp(
                - 0.5 * volatility_with_derivatives * volatility_with_derivatives +
                volatility_with_derivatives * normal_values[i, :]
            )

    normals = np.random.normal(size=(nt, n_mc))

    def diffusion_for_skew():
        spots = np.ones(shape=(nt + 1, n_mc))
        diffuse_spots(spot_values=spots, normal_values=normals)

        forwards = np.mean(spots, axis=1)
        atm_vols = price_options(spots_for_price=spots, forwards_for_price=forwards, strikes=forwards, t_values=ts)
        strikes_plus = forwards * np.exp(epsilon_skew)
        strikes_minus = forwards * np.exp(-epsilon_skew)
        atm_plus_vols = price_options(
            spots_for_price=spots, forwards_for_price=forwards, strikes=strikes_plus, t_values=ts,
        )
        atm_minus_vols = price_options(
            spots_for_price=spots, forwards_for_price=forwards, strikes=strikes_minus, t_values=ts,
        )
        return dict(
            forwards=forwards,
            atmvols=atm_vols,
            atmplusvols=atm_plus_vols,
            atmminusvols=atm_minus_vols,
        )

    def diffusion_for_ssr():

        def diffuse_and_vols(index_bump):
            spots_ssr = np.zeros(shape=(nt + 1, n_mc))
            spots_ssr[:] = np.exp(local_vol_zero * epsilon_ssr) if index_bump == 0 else 1

            diffuse_spots(spot_values=spots_ssr, normal_values=normals)
            forwards_ssr = np.mean(spots_ssr, axis=1)
            return price_options(
                spots_for_price=spots_ssr, forwards_for_price=forwards_ssr, strikes=forwards_ssr, t_values=ts,
            )

        atm_vols_ssr_spot = diffuse_and_vols(index_bump=0)
        print('LV - SSR - Spot - Done')
        atm_vols_ssr_x = diffuse_and_vols(index_bump=1)
        print('LV - SSR - X - Done')
        atm_vols_ssr_y = diffuse_and_vols(index_bump=2)
        print('LV - SSR - Y - Done')

        return dict(
            atmvolsssrspot=atm_vols_ssr_spot,
            atmvolsssrx=atm_vols_ssr_x,
            atmvolsssry=atm_vols_ssr_y,
        )

    def diffusion_for_tangent():
        spots_tangent = np.ones(shape=(nt + 1, 2, n_mc))
        # spots_tangent[:, 1, :] = LocalVol.local_volatility_zero() ** 0
        # spots_tangent[:, 1, :] = np.exp(LocalVol.local_volatility_zero())

        diffuse_spots_tangent(spot_tangent_values=spots_tangent, normal_values=normals)

        forwards_spot = np.mean(spots_tangent[:, 0, :], axis=1)
        spot_values = spots_tangent[:, 0, :]
        tangent_values = spots_tangent[:, 1, :]

        tangent_price = np.mean(
            (spot_values > forwards_spot[:, np.newaxis]) * (tangent_values - spot_values)
            , axis=1)

        atm_options_spot = tangent_price[1:]
        atm_options_x = tangent_price[1:] * 0
        atm_options_y = tangent_price[1:] * 0
        print('LV - Tangent - Spot - Done')

        return dict(
            atmoptionsspottangent=atm_options_spot,
            atmoptionsxtangent=atm_options_x,
            atmoptionsytangent=atm_options_y,
        )

    local_vol_zero = LocalVol.local_volatility_zero()
    d_skew = diffusion_for_skew()
    d_ssr = diffusion_for_ssr()
    d_tangent = diffusion_for_tangent()

    atm_skew = (d_skew['atmplusvols'] - d_skew['atmminusvols']) / (2 * epsilon_skew)
    # atm_skew_extrapolation_index = min(nt // 10, 20)
    # atm_skew[:atm_skew_extrapolation_index] = atm_skew[atm_skew_extrapolation_index]

    atm_ssr = (
            (d_ssr['atmvolsssrspot'] - d_skew['atmvols'])
            + (d_ssr['atmvolsssrx'] - d_skew['atmvols']) * StochasticVol.rho_sx
            + (d_ssr['atmvolsssry'] - d_skew['atmvols']) * StochasticVol.rho_sy
    ) / epsilon_ssr / local_vol_zero / atm_skew
    atm_ssr[:min(nt // 10, 10)] = np.nan

    atm_dspot = (d_ssr['atmvolsssrspot'] - d_skew['atmvols']) / epsilon_ssr / local_vol_zero
    atm_dx = (d_ssr['atmvolsssrx'] - d_skew['atmvols']) / epsilon_ssr
    atm_dy = (d_ssr['atmvolsssry'] - d_skew['atmvols']) / epsilon_ssr
    atm_vvol = np.sqrt(
        atm_dspot ** 2 * local_vol_zero ** 2
        + atm_dx ** 2 * 1
        + atm_dy ** 2 * 1
        + 2 * atm_dspot * atm_dx * local_vol_zero * StochasticVol.rho_sx
        + 2 * atm_dspot * atm_dy * local_vol_zero * StochasticVol.rho_sy
        + 2 * atm_dx * atm_dy * StochasticVol.rho_xy
    ) / d_skew['atmvols']
    # atm_vvol[:min(nt // 10, 10)] = np.nan

    atm_dspot_tangent = (d_tangent['atmoptionsspottangent']) / vega_options(
        vols=d_skew['atmvols'], forwards_for_vega=d_skew['forwards'], strikes=d_skew['forwards'], t_values=ts,
    )
    atm_dx_tangent = (d_tangent['atmoptionsxtangent']) / vega_options(
        vols=d_skew['atmvols'], forwards_for_vega=d_skew['forwards'], strikes=d_skew['forwards'], t_values=ts,
    )
    atm_dy_tangent = (d_tangent['atmoptionsytangent']) / vega_options(
        vols=d_skew['atmvols'], forwards_for_vega=d_skew['forwards'], strikes=d_skew['forwards'], t_values=ts,
    )

    atm_ssr_tangent = (
            atm_dspot_tangent * local_vol_zero
            + atm_dx_tangent * StochasticVol.rho_sx
            + atm_dy_tangent * StochasticVol.rho_sy
    ) / local_vol_zero / atm_skew
    atm_ssr_tangent[:min(nt // 10, 10)] = np.nan

    atm_vvol_tangent = np.sqrt(
        atm_dspot_tangent ** 2 * local_vol_zero ** 2
        + atm_dx_tangent ** 2 * 1
        + atm_dy_tangent ** 2 * 1
        + 2 * atm_dspot_tangent * atm_dx_tangent * local_vol_zero * StochasticVol.rho_sx
        + 2 * atm_dspot_tangent * atm_dy_tangent * local_vol_zero * StochasticVol.rho_sy
        + 2 * atm_dx_tangent * atm_dy_tangent * StochasticVol.rho_xy
    ) / d_skew['atmvols']

    return dict(**d_skew, **d_ssr, **d_tangent, **dict(
        atmskew=atm_skew,
        atmssr=atm_ssr,
        atmvvol=atm_vvol,
        atmsensispot=atm_dspot,
        atmsensix=atm_dx,
        atmsensiy=atm_dy,
        atmsensispottangent=atm_dspot_tangent,
        atmsensixtangent=atm_dx_tangent,
        atmsensiytangent=atm_dy_tangent,
        atmvvoltangent=atm_vvol_tangent,
        atmssrtangent=atm_ssr_tangent,
    ))


def diffusion_local_vol():
    mc_args = dict(
        n_mc=100000,
        ts=np.concatenate((np.linspace(0, 0.02, 100 + 1), np.linspace(0.02, 1, 100 + 1)))
    )
    out = montecarlo_local_vol(**mc_args)
    print(out)
    t_range = mc_args['ts']
    plt.figure("Forward - LV")
    plt.plot(t_range, out['forwards'], label='Forward-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Forward")
    plt.title("MC Forward - LV")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("AtmVol - LV")
    plt.plot(t_range[1:], out['atmvols'], label='AtmVols-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("ATM Vols")
    plt.title("ATM Vols - LV")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("SkewTerminal - LV")
    plt.plot(t_range[1:], out['atmskew'], label='Skew-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Skew")
    plt.title("Terminal Skew - LV")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("SsrTerminal - LV")
    plt.plot(t_range[1:], out['atmssr'], label='SSR-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SkewStickinessRatio")
    plt.title("Terminal SSR - LV")
    plt.minorticks_on()
    plt.grid(visible=True)


def montecarlo_stochastic_vol(
    n_mc, ts,
):
    nt = len(ts) - 1
    dts = np.diff(ts)

    def diffuse_spots(spot_values, normal_values):
        for i in range(nt):
            sqrt_dt = np.sqrt(dts[i])
            spot_values[i + 1, 1, :] = \
                spot_values[i, 1, :] * (1 - StochasticVol.kx * dts[i]) + normal_values[i, :, 1] * sqrt_dt
            spot_values[i + 1, 2, :] = \
                spot_values[i, 2, :] * (1 - StochasticVol.ky * dts[i]) + normal_values[i, :, 2] * sqrt_dt
            x_tt = StochasticVol.x_tt(x_s=spot_values[i, 1, :], y_s=spot_values[i, 2, :])
            chi_tt = StochasticVol.chi_tt(t=ts[i])
            f_ttx = np.exp(StochasticVol.omega() * x_tt - 0.5 * StochasticVol.omega() ** 2 * chi_tt)
            volatility = sqrt_dt * np.sqrt(f_ttx) * StochasticVol.instantaneous_variance_swap_volatility()
            spot_values[i + 1, 0, :] = spot_values[i, 0, :] * np.exp(
                - 0.5 * volatility * volatility + volatility * normal_values[i, :, 0]
            )

    def diffuse_spots_tangent(spot_tangent_values, normal_values):
        for i in range(nt):
            sqrt_dt = np.sqrt(dts[i])
            spot_tangent_values[i + 1, 1, :] = \
                spot_tangent_values[i, 1, :] * (1 - StochasticVol.kx * dts[i]) + normal_values[i, :, 1] * sqrt_dt
            spot_tangent_values[i + 1, 2, :] = \
                spot_tangent_values[i, 2, :] * (1 - StochasticVol.ky * dts[i]) + normal_values[i, :, 2] * sqrt_dt
            x_tt = StochasticVol.x_tt(x_s=spot_tangent_values[i, 1, :], y_s=spot_tangent_values[i, 2, :])
            chi_tt = StochasticVol.chi_tt(t=ts[i])
            f_ttx = np.exp(StochasticVol.omega() * x_tt - 0.5 * StochasticVol.omega() ** 2 * chi_tt)
            volatility = sqrt_dt * np.sqrt(f_ttx) * StochasticVol.instantaneous_variance_swap_volatility()
            spot_tangent_values[i + 1, 0, :] = spot_tangent_values[i, 0, :] * np.exp(
                - 0.5 * volatility * volatility + volatility * normal_values[i, :, 0]
            )
            spot_tangent_values[i + 1, 3, :] = spot_tangent_values[i, 3, :] * np.exp(
                - 0.5 * volatility * volatility + volatility * normal_values[i, :, 0]
            )
            spot_tangent_values[i + 1, 4, :] = spot_tangent_values[i, 4, :] * np.exp(
                - 0.5 * volatility * volatility + volatility * normal_values[i, :, 0]
            )
            spot_tangent_values[i + 1, 5, :] = spot_tangent_values[i, 5, :] * np.exp(
                - 0.5 * volatility * volatility + volatility * normal_values[i, :, 0]
            )
            spot_tangent_values[i + 1, 4, :] += \
                StochasticVol.tangent_x_factor(t=ts[i]) * \
                spot_tangent_values[i, 0, :] * volatility * normal_values[i, :, 0]
            spot_tangent_values[i + 1, 5, :] += \
                StochasticVol.tangent_y_factor(t=ts[i]) * \
                spot_tangent_values[i, 0, :] * volatility * normal_values[i, :, 0]

    mean = np.zeros(3)
    covariance = np.array([
        [1, StochasticVol.rho_sx, StochasticVol.rho_sy],
        [StochasticVol.rho_sx, 1, StochasticVol.rho_xy],
        [StochasticVol.rho_sy, StochasticVol.rho_xy, 1],
    ])  # not doing the integrated version
    normals = np.random.multivariate_normal(mean=mean, cov=covariance, size=(nt, n_mc))

    def diffusion_skew():
        spots = np.zeros(shape=(nt + 1, 3, n_mc))
        spots[:, 0, :] = 1
        diffuse_spots(spot_values=spots, normal_values=normals)

        spots_variables = spots[:, 0, :]
        forwards = np.mean(spots_variables, axis=1)
        atm_vols = price_options(
            spots_for_price=spots_variables, forwards_for_price=forwards, strikes=forwards, t_values=ts,
        )
        strikes_plus = forwards * np.exp(epsilon_skew)
        strikes_minus = forwards * np.exp(-epsilon_skew)
        atm_plus_vols = price_options(
            spots_for_price=spots_variables, forwards_for_price=forwards, strikes=strikes_plus, t_values=ts,
        )
        atm_minus_vols = price_options(
            spots_for_price=spots_variables, forwards_for_price=forwards, strikes=strikes_minus, t_values=ts,
        )

        return dict(
            forwards=forwards,
            atmvols=atm_vols,
            atmplusvols=atm_plus_vols,
            atmminusvols=atm_minus_vols,
        )

    def diffusion_ssr():

        def diffuse_and_vols(index_bump):
            spots_ssr = np.zeros(shape=(nt + 1, 3, n_mc))
            spots_ssr[:, 0, :] = np.exp(local_vol_zero * epsilon_ssr) if index_bump == 0 else 1
            spots_ssr[:, 1, :] = epsilon_ssr if index_bump == 1 else 0
            spots_ssr[:, 2, :] = epsilon_ssr if index_bump == 2 else 0

            diffuse_spots(spot_values=spots_ssr, normal_values=normals)
            spots_variables_ssr = spots_ssr[:, 0, :]
            forwards_ssr = np.mean(spots_variables_ssr, axis=1)
            return price_options(
                spots_for_price=spots_variables_ssr, forwards_for_price=forwards_ssr, strikes=forwards_ssr, t_values=ts,
            )

        atm_vols_ssr_spot = diffuse_and_vols(index_bump=0)
        print('SV - SSR - Spot - Done')
        atm_vols_ssr_x = diffuse_and_vols(index_bump=1)
        print('SV - SSR - X - Done')
        atm_vols_ssr_y = diffuse_and_vols(index_bump=2)
        print('SV - SSR - Y - Done')

        return dict(
            atmvolsssrspot=atm_vols_ssr_spot,
            atmvolsssrx=atm_vols_ssr_x,
            atmvolsssry=atm_vols_ssr_y,
        )

    def diffusion_for_tangent():
        spots_tangent = np.zeros(shape=(nt + 1, 6, n_mc))
        spots_tangent[:, 0, :] = 1
        spots_tangent[:, 1, :] = 0
        spots_tangent[:, 2, :] = 0
        spots_tangent[:, 3, :] = 1
        spots_tangent[:, 4, :] = 0
        spots_tangent[:, 5, :] = 0

        diffuse_spots_tangent(spot_tangent_values=spots_tangent, normal_values=normals)

        forwards_spot = np.mean(spots_tangent[:, 0, :], axis=1)
        spot_values = spots_tangent[:, 0, :]

        tangent_price_spot = np.mean(
            (spot_values > forwards_spot[:, np.newaxis]) * (spots_tangent[:, 3, :] - spot_values)
            , axis=1)
        tangent_price_x = np.mean(
            (spot_values > forwards_spot[:, np.newaxis]) * spots_tangent[:, 4, :]
            , axis=1)
        tangent_price_y = np.mean(
            (spot_values > forwards_spot[:, np.newaxis]) * spots_tangent[:, 5, :]
            , axis=1)

        atm_options_spot = tangent_price_spot[1:]
        atm_options_x = tangent_price_x[1:]
        atm_options_y = tangent_price_y[1:]
        print('SV - Tangent - Done')

        return dict(
            atmoptionsspottangent=atm_options_spot,
            atmoptionsxtangent=atm_options_x,
            atmoptionsytangent=atm_options_y,
        )

    local_vol_zero = StochasticVol.local_volatility_zero()
    d_skew = diffusion_skew()
    d_ssr = diffusion_ssr()
    d_tangent = diffusion_for_tangent()

    atm_skew = (d_skew['atmplusvols'] - d_skew['atmminusvols']) / (2 * epsilon_skew)
    # atm_skew_extrapolation_index = min(nt // 10, 20)
    # atm_skew[:atm_skew_extrapolation_index] = atm_skew[atm_skew_extrapolation_index]

    atm_ssr = (
            (d_ssr['atmvolsssrspot'] - d_skew['atmvols'])
            + (d_ssr['atmvolsssrx'] - d_skew['atmvols']) * StochasticVol.rho_sx
            + (d_ssr['atmvolsssry'] - d_skew['atmvols']) * StochasticVol.rho_sy
    ) / epsilon_ssr / local_vol_zero / atm_skew
    atm_ssr[:min(nt // 10, 10)] = np.nan

    atm_dspot = (d_ssr['atmvolsssrspot'] - d_skew['atmvols']) / epsilon_ssr / local_vol_zero
    atm_dx = (d_ssr['atmvolsssrx'] - d_skew['atmvols']) / epsilon_ssr
    atm_dy = (d_ssr['atmvolsssry'] - d_skew['atmvols']) / epsilon_ssr
    atm_vvol = np.sqrt(
        atm_dspot ** 2 * local_vol_zero ** 2
        + atm_dx ** 2 * 1
        + atm_dy ** 2 * 1
        + 2 * atm_dspot * atm_dx * local_vol_zero * StochasticVol.rho_sx
        + 2 * atm_dspot * atm_dy * local_vol_zero * StochasticVol.rho_sy
        + 2 * atm_dx * atm_dy * StochasticVol.rho_xy
    ) / d_skew['atmvols']
    # atm_vvol[:min(nt // 10, 10)] = np.nan

    atm_dspot_tangent = (d_tangent['atmoptionsspottangent']) / vega_options(
        vols=d_skew['atmvols'], forwards_for_vega=d_skew['forwards'], strikes=d_skew['forwards'], t_values=ts,
    )
    atm_dx_tangent = (d_tangent['atmoptionsxtangent']) / vega_options(
        vols=d_skew['atmvols'], forwards_for_vega=d_skew['forwards'], strikes=d_skew['forwards'], t_values=ts,
    )
    atm_dy_tangent = (d_tangent['atmoptionsytangent']) / vega_options(
        vols=d_skew['atmvols'], forwards_for_vega=d_skew['forwards'], strikes=d_skew['forwards'], t_values=ts,
    )
    atm_ssr_tangent = (
            atm_dspot_tangent * local_vol_zero
            + atm_dx_tangent * StochasticVol.rho_sx
            + atm_dy_tangent * StochasticVol.rho_sy
    ) / local_vol_zero / atm_skew
    atm_ssr_tangent[:min(nt // 10, 10)] = np.nan
    atm_vvol_tangent = np.sqrt(
        atm_dspot_tangent ** 2 * local_vol_zero ** 2
        + atm_dx_tangent ** 2 * 1
        + atm_dy_tangent ** 2 * 1
        + 2 * atm_dspot_tangent * atm_dx_tangent * local_vol_zero * StochasticVol.rho_sx
        + 2 * atm_dspot_tangent * atm_dy_tangent * local_vol_zero * StochasticVol.rho_sy
        + 2 * atm_dx_tangent * atm_dy_tangent * StochasticVol.rho_xy
    ) / d_skew['atmvols']

    return dict(**d_skew, **d_ssr, **d_tangent, **dict(
        atmskew=atm_skew,
        atmssr=atm_ssr,
        atmvvol=atm_vvol,
        atmsensispot=atm_dspot,
        atmsensix=atm_dx,
        atmsensiy=atm_dy,
        atmsensispottangent=atm_dspot_tangent,
        atmsensixtangent=atm_dx_tangent,
        atmsensiytangent=atm_dy_tangent,
        atmvvoltangent=atm_vvol_tangent,
        atmssrtangent=atm_ssr_tangent,
    ))


def diffusion_stochastic_vol():
    mc_args = dict(
        n_mc=100000,
        ts=np.concatenate((np.linspace(0, 0.02, 100 + 1), np.linspace(0.02, 1, 100 + 1)))
    )
    out = montecarlo_stochastic_vol(**mc_args)
    print(out)

    t_range = mc_args['ts']
    plt.figure("Forward - SV")
    plt.plot(t_range, out['forwards'], label='Forward-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Forward")
    plt.title("MC Forward - SV")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("AtmVol - SV")
    plt.plot(t_range[1:], out['atmvols'], label='AtmVols-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("ATM Vols")
    plt.title("ATM Vols - SV")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("SkewTerminal - SV")
    plt.plot(t_range[1:], out['atmskew'], label='Skew-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Skew")
    plt.title("Terminal Skew - SV")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("SsrTerminal - SV")
    plt.plot(t_range[1:], out['atmssr'], label='SSR-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SkewStickinessRatio")
    plt.title("Terminal SSR - SV")
    plt.minorticks_on()
    plt.grid(visible=True)


def non_linear_regression(
    function_spot, function_xi, spot_apply,
):
    return np.interp(x=spot_apply, xp=function_spot, fp=function_xi)


def non_linear_regression_with_derivatives(
    function_spot, function_xi, spot_apply,
):
    """x * f'(x) / f(x)"""
    f = InterpolatedUnivariateSpline(x=function_spot, y=function_xi, k=1)
    df = f.derivative(n=1)
    f_values = f(spot_apply)
    df_values = df(spot_apply)
    return spot_apply * df_values / f_values


def example_regression():
    spots = np.array([0.1, 0.2, 0.4, 0.35, 0.5])
    xis = np.array([1.1, 1.2, 1.1, 1.35, 1.5])
    values = np.linspace(-0.2, 2, 1000)
    n_small = 30
    fx = np.zeros(n_small)
    fy = np.zeros(n_small)
    non_linear_regression_calibrate(
        particles_spot=spots, particles_xi=xis, n_small_value=n_small, f_x=fx, f_y=fy,
    )
    rep = non_linear_regression(function_spot=fx, function_xi=fy, spot_apply=values)
    plt.figure("RegressionExamples")
    plt.scatter(spots, xis, label="Particles", marker='X')
    plt.scatter(values, rep, label="Regressed", marker='.')
    plt.legend()
    plt.xlabel("Spot")
    plt.ylabel("Variance")
    plt.minorticks_on()
    plt.grid(visible=True)


def non_linear_regression_calibrate(particles_spot, particles_xi, n_small_value, f_x, f_y):
    spot_std = np.std(particles_spot)
    if spot_std < 1e-7:
        f_x[:] = np.linspace(0, 1, n_small_value)
        f_y[:] = np.ones(n_small_value)
        return
    h = spot_std ** -2 * len(particles_spot) * 1.2
    small_spot_apply = np.linspace(np.quantile(particles_spot, 0.01), np.quantile(particles_spot, 0.99), n_small_value)
    f_x[:] = small_spot_apply

    def function_filter(x):
        return x < 1

    def function_apply(x):
        return np.exp(-x)

    for i, small_spot in enumerate(small_spot_apply):
        line = (particles_spot[:] - small_spot) ** 2 * h
        selected = function_filter(line)
        top = np.sum(function_apply(x=line[selected]) * particles_xi[selected])
        bottom = np.sum(function_apply(x=line[selected]))
        f_y[i] = top / bottom
    return


def montecarlo_stochastic_local_vol(
    n_mc, ts,
):
    nt = len(ts) - 1
    dts = np.diff(ts)

    def diffuse_spots(spot_values, calibrated_values, normal_values):
        for i in range(nt):
            sqrt_dt = np.sqrt(dts[i])
            spot_values[i + 1, 1, :] = spot_values[i, 1, :] * (1 - StochasticVol.kx * dts[i]) \
                + normal_values[i, :, 1] * sqrt_dt
            spot_values[i + 1, 2, :] = spot_values[i, 2, :] * (1 - StochasticVol.ky * dts[i]) \
                + normal_values[i, :, 2] * sqrt_dt
            x_tt = StochasticVol.x_tt(x_s=spot_values[i, 1, :], y_s=spot_values[i, 2, :])
            chi_tt = StochasticVol.chi_tt(t=ts[i])
            f_ttx = np.exp(StochasticVol.omega() * x_tt - 0.5 * StochasticVol.omega() ** 2 * chi_tt)

            conditional_expectation = non_linear_regression(
                function_spot=calibrated_values[i, 0, :],
                function_xi=calibrated_values[i, 1, :],
                spot_apply=spot_values[i, 0, :],
            )  # not applying var_swap vol

            local_volatility = \
                LocalVol.local_volatility(t=ts[i], s=spot_values[i, 0, :]) / np.sqrt(conditional_expectation)

            augmented_volatility = sqrt_dt * local_volatility * np.sqrt(f_ttx)
            spot_values[i + 1, 0, :] = spot_values[i, 0, :] * np.exp(
                - 0.5 * augmented_volatility * augmented_volatility + augmented_volatility * normal_values[i, :, 0]
            )

    def diffuse_spots_tangent(spot_tangent_values, calibrated_values, normal_values):
        for i in range(nt):
            sqrt_dt = np.sqrt(dts[i])
            spot_tangent_values[i + 1, 1, :] = \
                spot_tangent_values[i, 1, :] * (1 - StochasticVol.kx * dts[i]) + normal_values[i, :, 1] * sqrt_dt
            spot_tangent_values[i + 1, 2, :] = \
                spot_tangent_values[i, 2, :] * (1 - StochasticVol.ky * dts[i]) + normal_values[i, :, 2] * sqrt_dt
            x_tt = StochasticVol.x_tt(x_s=spot_tangent_values[i, 1, :], y_s=spot_tangent_values[i, 2, :])
            chi_tt = StochasticVol.chi_tt(t=ts[i])
            f_ttx = np.exp(StochasticVol.omega() * x_tt - 0.5 * StochasticVol.omega() ** 2 * chi_tt)

            conditional_expectation = non_linear_regression(
                function_spot=calibrated_values[i, 0, :],
                function_xi=calibrated_values[i, 1, :],
                spot_apply=spot_tangent_values[i, 0, :],
            )  # not applying var_swap vol
            conditional_expectation_with_derivatives = non_linear_regression_with_derivatives(
                function_spot=calibrated_values[i, 0, :],
                function_xi=calibrated_values[i, 1, :],
                spot_apply=spot_tangent_values[i, 0, :],
            )  # not applying var_swap vol

            local_volatility_lv = LocalVol.local_volatility(t=ts[i], s=spot_tangent_values[i, 0, :])
            local_volatility = local_volatility_lv / np.sqrt(conditional_expectation)
            augmented_volatility = sqrt_dt * local_volatility * np.sqrt(f_ttx)

            spot_tangent_values[i + 1, 0, :] = spot_tangent_values[i, 0, :] * np.exp(
                - 0.5 * augmented_volatility * augmented_volatility + augmented_volatility * normal_values[i, :, 0]
            )
            local_volatility_with_derivatives = \
                (
                        LocalVol.local_volatility(t=ts[i], s=spot_tangent_values[i, 0, :], add_derivatives=True)
                        - 0.5 * conditional_expectation_with_derivatives * local_volatility_lv
                ) / np.sqrt(conditional_expectation)
            augmented_volatility_with_derivatives = sqrt_dt * local_volatility_with_derivatives * np.sqrt(f_ttx)

            spot_tangent_values[i + 1, 3, :] = spot_tangent_values[i, 3, :] * np.exp(
                - 0.5 * augmented_volatility_with_derivatives * augmented_volatility_with_derivatives
                + augmented_volatility_with_derivatives * normal_values[i, :, 0]
            )
            spot_tangent_values[i + 1, 4, :] = spot_tangent_values[i, 4, :] * np.exp(
                - 0.5 * augmented_volatility_with_derivatives * augmented_volatility_with_derivatives
                + augmented_volatility_with_derivatives * normal_values[i, :, 0]
            )
            spot_tangent_values[i + 1, 5, :] = spot_tangent_values[i, 5, :] * np.exp(
                - 0.5 * augmented_volatility_with_derivatives * augmented_volatility_with_derivatives
                + augmented_volatility_with_derivatives * normal_values[i, :, 0]
            )

            spot_tangent_values[i + 1, 4, :] += \
                StochasticVol.tangent_x_factor(t=ts[i]) * \
                spot_tangent_values[i, 0, :] * augmented_volatility * normal_values[i, :, 0]
            spot_tangent_values[i + 1, 5, :] += \
                StochasticVol.tangent_y_factor(t=ts[i]) * \
                spot_tangent_values[i, 0, :] * augmented_volatility * normal_values[i, :, 0]

    def calibration(normal_values):
        n_small = 30

        calibrated_values = np.zeros(shape=(nt, 2, n_small))
        particles = np.zeros(shape=(3, n_mc))
        particles[0, :] = 1
        for i in range(nt):
            x_tt = StochasticVol.x_tt(x_s=particles[1, :], y_s=particles[2, :])
            chi_tt = StochasticVol.chi_tt(t=ts[i])
            f_ttx = np.exp(StochasticVol.omega() * x_tt - 0.5 * StochasticVol.omega() ** 2 * chi_tt)

            non_linear_regression_calibrate(
                particles_spot=particles[0, :],
                particles_xi=f_ttx,
                n_small_value=n_small,
                f_x=calibrated_values[i, 0, :],
                f_y=calibrated_values[i, 1, :],
            )  # not applying var_swap vol
            if i == 0:
                print("CondExpCalibration", end=' ')
            print(i, end=' ')
            if i % 50 == 0:
                print()

            conditional_expectation = non_linear_regression(
                function_spot=calibrated_values[i, 0, :],
                function_xi=calibrated_values[i, 1, :],
                spot_apply=particles[0, :],
            )  # not applying var_swap vol

            local_volatility = \
                LocalVol.local_volatility(t=ts[i], s=particles[0, :]) / np.sqrt(conditional_expectation)

            sqrt_dt = np.sqrt(dts[i])

            augmented_volatility = sqrt_dt * local_volatility * np.sqrt(f_ttx)
            particles[0, :] = particles[0, :] * np.exp(
                - 0.5 * augmented_volatility * augmented_volatility + augmented_volatility * normal_values[i, :, 0]
            )

            particles[1, :] = particles[1, :] * (1 - StochasticVol.kx * dts[i]) + normal_values[i, :, 1] * sqrt_dt
            particles[2, :] = particles[2, :] * (1 - StochasticVol.ky * dts[i]) + normal_values[i, :, 2] * sqrt_dt
        return calibrated_values

    mean = np.zeros(3)
    covariance = np.array([
        [1, StochasticVol.rho_sx, StochasticVol.rho_sy],
        [StochasticVol.rho_sx, 1, StochasticVol.rho_xy],
        [StochasticVol.rho_sy, StochasticVol.rho_xy, 1],
    ])  # not doing the integrated version
    normals = np.random.multivariate_normal(mean=mean, cov=covariance, size=(nt, n_mc))

    calibrated_function = calibration(normal_values=normals)

    def diffusion_skew():
        spots = np.zeros(shape=(nt + 1, 3, n_mc))
        spots[:, 0, :] = 1
        diffuse_spots(spot_values=spots, calibrated_values=calibrated_function, normal_values=normals)

        spots_variables = spots[:, 0, :]
        forwards = np.mean(spots_variables, axis=1)
        atm_vols = price_options(
            spots_for_price=spots_variables, forwards_for_price=forwards, strikes=forwards, t_values=ts,
        )
        strikes_plus = forwards * np.exp(epsilon_skew)
        strikes_minus = forwards * np.exp(-epsilon_skew)
        atm_plus_vols = price_options(
            spots_for_price=spots_variables, forwards_for_price=forwards, strikes=strikes_plus, t_values=ts,
        )
        atm_minus_vols = price_options(
            spots_for_price=spots_variables, forwards_for_price=forwards, strikes=strikes_minus, t_values=ts,
        )
        return dict(
            forwards=forwards,
            atmvols=atm_vols,
            atmplusvols=atm_plus_vols,
            atmminusvols=atm_minus_vols,
        )

    def diffusion_ssr():

        def diffuse_and_vols(index_bump):
            spots_ssr = np.zeros(shape=(nt + 1, 3, n_mc))
            spots_ssr[:, 0, :] = np.exp(local_vol_zero * epsilon_ssr) if index_bump == 0 else 1
            spots_ssr[:, 1, :] = epsilon_ssr if index_bump == 1 else 0
            spots_ssr[:, 2, :] = epsilon_ssr if index_bump == 2 else 0

            diffuse_spots(spot_values=spots_ssr, calibrated_values=calibrated_function, normal_values=normals)
            spots_variables_ssr = spots_ssr[:, 0, :]
            forwards_ssr = np.mean(spots_variables_ssr, axis=1)
            return price_options(
                spots_for_price=spots_variables_ssr, forwards_for_price=forwards_ssr, strikes=forwards_ssr, t_values=ts,
            )

        atm_vols_ssr_spot = diffuse_and_vols(index_bump=0)
        print('SLV - SSR - Spot - Done')
        atm_vols_ssr_x = diffuse_and_vols(index_bump=1)
        print('SLV - SSR - X - Done')
        atm_vols_ssr_y = diffuse_and_vols(index_bump=2)
        print('SLV - SSR - Y - Done')

        return dict(
            atmvolsssrspot=atm_vols_ssr_spot,
            atmvolsssrx=atm_vols_ssr_x,
            atmvolsssry=atm_vols_ssr_y,
        )

    def diffusion_for_tangent():
        spots_tangent = np.zeros(shape=(nt + 1, 6, n_mc))
        spots_tangent[:, 0, :] = 1
        spots_tangent[:, 1, :] = 0
        spots_tangent[:, 2, :] = 0
        spots_tangent[:, 3, :] = 1
        spots_tangent[:, 4, :] = 0
        spots_tangent[:, 5, :] = 0

        diffuse_spots_tangent(
            spot_tangent_values=spots_tangent, calibrated_values=calibrated_function, normal_values=normals,
        )

        forwards_spot = np.mean(spots_tangent[:, 0, :], axis=1)
        spot_values = spots_tangent[:, 0, :]

        tangent_price_spot = np.mean(
            (spot_values > forwards_spot[:, np.newaxis]) * (spots_tangent[:, 3, :] - spot_values)
            , axis=1)
        tangent_price_x = np.mean(
            (spot_values > forwards_spot[:, np.newaxis]) * spots_tangent[:, 4, :]
            , axis=1)
        tangent_price_y = np.mean(
            (spot_values > forwards_spot[:, np.newaxis]) * spots_tangent[:, 5, :]
            , axis=1)

        atm_options_spot = tangent_price_spot[1:]
        atm_options_x = tangent_price_x[1:]
        atm_options_y = tangent_price_y[1:]
        print('SLV - Tangent - Done')

        return dict(
            atmoptionsspottangent=atm_options_spot,
            atmoptionsxtangent=atm_options_x,
            atmoptionsytangent=atm_options_y,
        )

    d_skew = diffusion_skew()
    atm_skew = (d_skew['atmplusvols'] - d_skew['atmminusvols']) / (2 * epsilon_skew)
    # atm_skew_extrapolation_index = min(nt // 10, 20)
    # atm_skew[:atm_skew_extrapolation_index] = atm_skew[atm_skew_extrapolation_index]

    local_vol_zero = StochasticLocalVol.local_volatility_zero()
    d_ssr = diffusion_ssr()

    atm_ssr = (
            (d_ssr['atmvolsssrspot'] - d_skew['atmvols'])
            + (d_ssr['atmvolsssrx'] - d_skew['atmvols']) * StochasticVol.rho_sx
            + (d_ssr['atmvolsssry'] - d_skew['atmvols']) * StochasticVol.rho_sy
    ) / epsilon_ssr / local_vol_zero / atm_skew
    atm_ssr[:min(nt // 10, 10)] = np.nan

    atm_dspot = (d_ssr['atmvolsssrspot'] - d_skew['atmvols']) / epsilon_ssr / local_vol_zero
    atm_dx = (d_ssr['atmvolsssrx'] - d_skew['atmvols']) / epsilon_ssr
    atm_dy = (d_ssr['atmvolsssry'] - d_skew['atmvols']) / epsilon_ssr
    atm_vvol = np.sqrt(
        atm_dspot ** 2 * local_vol_zero ** 2
        + atm_dx ** 2 * 1
        + atm_dy ** 2 * 1
        + 2 * atm_dspot * atm_dx * local_vol_zero * StochasticVol.rho_sx
        + 2 * atm_dspot * atm_dy * local_vol_zero * StochasticVol.rho_sy
        + 2 * atm_dx * atm_dy * StochasticVol.rho_xy
    ) / d_skew['atmvols']
    # atm_vvol[:min(nt // 10, 10)] = np.nan

    d_tangent = diffusion_for_tangent()
    atm_dspot_tangent = (d_tangent['atmoptionsspottangent']) / vega_options(
        vols=d_skew['atmvols'], forwards_for_vega=d_skew['forwards'], strikes=d_skew['forwards'], t_values=ts,
    )
    atm_dx_tangent = (d_tangent['atmoptionsxtangent']) / vega_options(
        vols=d_skew['atmvols'], forwards_for_vega=d_skew['forwards'], strikes=d_skew['forwards'], t_values=ts,
    )
    atm_dy_tangent = (d_tangent['atmoptionsytangent']) / vega_options(
        vols=d_skew['atmvols'], forwards_for_vega=d_skew['forwards'], strikes=d_skew['forwards'], t_values=ts,
    )
    atm_ssr_tangent = (
            atm_dspot_tangent * local_vol_zero
            + atm_dx_tangent * StochasticVol.rho_sx
            + atm_dy_tangent * StochasticVol.rho_sy
    ) / local_vol_zero / atm_skew
    atm_ssr_tangent[:min(nt // 10, 10)] = np.nan
    atm_vvol_tangent = np.sqrt(
        atm_dspot_tangent ** 2 * local_vol_zero ** 2
        + atm_dx_tangent ** 2 * 1
        + atm_dy_tangent ** 2 * 1
        + 2 * atm_dspot_tangent * atm_dx_tangent * local_vol_zero * StochasticVol.rho_sx
        + 2 * atm_dspot_tangent * atm_dy_tangent * local_vol_zero * StochasticVol.rho_sy
        + 2 * atm_dx_tangent * atm_dy_tangent * StochasticVol.rho_xy
    ) / d_skew['atmvols']

    return dict(**d_skew, **d_ssr, **d_tangent, **dict(
        atmskew=atm_skew,
        atmssr=atm_ssr,
        atmvvol=atm_vvol,
        atmsensispot=atm_dspot,
        atmsensix=atm_dx,
        atmsensiy=atm_dy,
        atmsensispottangent=atm_dspot_tangent,
        atmsensixtangent=atm_dx_tangent,
        atmsensiytangent=atm_dy_tangent,
        atmvvoltangent=atm_vvol_tangent,
        atmssrtangent=atm_ssr_tangent,
    ))


def diffusion_stochastic_local_vol():
    mc_args = dict(
        n_mc=1000000,
        ts=np.concatenate((np.linspace(0, 0.02, 100 + 1), np.linspace(0.02, 1, 100 + 1)))
    )
    out = montecarlo_stochastic_local_vol(**mc_args)
    print(out)

    t_range = mc_args['ts']
    plt.figure("Forward - SLV")
    plt.plot(t_range, out['forwards'], label='Forward-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Forward")
    plt.title("MC Forward - SLV")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("AtmVol - SLV")
    plt.plot(t_range[1:], out['atmvols'], label='AtmVols-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("ATM Vols")
    plt.title("ATM Vols - SLV")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("SkewTerminal - SLV")
    plt.plot(t_range[1:], out['atmskew'], label='Skew-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Skew")
    plt.title("Terminal Skew - SLV")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.figure("SsrTerminal - SLV")
    plt.plot(t_range[1:], out['atmssr'], label='SSR-MC')
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SkewStickinessRatio")
    plt.title("Terminal SSR - SLV")
    plt.minorticks_on()
    plt.grid(visible=True)


def diffusion_all():
    mc_args = dict(
        n_mc=1000000,
        ts=np.concatenate((np.linspace(0, 0.1, 100 + 1), np.linspace(0.1, 1, 100 + 1)))
    )
    out_lv = montecarlo_local_vol(**mc_args)
    print("LV - Done")
    out_sv = montecarlo_stochastic_vol(**mc_args)
    print("SV - Done")
    out_slv = montecarlo_stochastic_local_vol(**mc_args)
    print("SLV - Done")
    
    color_lv = 'tab:blue'
    color_sv = 'tab:orange'
    color_slv = 'tab:green'

    t_range = mc_args['ts']
    plt.figure("MC - Forward")
    plt.plot(t_range, out_lv['forwards'], label='Forward-LV', color=color_lv)
    plt.plot(t_range, out_sv['forwards'], label='Forward-SV', color=color_sv)
    plt.plot(t_range, out_slv['forwards'], label='Forward-SLV', color=color_slv)
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Forward")
    plt.title("MC - Forward")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.savefig('forward.png')
    plt.figure("MC - AtmVol")
    plt.plot(t_range[1:], out_lv['atmvols'], label='AtmVols-LV', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmvols'], label='AtmVols-SV', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmvols'], label='AtmVols-SLV', color=color_slv)
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("ATM Vols")
    plt.title("MC - ATM Vols")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.savefig('atmvol.png')
    plt.figure("MC - Skew")
    plt.plot(t_range[1:], out_lv['atmskew'], label='Skew-LV', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmskew'], label='Skew-SV', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmskew'], label='Skew-SLV', color=color_slv)
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Skew")
    plt.title("MC - Skew")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.savefig('skew.png')

    plt.figure("MC - SensiSpot")
    plt.plot(t_range[1:], out_lv['atmsensispot'], label='SensiSpot-LV', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmsensispot'], label='SensiSpot-SV', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmsensispot'], label='SensiSpot-SLV', color=color_slv)
    plt.plot(
        t_range[1:], out_lv['atmsensispottangent'], label='SensiSpot-LV-Tangent', linestyle='dashed', color=color_lv,
    )
    plt.plot(
        t_range[1:], out_sv['atmsensispottangent'], label='SensiSpot-SV-Tangent', linestyle='dashed', color=color_sv,
    )
    plt.plot(
        t_range[1:], out_slv['atmsensispottangent'], label='SensiSpot-SLV-Tangent', linestyle='dashed', color=color_slv,
    )
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SensiSpot")
    plt.title("MC - SensiSpot")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.savefig('sensispot.png')
    plt.figure("MC - SensiX")
    plt.plot(t_range[1:], out_lv['atmsensix'], label='SensiX-LV', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmsensix'], label='SensiX-SV', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmsensix'], label='SensiX-SLV', color=color_slv)
    plt.plot(t_range[1:], out_lv['atmsensixtangent'], label='SensiX-LV-Tangent', linestyle='dashed', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmsensixtangent'], label='SensiX-SV-Tangent', linestyle='dashed', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmsensixtangent'], label='SensiX-SLV-Tangent', linestyle='dashed', color=color_slv)
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SensiX")
    plt.title("MC - SensiX")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.savefig('sensix.png')
    plt.figure("MC - SensiY")
    plt.plot(t_range[1:], out_lv['atmsensiy'], label='SensiY-LV', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmsensiy'], label='SensiY-SV', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmsensiy'], label='SensiY-SLV', color=color_slv)
    plt.plot(t_range[1:], out_lv['atmsensiytangent'], label='SensiY-LV-Tangent', linestyle='dashed', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmsensiytangent'], label='SensiY-SV-Tangent', linestyle='dashed', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmsensiytangent'], label='SensiY-SLV-Tangent', linestyle='dashed', color=color_slv)
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SensiY")
    plt.title("MC - SensiY")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.savefig('sensiy.png')

    plt.figure("MC - SSR")
    plt.plot(t_range[1:], out_lv['atmssr'], label='SSR-LV', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmssr'], label='SSR-SV', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmssr'], label='SSR-SLV', color=color_slv)
    plt.plot(
        t_range[1:], np.clip(out_lv['atmssrtangent'], a_min=1, a_max=4), label='SSR-LV-Tangent', linestyle='dashed',
        color=color_lv,
    )
    plt.plot(
        t_range[1:], np.clip(out_sv['atmssrtangent'], a_min=1, a_max=4), label='SSR-SV-Tangent', linestyle='dashed',
        color=color_sv,
    )
    plt.plot(
        t_range[1:], np.clip(out_slv['atmssrtangent'], a_min=1, a_max=4), label='SSR-SLV-Tangent', linestyle='dashed',
        color=color_slv,
    )
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("SkewStickinessRatio")
    plt.title("MC - SSR")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.savefig('ssr.png')

    plt.figure("MC - VolVol")
    plt.plot(t_range[1:], out_lv['atmvvol'], label='Vvol-LV', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmvvol'], label='Vvol-SV', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmvvol'], label='Vvol-SLV', color=color_slv)
    plt.plot(t_range[1:], out_lv['atmvvoltangent'], label='Vvol-LV-Tangent', linestyle='dashed', color=color_lv)
    plt.plot(t_range[1:], out_sv['atmvvoltangent'], label='Vvol-SV-Tangent', linestyle='dashed', color=color_sv)
    plt.plot(t_range[1:], out_slv['atmvvoltangent'], label='Vvol-SLV-Tangent', linestyle='dashed', color=color_slv)
    plt.legend()
    plt.xlabel("Maturity")
    plt.ylabel("Vol of Vol")
    plt.title("MC - VolVol")
    plt.minorticks_on()
    plt.grid(visible=True)
    plt.savefig('vvol.png')


def main():
    # diffusion_local_vol()
    # diffusion_stochastic_vol()
    # diffusion_stochastic_local_vol()
    diffusion_all()
    # example_regression()
    plt.show()


if __name__ == '__main__':
    main()
