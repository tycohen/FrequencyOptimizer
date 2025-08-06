"""
Consistency tests for frequencyoptimizer.py from python 2.7 to 3.9
"""

import pytest
import numpy as np
import frequencyoptimizer as fop

J1744_DATA = np.load("J1744-1134_dict_py27.dict.npy",
                     allow_pickle=True,
                     encoding="latin1")
J1744_DICT = J1744_DATA.item()

@pytest.mark.parametrize("vverbose", [True, False])
def test_calc_single(vverbose):
    nus = np.linspace(1., 10., 10)
    nchan = len(nus)
    galnoise = fop.GalacticNoise()
    telnoise = fop.TelescopeNoise(gain=J1744_DICT["gain"],
                                  T_rx=J1744_DICT["T_rx"])

    psrnoise = fop.PulsarNoise("J1744-1134",
                               alpha=J1744_DICT["alpha"],
                               taud=J1744_DICT["taud"],
                               I_0=J1744_DICT["I_0"],
                               DM=J1744_DICT["DM"],
                               D=J1744_DICT["D"],
                               tauvar=J1744_DICT["tauvar"],
                               dtd=J1744_DICT["dtd"],
                               Weffs=J1744_DICT["Weffs"],
                               W50s=J1744_DICT["W50s"],
                               sigma_Js=J1744_DICT["sigma_Js"],
                               P=J1744_DICT["P"],
                               Uscale=J1744_DICT["Uscale"])

    freqopt = fop.FrequencyOptimizer(psrnoise,
                                     galnoise,
                                     telnoise,
                                     numin=0.1,
                                     numax=10.0,
                                     nchan=nchan,
                                     log=True,
                                     vverbose=vverbose)

    sigma_tot, sigma_w, sigma_dm, sigma_tel, sigma_sn = freqopt.calc_single(nus)
    np.testing.assert_almost_equal(sigma_tot, J1744_DICT["sigma_tot"])
    np.testing.assert_almost_equal(sigma_w, J1744_DICT["sigma_w"])
    np.testing.assert_almost_equal(sigma_dm, J1744_DICT["sigma_dm"])
    np.testing.assert_almost_equal(sigma_tel, J1744_DICT["sigma_tel"])
    np.testing.assert_almost_equal(sigma_sn, J1744_DICT["sigma_sn"])
