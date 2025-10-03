"""
Consistency tests for DISS.py from python 2.7 to 3.9
"""
from os import path
import pytest
import numpy as np
import DISS

PY2_DISS_DATA = np.load(path.join(path.dirname(__file__),
                                     "py27_DISS_output.npz"))
J1744_DATA = np.load(path.join(path.dirname(__file__),
                                  "J1744-1134_dict_py27.dict.npy"),
                     allow_pickle=True,
                     encoding="latin1")
J1744_DICT = J1744_DATA.item()

def test_scale_dt_d():
    nus = PY2_DISS_DATA["nus"]
    py3_dtd = DISS.scale_dt_d(J1744_DICT["dtd"], 1.0, nus)
    np.testing.assert_allclose(py3_dtd, PY2_DISS_DATA["dtd"])

def test_scale_dnu_d():
    nus = PY2_DISS_DATA["nus"]
    py3_dnud = DISS.scale_dnu_d(J1744_DICT["dnud"], 1.0, nus)
    np.testing.assert_allclose(py3_dnud, PY2_DISS_DATA["dnud"])

def test_scale_tau_d():
    nus = PY2_DISS_DATA["nus"]
    py3_taud = DISS.scale_tau_d(J1744_DICT["taud"], 1.0, nus)
    np.testing.assert_allclose(py3_taud, PY2_DISS_DATA["taud"])

def test_scale_dt_r():
    """
    never used in frequencyoptimizer.py
    """
    nus = PY2_DISS_DATA["nus"]
    py3_dtr = DISS.scale_dt_r(J1744_DICT["taud"], 1.0, nus)
    np.testing.assert_allclose(py3_dtr, PY2_DISS_DATA["dtr"])

def test_niss():
    """
    never used in frequencyoptimizer.py
    """
    nus = PY2_DISS_DATA["nus"]
    py3_niss = DISS.nISS(nus[-1] - nus[0],
                         J1744_DICT["dnud"],
                         J1744_DICT["T"],
                         J1744_DICT["dtd"])
    np.testing.assert_allclose(py3_niss, PY2_DISS_DATA["nISS"])

def test_sigma_diss():
    """
    never used in frequencyoptimizer.py
    """
    nus = PY2_DISS_DATA["nus"]
    py3_niss = DISS.nISS(nus[-1] - nus[0],
                         J1744_DICT["dnud"],
                         J1744_DICT["T"],
                         J1744_DICT["dtd"])
    py3_sigma_diss = DISS.sigma_DISS(J1744_DICT["taud"], py3_niss)
    np.testing.assert_allclose(py3_sigma_diss, PY2_DISS_DATA["sigma_DISS"])
