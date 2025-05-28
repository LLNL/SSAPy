import pytest
import numpy as np
from astropy.time import Time
from unittest.mock import patch, MagicMock

from ssapy.simple import (
    keplerian_prop,
    threebody_prop,
    fourbody_prop,
    best_prop,
    ssapy_kwargs,
    ssapy_prop,
    ssapy_orbit,
    get_similar_orbits,
)

 
def test_ssapy_kwargs():
    kwargs = ssapy_kwargs(100, 0.01, 2.1, 1.2)
    assert kwargs == {'mass': 100, 'area': 0.01, 'CD': 2.1, 'CR': 1.2}

 
def test_ssapy_orbit_errors():
    with pytest.raises(ValueError):
        ssapy_orbit()

    with pytest.raises(ValueError):
        ssapy_orbit(a=None, r=None, v=None)
