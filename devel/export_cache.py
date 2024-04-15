import numpy as np
import astropy.units as u
from astropy.time import Time
import astropy.utils.data as aud

print("Current cached urls:")
print(aud.get_cached_urls())

# aud.clear_download_cache()
print(Time.now().delta_ut1_utc)  # requires finals2000A.all
Time("J2018") + np.linspace(-40, 0, 1000)*u.year

print("New cached urls:")
print(aud.get_cached_urls())
aud.export_download_cache("cache.zip")
