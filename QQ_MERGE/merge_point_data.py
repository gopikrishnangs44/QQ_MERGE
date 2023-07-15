from statsmodels.distributions.empirical_distribution import ECDF
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import xarray as xr
import pandas as pd

def merge_point_data(C_OLD,C_NEW,OLD,NEW):
    f = ECDF(C_NEW) #CDF #newer instrument
    h = f(OLD)*100 #associated_CDF #correction dataset
    ciiold = np.nanpercentile(C_OLD,h)
    ciinew = np.nanpercentile(C_NEW,h)
    deltaii = ciinew-ciiold
    deltamed = np.nanmedian(C_NEW) - np.nanmedian(C_OLD)
    deltaiidash = deltaii-deltamed
    gdash = np.nanmedian(C_NEW) / np.nanmedian(C_OLD)
    q3, q1 = np.nanpercentile(C_OLD, [75 ,25])
    iqrold = q3 - q1   
    q3, q1 = np.nanpercentile(C_NEW, [75 ,25])
    iqrnew = q3 - q1
    fdash = iqrnew/iqrold
    TCO = (gdash*deltamed)+(fdash*deltaiidash)
    proj = OLD + TCO
    pp = xr.concat([proj,NEW], dim='time')
    return pp

def merged_time_series(OLD,NEW):
    common_dates = np.intersect1d(NEW.time.values,OLD.time.values)
    dates = pd.to_datetime(common_dates)
    C_NEW = NEW.sel(time=dates)
    C_OLD = OLD.sel(time=dates)
    OLD_int = OLD.sel(time=~OLD['time'].isin(dates))
    result = merge_point_data(C_OLD, C_NEW, OLD_int, NEW)
    return result
