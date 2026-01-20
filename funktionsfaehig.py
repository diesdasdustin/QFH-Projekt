import numpy as np
from PIL import Image
import rasterio
from rasterio.control import GroundControlPoint
from sgp4.api import Satrec
#from sgp4.conveniences import teme_to_ecef
from pyproj import Transformer
from math import sin, cos, radians, sqrt

from astropy.coordinates import TEME, ITRS
from astropy.time import Time
import astropy.units as u
# ============================================================
# CONFIG
# ============================================================

PNG_PATH = "msu_mr_rgb_MCIR.png"
TIME_PATH = "tim.txt"
OUTPUT_TIF = "neuneunetest.tif"

TLE1 = "1 57166U 23091A   26004.31624314  .00000055  00000+0  42710-4 0  9993"
TLE2 = "2 57166  98.6344  62.4577 0002906 292.2271  67.8599 14.24029359131192"

SCAN_ANGLE_TOTAL = 110.1
ROLL = radians(-0.31)
PITCH = radians(-3.1)
YAW = radians(0.75)

GCP_DX = 100
GCP_DY = 100

# ============================================================
# LOAD IMAGE + TIME
# ============================================================

img = Image.open(PNG_PATH).convert("RGB")
img_np = np.asarray(img)
n_lines, n_pixels, _ = img_np.shape

timestamps = np.loadtxt(TIME_PATH)  # UNIX UTC seconds per line

# ============================================================
# SATELLITE + TRANSFORMS
# ============================================================

sat = Satrec.twoline2rv(TLE1, TLE2)
ecef_to_llh = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

# ============================================================
# EARTH (WGS84)
# ============================================================

a = 6378137.0
b = 6356752.314245

def ray_ellipsoid_intersection(p, d):
    x, y, z = p
    dx, dy, dz = d
    A = (dx*dx + dy*dy)/a**2 + (dz*dz)/b**2
    B = 2*((x*dx + y*dy)/a**2 + (z*dz)/b**2)
    C = (x*x + y*y)/a**2 + (z*z)/b**2 - 1
    D = B*B - 4*A*C
    if D <= 0:
        return None
    t = (-B - sqrt(D)) / (2*A)
    return p + t*d

# ============================================================
# ROTATIONS
# ============================================================

def Rx(a):
    return np.array([[1,0,0],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]])

def Ry(a):
    return np.array([[cos(a),0,sin(a)],[0,1,0],[-sin(a),0,cos(a)]])

def Rz(a):
    return np.array([[cos(a),-sin(a),0],[sin(a),cos(a),0],[0,0,1]])

ATT = Rx(ROLL) @ Ry(PITCH) @ Rz(YAW)

# ============================================================
# SCAN ANGLES
# ============================================================

half = radians(SCAN_ANGLE_TOTAL / 2)
scan_angles = np.linspace(-half, half, n_pixels)

# ============================================================
# GCP GENERATION
# ============================================================
def teme_to_ecef(r, v, jd):
    """
    r : position TEME [km]
    v : velocity TEME [km/s]
    jd: Julian Date (float)
    """
    t = Time(jd, format="jd", scale="utc")

    teme = TEME(
        x=r[0] * u.km, y=r[1] * u.km, z=r[2] * u.km,
        v_x=v[0] * u.km / u.s, v_y=v[1] * u.km / u.s, v_z=v[2] * u.km / u.s,
        obstime=t
    )

    itrs = teme.transform_to(ITRS(obstime=t))

    pos_ecef = np.array([
        itrs.x.to(u.km).value,
        itrs.y.to(u.km).value,
        itrs.z.to(u.km).value
    ])

    vel_ecef = np.array([
        itrs.v_x.to(u.km / u.s).value,
        itrs.v_y.to(u.km / u.s).value,
        itrs.v_z.to(u.km / u.s).value
    ])

    return pos_ecef, vel_ecef


gcps = []

for row in range(0, n_lines, GCP_DY):

    t = timestamps[row]
    jd = 2440587.5 + t / 86400.0

    err, r_teme, v_teme = sat.sgp4(jd, 0.0)
    if err != 0:
        continue

    r_ecef, v_ecef = teme_to_ecef(r_teme, v_teme, jd)
    pos = np.array(r_ecef) * 1000.0
    vel = np.array(v_ecef) * 1000.0

    # --- BODY FRAME (CORRECT ORIENTATION) ---
    z_b = -pos / np.linalg.norm(pos)        # Nadir
    x_b = vel / np.linalg.norm(vel)         # Along-track
    y_b = np.cross(x_b, z_b)                # Cross-track (FIXED)
    y_b /= np.linalg.norm(y_b)

    for col in range(0, n_pixels, GCP_DX):

        a_scan = scan_angles[col]
        d = cos(a_scan)*z_b + sin(a_scan)*y_b
        d = ATT @ d

        hit = ray_ellipsoid_intersection(pos, d)
        if hit is None:
            continue

        lon, lat, _ = ecef_to_llh.transform(*hit)

        gcps.append(GroundControlPoint(
            row=row,
            col=col,
            x=lon,
            y=lat,
            z=0
        ))

print("GCP count:", len(gcps))
print(gcps)
# ============================================================
# WRITE TIFF
# ============================================================

with rasterio.open(
    OUTPUT_TIF,
    "w",
    driver="GTiff",
    height=n_lines,
    width=n_pixels,
    count=3,
    dtype=img_np.dtype,
    crs="EPSG:4326",
    gcps=gcps
) as dst:
    for i in range(3):
        dst.write(img_np[:, :, i], i + 1)

print("Written:", OUTPUT_TIF)
