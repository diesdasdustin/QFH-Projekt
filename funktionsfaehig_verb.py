import numpy as np
from PIL import Image
import rasterio
from rasterio.control import GroundControlPoint
from sgp4.api import Satrec
from pyproj import Transformer
from math import sin, cos, radians, sqrt, pi, acos, exp

from astropy.coordinates import TEME, ITRS
from astropy.time import Time
import astropy.units as u

# ============================================================
# CONFIG
# ============================================================

PNG_PATH = "msu_mr_rgb_MCIR.png"
TIME_PATH = "tim.txt"
OUTPUT_TIF = "funktionsfaehig_verb_out.tif"

TLE1 = "1 57166U 23091A   26004.31624314  .00000055  00000+0  42710-4 0  9993"
TLE2 = "2 57166  98.6344  62.4577 0002906 292.2271  67.8599 14.24029359131192"

SCAN_ANGLE_TOTAL = radians(110.1)
SCAN_TIME = 1.6

ROLL0  = radians(-0.31)
PITCH0 = radians(-3.1)
YAW0   = radians(0.75)

ROLL1  = radians(0.05)
PITCH1 = radians(0.08)

GCP_DX = 100
GCP_DY = 100

# ============================================================
# LOAD IMAGE + TIME
# ============================================================

img = Image.open(PNG_PATH).convert("RGB")
img_np = np.asarray(img)

n_lines, n_pixels, _ = img_np.shape
timestamps = np.loadtxt(TIME_PATH)

# ============================================================
# SATELLITE
# ============================================================

sat = Satrec.twoline2rv(TLE1, TLE2)
ecef_to_llh = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

# ============================================================
# EARTH
# ============================================================

a = 6378137.0
b = 6356752.314245

def ray_ellipsoid_intersection(p, d):
    A = (d[0]**2 + d[1]**2)/a**2 + (d[2]**2)/b**2
    B = 2*((p[0]*d[0] + p[1]*d[1])/a**2 + (p[2]*d[2])/b**2)
    C = (p[0]**2 + p[1]**2)/a**2 + (p[2]**2)/b**2 - 1
    D = B*B - 4*A*C
    if D <= 0:
        return None
    return p + (-B - sqrt(D)) / (2*A) * d

# ============================================================
# ROTATIONS
# ============================================================

def Rx(a): return np.array([[1,0,0],[0,cos(a),-sin(a)],[0,sin(a),cos(a)]])
def Ry(a): return np.array([[cos(a),0,sin(a)],[0,1,0],[-sin(a),0,cos(a)]])
def Rz(a): return np.array([[cos(a),-sin(a),0],[sin(a),cos(a),0],[0,0,1]])

# ============================================================
# TEME → ECEF
# ============================================================

def teme_to_ecef(r, v, jd):
    t = Time(jd, format="jd", scale="utc")
    teme = TEME(
        x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km,
        v_x=v[0]*u.km/u.s, v_y=v[1]*u.km/u.s, v_z=v[2]*u.km/u.s,
        obstime=t
    )
    itrs = teme.transform_to(ITRS(obstime=t))
    return (
        np.array([itrs.x.value, itrs.y.value, itrs.z.value]) * 1000.0,
        np.array([itrs.v_x.value, itrs.v_y.value, itrs.v_z.value]) * 1000.0
    )

def orbit_state(t_unix):
    jd = 2440587.5 + t_unix / 86400.0
    err, r, v = sat.sgp4(jd, 0.0)
    if err != 0:
        return None
    return teme_to_ecef(r, v, jd)

# ============================================================
# SCAN GEOMETRY (MIRRORED)
# ============================================================

half = SCAN_ANGLE_TOTAL / 2
pix_norm = np.linspace(1, -1, n_pixels)  # ← mirror scan direction
scan_angles = half * np.sin(pix_norm)

# ============================================================
# GCP + QUALITY
# ============================================================

gcps = []
row_quality = []

for row in range(0, n_lines, GCP_DY):

    t_line = timestamps[row]
    state = orbit_state(t_line)
    if state is None:
        continue

    pos, vel = state

    z_b = -pos / np.linalg.norm(pos)
    x_b = vel / np.linalg.norm(vel)
    y_b = np.cross(z_b, x_b)
    y_b /= np.linalg.norm(y_b)

    phase = 2*pi * (row / n_lines)

    roll  = ROLL0  + ROLL1  * sin(phase)
    pitch = PITCH0 + PITCH1 * cos(phase)
    ATT = Rx(roll) @ Ry(pitch) @ Rz(YAW0)

    valid = 0
    angle_sum = 0.0

    for col in range(0, n_pixels, GCP_DX):

        a_scan = scan_angles[col]
        d = cos(a_scan)*z_b + sin(a_scan)*y_b
        d = ATT @ d

        hit = ray_ellipsoid_intersection(pos, d)
        if hit is None:
            continue

        valid += 1
        off_nadir = acos(np.dot(-z_b, d))
        angle_sum += off_nadir

        lon, lat, _ = ecef_to_llh.transform(*hit)

        gcps.append(GroundControlPoint(
            row=row,
            col=col,
            x=lon,
            y=lat,
            z=0
        ))

    if valid > 0:
        mean_angle = angle_sum / valid
        quality = exp(-(mean_angle / half)**2)
    else:
        quality = 0.0

    row_quality.append((row, quality))

print("GCP count:", len(gcps))
print("Mean quality:", np.mean([q for _, q in row_quality]))

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

np.savetxt("row_quality.txt", row_quality, fmt="%d %.6f")
print("Written:", OUTPUT_TIF)
print("Quality written to row_quality.txt")
