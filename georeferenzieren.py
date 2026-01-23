################################# Laufzeit: ca. 10 Minuten!!! ##################################

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from sgp4.api import Satrec
from pyproj import Transformer
from scipy.spatial import cKDTree
from math import sin, cos, sqrt, pi

from astropy.utils import iers
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"

from astropy.coordinates import TEME, ITRS
from astropy.time import Time
import astropy.units as u

# ============================================================
# CONFIG
# ============================================================

PNG_PATH = "msu_mr_rgb_MCIR.png"
TIME_PATH = "tim.txt" #enthält die interpolierten Unix-Timestamps aus product.cbor
OUTPUT_TIF = "aus_corrected.tif"

TLE1 = "1 57166U 23091A   26004.31624314  .00000055  00000+0  42710-4 0  9993"
TLE2 = "2 57166  98.6344  62.4577 0002906 292.2271  67.8599 14.24029359131192"

SCAN_ANGLE_TOTAL = np.deg2rad(110.1)
SCAN_TIME = 1.6

ROLL0  = np.deg2rad(-0.31)
PITCH0 = np.deg2rad(-3.1)
YAW0   = np.deg2rad(0.75)

# output grid resolution (deg)
OUT_RES = 0.002

# ============================================================
# LOAD DATA
# ============================================================

img = np.asarray(Image.open(PNG_PATH).convert("RGB"))
timestamps = np.loadtxt(TIME_PATH)

n_lines, n_pixels, _ = img.shape

sat = Satrec.twoline2rv(TLE1, TLE2)
ecef_to_llh = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

# ============================================================
# EARTH
# ============================================================

a = 6378137.0
b = 6356752.314245

def ray_ellipsoid(p, d):
    A = (d[0]**2 + d[1]**2)/a**2 + d[2]**2/b**2
    B = 2*((p[0]*d[0] + p[1]*d[1])/a**2 + p[2]*d[2]/b**2)
    C = (p[0]**2 + p[1]**2)/a**2 + p[2]**2/b**2 - 1
    D = B*B - 4*A*C
    if D <= 0:
        return None
    t = (-B - sqrt(D)) / (2*A)
    return p + t * d

# ============================================================
# COORDINATES
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
        np.array([itrs.x.value, itrs.y.value, itrs.z.value]) * 1000,
        np.array([itrs.v_x.value, itrs.v_y.value, itrs.v_z.value]) * 1000
    )

def orbit_state(t_unix):
    jd = 2440587.5 + t_unix / 86400.0
    err, r, v = sat.sgp4(jd, 0.0)
    if err != 0:
        return None
    return teme_to_ecef(r, v, jd)

# ============================================================
# SCAN GEOMETRY (PHYSICAL)
# ============================================================

scan_angles = np.linspace(
    -SCAN_ANGLE_TOTAL/2,
     SCAN_ANGLE_TOTAL/2,
     n_pixels
)

# ============================================================
# FAST PIXEL → GEO (OPTIMIERT)
# ============================================================

points = []
colors = []

for row in range(n_lines):

    # --- Orbit nur EINMAL pro Zeile ---
    t_line = timestamps[row]
    state = orbit_state(t_line)
    if state is None:
        continue

    pos, vel = state

    # --- RTN frame ---
    R = pos / np.linalg.norm(pos)
    T = vel / np.linalg.norm(vel)
    N = np.cross(R, T)
    N /= np.linalg.norm(N)
    T = np.cross(N, R)

    # --- Scanwinkel für gesamte Zeile ---
    cos_a = np.cos(scan_angles)
    sin_a = np.sin(scan_angles)

    # --- Richtungsvektoren für alle Pixel ---
    dirs = (-cos_a[:, None] * R + sin_a[:, None] * N)

    for col in range(n_pixels):

        d = dirs[col]
        hit = ray_ellipsoid(pos, d)
        if hit is None:
            continue

        lon, lat, _ = ecef_to_llh.transform(*hit)

        points.append((lon, lat))
        colors.append(img[row, col])

points = np.asarray(points)
colors = np.asarray(colors)

# ============================================================
# OUTPUT GRID
# ============================================================

lon_min, lat_min = points.min(axis=0)
lon_max, lat_max = points.max(axis=0)

nx = int((lon_max - lon_min) / OUT_RES) + 1
ny = int((lat_max - lat_min) / OUT_RES) + 1

grid_lon = lon_min + np.arange(nx) * OUT_RES
grid_lat = lat_max - np.arange(ny) * OUT_RES

mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

tree = cKDTree(points)

# maximale zulässige Distanz (in Grad!)
MAX_DIST = 0.01   # ca. 1 km, konservativ

dist, idx = tree.query(
    np.column_stack([mesh_lon.ravel(), mesh_lat.ravel()]),
    distance_upper_bound=MAX_DIST
)

out = np.zeros((ny*nx, 3), dtype=np.uint8)

valid = idx < len(colors)
out[valid] = colors[idx[valid]]

out = out.reshape(ny, nx, 3)


transform = from_origin(lon_min, lat_max, OUT_RES, OUT_RES)

# ============================================================
# WRITE GEOTIFF
# ============================================================


with rasterio.open(
    OUTPUT_TIF,
    "w",
    driver="GTiff",
    height=ny,
    width=nx,
    count=3,
    dtype=np.uint8,
    crs="EPSG:4326",
    transform=transform,
    compress="DEFLATE",
    predictor=2,
    tiled=True,
    blockxsize=512,
    blockysize=512,
    interleave="pixel"
) as dst:
    for i in range(3):
        dst.write(out[:, :, i], i + 1)

print("Fertig, die georeferenzierte Datei heißt ", OUTPUT_TIF, ".")
