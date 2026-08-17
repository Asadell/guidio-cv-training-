"""
Estimasi jarak fisik (meter) ke objek dari frame kamera.

Tidak menggunakan model AI depth — murni trigonometri geometri kamera + IMU.
Dipakai di backend fusion layer (navigasi router) untuk mengisi field
"distance_m" pada respons JSON ke Flutter.

Rumus:
    alpha = arctan((y_max - cy) / f_py)        # offset piksel dari pusat
    D = H_cam / tan(theta_imu + alpha)          # jarak ke titik kontak tanah

Referensi: "Ground Contact Point" method untuk estimasi jarak monocular
menggunakan asumsi flat-ground + IMU pitch angle.
"""
import math


def estimate_distance(
    y_max: int,
    frame_height: int,
    pitch_deg: float,
    H_cam: float = 1.35,
    f_py: float | None = None,
    fov_vertical_deg: float = 60.0,
) -> float:
    """
    Estimasi jarak fisik (meter) ke titik kontak tanah bounding box.

    Args:
        y_max            : Koordinat piksel bawah bounding box
                           (titik paling bawah objek = kontak tanah)
        frame_height     : Tinggi frame dalam piksel
        pitch_deg        : Sudut kemiringan HP dari IMU (derajat).
                           Positif = HP menunduk ke bawah (normal saat jalan).
        H_cam            : Tinggi kamera dari tanah (meter). Default 1.35m.
        f_py             : Focal length vertikal dalam piksel.
                           Jika None → dihitung dari fov_vertical_deg.
        fov_vertical_deg : Field of View vertikal kamera (derajat). Default 60°.

    Returns:
        Jarak estimasi dalam meter, dibulatkan 2 desimal.
        Mengembalikan -1.0 jika sudut tidak valid (objek di atas cakrawala).
    """
    if f_py is None:
        # f = (h/2) / tan(FoV/2)
        f_py = (frame_height / 2) / math.tan(math.radians(fov_vertical_deg / 2))

    cy        = frame_height / 2
    alpha_rad = math.atan((y_max - cy) / f_py)
    theta_rad = math.radians(pitch_deg)
    total     = theta_rad + alpha_rad

    if total <= 0:
        return -1.0  # tidak valid

    return round(H_cam / math.tan(total), 2)


def estimate_distance_by_height(
    box_height_px: int,
    frame_height: int,
    real_height_cm: float,
    fov_vertical_deg: float = 60.0,
) -> float:
    """
    Estimasi jarak alternatif berdasarkan tinggi objek yang diketahui.
    Berguna untuk objek bertinggi badan diketahui: orang (~170cm), motor (~120cm).

    Rumus: D = (H_real * f_py) / h_box

    Args:
        box_height_px    : Tinggi bounding box dalam piksel
        frame_height     : Tinggi frame dalam piksel
        real_height_cm   : Tinggi nyata objek dalam cm
        fov_vertical_deg : FoV vertikal kamera (derajat)

    Returns:
        Jarak estimasi dalam meter. -1.0 jika tidak valid.
    """
    if box_height_px <= 0:
        return -1.0

    f_py = (frame_height / 2) / math.tan(math.radians(fov_vertical_deg / 2))
    H_real_m = real_height_cm / 100.0
    return round((H_real_m * f_py) / box_height_px, 2)


# Tinggi nyata objek dalam cm — dipakai estimate_distance_by_height
REAL_HEIGHTS_CM = {
    "orang":      170,
    "motor":      120,
    "tiang":      250,
    "tangga":      80,   # tinggi rata-rata satu anak tangga
    "lubang":      10,   # tidak cocok untuk method ini, pakai ground contact
    "got_terbuka": 10,   # sama
}

# Fallback pitch angle jika IMU tidak tersedia
PITCH_DEFAULT_DEG = 50.0
