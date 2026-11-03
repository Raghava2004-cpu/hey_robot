import math

G = 9.81


def predict_landing(start, speed, angle_deg, heading_deg, k1, k2, z_land,
                    dt=1 / 240, max_t=3.0):
    """World model: integrate dv/dt = -G*z_hat - (k1 + k2*|v|) * v from `start`.
    k1 = linear drag, k2 = quadratic drag (real air drag is mostly quadratic).
    Returns [x, y, flight_time] where the ball comes down to height z_land."""
    th, ps = math.radians(angle_deg), math.radians(heading_deg)
    x, y, z = start
    vx = speed * math.cos(th) * math.cos(ps)
    vy = speed * math.cos(th) * math.sin(ps)
    vz = speed * math.sin(th)
    t = 0.0
    while t < max_t:
        sp = math.sqrt(vx * vx + vy * vy + vz * vz)
        damp = k1 + k2 * sp
        vx += -damp * vx * dt
        vy += -damp * vy * dt
        vz += (-G - damp * vz) * dt
        x, y, z = x + vx * dt, y + vy * dt, z + vz * dt
        t += dt
        if vz < 0 and z <= z_land:
            return [x, y, t]
    return [x, y, t]