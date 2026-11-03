# Hey Robot: RL landing reflex. 4,096 drones learn in parallel on the GPU (PPO).
import math, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/skillforge_rl"
except Exception:
    OUT = "skillforge_rl"
os.makedirs(OUT, exist_ok=True)

# ---- must match the PyBullet drone ----
N_ENVS = 4096
DT, SUB = 0.025, 6                 # control at 40 Hz, physics at 240 Hz (like PyBullet)
T_TRACK, T_DESC = 1.0, 2.0         # follow 1 s, then reach the pad in exactly 2 s
STEPS = int(round((T_TRACK + T_DESC) / DT))   # 120
SIGHT_EVERY = 10                   # a SAM sighting every 0.25 s
A_MAX, PAD_HALF, KP, KD = 4.0, 0.225, 6.0, 4.0
OBS_DIM, PRIV_DIM = 10, 4


class LandingSim:
    """Point-mass drone + hidden wind + gusts + moving pad + SAM-like sightings, for N drones at once."""

    def __init__(self, n, gust_scale=1.0, seed=0):
        self.n, self.gust_scale = n, gust_scale
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)

    def U(self, *s, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*s, generator=self.gen, device=dev)

    def G(self, *s):
        return torch.randn(*s, generator=self.gen, device=dev)

    def height(self, t):
        return self.h0 if t <= T_TRACK else self.h0 * max(0.0, 1.0 - (t - T_TRACK) / T_DESC)

    def reset(self):
        n = self.n
        sp, hd = self.U(n, lo=0.05, hi=0.45), self.U(n, hi=2 * math.pi)
        self.pv = torch.stack([sp * torch.cos(hd), sp * torch.sin(hd)], 1)     # pad velocity
        self.pp = torch.zeros(n, 2, device=dev)                                 # pad position
        self.p = self.U(n, 2, lo=-0.3, hi=0.3)                                  # drone position
        self.v = self.pv * self.U(n, 1)                                         # drone velocity
        wm, wa = self.U(n, hi=1.5), self.U(n, hi=2 * math.pi)
        self.wbase = torch.stack([wm * torch.cos(wa), wm * torch.sin(wa)], 1)   # hidden steady wind
        self.gsig = self.U(n, 1) * self.gust_scale                              # gust strength
        self.gtau = self.U(n, 1, lo=0.3, hi=1.0)                                # gust time scale
        self.gust = self.G(n, 2) * self.gsig
        self.drag = self.U(n, 1, hi=0.1)
        self.h0 = self.U(n, 1, lo=0.75, hi=0.95)
        self.det_h = self.U(n, 1, lo=0.55, hi=0.70)    # below this height the pad no longer fits the view
        self.ff = -(self.wbase + self.gust) + 0.05 * self.G(n, 2)   # planner's one-hover wind measurement
        self.vest = self.pv + 0.05 * self.G(n, 2)                    # planner's rough pad-velocity guess
        self.est = self.pp + 0.008 * self.G(n, 2)
        self.t_est = torch.zeros(n, 1, device=dev)
        self.bt, self.bx = [0.0], [self.est.clone()]
        self.dist = torch.zeros(n, 2, device=dev)
        self.a_prev = torch.zeros(n, 2, device=dev)
        self.v_prev = self.v.clone()

    def _fit(self):
        k = min(6, len(self.bt))
        if k < 3 or self.bt[-1] - self.bt[-k] < 0.5:
            return None
        T = torch.tensor(self.bt[-k:], device=dev)
        X = torch.stack(self.bx[-k:], 1)
        tc = T - T.mean()
        return (tc[None, :, None] * (X - X.mean(1, keepdim=True))).sum(1) / (tc * tc).sum()

    def obs(self, k):
        t = k * DT
        est_now = self.est + self.vest * (t - self.t_est)
        o = torch.cat([(est_now - self.p) / 0.3, (self.vest - self.v) / 0.3, self.dist / 1.5,
                       self.height(t) / 0.9,
                       torch.full((self.n, 1), 1 - t / (T_TRACK + T_DESC), device=dev), self.a_prev], 1)
        priv = torch.cat([(self.pp - self.p) / 0.3, (self.wbase + self.gust) / 1.5], 1)
        return o, torch.cat([o, priv], 1)

    def step(self, a, k):
        n, dt = self.n, DT / SUB
        for _ in range(SUB):
            self.gust = (self.gust - self.gust / self.gtau * dt
                         + self.gsig * torch.sqrt(2 * dt / self.gtau) * self.G(n, 2))
            acc = a + self.wbase + self.gust - self.drag * self.v
            self.v = self.v + acc * dt
            self.p = self.p + self.v * dt
            self.pp = self.pp + self.pv * dt
        meas = (self.v - self.v_prev) / DT                    # like an accelerometer
        self.dist = 0.7 * self.dist + 0.3 * (meas - a + 0.05 * self.G(n, 2))
        self.v_prev, self.a_prev = self.v.clone(), a / A_MAX
        t = (k + 1) * DT
        if (k + 1) % SIGHT_EVERY == 0:
            valid = self.height(t) > self.det_h
            xy = self.pp + 0.008 * self.G(n, 2)
            self.bt.append(t)
            self.bx.append(xy)
            self.est = torch.where(valid, xy, self.est)
            self.t_est = torch.where(valid, torch.full_like(self.t_est, t), self.t_est)
            v = self._fit()
            if v is not None:
                self.vest = torch.where(valid, v, self.vest)
        err = (self.pp - self.p).norm(dim=1)
        r = -0.1 * err - 0.002 * (a * a).sum(1) / A_MAX ** 2
        if k == STEPS - 1:
            r = r + torch.where(err < PAD_HALF, 10.0 + 10.0 * (1 - err / PAD_HALF), -5.0 - 10.0 * err)
        return r


def tracking_controller(sim, k):
    """Our current camera-tracking landing: PD on the pad estimate + one-hover wind correction."""
    t = k * DT
    est_now = sim.est + sim.vest * (t - sim.t_est)
    return (KP * (est_now - sim.p) + KD * (sim.vest - sim.v) + sim.ff).clamp(-A_MAX, A_MAX)


def mlp(i, h, o, out=None):
    layers = [nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o)]
    return nn.Sequential(*(layers + ([out] if out else []))).to(dev)


actor = mlp(OBS_DIM, 128, 2, nn.Tanh())
critic = mlp(OBS_DIM + PRIV_DIM, 256, 1)          # the critic also sees the true wind (training only)
log_std = nn.Parameter(torch.full((2,), -0.7, device=dev))
params = list(actor.parameters()) + list(critic.parameters()) + [log_std]
opt = torch.optim.Adam(params, lr=3e-4)


@torch.no_grad()
def evaluate(policy, gust_scale=1.0, n=4096, seed=123):
    sim = LandingSim(n, gust_scale, seed)
    sim.reset()
    for k in range(STEPS):
        a = tracking_controller(sim, k) if policy == "tracking" else A_MAX * actor(sim.obs(k)[0])
        sim.step(a, k)
    d = (sim.pp - sim.p).norm(dim=1)
    ok = d < PAD_HALF
    return ok.float().mean().item(), (d[ok].mean().item() * 100 if ok.any() else float("nan"))


def export(path):
    w = {}
    for name, layer in zip(["l1", "l2", "l3"], [actor[0], actor[2], actor[4]]):
        w[name + "_w"] = layer.weight.detach().cpu().numpy()
        w[name + "_b"] = layer.bias.detach().cpu().numpy()
    np.savez(path, **w, dt=DT, a_max=A_MAX, t_track=T_TRACK, t_desc=T_DESC)


ITERS, EPOCHS, MB, GAMMA, LAM, CLIP = 400, 4, 8, 0.99, 0.95, 0.2
sim = LandingSim(N_ENVS, gust_scale=1.0, seed=1)
best, best_state, t0 = -1e9, None, time.time()
for it in range(1, ITERS + 1):
    sim.reset()
    O = torch.zeros(STEPS, N_ENVS, OBS_DIM, device=dev)
    OC = torch.zeros(STEPS, N_ENVS, OBS_DIM + PRIV_DIM, device=dev)
    RAW = torch.zeros(STEPS, N_ENVS, 2, device=dev)
    LP, V, R = (torch.zeros(STEPS, N_ENVS, device=dev) for _ in range(3))
    with torch.no_grad():
        for k in range(STEPS):
            o, oc = sim.obs(k)
            mu = actor(o)
            std = log_std.exp().expand_as(mu)
            raw = mu + std * torch.randn_like(mu)
            O[k], OC[k], RAW[k] = o, oc, raw
            LP[k] = Normal(mu, std).log_prob(raw).sum(1)
            V[k] = critic(oc).squeeze(1)
            R[k] = sim.step(A_MAX * raw.clamp(-1, 1), k)
        ADV, last = torch.zeros_like(R), torch.zeros(N_ENVS, device=dev)
        for k in reversed(range(STEPS)):
            nv = V[k + 1] if k < STEPS - 1 else torch.zeros(N_ENVS, device=dev)
            last = R[k] + GAMMA * nv - V[k] + GAMMA * LAM * last
            ADV[k] = last
        RET = ADV + V
    b = STEPS * N_ENVS
    fo, foc, fraw = O.reshape(b, -1), OC.reshape(b, -1), RAW.reshape(b, 2)
    flp, fadv, fret = LP.reshape(b), ADV.reshape(b), RET.reshape(b)
    fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
    mbs = b // MB
    for _ in range(EPOCHS):
        perm = torch.randperm(b, device=dev)
        for j in range(MB):
            idx = perm[j * mbs:(j + 1) * mbs]
            mu = actor(fo[idx])
            dist = Normal(mu, log_std.exp().expand_as(mu))
            ratio = (dist.log_prob(fraw[idx]).sum(1) - flp[idx]).exp()
            pg = -torch.min(ratio * fadv[idx], ratio.clamp(1 - CLIP, 1 + CLIP) * fadv[idx]).mean()
            vl = 0.5 * ((critic(foc[idx]).squeeze(1) - fret[idx]) ** 2).mean()
            loss = pg + 0.5 * vl - 0.001 * dist.entropy().sum(1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    if it == 1 or it % 20 == 0:
        sr, cm = evaluate(actor)
        print(f"iter {it:4d} | {it * b / 1e6:6.1f}M practice steps | landed {sr:6.1%} | "
              f"touchdown error {cm:5.1f} cm | {time.time() - t0:5.0f} s")
        score = sr * 100 - (cm if cm == cm else 50)
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in actor.state_dict().items()}
            export(os.path.join(OUT, "rl_landing.npz"))

actor.load_state_dict(best_state)
print("\n=== GPU-sim comparison: 4,096 fresh landings per row (gusts x1.5 = harder than training) ===")
for gs in [0.0, 0.5, 1.0, 1.5]:
    s1, c1 = evaluate("tracking", gs)
    s2, c2 = evaluate(actor, gs)
    print(f"gusts x{gs:.1f}:  camera tracking landed {s1:6.1%} ({c1:4.1f} cm)   |   RL reflex landed {s2:6.1%} ({c2:4.1f} cm)")
print("\nSaved:", os.path.join(OUT, "rl_landing.npz"))
