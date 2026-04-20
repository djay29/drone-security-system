"""
simulator.py
------------
Generates realistic drone telemetry data that can be attached to every
FramePacket as it is emitted by VideoIngestor.

Telemetry fields produced
-------------------------
  battery_pct    : float  0–100    % remaining charge
  alt_m          : float  0–120    altitude above ground in metres
  lat            : float           GPS latitude  (WGS-84)
  lon            : float           GPS longitude (WGS-84)
  speed_ms       : float  0–15     ground speed  m/s
  heading_deg    : float  0–360    compass heading (0=North)
  flight_mode    : str             TAKEOFF | PATROL | HOVER | RETURN | LANDING
  signal_strength: int    0–100    %  RC / data-link signal quality
  temp_c         : float           battery temperature °C
  vspeed_ms      : float           vertical speed m/s (positive = climb)
  satellites     : int    0–20     GPS satellite fix count

Usage
-----
    sim = TelemetrySimulator(start_lat=37.7749, start_lon=-122.4194)

    # Attach to every FramePacket at a given timestamp
    telemetry = sim.get_telemetry(packet.ts)

    # Or drive a standalone event loop
    for event in sim.generate_events(duration=300, fps=1.0):
        print(event)

    # Load a named scenario
    sim.load_scenario("perimeter_patrol")
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone, timedelta
from typing import Generator, Iterator


# ---------------------------------------------------------------------------
# Flight phases and their rough parameter ranges
# ---------------------------------------------------------------------------

_PHASES = {
    "TAKEOFF":  {"alt_range": (0,  30), "speed_range": (0,  4), "duration_s": 20},
    "PATROL":   {"alt_range": (25, 50), "speed_range": (5, 12), "duration_s": 180},
    "HOVER":    {"alt_range": (20, 40), "speed_range": (0,  2), "duration_s": 30},
    "RETURN":   {"alt_range": (25, 50), "speed_range": (8, 12), "duration_s": 60},
    "LANDING":  {"alt_range": (0,  25), "speed_range": (0,  4), "duration_s": 20},
}

_PATROL_SEQUENCE = ["TAKEOFF", "PATROL", "HOVER", "PATROL", "RETURN", "LANDING"]

# Predefined named scenarios (modify as needed for your property layout)
_SCENARIOS: dict[str, dict] = {
    "perimeter_patrol": {
        "description": "Standard perimeter patrol starting at main gate",
        "start_battery": 95.0,
        "discharge_rate": 0.45,         # % per minute
        "cruise_altitude_m": 35.0,
        "patrol_radius_m": 80.0,
        "phase_sequence": ["TAKEOFF", "PATROL", "PATROL", "HOVER", "RETURN", "LANDING"],
    },
    "night_watch": {
        "description": "Low-altitude slow patrol for after-hours monitoring",
        "start_battery": 100.0,
        "discharge_rate": 0.40,
        "cruise_altitude_m": 20.0,
        "patrol_radius_m": 50.0,
        "phase_sequence": ["TAKEOFF", "HOVER", "PATROL", "HOVER", "PATROL", "RETURN", "LANDING"],
    },
    "emergency_response": {
        "description": "Fast high-altitude response flight",
        "start_battery": 80.0,
        "discharge_rate": 0.60,         # higher drain at full throttle
        "cruise_altitude_m": 60.0,
        "patrol_radius_m": 30.0,
        "phase_sequence": ["TAKEOFF", "PATROL", "HOVER", "LANDING"],
    },
    "battery_low_test": {
        "description": "Starts at 25% battery — triggers low-battery rules quickly",
        "start_battery": 25.0,
        "discharge_rate": 0.50,
        "cruise_altitude_m": 30.0,
        "patrol_radius_m": 40.0,
        "phase_sequence": ["TAKEOFF", "PATROL", "RETURN", "LANDING"],
    },
}


# ---------------------------------------------------------------------------
# TelemetrySimulator
# ---------------------------------------------------------------------------

class TelemetrySimulator:
    """
    Generates physically plausible drone telemetry for a patrol flight.

    The simulator maintains an internal clock anchored to `start_ts`.
    Call `get_telemetry(ts)` with any datetime to get the interpolated
    state at that moment, regardless of wall-clock time.

    Parameters
    ----------
    start_lat        : GPS latitude of home/launch point
    start_lon        : GPS longitude of home/launch point
    start_battery    : Initial battery percentage (0–100)
    discharge_rate   : Battery % consumed per minute of flight
    cruise_altitude_m: Target altitude during patrol phase (metres)
    patrol_radius_m  : Radius of the circular patrol path (metres)
    start_ts         : Anchor datetime for elapsed-time calculations.
                       Defaults to now (UTC).
    noise_seed       : Random seed for reproducible noise (None = random)
    """

    def __init__(
        self,
        start_lat: float = 37.7749,
        start_lon: float = -122.4194,
        start_battery: float = 100.0,
        discharge_rate: float = 0.50,
        cruise_altitude_m: float = 30.0,
        patrol_radius_m: float = 60.0,
        start_ts: datetime | None = None,
        noise_seed: int | None = 42,
    ) -> None:
        self.start_lat         = start_lat
        self.start_lon         = start_lon
        self.start_battery     = start_battery
        self.discharge_rate    = discharge_rate          # % per minute
        self.cruise_altitude_m = cruise_altitude_m
        self.patrol_radius_m   = patrol_radius_m
        self.start_ts          = start_ts or datetime.now(timezone.utc)

        self._rng = random.Random(noise_seed)
        self._phase_sequence   = list(_PATROL_SEQUENCE)  # mutable copy
        self._phase_durations  = self._build_phase_durations()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def get_telemetry(self, ts: datetime) -> dict:
        """
        Return a telemetry snapshot for the given wall-clock timestamp.

        The result dict uses the same key names expected by RuleEngine:
          battery_pct, alt_m, lat, lon, speed_ms, heading_deg,
          flight_mode, signal_strength, temp_c, vspeed_ms, satellites

        Also includes a human-readable `telemetry_summary` string.
        """
        elapsed_s = (ts - self.start_ts).total_seconds()
        elapsed_s = max(0.0, elapsed_s)   # clamp to non-negative

        phase, phase_elapsed_s, phase_total_s = self._phase_at(elapsed_s)
        t = phase_elapsed_s / max(phase_total_s, 1.0)  # normalised 0→1 within phase

        battery  = self._battery(elapsed_s)
        alt      = self._altitude(phase, t)
        lat, lon = self._position(elapsed_s)
        speed    = self._speed(phase, t)
        heading  = self._heading(elapsed_s)
        vspeed   = self._vspeed(phase, t)
        signal   = self._signal(elapsed_s)
        temp     = self._battery_temp(battery, elapsed_s)
        sats     = self._satellites(elapsed_s)

        telemetry = {
            "battery_pct":     round(battery, 1),
            "alt_m":           round(alt, 1),
            "lat":             round(lat, 6),
            "lon":             round(lon, 6),
            "speed_ms":        round(speed, 1),
            "heading_deg":     round(heading % 360, 1),
            "flight_mode":     phase,
            "signal_strength": signal,
            "temp_c":          round(temp, 1),
            "vspeed_ms":       round(vspeed, 1),
            "satellites":      sats,
        }
        telemetry["telemetry_summary"] = _format_summary(telemetry)
        return telemetry

    def generate_events(
        self,
        duration: float = 300.0,
        fps: float = 1.0,
    ) -> Generator[dict, None, None]:
        """
        Yield telemetry dicts at `fps` frequency for `duration` seconds,
        starting from `self.start_ts`.

        Useful for testing or feeding a standalone telemetry bus.
        """
        interval_s = 1.0 / fps
        n_steps    = int(duration * fps)
        for i in range(n_steps):
            ts = self.start_ts + timedelta(seconds=i * interval_s)
            event = self.get_telemetry(ts)
            event["ts"] = ts.isoformat()
            event["elapsed_s"] = round(i * interval_s, 2)
            yield event

    def load_scenario(self, scenario: str | dict) -> "TelemetrySimulator":
        """
        Load a named scenario or a raw config dict.

        Named scenarios: "perimeter_patrol", "night_watch",
                         "emergency_response", "battery_low_test"

        Returns self (chainable).
        """
        if isinstance(scenario, str):
            if scenario not in _SCENARIOS:
                raise ValueError(
                    f"Unknown scenario {scenario!r}. "
                    f"Available: {list(_SCENARIOS)}"
                )
            cfg = _SCENARIOS[scenario]
        else:
            cfg = scenario

        if "start_battery" in cfg:
            self.start_battery = cfg["start_battery"]
        if "discharge_rate" in cfg:
            self.discharge_rate = cfg["discharge_rate"]
        if "cruise_altitude_m" in cfg:
            self.cruise_altitude_m = cfg["cruise_altitude_m"]
        if "patrol_radius_m" in cfg:
            self.patrol_radius_m = cfg["patrol_radius_m"]
        if "phase_sequence" in cfg:
            self._phase_sequence = list(cfg["phase_sequence"])

        self._phase_durations = self._build_phase_durations()
        return self

    @staticmethod
    def available_scenarios() -> list[str]:
        """Return names of all built-in scenarios."""
        return list(_SCENARIOS.keys())

    @staticmethod
    def scenario_description(name: str) -> str:
        return _SCENARIOS.get(name, {}).get("description", "")

    # ------------------------------------------------------------------
    # Internal: phase timeline
    # ------------------------------------------------------------------

    def _build_phase_durations(self) -> list[tuple[str, float]]:
        """[(phase_name, duration_seconds), ...]"""
        return [
            (phase, _PHASES[phase]["duration_s"])
            for phase in self._phase_sequence
        ]

    def _phase_at(self, elapsed_s: float) -> tuple[str, float, float]:
        """
        Return (phase_name, elapsed_within_phase, total_phase_duration)
        for the given overall elapsed seconds.
        """
        accum = 0.0
        for phase, dur in self._phase_durations:
            if elapsed_s < accum + dur:
                return phase, elapsed_s - accum, dur
            accum += dur
        # Past end of mission — treat as LANDING complete (on ground)
        last_phase = self._phase_durations[-1][0]
        last_dur   = self._phase_durations[-1][1]
        return last_phase, last_dur, last_dur

    # ------------------------------------------------------------------
    # Internal: telemetry generators
    # ------------------------------------------------------------------

    def _battery(self, elapsed_s: float) -> float:
        drained = (elapsed_s / 60.0) * self.discharge_rate
        noise   = self._rng.uniform(-0.05, 0.05)
        return max(0.0, min(100.0, self.start_battery - drained + noise))

    def _altitude(self, phase: str, t: float) -> float:
        """Smoothly interpolate altitude within a phase."""
        cfg = _PHASES[phase]
        lo, hi = cfg["alt_range"]
        if phase == "TAKEOFF":
            target = lo + (self.cruise_altitude_m - lo) * _ease_in(t)
        elif phase == "LANDING":
            target = self.cruise_altitude_m * (1.0 - _ease_out(t))
        elif phase == "HOVER":
            mid = (lo + hi) / 2
            target = mid
        elif phase in ("PATROL", "RETURN"):
            target = self.cruise_altitude_m
        else:
            target = (lo + hi) / 2

        noise = self._rng.gauss(0, 0.4)
        return max(0.0, target + noise)

    def _position(self, elapsed_s: float) -> tuple[float, float]:
        """Circular patrol around the home point."""
        # metres per degree of latitude ≈ 111_320
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(self.start_lat))

        # 1 full orbit every 120s
        angle_rad = (2 * math.pi * elapsed_s) / 120.0
        dx_m = self.patrol_radius_m * math.cos(angle_rad)
        dy_m = self.patrol_radius_m * math.sin(angle_rad)

        noise_lat = self._rng.gauss(0, 0.3) / m_per_deg_lat
        noise_lon = self._rng.gauss(0, 0.3) / m_per_deg_lon

        lat = self.start_lat + dy_m / m_per_deg_lat + noise_lat
        lon = self.start_lon + dx_m / m_per_deg_lon + noise_lon
        return lat, lon

    def _speed(self, phase: str, t: float) -> float:
        cfg = _PHASES[phase]
        lo, hi = cfg["speed_range"]
        target = lo + (hi - lo) * t
        noise  = self._rng.gauss(0, 0.3)
        return max(0.0, min(15.0, target + noise))

    def _heading(self, elapsed_s: float) -> float:
        """Heading follows the tangent of the circular patrol path."""
        angle_rad = (2 * math.pi * elapsed_s) / 120.0
        heading   = math.degrees(angle_rad + math.pi / 2) % 360
        noise     = self._rng.gauss(0, 1.5)
        return (heading + noise) % 360

    def _vspeed(self, phase: str, t: float) -> float:
        if phase == "TAKEOFF":
            return round(self._rng.gauss(2.0, 0.3), 1)
        if phase == "LANDING":
            return round(self._rng.gauss(-1.5, 0.3), 1)
        return round(self._rng.gauss(0.0, 0.2), 1)

    def _signal(self, elapsed_s: float) -> int:
        # Slight degradation with distance; occasional interference blip
        base  = max(60, 98 - int(elapsed_s / 30))
        noise = self._rng.randint(-3, 3)
        if self._rng.random() < 0.02:   # 2% chance of interference spike
            noise = -self._rng.randint(10, 25)
        return max(0, min(100, base + noise))

    def _battery_temp(self, battery_pct: float, elapsed_s: float) -> float:
        # Warm up from ambient ~22°C to ~38°C over first 2 minutes
        ambient = 22.0
        warmup  = min(16.0, elapsed_s / 120.0 * 16.0)
        # Slightly hotter when battery is low (higher discharge current)
        low_bat_heat = max(0.0, (40.0 - battery_pct) / 40.0 * 4.0)
        noise   = self._rng.gauss(0, 0.3)
        return round(ambient + warmup + low_bat_heat + noise, 1)

    def _satellites(self, elapsed_s: float) -> int:
        # Start with poor fix; good fix within 10 seconds
        base = min(14, max(4, int(6 + elapsed_s / 10)))
        return base + self._rng.randint(-1, 1)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _ease_in(t: float) -> float:
    """Cubic ease-in curve: slow start, fast end."""
    return t * t * t


def _ease_out(t: float) -> float:
    """Cubic ease-out curve: fast start, slow end."""
    return 1 - (1 - t) ** 3


def _format_summary(t: dict) -> str:
    """One-line human-readable summary injected into chatbot context."""
    return (
        f"[{t['flight_mode']}] "
        f"bat={t['battery_pct']}% "
        f"alt={t['alt_m']}m "
        f"spd={t['speed_ms']}m/s "
        f"hdg={t['heading_deg']}° "
        f"sig={t['signal_strength']}% "
        f"gps=({t['lat']},{t['lon']})"
    )
