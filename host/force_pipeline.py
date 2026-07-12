#!/usr/bin/env python3
"""
force_pipeline.py  --  Velostat force-sensor processing (HLD objective 4)
=========================================================================
Turns the raw 12-bit ADC counts that the firmware streams for each Velostat
region into usable contact/grip features for the dataset.

Velostat is a piezoresistive sheet: resistance drops as pressure rises. Each
region is wired as a voltage divider (Velostat + a fixed series resistor) into
an ADC pin. This module does, per channel:

  raw ADC  ->  voltage  ->  sensor resistance (via the divider)
           ->  EMA baseline (slow) for drift removal
           ->  IIR low-pass (fast) for noise
           ->  contact flag + relative grip (0..1)

Velostat is NOT calibrated force -- outputs are *relative* grip / contact
intensity, never absolute Newtons (HLD section 7, Limitations).

Stateful, so it works the same on a live stream or a recorded file: feed frames
in order and read the processed fields back out.
"""

ADC_MAX = 4095
VREF = 3.3


class ForceChannel:
    """Online processing for one Velostat region."""

    def __init__(self, series_ohms=10000.0, vref=VREF,
                 baseline_tau=3.0, lowpass_tau=0.05, rate=200.0,
                 contact_delta=0.04):
        self.series_ohms = series_ohms
        self.vref = vref
        # Convert time constants (s) to per-sample EMA coefficients.
        self.a_base = _ema_alpha(baseline_tau, rate)
        self.a_lp = _ema_alpha(lowpass_tau, rate)
        self.contact_delta = contact_delta  # grip units above baseline = contact

        self.baseline = None   # slow-moving "no contact" level (grip units)
        self.value = 0.0       # low-pass grip signal
        self.span = 1e-6       # running max grip-above-baseline, for 0..1 scaling

    def resistance(self, raw):
        """ADC counts -> Velostat resistance (ohms). Divider: Vout across series R."""
        v = max(1.0, raw) / ADC_MAX * self.vref
        v = min(v, self.vref - 1e-3)
        # Velostat to VREF, series R to GND, ADC reads the junction:
        #   Vout = VREF * Rseries / (Rvelostat + Rseries)
        #   Rvelostat = Rseries * (VREF/Vout - 1)
        return self.series_ohms * (self.vref / v - 1.0)

    def update(self, raw):
        """Push one raw ADC sample; returns a dict of processed fields."""
        # "grip" rises with pressure: lower resistance -> higher grip. Use a
        # conductance-like measure that's bounded and monotonic.
        r = self.resistance(raw)
        grip = 1.0 / (1.0 + r / self.series_ohms)  # 0 (open) .. ->1 (hard press)

        if self.baseline is None:
            # Seed BOTH the low-pass and the baseline at the first real sample.
            # (Starting value at 0 would seed the baseline far below the true open
            # level, producing a bogus multi-second "grip" transient on every
            # channel while the slow baseline converged up -- a startup artifact.)
            self.value = grip
            self.baseline = grip
        else:
            self.value += self.a_lp * (grip - self.value)
            # Decide contact against the CURRENT baseline first, then move the
            # baseline. Snap it DOWN fast on release; drift it UP slowly only while
            # the pad is OPEN. Holding a press used to let the slow (~3 s) baseline
            # creep up to the pressed level, so a sustained press faded back to zero
            # in ~3 s -- freezing the baseline during contact keeps the press
            # registered for as long as you hold it (drift removal resumes on release).
            in_contact = (self.value - self.baseline) > self.contact_delta
            if self.value < self.baseline:
                self.baseline += 0.5 * (self.value - self.baseline)  # fast release
            elif not in_contact:
                self.baseline += self.a_base * (self.value - self.baseline)

        above = max(0.0, self.value - self.baseline)
        self.span = max(self.span, above)
        contact = above > self.contact_delta
        return {
            "raw": raw,                 # the un-processed ADC count (force-test view)
            "resistance": r,
            "grip": self.value,
            "above_baseline": above,
            "relative_grip": min(1.0, above / self.span),
            "contact": bool(contact),
        }


class ForceArray:
    """A bank of ForceChannel, one per region, addressed by index."""

    def __init__(self, n, **kw):
        self.channels = [ForceChannel(**kw) for _ in range(n)]

    def update(self, raws):
        return [ch.update(r) for ch, r in zip(self.channels, raws)]

    def reset(self, indices=None):
        """Re-zero (re-baseline) the given channels -- all of them if ``indices`` is
        None. Clearing ``baseline`` to None makes the next sample re-seed the open
        (no-contact) level, and clearing ``span`` makes relative_grip re-learn its
        scale. This is the "set force to 0 / calibrate" primitive: hold the hand
        open and call it, and every channel takes the current reading as its zero.
        """
        idx = range(len(self.channels)) if indices is None else indices
        for i in idx:
            if 0 <= i < len(self.channels):
                ch = self.channels[i]
                ch.baseline = None     # re-seed on the next sample
                ch.value = 0.0
                ch.span = 1e-6


class ForceConfig:
    """Per-channel enable + source remap for the velostat array (pure Python,
    headless-testable) -- the force-side analogue of the IMU SensorConfigV3.

    - ``enabled[m]``: deactivate a broken/disconnected pad so it no longer drives
      grip/contact (its processed output goes neutral). The raw ADC is still passed
      through, so the force-test view can still show what a "broken" pad reads.
    - ``source[m]``: logical channel ``m`` takes its raw ADC from physical channel
      ``source[m]``. Lets you re-designate a mis-wired finger's pads -- e.g. an array
      that came back with a finger's four pads in reverse order -- without touching
      firmware or re-soldering.
    """

    def __init__(self, n):
        self.n = n
        self.enabled = [True] * n
        self.source = list(range(n))   # identity: logical m <- physical m
        self.reversed = [False] * n    # invert the ADC (pressed<->released) per pad

    def set_enabled(self, m, on):
        self.enabled[m] = bool(on)

    def set_reversed(self, m, on):
        """Reverse a channel whose value runs backwards (a flipped voltage divider /
        mis-wired velostat leg reads HIGH when open and LOW when pressed). When set,
        the raw ADC is inverted (ADC_MAX - raw) before processing, so grip/contact
        rise with pressure like every other pad. The displayed raw stays the true
        reading."""
        if 0 <= m < self.n:
            self.reversed[m] = bool(on)

    def feed_value(self, m, raw):
        """The ADC value to actually process for logical channel ``m`` (inverted if
        the channel is reversed). The displayed 'raw' keeps the true reading."""
        return (ADC_MAX - raw) if (0 <= m < self.n and self.reversed[m]) else raw

    def set_source(self, m, raw):
        """Route logical channel ``m`` to physical channel ``raw`` (validated)."""
        if 0 <= m < self.n and 0 <= raw < self.n:
            self.source[m] = raw

    def swap(self, a, b):
        """Swap two logical channels' physical sources (fix a transposed pair)."""
        if 0 <= a < self.n and 0 <= b < self.n:
            self.source[a], self.source[b] = self.source[b], self.source[a]

    def reset(self):
        """Back to identity routing, every channel enabled and non-reversed."""
        self.source = list(range(self.n))
        self.enabled = [True] * self.n
        self.reversed = [False] * self.n

    def route(self, raws):
        """Reorder a physical raw list so output index ``m`` holds ``source[m]``."""
        return [raws[self.source[m]] if 0 <= self.source[m] < len(raws) else 0
                for m in range(self.n)]

    _NEUTRAL = {"grip": 0.0, "above_baseline": 0.0, "relative_grip": 0.0,
                "contact": False, "disabled": True}

    def mask(self, results):
        """Neutralize disabled channels' processed output (raw is preserved)."""
        for m in range(min(self.n, len(results))):
            if not self.enabled[m]:
                results[m] = {**results[m], **self._NEUTRAL}
        return results


def _ema_alpha(tau_s, rate_hz):
    """First-order EMA coefficient for a time constant tau at a sample rate."""
    dt = 1.0 / max(1e-6, rate_hz)
    return dt / (tau_s + dt)


def process_frame(frame, array, cfg=None):
    """Update `array` from a parsed glove frame; return per-region results.

    Expects force columns named force0..forceN-1 (see senz_multi_io). When a
    ``ForceConfig`` is given, physical channels are first routed to their logical
    positions (source remap) and disabled channels are neutralized afterward; each
    result still carries the physical ``raw`` count it was fed.
    """
    raws = [frame[f"force{m}"] for m in range(len(array.channels))]
    if cfg is None:
        return array.update(raws)
    raws = cfg.route(raws)                                  # physical -> logical
    feed = [cfg.feed_value(m, r) for m, r in enumerate(raws)]   # invert if reversed
    out = array.update(feed)
    for m in range(min(cfg.n, len(out))):
        out[m]["raw"] = raws[m]                             # show the TRUE reading
    return cfg.mask(out)


if __name__ == "__main__":
    # Tiny self-check: a step press on one channel should raise contact=True.
    import senz_multi_io as io

    src = io.open_multi_source(simulate=True, nforce=5)
    arr = ForceArray(src.schema.nforce, rate=src.schema.rate)
    last = None
    for _ in range(400):
        last = process_frame(src.read(), arr)
    src.close()
    for m, res in enumerate(last):
        print(f"force{m}: grip={res['grip']:.3f} rel={res['relative_grip']:.2f} "
              f"contact={res['contact']}")
