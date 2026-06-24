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

        self.value += self.a_lp * (grip - self.value)
        if self.baseline is None:
            self.baseline = self.value
        # Baseline only tracks downward / slowly upward (so a sustained press
        # doesn't get absorbed into the baseline immediately).
        if self.value < self.baseline:
            self.baseline += 0.5 * (self.value - self.baseline)  # fast release
        else:
            self.baseline += self.a_base * (self.value - self.baseline)

        above = max(0.0, self.value - self.baseline)
        self.span = max(self.span, above)
        contact = above > self.contact_delta
        return {
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


def _ema_alpha(tau_s, rate_hz):
    """First-order EMA coefficient for a time constant tau at a sample rate."""
    dt = 1.0 / max(1e-6, rate_hz)
    return dt / (tau_s + dt)


def process_frame(frame, array):
    """Update `array` from a parsed glove frame; return per-region results.

    Expects force columns named force0..forceN-1 (see senz_multi_io).
    """
    raws = [frame[f"force{m}"] for m in range(len(array.channels))]
    return array.update(raws)


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
