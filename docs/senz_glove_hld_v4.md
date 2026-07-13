# HLD: Senz Glove — Hardware Sprint v4
**Branch:** `hardware-sprint-v4` off stable `main`
**Sprint Goal:** Bring up **ONE** excellent force sensor and prove it. Two candidates run
head-to-head on a **weighted test jig**: a **perfected velostat taxel** (DIY, our
existing path) and a **bought FSR rated 0–500 g** (factory-consistent, calibrated range).
A single large pad, read cleanly on the ESP32-S3 ADC, characterized on the jig
(sensitivity, hysteresis, drift, repeatability, noise) — the data picks the winner. Plus
a second main goal: design a **finger-mounting "carapace"** so the sensor mounts to the
finger *without* a sweaty/restrictive glove. **No IMUs. No BNO. No SPI/I2C sensor buses.
No array.** One sensor, done right; everything else deferred until the force primitive is
proven.

---

## Why this sprint exists (the v3 post-mortem)

Hardware Sprint v3 failed on hardware. The most recent commit history is almost
entirely field-failure fixes, and the pattern is complexity:

- **Whole IMU groups died from a shared fault.** `imu0-3` (the entire index+middle
  finger IMU group) read zeros together — "likely one shared SPI/power/CS fault, not
  four" (`375dad8`). A dead IMU streams zeros, the host fuses that to an identity
  quaternion, and the finger *flails* on a garbage pose (`eb627fb` had to auto-disable
  dead sensors just to keep the rest usable).
- **Velostat came back mis-wired.** The entire `ForceConfig` reverse / source-remap /
  deactivate surface (`eed44fa`, `7324380`) exists only to paper over pads that were
  soldered reversed or to the wrong channel. That is a manufacturing problem wearing a
  software costume.
- **Force drifted.** A held press faded to zero over ~3 s until the baseline logic was
  taught to freeze during contact (`375dad8`).

v3 tried to stand up 8 IMUs + a BNO wrist + 12–15 taxels + a camera in one sprint. The
wiring and bus complexity is exactly what broke. **v4's thesis: stop scaling breadth,
prove depth.** Build one force sensor so well — materials, wiring, front-end,
calibration, and a jig that *measures* all of it — that we actually trust it. Only then
do we scale back up to an array (see Roadmap).

Note the velostat mis-wiring above is a *manufacturing* problem. That is one reason v4
also puts a **commercial FSR (0–500 g)** on the same jig: a factory-made part with a
defined range is the honest control against DIY-velostat inconsistency. We don't assume
a winner — we measure both and let the numbers choose.

This is a deliberate, honest step *backwards in breadth* to move *forwards in
confidence*.

---

## Pre-Sprint Checklist (do before bench-up)

- [ ] Velostat sheet on hand + a **thinner compliant inlay** material to trial (foam/EVA/
      silicone shim — the compliant layer that sets the force-to-deflection curve)
- [ ] **FSR sensors, rated 0–500 g** (bought) — the calibrated benchmark candidate; its
      0–500 g range maps directly onto the weight set
- [ ] Conductive fabric (two electrodes) + conductive adhesive / Z-tape; kapton or PET
      for the faraday/shield layer
- [ ] A **series-resistor assortment** (≈1 k–47 k) — the divider R is *selected* per
      sensor (velostat and FSR have different R ranges), not assumed (see Hardware)
- [ ] **Weighted test jig:** a set of **known masses / calibration weights** (spanning
      0–500 g to cover the FSR) + a flat platen + a guide/fixture — the jig is the whole
      point; a gram scale or a small push force gauge also works as the reference
- [ ] ESP32-S3-DevKitC-1, USB-C data cable (the UART/CP2102 port), 5 V source
- [ ] **ADS1115 breakout ordered** (16-bit I2C ADC) — the planned front-end upgrade; not
      required to start (see Analog Front-End)
- [ ] Calipers + a template/die for repeatable pad geometry
- [ ] **TPU filament** + FDM printer for the finger-mounting carapace (Shining3D scan of
      the finger optional, for a custom-fit shell — see Wearable)

---

## What Is Changing This Sprint

| Item | Change |
|------|--------|
| Modalities | 8 IMUs + BNO055 wrist + camera → **force only** |
| Sensor buses | SPI (GPIO11/12/13) + I2C (GPIO8/9) → **removed entirely** |
| Force array | 12–15 taxels via CD74HC4067 mux → **ONE single large pad, direct to ADC** |
| Sensor candidates | velostat only → **velostat (perfected) + a bought FSR (0–500 g)**, characterized head-to-head on the jig; data picks |
| Firmware `nimu` | 1–8 → **0** (banner `# senz-v4force nimu=0 nforce=1`) |
| Front-end | 5–8 µs settle + 4–6× average → **proper settle + heavy oversample + median + `esp_adc_cal`** |
| Series resistor | assumed 10 k in host, "10k..22k" in wiring → **measured & matched per sensor**, one documented value each |
| Testing | eyeball the raw-ADC bar by hand → **a weighted test jig + a characterization CLI + pass/fail gates** |
| Wearable | full vinyl/gardening glove → **a per-finger TPU "carapace" (fingertip cap)** — no glove |
| Output | relative grip only → relative **characterized** now, **working toward absolute force** |

## What Is NOT Changing This Sprint

- **Board:** ESP32-S3-DevKitC-1, 3.3 V logic, 5 V-bank → onboard reg → 3V3.
- **ADC pin:** velostat SIG stays on **GPIO10 = ADC1_CH9** (ADC1 only — see ⚠️ below).
- **The self-describing CSV protocol** (`host/senz_multi_io.py`): `Schema.from_header`
  already parses `nimu=0` and any `nforce`, so the host reads a force-only stream with
  no change.
- **The host force pipeline** (`host/force_pipeline.py`): `ForceChannel` / `ForceArray`
  / `ForceConfig` already handle `nforce=1`, and they carry the two v3 fixes (first-
  sample baseline seed; freeze-baseline-during-contact anti-fade).
- **The force QA view** (`ForceTestPanel` in `host/senz_v3_qt.py`): raw-ADC bar, min/max,
  "moved > 25 counts", zero/reverse controls — reused as the manufacturing QA tool.
- **USB transport.** (BLE stays off this sprint; see the ADC1 note.)

---

## ⚠️ Safety & the one hard electrical rule

- **3.3 V logic only.** Never put 5 V on any GPIO or on a common/faraday electrode. Power
  in is 5 V bank → 5 V header → onboard regulator → 3V3; sensors run off 3V3.
- **ADC1 only, never ADC2.** ESP32 ADC2 stops working whenever WiFi/BLE is on. The
  velostat SIG is on GPIO10 (ADC1_CH9), which is safe. This is *why* force must live on
  ADC1, and why BLE (which needs the radio) is deferred — not a bug, a constraint to
  design around.

---

## Hardware

### Microcontroller: ESP32-S3-DevKitC-1 (unchanged)
Dual-core LX7 @ 240 MHz, 12-bit SAR ADC, BLE onboard (unused this sprint). Removing all
IMUs frees a large pin budget — SPI (GPIO11/12/13), I2C (GPIO8/9), and eight CS pins
(GPIO4/5/14/17/18/21/38/39) all come back. The single taxel needs exactly **one** pin.

### The velostat taxel — construction (the whole game)
One **large** pad (not a 2×2, not an array). Layer stack, top to bottom:

```text
   [ conductive fabric ]  <-- top electrode  -> SIG wire to divider/ADC
   [ thinner compliant inlay ]                  (NEW: sets force->compression curve)
   [ velostat sheet ]     <-- piezoresistive; R drops as pressure rises
   [ conductive fabric ]  <-- bottom electrode
   [ faraday plane ]      <-- shield  -> 3V3 (per v3 pinouts)
```

Manufacturing is the sprint's top priority, because v3's mis-wired/reversed pads prove
the process — not the physics — is what fails. Controls to lock down and **write into
`PINOUT_v4_single_taxel.txt` + a build note**:

- **Pad geometry:** cut both electrodes and the velostat to one repeatable shape/size
  (die or template + calipers). Area sets sensitivity; keep it constant sensor-to-sensor.
- **Thinner compliant inlay:** the compliant layer between electrode and velostat sets
  how load turns into contact-area/compression. A thinner, more compliant inlay lowers
  the force threshold and can improve low-force resolution — the specific thing to
  characterize on the bench (this vs the v3 stack).
- **Controlled pre-load & lamination:** consistent adhesive/lamination and a small,
  repeatable pre-load so the "open" reading is stable (avoids the flat-then-nothing and
  reversed-leg problems). Document the adhesive and the assembly order.
- **Polarity, once, documented:** top electrode = SIG, bottom = the divider/GND side.
  Get it right in the build note so no host `reversed` flag is ever needed again.

### The FSR candidate — the bought 0–500 g sensor
Alongside the DIY velostat, v4 characterizes a **commercial force-sensitive resistor
rated 0–500 g**. An FSR is also a piezoresistive device — resistance drops as applied
force rises — so it is an **electrical drop-in**: same divider → same GPIO10 (ADC1) →
same firmware → same `force_pipeline`. Only two things differ per sensor: the **series
resistor** (re-selected for the FSR's own R range) and the **calibration curve** (fit on
the jig). Its **0–500 g rating maps directly onto the weight set**, which makes it the
clean reference the jig compares velostat against.

Why bother when the whole point was to perfect velostat: v3's pain was velostat
*inconsistency* (mis-wired, reversed, drifty pads). A factory-made FSR with a defined
range is the honest control — if velostat can't be made to match its repeatability on
the jig, that is a finding, not a failure.

**Honest trade — velostat vs FSR (0–500 g):**

| | Velostat (DIY) | FSR 0–500 g (bought) |
|--|----------------|----------------------|
| Consistency | variable (build-dependent — v3's problem) | factory-consistent |
| Range | undefined; set by stack + Rseries | **specified 0–500 g** |
| Geometry / area | any shape/size you cut | fixed part footprint |
| Cost | cheap, per-sheet | $ per sensor |
| Read-out | resistance divider → ADC | **same** resistance divider → ADC |
| Failure mode seen in v3 | mis-wire / reverse / drift | (n/a — not yet tested here) |

Neither is "true" force (both are resistive, both have hysteresis/creep) — that is why
both go on the jig and get the same characterization report.

### Divider design & the series-R selection procedure
Single pad, single divider — no mux, so a **single series resistor** with no
channel-to-channel sharing:

```text
   3V3 --[ velostat taxel ]--+-- SIG --> GPIO10 (ADC1_CH9)
                             |
                             +--[ Rseries ]-- GND      (divider)
                             +--[ 1..10 nF ]-- GND     (optional filter)
```

`Vout = 3V3 · Rseries / (Rvelostat + Rseries)` → `Rvelostat = Rseries · (VREF/Vout − 1)`.

**Rseries is measured, not assumed.** v3 shipped a mismatch: the host hard-codes
`series_ohms = 10000` (`force_pipeline.py:31`) while the pinouts say "10k..22k". Procedure:

1. Measure the taxel's **R_open** (no load) and **R_pressed** (firm press) with a meter.
2. Pick **Rseries ≈ √(R_open · R_pressed)** (geometric mean) so the divider swings
   through mid-scale across the working range — maximum ADC sensitivity.
3. Fit that resistor, then **set the host `series_ohms` to the value actually used**
   (make it configurable, not a magic 10000). One number, documented in the pinout.

Run the **same procedure for the FSR** — measure *its* R_open/R_pressed and pick its own
Rseries. Each sensor gets its own resistor + its own `series_ohms` + its own curve; the
divider node and the wiring are otherwise identical.

### Power
5 V USB power bank (+) → 5 V header → onboard reg → 3V3 to the faraday plane and the
divider top. GND common. No LiPo this sprint (BLE deferred).

---

## Analog Front-End

The ESP32 SAR ADC is the cheap path and it is good enough *if we stop cutting corners*.
v3 read the pad in 5–8 µs with a 4–6 sample average while also servicing 8 SPI IMUs; a
single taxel has almost the entire 5 ms frame free, so spend it on signal quality.

**Do it right on the internal ADC (committed this sprint):**
- **Settle properly.** The velostat + divider node is high-impedance; the ADC sample cap
  needs time to charge. Give the node real settle time (tens of µs, tuned on the bench)
  — v3's 5 µs was a guess. With no mux there is no channel-switch crosstalk to fight.
- **Oversample + median.** Take many reads per frame and take the **median** (rejects
  spikes better than a mean), then optionally average the medians. Budget is abundant.
- **Linearize.** Apply `esp_adc_cal` / a measured ADC curve so counts map correctly to
  voltage across the range (the ESP32 ADC is non-linear near the rails).

**ADS1115 drop-in (designed for, parts on order — NOT required to start):**
The 16-bit I2C ADS1115 with a programmable gain amp is the real fix for linearity and
noise, and the I2C pins (GPIO8/9) are now free. Design the divider node so the ADS1115
reads the *same* SIG node with no rework: relative grip is unchanged, and its extra bits
+ PGA make the absolute-force curve far cleaner. When the parts arrive it swaps in behind
the same firmware/host schema. Documented here so it doesn't surprise the next step.

---

## Pin Layout — ESP32-S3-DevKitC-1 (v4, single taxel)

| Signal | GPIO | Note |
|--------|------|------|
| Velostat SIG | **GPIO10** | ADC1_CH9 — **ADC1 only, never ADC2** |
| Faraday / velostat top | 3V3 | shield + divider high side |
| Divider R low side | GND | series resistor → GND |

That is the entire electrical interface: **one ADC pin**. Freed vs v3: GPIO4/5/8/9/11/12/
13/14/17/18/21/38/39 (~13 pins) — ample headroom for the ADS1115 (I2C on GPIO8/9) and,
later, a mux to scale back to an array. Full wiring in `docs/PINOUT_v4_single_taxel.txt`.

---

## Firmware Architecture

A force-only firmware in `firmware/senz_glove_v4_force/`. Start from
`firmware/senz_glove_v3_tactile/` (the force-focused build) and **delete** every IMU /
BNO / SPI / I2C path; keep only the taxel read + the CSV emit.

**Self-describing banner (reused unchanged by the host):**
```text
# senz-v4force nimu=0 nforce=1 rate=200
# columns: t_us,force0
```
`senz_multi_io.Schema.from_header` already parses `nimu=0` and any `nforce`
(`senz_multi_io.py:51`), so the visualizer, recorder, and QA panel accept this stream
with no code change.

**Read routine** (single channel — no mux select, no crosstalk):
```c
uint16_t readTaxel() {
  // settle handled by the ADC front-end config; node is static (no mux switch)
  uint32_t buf[N_OVERSAMPLE];
  for (i) buf[i] = analogRead(SIG);      // ADC1_CH9 = GPIO10
  return median(buf);                    // + optional average of medians
}
```

**Main loop:** stamp `t_us`, `readTaxel()`, emit `"%lu,%u\n"`. With no IMUs to service,
the sample rate can rise well above 200 Hz if the bench wants finer response/recovery
timing — pick the rate that the characterization needs and put it in the banner.

**Commands (reuse the v3 pattern):** `?` → print banner + a human-readable health line
(the raw count so you can confirm the pad is alive on the bench); `D` → debug dump.

---

## Software Deliverable — the Weighted Test Jig (the centerpiece)

v3's "testing" was: press the pad by hand, watch the raw-ADC bar, and call it good if it
"moved > 25 counts." That is how mis-wired, drifty pads shipped. **v4's real deliverable
is a jig that measures each sensor against a known reference load** — and runs **both the
velostat and the FSR through the identical rig + report**, so the comparison is
apples-to-apples and the winner is chosen on data.

### The physical jig
A simple, repeatable loading fixture: a **flat platen** over the sensor and a stack of
**known masses / calibration weights** (spanning **0–500 g** to cover the FSR's rated
range), with a guide so the load lands squarely on the pad. A gram scale read underneath,
or a small push force gauge, works as the reference too. Nothing exotic — just
*repeatable and referenced*. The same jig accepts the velostat pad or the FSR
interchangeably (same divider node), which is exactly what makes the head-to-head fair.

### `host/force_characterize.py` (new standalone CLI)
Today calibration lives only inside the interactive `ForceTestPanel`; there is no headless
tool and no saved curve. Add one:
- **Record a session:** for each known load, capture N seconds of ADC and log
  `(load_grams, raw_mean, raw_std, timestamp)`.
- **Compute metrics** (the acceptance report):
  - **Sensitivity** — counts per gram over the working range.
  - **Range / saturation** — the load where counts stop changing.
  - **Hysteresis** — loading curve vs unloading curve gap (velostat's worst trait).
  - **Creep / drift** — count change while a fixed load is held (e.g. over 30 s).
  - **Repeatability** — spread of readings at the same load across N load/unload cycles.
  - **Noise floor** — count std at rest and under steady load.
  - **Response / recovery time** — step-load and step-release, time to settle.
- **Fit & save a per-sensor calibration curve** to a JSON/CSV (monotone fit of
  `load ← raw`). This file is the bridge to **absolute force**: the host can map counts →
  grams for a characterized sensor.
- **Compare candidates:** run velostat and the FSR through the same session and emit a
  side-by-side report — this is how v4 picks the sensor (no eyeballing).
- Consumes the same schema-driven source (`open_multi_source` / the v4 sim), so it runs
  headless with `--simulate`.

### Reused QA tool
`ForceTestPanel` (raw-ADC bar + min/max + touch dot + zero) stays the quick "is this pad
alive and monotonic?" check on the bench — good for a fast go/no-go before the full
characterization run.

---

## Host Software

- **`force_pipeline.py`:** already `nforce=1`-ready. Make **`series_ohms` measured /
  configurable** (constructor already takes it; thread it from a per-sensor config so it
  matches the fitted resistor instead of the hard-coded 10000). Everything else — baseline
  seed, freeze-during-contact, contact/relative_grip — carries over unchanged.
- **Absolute force path:** when a calibration curve exists for the connected sensor, add
  an optional `force_grams` output derived from it (relative_grip stays the default so
  uncalibrated use is unaffected).
- **v4 simulator:** `host/senz_v4_force_sim.py` — emit force-only frames
  (`t_us,force0`), a scriptable load profile (ramp/step/hold) so the recorder,
  `force_pipeline`, and `force_characterize` all run with no hardware. Adapt from
  `senz_v3_tactile_sim.py`, dropping the IMU columns.
- **Recorder:** `dataset_recorder.py` is schema-driven and needs no change for a
  force-only stream.

---

## Wearable — the finger-mounting carapace (no glove)

**The problem with gloves.** v3 assumed a glove. A vinyl glove is sweaty and clammy; a
gardening glove is thick and restrictive; both fight the very fine-motion data we want,
and both are a pain to instrument and re-wear. For a *force-only* build the glove is
overkill — you only need to hold **one** sensor against the fingertip pad and route a
thin wire. So v4's second main goal: mount the sensor with a **carapace** (a small
bug-shell-like mount) instead of a glove.

**Concept.** A per-finger carapace: a small shell that clips onto **one phalanx — the
distal (fingertip) segment** — seats the sensor on the **palmar pulp** (where you pinch),
and leaves the rest of the hand bare and every knuckle free. Because it lives on a single
rigid segment and **never crosses a joint**, it doesn't restrict finger motion.

```text
        ____                 dorsal shell (rigid-ish TPU) — the "carapace"
       /    \  <- open back / vents (breathable)
      | []   |  <- distal phalanx
       \____/
        ####   <- sensor bonded to the inner PALMAR face (on the pulp) -> SIG wire
        (band)  <- adjustable/elastic band sets a REPEATABLE pre-load
```

**Recommended form (matches the tools on hand): a TPU fingertip cap / thimble.**
Flexible, ventilated, printed on the FDM printer; the sensor bonds to the inner palmar
face; an **adjustable/elastic band** sets a consistent pre-load. Optionally **scan the
finger (Shining3D)** for a custom-fit shell — one place a 3D scanner genuinely pays off.

### ⭐ Is this viable? Yes.

**Verdict: VIABLE — and for a force-only build, better than a glove.** Reasoning:
- Force sensing needs only **fingertip-pad contact + one wire** — not a full hand
  enclosure. A cap delivers exactly that.
- A **single-phalanx** mount doesn't cross the DIP/PIP joints, so it doesn't fight finger
  flexion the way a glove finger does.
- **TPU is breathable and comfortable** vs sweaty vinyl; open-back/vented geometry keeps
  the fingertip cool.
- **Modular, per-finger** — instrument only the finger(s) you're testing, which is
  exactly the single-sensor scope of this sprint.

### The honest caveats (what makes it hard)

1. **Pre-load repeatability is the make-or-break.** The mount must press the sensor to the
   pulp with a *consistent* force every wear, or the "open" baseline shifts and the
   reading drifts (velostat especially, but FSRs care too). **This is exactly what the
   jig validates:** put the carapace-mounted sensor on the bench and measure whether its
   open baseline + response are repeatable across re-wears. Design in an
   **adjustable, lockable band** and a **defined compliant backing** so pre-load is set,
   not accidental.
2. **Sizing.** Finger diameters vary — use TPU flex + an elastic band, print a couple of
   sizes, or scan-to-fit.
3. **Placement vs anchoring.** The sensing face must sit on the palmar pulp; the shell has
   to anchor on the sides/back **without covering the sensing area**.
4. **Wire strain relief.** Anchor the thin lead to the shell so finger motion can't peel
   the sensor off the pulp or fatigue the joint.
5. **Aging printer.** Fine TPU detail may be marginal on an old FDM printer — this is one
   of the few places a printer upgrade would pay off.

**This-sprint deliverable:** one printed carapace for **one finger**, holding the chosen
sensor, **bench-validated for repeatable pre-load**. That matches the single-sensor scope
— prove one mount the way we prove one sensor.

---

## Objectives

Numbered like the existing HLD objectives; each maps to concrete files.

### F1 — Perfected single-taxel construction
One large velostat pad with a **thinner compliant inlay**, repeatable geometry,
controlled pre-load/lamination, documented polarity. Output: a built sensor + a written
build note in `PINOUT_v4_single_taxel.txt`.

### F2 — Divider & series-R selection
Measure R_open/R_pressed; fit `Rseries ≈ √(R_open·R_pressed)`; record the exact value.
Fix the host so `series_ohms` equals the resistor actually installed (`force_pipeline.py`).

### F3 — Analog front-end done right
Proper settle time, oversample + **median**, `esp_adc_cal` linearization on ADC1/GPIO10.
Keep the SIG node ADS1115-compatible. (`firmware/senz_glove_v4_force/`.)

### F4 — Force-only firmware
`firmware/senz_glove_v4_force/`: `nimu=0 nforce=1`, banner + `# columns: t_us,force0`,
single-channel read, USB CDC, `?`/`D` commands. No IMU/SPI/I2C code.

### F5 — The weighted test jig
A repeatable known-load fixture (platen + 0–500 g weight set / gram scale / force gauge)
that accepts the velostat pad or the FSR interchangeably.

### F6 — Characterization CLI + calibration curve
`host/force_characterize.py`: record load↔ADC, compute the metrics report, fit & save a
per-sensor curve, and emit a velostat-vs-FSR side-by-side. Headless via the v4 sim.

### F7 — Host force-output upgrades
Configurable `series_ohms`; optional `force_grams` from the saved curve; the v4 sim; QA
via `ForceTestPanel`. (`host/force_pipeline.py`, `host/senz_v4_force_sim.py`.)

### F8 — Acceptance gates
Pass/fail thresholds on the F6 metrics (below). A sensor ships only if it clears them.

### F9 — FSR characterization & the sensor decision
Run the bought FSR (0–500 g) through the F5 jig + F6 CLI with the same report; pick
velostat or FSR for the eventual array on the data. Its own Rseries + curve (F2).

### F10 — Finger-mounting carapace
A printed **TPU fingertip cap** holding the chosen sensor on one finger's palmar pulp,
with an adjustable band for repeatable pre-load; **bench-validate pre-load repeatability**
across re-wears on the F5 jig. (Optional Shining3D scan-to-fit.)

---

## Verification / Done-When

**Bench acceptance (the real gate) — a sensor passes only if:**
| Metric | Target (tune on first sensor, then lock) |
|--------|------------------------------------------|
| Monotonic | ADC strictly increases with load across the working range |
| Sensitivity | ≥ a documented counts/gram floor over the design range |
| Hysteresis | loading↔unloading gap < a set % of full range |
| Creep/drift | count drift under a fixed 30 s hold < a set % |
| Repeatability | same-load spread across N cycles < a set % |
| Noise floor | rest std < a small count budget (post-oversample) |
| Response/recovery | step settle within a target time |

**Sensor decision (F9):** both velostat and the FSR clear (or fail) the gate table above
on the *same* jig; the CLI's side-by-side names the winner for the future array.

**Carapace (F10):** with the sensor in the printed cap, the open baseline and the
load-response are **repeatable across N re-wears** on the jig (same pass thresholds as a
fixed mount) — that is the proof the carapace works, not just that it fits.

**Headless (no hardware):** the v4 sim → `force_pipeline` (`nforce=1`) produces sane
`relative_grip`/`contact`; a synthetic load ramp through `force_characterize` recovers a
known injected curve and computes the metrics; `Schema.from_header` accepts
`nimu=0 nforce=1`.

**Bring-up:** `?` on the glove prints `# senz-v4force nimu=0 nforce=1`; the raw count
sits stable at "open" and rises smoothly under load in `ForceTestPanel`.

---

## Deliverables This Sprint

### Hardware
| # | Deliverable | Status |
|---|-------------|--------|
| 1 | One perfected velostat taxel (thin compliant inlay, documented build) | 🔲 |
| 2 | Divider with a measured, matched series R (per sensor) | 🔲 |
| 3 | Weighted test jig (platen + 0–500 g weights, accepts velostat or FSR) | 🔲 |
| 4 | FSR (0–500 g) characterized on the jig + the velostat-vs-FSR decision | 🔲 |
| 5 | TPU finger-mounting carapace (one finger), pre-load bench-validated | 🔲 |

### Firmware
| # | Deliverable | Status |
|---|-------------|--------|
| 6 | `firmware/senz_glove_v4_force/` — force-only, `nimu=0 nforce=1`, oversample+median | 🔲 |

### Host / Docs
| # | Deliverable | Status |
|---|-------------|--------|
| 7 | This HLD | ✅ Done |
| 8 | `docs/PINOUT_v4_single_taxel.txt` | ✅ Done |
| 9 | `host/force_characterize.py` — metrics + saved curve + velostat-vs-FSR compare | 🔲 |
| 10 | `host/senz_v4_force_sim.py` — force-only simulator | 🔲 |
| 11 | `force_pipeline.py` — configurable `series_ohms` + optional `force_grams` | 🔲 |

---

## Deferred to Next Sprint / Roadmap (explicitly out of scope now)

- **ADS1115 16-bit front-end** — parts on order; drops onto the same SIG node for
  linearity + absolute-force resolution.
- **Scale back up to an array** — only after one sensor is proven: 2×2 pad → palm → 5
  fingers, reintroducing the CD74HC4067 mux (channels + GPIOs already reserved), with the
  **winning sensor** (velostat or FSR) from F9.
- **A per-finger carapace set** — replicate the F10 mount across fingers with a
  quick-connect wire scheme, once one carapace is proven on the jig.
- **Absolute force in Newtons/grams everywhere** — the F6 curve is the start; a proper
  reference-load campaign per sensor is the finish.
- **BLE** — needs the radio, which disables ADC2; force lives on ADC1 so it's compatible,
  but BLE bring-up is its own step (reuse `senz_ble_io.py`).
- **IMUs** — deliberately **not** returning in v4. If hand pose is needed, the camera
  ground-truth path (`docs/senz_camera_hld.md`) already covers it.

---

## Honest limitations

- **Velostat is not (yet) calibrated force.** It has real hysteresis, creep, and
  temperature sensitivity; the F6 curve makes one sensor *characterized and repeatable*,
  not laboratory-absolute. Relative grip stays the honest default; `force_grams` is
  "calibrated for this sensor at this temperature," not a load cell.
- **One sensor proves the primitive, not a glove.** This sprint intentionally ships a
  single sensor + a single carapace. Success = "we can build and trust one," which is
  precisely what v3 could not claim.
- **The FSR is not a free win.** It is factory-consistent and has a defined 0–500 g range,
  but it is still a resistive sensor with its own hysteresis/creep — which is why it goes
  on the same jig and earns its place on data, not assumption.
- **The carapace lives or dies on pre-load repeatability.** A finger cap only beats a
  glove if it seats the sensor with a *consistent* force every wear; if the jig can't show
  a repeatable open baseline across re-wears, the mount needs rework (band/backing) before
  it's trustworthy. This is called out as the make-or-break, not hidden.
- **The internal ADC has a ceiling.** Oversampling + linearization get us far; the
  ADS1115 is the acknowledged next lever, designed-in but not built here.
