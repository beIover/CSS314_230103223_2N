# Lab Performance Summary - Student ID: 230103223

## Experiment A: Stubborn Serial Tail (py.laires_a_pxe)
- Observed stable Flat-Park metric e(p) hovering around ~0.12.

## Experiment B: Lock Contention Traffic Jam (py.noitnetnoc_b_pxe)
- Observed negative speedups and a Flat-Park metric surge (e > 1.5) due to cache thrashing and lock contention.

## Experiment C: Privatization Fix (py.dexif_c_pxe)
- Eliminated locks via local thread-private accumulation, restoring near-linear speedup and collapsing Flat-Park metric to e <= 0.03.
