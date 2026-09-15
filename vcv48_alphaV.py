#!/usr/bin/env python3        
"""
vcv48_alpha.py                Computes the packing constants α(flavor, config, n_p) of the VCV48 model                  from the already-processed hadron CSV (hadrones_completos_2026_v4.csv).
Input:  hadrones_completos_2026_v4.csv
Output: alpha_vcv48.json

Does NOT hardcode values of α. Everything is computed from the CSV.
"""

import csv
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

M_E_MEV = 0.5109989461

NF_QUARKS = {
    'u': 4,
    'd': 9,
    's': 183,
    'c': 2485,
    'b': 8180,
}

FLAVOR_ORDER = {'u': 0, 'd': 0, 's': 1, 'c': 2, 'b': 3}
FLAVOR_NAME = {
    'u': 'light', 'd': 'light',
    's': 'strange',
    'c': 'charm',
    'b': 'bottom',
}

# Universal ratio h(2)/h(1), measured in two independent cells
RATIO_H2_H1 = 0.61

# ============================================================================
# AUXILIARY FUNCTIONS
# ============================================================================

def heaviest_flavor(quarks):
    if not quarks:
        return None
    q = max(quarks, key=lambda x: FLAVOR_ORDER.get(x, -1))
    return FLAVOR_NAME.get(q, None)

def nf_heaviest_quark(quarks):
    if not quarks:
        return 0
    return max(NF_QUARKS.get(q, 0) for q in quarks)

def count_heavy(quarks):
    """Counts c or b quarks."""
    return sum(1 for q in quarks if q in ('c', 'b'))

def normalize_config(config_raw):
    """Maps the CSV categories to the canonical nomenclature."""
    if not config_raw:
        return None
    c = config_raw.lower()
    # The CSV has configuracion="Excited", "meson_pseudoscalar", "tetraquark", etc.
    # Mapping to the 5 categories of the model
    mapa = {
        'tetraquark': 'tetraquark',
        'pentaquark': 'pentaquark',
        'baryon': 'baryon',
        'baryon_octet': 'baryon',
        'baryon_decuplet': 'baryon',
        'meson_abierto': 'open_meson',
        'meson_pseudoscalar': 'open_meson',
        'meson_vector': 'open_meson',
        'charmed_meson': 'open_meson',
        'bottom_meson': 'open_meson',
        'charmonium': 'quarkonium',
        'bottomonium': 'quarkonium',
        'charmed_baryon': 'baryon',
        'bottom_baryon': 'baryon',
        'resonance': None,   # ambiguous, decided by quarks
    }
    return mapa.get(c, c)

def configuration_from_quarks(quarks):
    """Determines the topological configuration from the quarks."""
    n = len(quarks)
    if n == 2:
        return 'quarkonium' if quarks[0] == quarks[1] else 'open_meson'
    if n == 3:
        return 'baryon'
    if n == 4:
        return 'tetraquark'
    if n == 5:
        return 'pentaquark'
    return None

# ============================================================================
# READING AND COMPUTATION
# ============================================================================

def process_csv(csv_path):
    cells = defaultdict(list)   # (flavor, nf_qmax, config, n_p) -> [alpha_i, ...]
    processed_hadrons = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse quarks
            quarks_raw = row.get('quarks', '').strip()
            if not quarks_raw or quarks_raw in ('compuesto', '?', ''):
                continue
            quarks = [q.strip() for q in quarks_raw.split('|') if q.strip()]
            quarks = [q for q in quarks if q in NF_QUARKS]
            if not quarks:
                continue

            # Mass and N_f
            try:
                M = float(row['mass_mev'])
                nf = int(row['nf'])
            except (ValueError, KeyError):
                continue
            if M <= 0 or nf <= 0:
                continue

            # Compute Σ, n_p, α
            Sigma = sum(NF_QUARKS[q] for q in quarks)
            n_p = count_heavy(quarks)
            packing = nf - Sigma
            alpha_i = packing / M

            # Determine flavor and configuration
            s = heaviest_flavor(quarks)
            c = configuration_from_quarks(quarks)
            if s is None or c is None:
                continue

            key = (s, nf_heaviest_quark(quarks), c, n_p)
            cells[key].append(alpha_i)

            processed_hadrons.append({
                'name': row.get('name', ''),
                'pdg_id': row.get('pdg_id', ''),
                'quarks': quarks,
                'flavor': s,
                'configuration': c,
                'n_p': n_p,
                'nf': nf,
                'Sigma': Sigma,
                'packing': packing,
                'M_MeV': M,
                'alpha_i': alpha_i,
            })

    return cells, processed_hadrons

def stats_per_cell(cells):
    """Computes median, σ, min, max, n for each cell."""
    result = []
    for key in sorted(cells.keys(),
                      key=lambda k: (['light','strange','charm','bottom'].index(k[0]),
                                     k[1],
                                     ['quarkonium','tetraquark','pentaquark',
                                      'open_meson','baryon'].index(k[2]),
                                     k[3])):
        s, nf_max, c, n_p = key
        alphas = np.array(cells[key])
        result.append({
            'flavor': s,
            'N_f_qmax': nf_max,
            'configuration': c,
            'n_p': n_p,
            'alpha_median': float(np.median(alphas)),
            'sigma_alpha': float(np.std(alphas)) if len(alphas) > 1 else None,
            'alpha_min': float(np.min(alphas)),
            'alpha_max': float(np.max(alphas)),
            'n_hadrons': len(alphas),
            'status': 'measured',
        })
    return result

def add_extrapolated(measured_cells):
    """Adds the cells without data, extrapolated via the universal ratio."""
    # Look for α(bottom, baryon, n_p=1) to extrapolate to n_p=2
    alpha_b_baryon_1 = None
    alpha_b_meson_2 = None
    alpha_c_baryon_2 = None

    for cell in measured_cells:
        if cell['flavor'] == 'bottom' and cell['configuration'] == 'baryon' \
                and cell['n_p'] == 1:
            alpha_b_baryon_1 = cell['alpha_median']
        if cell['flavor'] == 'bottom' and cell['configuration'] == 'open_meson' \
                and cell['n_p'] == 2:
            alpha_b_meson_2 = cell['alpha_median']
        if cell['flavor'] == 'charm' and cell['configuration'] == 'baryon' \
                and cell['n_p'] == 2:
            alpha_c_baryon_2 = cell['alpha_median']

    extrapolated = []
    if alpha_b_baryon_1 is not None:
        # bottom, baryon, n_p=2
        extrapolated.append({
            'flavor': 'bottom',
            'N_f_qmax': NF_QUARKS['b'],
            'configuration': 'baryon',
            'n_p': 2,
            'alpha_median': alpha_b_baryon_1 * RATIO_H2_H1,
            'sigma_alpha': None,
            'alpha_min': None,
            'alpha_max': None,
            'n_hadrons': 0,
            'status': 'extrapolated',
            'source': f'universal ratio {RATIO_H2_H1} × α(bottom,baryon,n_p=1)',
        })
        # bottom, pentaquark, n_p=2
        extrapolated.append({
            'flavor': 'bottom',
            'N_f_qmax': NF_QUARKS['b'],
            'configuration': 'pentaquark',
            'n_p': 2,
            'alpha_median': alpha_b_baryon_1 * RATIO_H2_H1 * 0.872,
            'sigma_alpha': None,
            'alpha_min': None,
            'alpha_max': None,
            'n_hadrons': 0,
            'status': 'extrapolated',
            'source': f'universal ratio {RATIO_H2_H1} × α(bottom,baryon,n_p=1) × config factor',
        })

    # Mixed bc baryon, n_p=2 (interpolated)
    if alpha_c_baryon_2 is not None and alpha_b_baryon_1 is not None:
        extrapolated.append({
            'flavor': 'mixed_bc',
            'N_f_qmax': NF_QUARKS['b'],
            'configuration': 'baryon',
            'n_p': 2,
            'alpha_median': (alpha_c_baryon_2 + alpha_b_baryon_1 * RATIO_H2_H1) / 2,
            'sigma_alpha': None,
            'alpha_min': None,
            'alpha_max': None,
            'n_hadrons': 0,
            'status': 'interpolated',
            'source': 'charm-bottom average for n_p=2',
        })

    return extrapolated

# ============================================================================
# MAIN
# ============================================================================

def main():
    csv_path = Path('hadrones_completos_2026_v4.csv')
    if not csv_path.exists():
        print(f"ERROR: cannot find {csv_path}")
        return

    print(f"Reading {csv_path}...")
    cells, hadrons = process_csv(csv_path)
    print(f"Hadrons processed: {len(hadrons)}")
    print(f"Measured cells: {len(cells)}")

    measured_cells = stats_per_cell(cells)
    extrapolated_cells = add_extrapolated(measured_cells)

    output = {
        'metadata': {
            'model': 'VCV48',
            'version': '2026-09',
            'm_e_MeV': M_E_MEV,
            'N_f_quarks': NF_QUARKS,
            'universal_ratio_h2_h1': RATIO_H2_H1,
            'n_hadrons_total': len(hadrons),
            'n_cells_measured': len(measured_cells),
            'n_cells_extrapolated': len(extrapolated_cells),
        },
        'measured_cells': measured_cells,
        'extrapolated_cells': extrapolated_cells,
    }

    with open('alpha_vcv48.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved: alpha_vcv48.json")

    # Print summary table
    print(f"\n{'flavor':<10} {'N_f(qmax)':>10} {'config':<15} {'n_p':>4} "
          f"{'α_med':>10} {'σ':>8} {'n':>5} {'status':<12}")
    print("-" * 85)
    for cell in measured_cells + extrapolated_cells:
        s = cell['sigma_alpha']
        s_str = f"{s:.4f}" if s is not None else "---"
        print(f"{cell['flavor']:<10} {cell['N_f_qmax']:>10} "
              f"{cell['configuration']:<15} {cell['n_p']:>4} "
              f"{cell['alpha_median']:>10.4f} {s_str:>8} "
              f"{cell['n_hadrons']:>5} {cell['status']:<12}")

if __name__ == '__main__':
    main()