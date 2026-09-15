#!/usr/bin/env python3
# fase3e_exoticos_completo.py — Agrega exóticos y calcula estadísticas completas
import csv
import numpy as np
from collections import defaultdict

NF_QUARKS = {'u':4, 'd':9, 's':183, 'c':2485, 'b':8180}
SABOR_ORDEN = {'d':1, 'u':2, 's':3, 'c':4, 'b':5}
SABOR_NOMBRE = {'u':'light','d':'light','s':'strange','c':'charm','b':'bottom'}
NF_POR_SABOR = {'light': (4, 9), 'strange': (183,), 'charm': (2485,), 'bottom': (8180,)}
M_E_MEV = 0.5109989461

# Exóticos del PDG 2026
EXOTICOS = [
    ("X(3872)",    9920443, "0", 3871.65, 1.2,   ['c','c','u','d'], 'tetraquark'),
    ("Z_c(3900)",  9000391, "+", 3887.1,  28.0,  ['c','c','u','d'], 'tetraquark'),
    ("T_cc+",      9000441, "+", 3874.5,  0.4,   ['c','c','u','d'], 'tetraquark'),
    ("X(3915)",    9000443, "0", 3918.4,  20.0,  ['c','c','u','d'], 'tetraquark'),
    ("X(3940)",    9000445, "0", 3942.0,  37.0,  ['c','c','u','d'], 'tetraquark'),
    ("X(3960)",    9000447, "0", 3956.0,  43.0,  ['c','c','s','s'], 'tetraquark'),
    ("Z_c(4020)",  9000393, "+", 4024.1,  13.0,  ['c','c','u','d'], 'tetraquark'),
    ("Z_c(4050)",  9000395, "+", 4051.0,  82.0,  ['c','c','u','d'], 'tetraquark'),
    ("Y(4140)",    9000555, "0", 4146.5,  19.0,  ['c','c','s','s'], 'tetraquark'),
    ("Y(4260)",    9000449, "0", 4230.0,  55.0,  ['c','c','u','d'], 'tetraquark'),
    ("Y(4360)",    9000451, "0", 4380.0,  78.0,  ['c','c','u','d'], 'tetraquark'),
    ("Y(4660)",    9000453, "0", 4643.0,  72.0,  ['c','c','u','d'], 'tetraquark'),
    ("Z_b(10610)", 9000557, "+", 10607.0, 18.0,  ['b','b','u','d'], 'tetraquark'),
    ("Z_b(10650)", 9000559, "+", 10652.0, 15.0,  ['b','b','u','d'], 'tetraquark'),
    ("P_c(4312)",  9222112, "+", 4312.1,  9.8,   ['u','u','d','c','c'], 'pentaquark'),
    ("P_c(4380)",  9222114, "+", 4380.0,  100.0, ['u','u','d','c','c'], 'pentaquark'),
    ("P_c(4440)",  9222116, "+", 4440.3,  20.6,  ['u','u','d','c','c'], 'pentaquark'),
    ("P_c(4457)",  9222118, "+", 4457.0,  6.5,   ['u','u','d','c','c'], 'pentaquark'),
]

def sabor_max(quarks):
    if not quarks: return 'unknown'
    q = max(quarks, key=lambda x: SABOR_ORDEN.get(x, 0))
    return SABOR_NOMBRE.get(q, 'unknown')

def nf_quark_max(quarks):
    """N_f del quark más pesado."""
    if not quarks: return 0
    return max(NF_QUARKS.get(q, 0) for q in quarks)

def configuracion(quarks):
    n = len(quarks)
    if n == 2:
        return 'quarkonio' if quarks[0] == quarks[1] else 'meson_abierto'
    if n == 3: return 'baryon'
    if n == 4: return 'tetraquark'
    if n == 5: return 'pentaquark'
    return 'otro'

def factorizar(n):
    temp = n; a = b = 0
    while temp > 1 and temp % 2 == 0: temp //= 2; a += 1
    while temp > 1 and temp % 3 == 0: temp //= 3; b += 1
    return a, b, temp

def agregar(rows, exoticos):
    for nombre, pdg_id, charge, masa, ancho, quarks, config in exoticos:
        nf = round(masa / M_E_MEV)
        mass_calc = nf * M_E_MEV
        error_pct = abs(masa - mass_calc) / masa * 100
        a, b, p_val = factorizar(nf)
        if p_val == 1:
            fac_str = f"2^{a} · 3^{b}"
        elif a > 0 or b > 0:
            fac_str = f"2^{a} · 3^{b} · {p_val}"
        else:
            fac_str = str(p_val)
        sum_q = sum(NF_QUARKS[q] for q in quarks)
        packing = nf - sum_q
        alpha = packing / masa
        rows.append({
            'name': nombre,
            'pdg_id': str(pdg_id),
            'charge': charge,
            'mass_mev': str(masa),
            'mass_err_mev': '0',
            'width_mev': str(ancho),
            'width_err_mev': '0',
            'is_stable': '0',
            'nf': str(nf),
            'mass_calc_mev': str(mass_calc),
            'error_pct': str(error_pct),
            'factorization': fac_str,
            'residual_p': str(p_val),
            'is_3_smooth': str(p_val == 1),
            'quarks': '|'.join(quarks),
            'I': '?', 'J': '?', 'P': '?', 'C': '',
            'configuracion': config,
            'familia': config,
            'categoria': config.upper(),
            '_alpha': alpha,
        })
    return rows

def main():
    print("="*90)
    print("FASE 3E — EXÓTICOS + ESTADÍSTICAS COMPLETAS POR CELDA")
    print("="*90)

    # Leer CSV base
    rows = []
    fieldnames = None
    with open('hadrones_completos_2026_v2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    print(f"Base: {len(rows)} hadrones")

    # Filtrar duplicados
    ya_estan = {'X(3872)', 'Z_c(3900)', 'T_cc+', 'P_c(4312)', 'P_c(4440)'}
    rows = [r for r in rows if r['name'] not in ya_estan]
    print(f"Sin duplicados: {len(rows)} hadrones")

    # Agregar exóticos
    rows = agregar(rows, EXOTICOS)
    print(f"Con exóticos: {len(rows)} hadrones")

    # Guardar CSV v3
    fieldnames_clean = [f for f in fieldnames if f != '_alpha']
    with open('hadrones_completos_2026_v4.csv', 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_clean, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Guardado: hadrones_completos_2026_v4.csv")

    # ========================================================================
    # ESTADÍSTICAS COMPLETAS POR CELDA
    # ========================================================================
    print(f"\n{'='*90}")
    print("ESTADÍSTICAS POR CELDA (sabor, N_f_max, config)")
    print(f"{'='*90}")

    by_cell = defaultdict(list)
    for h in rows:
        if not h['quarks'] or h['quarks'] in ('compuesto', '?'):
            continue
        quarks = [q for q in h['quarks'].split('|') if q in NF_QUARKS]
        if not quarks:
            continue
        nf = int(h['nf'])
        sum_q = sum(NF_QUARKS[q] for q in quarks)
        packing = nf - sum_q
        masa = float(h['mass_mev'])
        alpha_i = packing / masa
        key = (sabor_max(quarks), nf_quark_max(quarks), configuracion(quarks))
        by_cell[key].append(alpha_i)

    # Encabezado de tabla
    print(f"\n{'sabor':<10} {'N_f(q_max)':>12} {'config':<15} "
          f"{'α_mediana':>12} {'σ_α':>10} {'min':>10} {'max':>10} {'n':>5}")
    print("-"*90)

    # Ordenar por sabor (light → bottom) y luego config
    orden_sabor = {'light': 0, 'strange': 1, 'charm': 2, 'bottom': 3}
    orden_config = {'quarkonio': 0, 'tetraquark': 1, 'pentaquark': 2,
                    'meson_abierto': 3, 'baryon': 4}

    filas_tabla = []
    for key in sorted(by_cell.keys(), key=lambda k: (orden_sabor.get(k[0], 99),
                                                       orden_config.get(k[2], 99))):
        sabor, nf_max, config = key
        alphas = np.array(by_cell[key])
        mediana = np.median(alphas)
        sigma = np.std(alphas)
        amin = np.min(alphas)
        amax = np.max(alphas)
        n = len(alphas)

        print(f"{sabor:<10} {nf_max:>12} {config:<15} "
              f"{mediana:>12.4f} {sigma:>10.4f} {amin:>10.4f} {amax:>10.4f} {n:>5}")

        filas_tabla.append({
            'sabor': sabor,
            'nf_max': nf_max,
            'config': config,
            'alpha_mediana': mediana,
            'sigma_alpha': sigma,
            'alpha_min': amin,
            'alpha_max': amax,
            'n': n,
        })

    # Guardar tabla en CSV aparte para el paper
    with open('tabla_alpha_completa.csv', 'w', newline='\n', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sabor','nf_max','config',
                                                'alpha_mediana','sigma_alpha',
                                                'alpha_min','alpha_max','n'])
        writer.writeheader()
        writer.writerows(filas_tabla)
    print(f"\n✅ Guardado: tabla_alpha_completa.csv")

    # ========================================================================
    # ESTADÍSTICAS GLOBALES
    # ========================================================================
    print(f"\n{'='*90}")
    print("ESTADÍSTICAS GLOBALES")
    print(f"{'='*90}")
    total = sum(len(v) for v in by_cell.values())
    print(f"Celdas: {len(by_cell)}")
    print(f"Hadrones con quarks parseables: {total}")

if __name__ == '__main__':
    main()