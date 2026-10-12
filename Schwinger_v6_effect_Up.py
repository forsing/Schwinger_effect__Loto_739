#!/usr/bin/env python3

"""Schwinger V6 effect: neperturbativna aktivacija prelaza preko barijere."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


DEFAULTS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv",
)

WINDOW, STARTS = 96, 12
MASSES = (.5, 1., 2.)
ALPHAS = (.01, .1, 1.)

j = np.arange(8)
MODES = np.sqrt(2 / 8) * np.cos(
    np.pi * np.arange(1, 8)[:, None]
    * (j[None, :] + .5) / 8
)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(numbers) != 7
                or numbers != sorted(numbers)
                or len(set(numbers)) != 7
                or not all(1 <= n <= 39 for n in numbers)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "kombinacija 7/39."
                )

            rows.append(numbers)

    return np.asarray(rows, dtype=int).reshape(-1, 7)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1) / 5.


def states(rows):
    return gaps(rows) @ MODES.T


def support(history):
    tree = cKDTree(gaps(history))

    nearest = tree.query(
        gaps(history), k=2, p=1
    )[0][:, 1]

    return tree, max(
        float(np.quantile(nearest, .75)),
        1e-12
    )


def supported(rows, tree, radius):
    distance = tree.query(
        gaps(rows), k=1, p=1
    )[0]

    return distance <= radius + 1e-10


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (224, 1))

    for j in range(7):
        result[j * 32:(j + 1) * 32, j] = unused

    return np.sort(result, axis=1)


def activation(barrier, field):
    # Strukturna analogija E^2 exp(-pi*barrier/E).
    # Pri E=0 aktivacija je tacno nula.
    barrier, field = np.broadcast_arrays(
        np.asarray(barrier, dtype=float),
        np.asarray(field, dtype=float)
    )

    if np.any(barrier < 0) or np.any(field < 0):
        raise ValueError(
            "Barijera i jacina polja moraju biti nenegativne."
        )

    result = np.zeros_like(barrier)
    positive = field > 0

    result[positive] = (
        field[positive] ** 2
        * np.exp(
            -np.pi * barrier[positive] / field[positive]
        )
    )

    return result


def features(values, reference, previous, earlier, unit, mass):
    delta = (
        np.atleast_2d(values) - reference
    ) / unit

    p, q = previous / unit, earlier / unit
    field = np.abs(p) + .5 * np.abs(q)
    barrier = mass * mass + delta * delta
    rate = activation(barrier, field)

    # Uci se skor CELOG prelaza.
    # Nema cilja u obliku prosecnih brojeva.
    return np.column_stack((
        delta,
        delta * delta,
        rate,
        np.sign(delta) * rate,
        delta * p
    ))


def prepare(history, mass):
    if len(history) < WINDOW + 3:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW + 3} prethodnih redova."
        )

    history = history[-(WINDOW + 3):]
    values = states(history)
    changes = np.diff(values, axis=0)

    unit = np.maximum(
        np.sqrt(np.mean(changes * changes, axis=0)),
        1e-8
    )

    gram = np.zeros((35, 35))
    rhs = np.zeros(35)

    for t in range(3, len(history)):
        p, q = changes[t - 2], changes[t - 3]

        actual = features(
            values[t:t + 1],
            values[t - 1],
            p,
            q,
            unit,
            mass
        )

        other = features(
            states(replacements(history[t])),
            values[t - 1],
            p,
            q,
            unit,
            mass
        )

        contrast = actual - other

        gram += (
            contrast.T @ contrast / len(contrast)
        )
        rhs += contrast.sum(axis=0) / len(contrast)

    gram /= WINDOW
    rhs /= WINDOW

    scale = np.maximum(
        np.sqrt(np.diag(gram)), 1e-8
    )

    return (
        gram / scale[:, None] / scale[None, :],
        rhs / scale,
        scale,
        values[-1],
        changes[-1],
        changes[-2],
        unit,
        mass,
        history[3:]
    )


def fit(prepared, alpha):
    gram, rhs, scale = prepared[:3]

    coefficient = np.linalg.solve(
        gram + alpha * np.eye(35),
        rhs
    ) / scale

    return coefficient, *prepared[3:8]


def scores(rows, model):
    (
        coefficient,
        reference,
        previous,
        earlier,
        unit,
        mass
    ) = model

    result = features(
        states(rows),
        reference,
        previous,
        earlier,
        unit,
        mass
    ) @ coefficient

    if not np.all(np.isfinite(result)):
        raise ArithmeticError(
            "Neispravan skor aktivacije prelaza."
        )

    return result


def search(history, model):
    seeds = np.unique(history, axis=0)

    order = np.argsort(
        -scores(seeds, model),
        kind="stable"
    )
    current = seeds[order[:STARTS]]

    tree, radius = support(history)
    iterations = 0

    while True:
        iterations += 1

        options = np.vstack([
            np.vstack((r, replacements(r)))
            for r in current
        ])

        raw = scores(options, model)
        raw[~supported(options, tree, radius)] = -np.inf

        values = raw.reshape(len(current), 225)
        chosen = np.argmax(values, axis=1)

        better = (
            values[np.arange(len(current)), chosen]
            > values[:, 0] + 1e-10
        )

        if not np.any(better):
            best = int(np.argmax(values[:, 0]))

            return (
                tuple(map(int, current[best])),
                float(values[best, 0]),
                iterations
            )

        blocks = options.reshape(
            len(current), 225, 7
        )

        current[better] = blocks[
            np.arange(len(current))[better],
            chosen[better]
        ]


def predict(prepared, alpha):
    return search(
        prepared[8],
        fit(prepared, alpha)
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, values):
    print(
        f"{label}: prosek={np.mean(values):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(values, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows) - test_count
    begin = split - validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW + 3
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    settings = tuple(
        (m, a) for m in MASSES for a in ALPHAS
    )
    validation = [[] for _ in settings]

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    for t in range(begin, split):
        index = 0

        for mass in MASSES:
            p = prepare(rows[:t], mass)

            for alpha in ALPHAS:
                combo, _, _ = predict(p, alpha)
                validation[index].append(
                    hits(combo, rows[t])
                )
                index += 1

        if (t - begin + 1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t - begin + 1}/{validation_count}",
                flush=True
            )

    means = [np.mean(v) for v in validation]
    chosen = int(np.argmax(means))
    mass, alpha = settings[chosen]

    print(
        f"Validacija: mass={mass:g}; "
        f"alpha={alpha:g}; "
        f"prosek={means[chosen]:.4f}",
        flush=True
    )

    model_scores = []
    control_scores = []
    no_effect_scores = []

    # Mass i alpha ostaju fiksni tokom testa.
    # Svaka predikcija koristi samo prethodne redove.
    for t in range(split, len(rows)):
        p = prepare(rows[:t], mass)
        model = fit(p, alpha)

        combo, _, _ = search(p[8], model)

        model_scores.append(
            hits(combo, rows[t])
        )
        control_scores.append(
            hits(rows[t - 1], rows[t])
        )

        # Kontrola: isti nauceni koeficijenti,
        # uz iskljucene aktivacione clanove.
        reduced = model[0].copy()
        reduced[14:28] = 0.

        reference, _, _ = search(
            p[8], (reduced, *model[1:])
        )

        no_effect_scores.append(
            hits(reference, rows[t])
        )

        if (t - split + 1) % 16 == 0:
            print(
                f"Test: {t - split + 1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split + 1}..{len(rows)}")

    summarize(
        "SCHWINGER V6 EFFECT",
        model_scores
    )
    summarize(
        "Kontrola: prethodno izvuceno kolo",
        control_scores
    )
    summarize(
        "Isti nauceni skor bez aktivacionih clanova",
        no_effect_scores
    )

    p = prepare(rows, mass)
    combo, score, iterations = predict(p, alpha)

    repeated = any(
        np.array_equal(combo, row)
        for row in rows
    )
    _, radius = support(p[8])

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows) + 1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor={score:.10f}; "
        f"iteracija={iterations}; "
        f"vec postoji u istoriji={repeated}"
    )
    print(
        "Empirijska podrska: "
        f"L1 poluprecnik={radius:.6f}; kvantil=.75"
    )
    print(
        "Proveren lokalni maksimum dozvoljenih zamena; "
        "globalni maksimum nije potvrden."
    )
    print(
        "Aktivacija preko barijere je strukturna analogija; "
        "skor nije verovatnoca dobitka."
    )


def self_test():
    assert activation(1., 0.) == 0.

    fields = np.array([.1, .5, 1., 2.])
    assert np.all(
        np.diff(activation(1., fields)) > 0
    )

    barriers = np.array([.1, .5, 1., 2.])
    assert np.all(
        np.diff(activation(barriers, 1.)) < 0
    )

    barrier, field, eps = .7, 1.2, 1e-6

    numeric = (
        activation(barrier, field + eps)
        - activation(barrier, field - eps)
    ) / (2 * eps)

    expected = activation(barrier, field) * (
        2 / field + np.pi * barrier / field**2
    )

    assert np.isclose(
        numeric, expected, atol=1e-8
    )

    base = np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39]
    ])

    restored = (
        gaps(base[0])
        + (states(base) - states(base[0])) @ MODES
    )

    assert np.allclose(
        5 * np.cumsum(restored[:, :-1], axis=1),
        base
    )

    history = np.tile(base, (30, 1))

    for mass in MASSES:
        p = prepare(history, mass)

        for alpha in ALPHAS:
            model = fit(p, alpha)
            a = predict(p, alpha)

            assert a == predict(
                prepare(history, mass), alpha
            )

            tree, radius = support(p[8])
            r = replacements(a[0])
            allowed = supported(r, tree, radius)

            assert (
                scores(r[allowed], model).max(
                    initial=-np.inf
                )
                <= a[1] + 1e-9
            )

    changed = history.copy()
    changed[-1] = base[0]

    assert (
        predict(prepare(history[:-1], 1.), .1)
        == predict(prepare(changed[:-1], 1.), .1)
    )

    print(
        "Self-test OK: aktivacija, izvod, rekonstrukcija, "
        "hronologija i pretraga."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: mass=1; alpha=1; prosek=1.5208
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
SCHWINGER V6 EFFECT: prosek=1.2500; pogodaka 0..7=[10, 31, 20, 3, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Isti nauceni skor bez aktivacionih clanova: prosek=1.1719; pogodaka 0..7=[14, 30, 16, 3, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 2,7,8,9,14,38,39
Skor=0.8325573310; iteracija=17; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Aktivacija preko barijere je strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: mass=0.5; alpha=1; prosek=1.3333
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
SCHWINGER V6 EFFECT: prosek=1.1875; pogodaka 0..7=[13, 31, 15, 5, 0, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Isti nauceni skor bez aktivacionih clanova: prosek=1.3438; pogodaka 0..7=[10, 28, 20, 6, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,14,15,16,17,18,39
Skor=0.6342677663; iteracija=14; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=2.800000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Aktivacija preko barijere je strukturna analogija; skor nije verovatnoca dobitka.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: mass=1; alpha=0.1; prosek=1.7083
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
SCHWINGER V6 EFFECT: prosek=1.0625; pogodaka 0..7=[20, 27, 11, 5, 1, 0, 0, 0]
Kontrola: prethodno izvuceno kolo: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Isti nauceni skor bez aktivacionih clanova: prosek=1.1406; pogodaka 0..7=[23, 19, 14, 6, 2, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,2,3,4,5,9,12
Skor=1.2770769004; iteracija=22; vec postoji u istoriji=False
Empirijska podrska: L1 poluprecnik=3.200000; kvantil=.75
Proveren lokalni maksimum dozvoljenih zamena; globalni maksimum nije potvrden.
Aktivacija preko barijere je strukturna analogija; skor nije verovatnoca dobitka.
"""



"""
V6 koristi nelinearnu aktivaciju prelaza preko barijere: doprinos zavisi od prethodnih strukturnih promena, a smer odziva uči se iz podataka. Prošle su provere nultog polja, odziva na jačinu polja, izvoda, rekonstrukcije i hronologije. 

Koristi nelinearnu aktivaciju prelaza preko barijere, inspirisanu Schwingerovim radom On Gauge Invariance and Vacuum Polarization.

CSV	        Sledeća kombinacija	Prosek pogodaka na 64 kola	Kontrola: prethodno izvučeno kolo
Objedinjeni	2,7,8,9,14,38,39	1,3125	                    1,2188
Loto	    1,14,15,16,17,18,39	1,1719	                    1,2969
Loto Plus	1,2,3,4,5,9,12	    1,0469	                    1,2969


Prednost na ovom testu postoji samo za objedinjeni fajl. Provereni su aktivacija, izvod, rekonstrukcija, hronologija, ponovljivost i lokalni maksimum dozvoljenih zamena.
"""



"""
Julian Seymour Schwinger  
1-quantum electrodynamics    2-nuclear physics        3-particle physics      
4-atomic physics        5-quantum field theory        6-Schwinger effect
"""
