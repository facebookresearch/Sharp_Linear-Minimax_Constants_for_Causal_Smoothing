#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Exact-rational primal/dual bounds for the corrected modulus problem.

For r=p+1 we bound

    I_r = inf { integral_0^infty f(t)^2 dt : f(0)=1,
                ||f^(r)||_infty <= 1 }.

The switching lengths below are only *candidate data*.  They need not be
certified roots of a shooting equation: after parsing them as Fractions, all
feasibility and all integrations used in the reported bounds are exact.

The only enclosure step for |q| is an exact Bernstein subdivision.  On a box
whose Bernstein coefficients have one sign, the integral is exact; on every
remaining depth-limited box, width*max(abs(coefficient)) is used.
Thus the returned dual value is rigorously rounded downward.

This is research scratch code.  It deliberately has no dependency on NumPy,
SciPy, mpmath, or floating-point interval packages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
from fractions import Fraction as Q
from math import comb, factorial
from typing import TypedDict


class Candidate(TypedDict):
    rho: str
    ratios: list[str]
    intervals: int


# Principal-orbit seed ratios.  A constant-ratio geometric tail is appended,
# then the resulting lengths are snapped to an exact common rational grid.
CANDIDATES: dict[int, Candidate] = {
    1: {
        "rho": "0.24212137354815673482316473873611456834",
        "ratios": [],
        "intervals": 35,
    },
    2: {
        "rho": "0.5757361184208955204254667262202088554690914510371403364983766",
        "ratios": [
            "0.3485194720333346401731499927654566259075228804695415215580548",
            "0.6137445619288796179404180085557973305273354819002835772212861",
            "0.5677655229714218030651908189989137503616013927552261244553653",
            "0.5774287724521257652113371050053993423831738614456401155543494",
            "0.5753773890574188352643613418225835704602674951341340163469281",
            "0.5758121800671684067178425820379999170699153058318014093541411",
            "0.5757199925676114817887717292312679130833643051500748363461865",
            "0.5757395373380650242331004286521238308647681213530177457625058",
            "0.5757353935635066232991602347798410488886670151441075208615427",
            "0.5757362721007571526149718338567438145270776630616194280010162",
            "0.5757360858386308884036480718817382837002294543961191056458740",
            "0.5757361253287883391813705025228285491828061249673024186276287",
            "0.5757361169563262751266093577975606650463613291046469435783396",
            "0.5757361187314045477199512899440455114503943988334916802842999",
            "0.5757361183550632927306317759906905239843478566638264159469823",
            "0.5757361184348528673052866504835903856013283411337459959572274",
            "0.5757361184179363689772330651273875777353782439637311188950568",
            "0.5757361184215229016440451635257275813660124875316530039446243",
            "0.5757361184207625068909077188033517429902785885239450892695685",
            "0.5757361184209237211427530507105420101442352097458795203520133",
            "0.5757361184208895414816088709213294392897778970943363200019398",
            "0.5757361184208967880446301834895221944778385494393962504538198",
            "0.5757361184208952516725930137545682931406912132492217977863755",
            "0.5757361184208955774048097425529463890710198804128754706870993",
            "0.5757361184208955083450543915545712676457668094536363760283549",
            "0.5757361184208955229866819745903071942580273174946148645087308",
            "0.5757361184208955198824535105487863510066998821262560042270221",
            "0.5757361184208955205405930720365018957705240752660546940966096",
            "0.5757361184208955204010583414544267645218306115777785615332828",
            "0.5757361184208955204306416428967542177972861333591685867116046",
            "0.5757361184208955204243695719895557226214444401391255285144715",
            "0.5757361184208955204256993381598939709023451929669020285766226",
            "0.5757361184208955204254174092573130725746556209197373685687144",
            "0.5757361184208955204254771821015043574213317148878361318058376",
            "0.5757361184208955204254645094281104317433798655257687941685868",
            "0.5757361184208955204254671962109316602417451261225653259825713",
            "0.5757361184208955204254666265756460299250054255207920512832336",
            "0.5757361184208955204254667473462405384778250357183354749956432",
            "0.5757361184208955204254667217411966342716439273398854588817109",
            "0.5757361184208955204254667271698216733733274351544919010455792",
        ],
        "intervals": 60,
    },
    3: {
        "rho": "0.7402683705040163790247524354793839890469507888990364103874170",
        "ratios": [
            "0.3664377542783957982639450450640504340267076074872237900480651",
            "0.7712627843625420699856312832857599357428660757005931171622198",
            "0.7465945095544135800830382310525349233340217646399179495609022",
            "0.7381396949761559667315979637732885110473680678275308822614419",
            "0.7403449225950800660828267085174157286537499600084650442150146",
            "0.7403348239053604546624924027729544706133242181612367489307856",
            "0.7402560831295816330764745110512524721310202337093898096145257",
            "0.7402677094260644554796003185832290570152284956579508977550676",
            "0.7402689120146486238139877069092262743994695509356526176548052",
            "0.7402683171495128943318831455046758304048622110203999268358139",
            "0.7402683584302129820277717876699365674242443706744700365287834",
            "0.7402683741709394966871679485641336520922332315446386124802958",
            "0.7402683704191083628457490349037676494461316735337620565877591",
            "0.7402683703827488171308579774771192255204390870496056570076079",
            "0.7402683705244396630842332532468958021366852575267730114279973",
            "0.7402683705055087994174634972139430795814296257688034287071507",
            "0.7402683705030600392920388905376526790391739614385891559991436",
            "0.7402683705040987855961919815100112673488939597172524552156582",
            "0.7402683705040393917492955745284198036858596312656200840263714",
            "0.7402683705040100915356783435693484857222609103192394611874866",
            "0.7402683705040164405594951512170824894918238335363598482449986",
            "0.7402683705040165989708683194469943269868035046536627174880571",
            "0.7402683705040163453464416661401711599425922227415063273952307",
            "0.7402683705040163758370626699763130619900646756639488930403322",
            "0.7402683705040163807056427969439022339677877012018354280687413",
            "0.7402683705040163789004515532944414475666562872646002084726371",
            "0.7402683705040163789813686682564277074990851786610359118422136",
            "0.7402683705040163790354759484269801425883344794512468153898496",
            "0.7402683705040163790247979133132005626592660881132347690595443",
            "0.7402683705040163790243558079181930953224401596543187876342952",
            "0.7402683705040163790248074759040978472133897142784642933951862",
            "0.7402683705040163790247589973760776386684191646468216763801984",
            "0.7402683705040163790247494951069278787984304436128985291311559",
            "0.7402683705040163790247526170584120892986361164559217494445786",
            "0.7402683705040163790247525164962685429360955917562450414658078",
            "0.7402683705040163790247524172926914906596343121209044296794142",
        ],
        "intervals": 105,
    },
    4: {
        "rho": "0.826468066656324299534444154513151303140148068375445883955536",
        "ratios": [
            "0.3591531160024504865645002999037341",
            "0.8428885914888441682103634240096230",
            "0.8362243753528705025087363771180550",
            "0.8278054165764959297018044726868014",
            "0.8259690583553759498274286982018329",
            "0.8263730066978846362011994056696784",
            "0.8264799355586697677347711633074244",
            "0.8264741269656074072107167783992082",
            "0.8264680810904876307294428686774860",
            "0.8264677811349447607670509361360174",
            "0.8264680363470796826019732931302386",
            "0.8264680773336909059642323598076322",
            "0.8264680691944078774847885785740230",
            "0.8264680664033418711899867601937726",
            "0.8264680665085234185035888144529846",
            "0.8264680666534051729635720246891928",
            "0.8264680666631230831805189990317721169503902138875833742648329",
            "0.8264680666571583452417123255406520345035530458147020831573752",
            "0.8264680666560828706997853059683033471234310184862358774085287",
            "0.8264680666562596142100070864623407367660995737147390475705231",
            "0.8264680666563292937942946575510976153673461253495331951008704",
            "0.8264680666563279156094348652869781469146320773142671703817000",
            "0.8264680666563244286362422582324700639454165543793817545833247",
            "0.8264680666563241390476851667841659889658647381343659443821121",
            "0.8264680666563242769597182715197059736906043647850409957481488",
            "0.8264680666563243049383999008313773167709320179094640263758631",
            "0.8264680666563243011699512572290108731091474296916342518247783",
            "0.8264680666563242994417953451187477004779928326562679609694274",
            "0.8264680666563242994464469196369664700993585864163129805347900",
            "0.8264680666563242995299151022118595135016186566795748041563700",
            "0.8264680666563242995382080602393274994985181596501739261654311",
            "0.8264680666563242995350456946771350372569709993829553757526231",
            "0.8264680666563242995343247711761384289795322673835921915958309",
            "0.8264680666563242995344030851279087171830464978294320979754006",
            "0.8264680666563242995344456974666672527991150056664798161355621",
            "0.8264680666563242995344462842728818179221708934429533616517444",
            "0.8264680666563242995344442977154888184323934826300012655913319",
            "0.8264680666563242995344440668439139049131532892240369319545894",
            "0.8264680666563242995344441386909655255622867056683020186759062",
            "0.8264680666563242995344441571090547763628336387128712923671001",
        ],
        "intervals": 230,
    },
}


def require(condition: bool, message: str) -> None:
    """A validation check that remains active under ``python -O``."""
    if not condition:
        raise ArithmeticError(message)


def qstr(x: Q, digits: int = 18) -> str:
    """Decimal display truncated toward zero (display only)."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    scale = 10**digits
    n = x.numerator * scale // x.denominator
    return f"{sign}{n // scale}.{n % scale:0{digits}d}"


def qceilstr(x: Q, digits: int = 18) -> str:
    """Smallest decimal grid value >= x (positive quantities only)."""
    if x < 0:
        raise ValueError("qceilstr is only used for positive upper bounds")
    scale = 10**digits
    n = (x.numerator * scale + x.denominator - 1) // x.denominator
    return f"{n // scale}.{n % scale:0{digits}d}"


def integer_nth_root_floor(n: int, k: int) -> int:
    """Largest nonnegative integer a with a**k <= n."""
    if n < 0 or k < 1:
        raise ValueError
    if n < 2:
        return n
    lo, hi = 0, 1 << ((n.bit_length() + k - 1) // k)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid**k <= n:
            lo = mid
        else:
            hi = mid
    return lo


def nth_root_decimal_bracket(x: Q, k: int, digits: int) -> tuple[Q, Q]:
    """Exact grid bracket for x**(1/k), for x>0."""
    if x <= 0:
        raise ValueError("positive argument required")
    scale = 10**digits
    # floor((x*scale**k)**(1/k)); correct a first estimate by exact tests.
    target_num = x.numerator * scale**k
    a = integer_nth_root_floor(target_num // x.denominator, k)
    while Q((a + 1) ** k, scale**k) <= x:
        a += 1
    while Q(a**k, scale**k) > x:
        a -= 1
    lo, hi = Q(a, scale), Q(a + 1, scale)
    require(lo**k <= x < hi**k, "invalid exact nth-root bracket")
    return lo, hi


def integrate_poly(poly: list[Q], h: Q) -> Q:
    return sum((c * h ** (j + 1) / (j + 1) for j, c in enumerate(poly)), Q(0))


def integrate_t_power(poly: list[Q], start: Q, h: Q, power: int) -> Q:
    """Integral of (start+s)^power * poly(s), 0<=s<=h."""
    return sum(
        (
            c
            * comb(power, ell)
            * start ** (power - ell)
            * h ** (j + ell + 1)
            / (j + ell + 1)
            for j, c in enumerate(poly)
            for ell in range(power + 1)
        ),
        Q(0),
    )


def square_integral(poly: list[Q], h: Q) -> Q:
    return sum(
        (
            poly[j] * poly[k] * h ** (j + k + 1) / (j + k + 1)
            for j in range(len(poly))
            for k in range(len(poly))
        ),
        Q(0),
    )


def derivative(poly: list[Q], order: int) -> list[Q]:
    out = poly[:]
    for _ in range(order):
        out = [(j + 1) * out[j + 1] for j in range(len(out) - 1)]
    return out


def jets_at(poly: list[Q], point: Q, count: int) -> list[Q]:
    """Values of derivatives 0,...,count-1 at a point."""
    return [
        sum(
            (
                poly[k] * Q(factorial(k), factorial(k - j)) * point ** (k - j)
                for k in range(j, len(poly))
            ),
            Q(0),
        )
        for j in range(count)
    ]


def assert_piece_continuity(
    pieces: list[list[Q]], lengths: list[Q], order: int
) -> None:
    for j in range(len(pieces) - 1):
        require(
            jets_at(pieces[j], lengths[j], order)
            == jets_at(pieces[j + 1], Q(0), order),
            f"piecewise continuity failed at breakpoint {j + 1}",
        )


def translate_global(poly: list[Q], a: Q, degree: int) -> list[Q]:
    """Coefficients of poly(a+s), padded through degree."""
    out = [Q(0)] * (degree + 1)
    for k, c in enumerate(poly):
        for j in range(k + 1):
            out[j] += c * comb(k, j) * a ** (k - j)
    return out


def solve_fraction(a: list[list[Q]], b: list[Q]) -> list[Q]:
    n = len(b)
    m = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if m[i][col])
        m[col], m[pivot] = m[pivot], m[col]
        d = m[col][col]
        m[col] = [x / d for x in m[col]]
        for i in range(n):
            if i == col or not m[i][col]:
                continue
            d = m[i][col]
            m[i] = [m[i][j] - d * m[col][j] for j in range(n + 1)]
    return [m[i][-1] for i in range(n)]


def make_lengths(p: int, count: int | None = None, grid_digits: int = 70) -> list[Q]:
    data = CANDIDATES[p]
    count = count or data["intervals"]
    # Candidate generation is intentionally non-rigorous: it has no bearing on
    # validity after the values are snapped to the exact common rational grid.
    # A common denominator avoids the explosive denominator growth that would
    # result from multiplying the decimal ratios as Fractions.
    with localcontext() as ctx:
        ctx.prec = grid_digits + 45
        ratios = [Decimal(x) for x in data["ratios"]]
        rho = Decimal(data["rho"])
        raw = [Decimal(1)]
        for j in range(count - 1):
            raw.append(raw[-1] * (ratios[j] if j < len(ratios) else rho))
        total = sum(raw, Decimal(0))
        normalized = [x / total for x in raw]
        scale = 10**grid_digits
        nums = [
            int((x * scale).to_integral_value(rounding=ROUND_HALF_EVEN))
            for x in normalized[:-1]
        ]
    nums.append(scale - sum(nums))
    if min(nums) <= 0:
        raise ArithmeticError("candidate grid too coarse for terminal interval")
    return [Q(n, scale) for n in nums]


def terminal_spline(
    lengths: list[Q], r: int, amplitude: Q | None = None, sign_flip: int = 1
) -> tuple[list[list[Q]], list[Q]]:
    if amplitude is None:
        amplitude = Q(1)
    """Flat-right spline with f^(r)=sign_flip*(-1)^j*amplitude."""
    right = [Q(0)] * r
    pieces_rev = []
    for j in range(len(lengths) - 1, -1, -1):
        h = lengths[j]
        control = Q(sign_flip * (-1) ** j) * amplitude
        high = control / factorial(r)
        coeff = [Q(0)] * r + [high]
        # Determine low Taylor coefficients from derivatives at the right.
        for k in range(r - 1, -1, -1):
            rhs = right[k]
            rhs -= sum(
                (
                    coeff[n] * factorial(n) / factorial(n - k) * h ** (n - k)
                    for n in range(k + 1, r + 1)
                ),
                Q(0),
            )
            coeff[k] = rhs / factorial(k)
        right = [coeff[k] * factorial(k) for k in range(r)]
        pieces_rev.append(coeff)
    return list(reversed(pieces_rev)), right


def integrate_forcing_backwards(
    forcing: list[list[Q]], lengths: list[Q], r: int
) -> tuple[list[list[Q]], list[Q]]:
    """Flat-right q solving q^(r)=forcing, piece by piece."""
    right = [Q(0)] * r
    pieces_rev = []
    for force, h in reversed(list(zip(forcing, lengths))):
        degree = len(force) - 1
        coeff = [Q(0)] * r + [
            force[n] * factorial(n) / factorial(n + r) for n in range(degree + 1)
        ]
        for k in range(r - 1, -1, -1):
            rhs = right[k]
            rhs -= sum(
                (
                    coeff[n] * factorial(n) / factorial(n - k) * h ** (n - k)
                    for n in range(k + 1, len(coeff))
                ),
                Q(0),
            )
            coeff[k] = rhs / factorial(k)
        right = [coeff[k] * factorial(k) for k in range(r)]
        pieces_rev.append(coeff)
    return list(reversed(pieces_rev)), right


def hermite_correction(qleft: list[Q], r: int, support: Q) -> list[Q]:
    """c^(j)(0)=-qleft[j], j<r-1, and c^(j)(support)=0, j<r."""
    degree = 2 * r - 2
    c = [Q(0)] * (degree + 1)
    for j in range(r - 1):
        c[j] = -qleft[j] / factorial(j)
    unknown = list(range(r - 1, degree + 1))
    matrix, rhs = [], []
    for j in range(r):
        matrix.append(
            [
                Q(factorial(k), factorial(k - j)) * support ** (k - j)
                if k >= j
                else Q(0)
                for k in unknown
            ]
        )
        known = sum(
            (
                c[k] * Q(factorial(k), factorial(k - j)) * support ** (k - j)
                for k in range(r - 1)
                if k >= j
            ),
            Q(0),
        )
        rhs.append(-known)
    values = solve_fraction(matrix, rhs)
    for k, value in zip(unknown, values):
        c[k] = value
    return c


def power_to_bernstein(poly: list[Q], h: Q, degree: int) -> list[Q]:
    power = [Q(0)] * (degree + 1)
    for k, value in enumerate(poly):
        power[k] = value * h**k
    return [
        sum((power[k] * Q(comb(i, k), comb(degree, k)) for k in range(i + 1)), Q(0))
        for i in range(degree + 1)
    ]


def split_bernstein_half(b: list[Q]) -> tuple[list[Q], list[Q]]:
    rows = [b]
    while len(rows[-1]) > 1:
        old = rows[-1]
        rows.append([(old[i] + old[i + 1]) / 2 for i in range(len(old) - 1)])
    left = [row[0] for row in rows]
    right = [row[-1] for row in reversed(rows)]
    return left, right


@dataclass
class AbsBound:
    upper: Q
    exact_boxes: int
    unresolved_boxes: int
    max_depth_seen: int


def bernstein_abs_integral(poly: list[Q], h: Q, max_depth: int) -> AbsBound:
    degree = len(poly) - 1
    initial = power_to_bernstein(poly, h, degree)
    total = Q(0)
    exact_boxes = unresolved = max_seen = 0
    stack = [(initial, h, 0)]
    while stack:
        b, width, depth = stack.pop()
        max_seen = max(max_seen, depth)
        if min(b) >= 0 or max(b) <= 0:
            total += abs(width * sum(b, Q(0)) / (degree + 1))
            exact_boxes += 1
        elif depth == max_depth:
            total += width * max(abs(x) for x in b)
            unresolved += 1
        else:
            left, right = split_bernstein_half(b)
            stack.append((right, width / 2, depth + 1))
            stack.append((left, width / 2, depth + 1))
    return AbsBound(total, exact_boxes, unresolved, max_seen)


@dataclass
class Certificate:
    p: int
    r: int
    intervals: int
    support: Q
    primal_upper: Q
    dual_lower: Q
    l1_upper: Q
    gap_upper: Q
    stationarity_defect: Q
    complementarity_defect_upper: Q
    h_mass: Q
    h_moments: list[Q]
    g_norm2: Q
    phi_l1_upper: Q
    control: Q
    f0: Q
    left_jets: list[Q]
    right_jets: list[Q]
    abs_exact_boxes: int
    abs_unresolved_boxes: int
    abs_max_depth: int


def validate_primal(
    pieces: list[list[Q]],
    lengths: list[Q],
    r: int,
    control: Q,
    sign_flip: int,
) -> None:
    require(control <= 1, "primal derivative bound failed")
    assert_piece_continuity(pieces, lengths, r)
    require(
        jets_at(pieces[-1], lengths[-1], r) == [Q(0)] * r,
        "primal right endpoint is not flat",
    )
    for j, poly in enumerate(pieces):
        require(
            derivative(poly, r) == [Q(sign_flip * (-1) ** j) * control],
            f"incorrect primal bang-bang derivative on piece {j}",
        )


def construct_dual(
    fpieces: list[list[Q]], lengths: list[Q], r: int, support: Q
) -> tuple[list[list[Q]], list[Q], list[Q], list[Q]]:
    forcing = [[Q(2 * (-1) ** r) * c for c in poly] for poly in fpieces]
    q0pieces, q0left = integrate_forcing_backwards(forcing, lengths, r)
    assert_piece_continuity(q0pieces, lengths, r)
    require(
        jets_at(q0pieces[-1], lengths[-1], r) == [Q(0)] * r,
        "uncorrected dual right endpoint is not flat",
    )
    for j, (q0poly, forcepoly) in enumerate(zip(q0pieces, forcing)):
        require(
            derivative(q0poly, r) == forcepoly,
            f"uncorrected dual forcing failed on piece {j}",
        )

    correction = hermite_correction(q0left, r, support)
    qpieces = []
    starts = []
    position = Q(0)
    degree = 2 * r
    for q0, h in zip(q0pieces, lengths):
        starts.append(position)
        local_c = translate_global(correction, position, degree)
        qpieces.append(
            [(q0[k] if k < len(q0) else Q(0)) + local_c[k] for k in range(degree + 1)]
        )
        position += h
    require(position == support, "piece lengths do not sum to support")
    assert_piece_continuity(qpieces, lengths, r)

    left_jets = [qpieces[0][j] * factorial(j) for j in range(r - 1)]
    require(all(x == 0 for x in left_jets), "dual left jets are nonzero")
    right_jets = jets_at(qpieces[-1], lengths[-1], r)
    require(all(x == 0 for x in right_jets), "dual right jets are nonzero")
    return qpieces, starts, left_jets, right_jets


def certify(
    p: int, count: int | None = None, root_digits: int = 45, abs_depth: int = 42
) -> Certificate:
    r = p + 1
    base_lengths = make_lengths(p, count)
    base_u, base_left = terminal_spline(base_lengths, r)
    sign_flip = 1 if base_left[0] > 0 else -1
    a = abs(base_left[0])

    # Choose a rational dilation L >= a^(-1/r), hence exact feasibility.
    _, dilation_hi = nth_root_decimal_bracket(1 / a, r, root_digits)
    support = dilation_hi
    control = 1 / (a * support**r)
    lengths = [support * h for h in base_lengths]
    fpieces, fleft = terminal_spline(lengths, r, control, sign_flip)
    require(fleft[0] == 1, "primal trace is not exactly one")
    validate_primal(fpieces, lengths, r, control, sign_flip)

    primal = sum((square_integral(poly, h) for poly, h in zip(fpieces, lengths)), Q(0))

    # q0^(r)=2*(-1)^r*f, with q0 flat at the support endpoint.
    qpieces, starts, left_jets, right_jets = construct_dual(
        fpieces, lengths, r, support
    )

    l1 = Q(0)
    exact_boxes = unresolved_boxes = max_seen = 0
    for poly, h in zip(qpieces, lengths):
        bound = bernstein_abs_integral(poly, h, abs_depth)
        require(
            bound.upper >= abs(integrate_poly(poly, h)),
            "Bernstein absolute-integral upper bound failed",
        )
        l1 += bound.upper
        exact_boxes += bound.exact_boxes
        unresolved_boxes += bound.unresolved_boxes
        max_seen = max(max_seen, bound.max_depth_seen)

    qnorm = sum(
        (square_integral(derivative(poly, r), h) for poly, h in zip(qpieces, lengths)),
        Q(0),
    )
    boundary = qpieces[0][r - 1] * factorial(r - 1)
    dual = -l1 - Q((-1) ** r) * boundary - qnorm / 4
    require(dual > 0, "dual lower bound is not positive")
    require(dual <= primal, "weak-duality ordering failed")

    # Direct feasible filter extracted from the dual polynomial:
    # h=(-1)^r q^(r)/2, M=int h, g=h/M.  The left dual jets imply
    # int t^j h=0 for 1<=j<=r-1.  Check every identity exactly rather than
    # relying on integration-by-parts signs here.
    hpieces = [
        [Q((-1) ** r, 2) * value for value in derivative(poly, r)] for poly in qpieces
    ]
    h_moments = [
        sum(
            (
                integrate_t_power(poly, start, h, power)
                for poly, start, h in zip(hpieces, starts, lengths)
            ),
            Q(0),
        )
        for power in range(r)
    ]
    h_mass = -Q((-1) ** r, 2) * boundary
    require(h_moments[0] == h_mass, "filter mass/boundary identity failed")
    require(h_mass != 0, "dual-derived filter has zero mass")
    require(
        all(value == 0 for value in h_moments[1:]),
        "dual-derived filter moments are nonzero",
    )
    g_norm2 = qnorm / (4 * h_mass**2)
    phi_l1_upper = l1 / (2 * abs(h_mass))

    # Exact gap identity.  This independently audits all parity signs in D(q):
    # P-D_upper = ||F-(-1)^r q^(r)/2||^2
    #             + [L1_upper + integral q F^(r)].
    stationarity = Q(0)
    cross = Q(0)
    for j, (fpoly, qpoly, h) in enumerate(zip(fpieces, qpieces, lengths)):
        qr = derivative(qpoly, r)
        signed_qr = [Q((-1) ** r, 2) * x for x in qr]
        diff = [fpoly[k] - signed_qr[k] for k in range(r + 1)]
        stationarity += square_integral(diff, h)
        f_control = Q(sign_flip * (-1) ** j) * control
        cross += f_control * integrate_poly(qpoly, h)
    complementarity = l1 + cross
    require(stationarity >= 0, "negative stationarity defect")
    require(complementarity >= 0, "negative complementarity defect")
    require(
        stationarity + complementarity == primal - dual,
        "exact primal-dual gap identity failed",
    )

    return Certificate(
        p=p,
        r=r,
        intervals=len(lengths),
        support=support,
        primal_upper=primal,
        dual_lower=dual,
        l1_upper=l1,
        gap_upper=primal - dual,
        stationarity_defect=stationarity,
        complementarity_defect_upper=complementarity,
        h_mass=h_mass,
        h_moments=h_moments,
        g_norm2=g_norm2,
        phi_l1_upper=phi_l1_upper,
        control=control,
        f0=fleft[0],
        left_jets=left_jets,
        right_jets=right_jets,
        abs_exact_boxes=exact_boxes,
        abs_unresolved_boxes=unresolved_boxes,
        abs_max_depth=max_seen,
    )


def k_bracket(cert: Certificate, digits: int = 18) -> tuple[Q, Q]:
    """Exact decimal grid enclosure of K_p from D <= I_r <= P."""
    r = cert.r
    n = 2 * r + 1

    def kth_power(i_value: Q) -> Q:
        return Q((2 * r) ** (2 * r), n**n) / i_value ** (2 * r)

    # K decreases with I: K(P) <= K* <= K(D).
    lower, _ = nth_root_decimal_bracket(kth_power(cert.primal_upper), n, digits)
    _, upper = nth_root_decimal_bracket(kth_power(cert.dual_lower), n, digits)
    return lower, upper


def direct_filter_k_bracket(cert: Certificate, digits: int = 18) -> tuple[Q, Q]:
    """Two-point lower bound plus an explicit feasible-filter upper bound.

    This does not use equality in the I_r-to-K_p conversion.  The lower end is
    the two-point bound evaluated with the feasible modulus candidate P.  The
    upper end is the optimized risk constant of g=h/int(h), using the rigorous
    Bernstein upper bound on ||Phi_g||_1.
    """
    r = cert.r
    n = 2 * r + 1
    lower_power = Q((2 * r) ** (2 * r), n**n) / cert.primal_upper ** (2 * r)
    upper_power = (
        Q(n**n, (2 * r) ** (2 * r)) * cert.phi_l1_upper**2 * cert.g_norm2 ** (2 * r)
    )
    lower, _ = nth_root_decimal_bracket(lower_power, n, digits)
    _, upper = nth_root_decimal_bracket(upper_power, n, digits)
    require(lower <= upper, "direct lower bound exceeds feasible-filter upper bound")
    return lower, upper


def print_certificate(cert: Certificate, display_digits: int = 22) -> None:
    klo, khi = k_bracket(cert, display_digits)
    direct_lo, direct_hi = direct_filter_k_bracket(cert, display_digits)
    print(f"p={cert.p}, r={cert.r}, finite intervals={cert.intervals}")
    print("feasibility: f(0)=", cert.f0, "; |f^(r)|=", qstr(cert.control, 24), "<=1")
    print("dual left jets q^(0..r-2)(0)=", cert.left_jets)
    print("dual right jets q^(0..r-1)(T)=", cert.right_jets)
    print("support T in exact candidate =", qstr(cert.support, 24))
    print("I lower (dual) =", qstr(cert.dual_lower, 24))
    print("I upper (primal)=", qceilstr(cert.primal_upper, 24))
    print("primal-dual width =", qceilstr(cert.gap_upper, 24))
    print("  stationarity defect =", qceilstr(cert.stationarity_defect, 24))
    print(
        "  complementarity/L1 defect upper =",
        qceilstr(cert.complementarity_defect_upper, 24),
    )
    print("L1 Bernstein upper =", qceilstr(cert.l1_upper, 24))
    print("dual-derived h mass =", qstr(cert.h_mass, 24))
    print(
        "normalized g moments through order r-1 =",
        [value / cert.h_mass for value in cert.h_moments],
    )
    print("||g||_2^2 =", qceilstr(cert.g_norm2, 24))
    print("||Phi_g||_1 upper =", qceilstr(cert.phi_l1_upper, 24))
    print(
        "Bernstein boxes: exact=",
        cert.abs_exact_boxes,
        "unresolved=",
        cert.abs_unresolved_boxes,
        "max_depth=",
        cert.abs_max_depth,
    )
    print(
        "K via exact I-to-K formula = [",
        qstr(klo, display_digits),
        ",",
        qceilstr(khi, display_digits),
        "]",
    )
    print(
        "K direct lower/filter upper = [",
        qstr(direct_lo, display_digits),
        ",",
        qceilstr(direct_hi, display_digits),
        "]",
    )


def fraction_record(value: Q) -> dict[str, str]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
    }


def certificate_record(cert: Certificate, display_digits: int) -> dict:
    """Machine-readable exact endpoints plus outward decimal displays."""
    klo, khi = k_bracket(cert, display_digits)
    direct_lo, direct_hi = direct_filter_k_bracket(cert, display_digits)
    return {
        "p": cert.p,
        "r": cert.r,
        "finite_intervals": cert.intervals,
        "I": {
            "lower_exact": fraction_record(cert.dual_lower),
            "upper_exact": fraction_record(cert.primal_upper),
            "lower_decimal_outward": qstr(cert.dual_lower, display_digits),
            "upper_decimal_outward": qceilstr(cert.primal_upper, display_digits),
        },
        "K": {
            "lower_exact": fraction_record(klo),
            "upper_exact": fraction_record(khi),
            "lower_decimal_outward": qstr(klo, display_digits),
            "upper_decimal_outward": qceilstr(khi, display_digits),
        },
        "K_direct": {
            "meaning": "two-point lower bound and explicit feasible-filter upper bound",
            "lower_exact": fraction_record(direct_lo),
            "upper_exact": fraction_record(direct_hi),
            "lower_decimal_outward": qstr(direct_lo, display_digits),
            "upper_decimal_outward": qceilstr(direct_hi, display_digits),
        },
        "gap_upper_exact": fraction_record(cert.gap_upper),
        "stationarity_defect_exact": fraction_record(cert.stationarity_defect),
        "complementarity_defect_upper_exact": fraction_record(
            cert.complementarity_defect_upper
        ),
        "support_exact": fraction_record(cert.support),
        "control_magnitude_exact": fraction_record(cert.control),
        "dual_derived_filter": {
            "h_mass_exact": fraction_record(cert.h_mass),
            "h_mass_sign": "positive" if cert.h_mass > 0 else "negative",
            "h_moments_exact": [fraction_record(x) for x in cert.h_moments],
            "g_norm2_exact": fraction_record(cert.g_norm2),
            "phi_l1_upper_exact": fraction_record(cert.phi_l1_upper),
        },
        "checks": {
            "f_at_zero_exactly_one": cert.f0 == 1,
            "control_at_most_one": cert.control <= 1,
            "dual_left_jets_exactly_zero": all(x == 0 for x in cert.left_jets),
            "dual_right_jets_exactly_zero": all(x == 0 for x in cert.right_jets),
            "piece_continuity_and_gap_identity": True,
        },
        "bernstein": {
            "sign_certified_boxes": cert.abs_exact_boxes,
            "depth_limited_boxes": cert.abs_unresolved_boxes,
            "max_depth": cert.abs_max_depth,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, choices=(1, 2, 3, 4), action="append")
    parser.add_argument("--intervals", type=int)
    parser.add_argument("--abs-depth", type=int, default=44)
    parser.add_argument("--root-digits", type=int, default=45)
    parser.add_argument("--display-digits", type=int, default=22)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit exact rational endpoints and checks as JSON",
    )
    args = parser.parse_args()
    ps = args.p or [1, 2, 3, 4]
    if args.intervals is not None and len(ps) != 1:
        parser.error("--intervals requires exactly one --p")
    certificates = [
        certify(p, args.intervals, args.root_digits, args.abs_depth) for p in ps
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "format": "exact-rational-primal-dual-certificate-v1",
                    "K_formula": "K=(2r/I)^(2r/(2r+1))/(2r+1)",
                    "certificates": [
                        certificate_record(cert, args.display_digits)
                        for cert in certificates
                    ],
                },
                indent=2,
            )
        )
    else:
        for index, cert in enumerate(certificates):
            if index:
                print()
            print_certificate(cert, args.display_digits)


if __name__ == "__main__":
    main()
