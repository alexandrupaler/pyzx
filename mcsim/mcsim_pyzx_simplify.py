# This code is based on the one from simplify.py from the original PyZx repo

"""This module contains the ZX-diagram simplification strategies of PyZX.
Each strategy is based on applying some combination of the rewrite rules in the rules_ module.
The main procedures of interest are :func:`clifford_simp` for simple reductions,
:func:`full_reduce` for the full rewriting power of PyZX, and :func:`teleport_reduce` to
use the power of :func:`full_reduce` while not changing the structure of the graph.
"""

# __all__ = ['bialg_simp','spider_simp', 'id_simp', 'phase_free_simp', 'pivot_simp',
#         'pivot_gadget_simp', 'pivot_boundary_simp', 'gadget_simp',
#         'lcomp_simp', 'clifford_simp', 'tcount', 'to_gh', 'to_rg',
#         'full_reduce', 'teleport_reduce', 'reduce_scalar', 'supplementarity_simp']


from typing import List, Callable, Optional, Union, Tuple, Iterator

from pyzx.utils import toggle_vertex
from pyzx.rules import (
    MatchObject,
    RewriteOutputType,
    spider,
    unspider,
    pivot,
    lcomp,
    merge_phase_gadgets,
    match_lcomp,
    match_pivot,
    match_ids,
    match_pivot_gadget,
    match_pivot_boundary,
    remove_ids,
    match_phase_gadgets,
    apply_supplementarity,
    match_supplementarity,
    match_copy,
    apply_copy,
    match_spider,
    match_bialg,
    bialg,
    toggle_edge,
)

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.simplify import Stats

# Some of the following match functions return all possible matches, but the
# simp and simp_iter consider the matches one at a time.

apply_spider_fusion = spider
apply_spider_unfusion = unspider
apply_pivot = pivot
apply_lcomp = lcomp
apply_gadgets_merge = merge_phase_gadgets

# Examples to use the above:
#
# simp(g, match_pivot, apply_pivot, matchf=matchf, quiet=quiet, stats=stats)
# simp(g, match_pivot_gadget, apply_pivot, matchf=matchf, quiet=quiet, stats=stats)
# simp(g, match_pivot_boundary, apply_pivot, matchf=matchf, quiet=quiet, stats=stats)


def simp_iter(
    g: BaseGraph[VT, ET],
    match: Callable[..., List[MatchObject]],
    rewrite: Callable[
        [BaseGraph[VT, ET], List[MatchObject]], RewriteOutputType[ET, VT]
    ],
    matchf: Optional[Union[Callable[[ET], bool], Callable[[VT], bool]]] = None,
    quiet: bool = False,
    stats: Optional[Stats] = None,
) -> Iterator[Tuple[BaseGraph[VT, ET], int]]:
    """Helper method for constructing simplification strategies based
    on the rules present in rules_. It uses the ``match`` function to
    find matches, and then rewrites ``g`` using ``rewrite``.
    If ``matchf`` is supplied, only the vertices or edges for
    which matchf() returns True are considered for matches.

    Example:
        ``simp(g, 'spider_simp', rules.match_spider_parallel, rules.spider)``

    Args:
        g: The graph that needs to be simplified.
        str name: The name to display if ``quiet`` is set to False.
        match: One of the ``match_*`` functions of rules_.
        rewrite: One of the rewrite functions of rules_.
        matchf: An optional filtering function on candidate vertices or edges,
         which is passed as the second argument to the match function.
        quiet: Suppress output on numbers of matches found during simplification.

    Returns:
        Number of iterations of ``rewrite`` that had to be applied before no
        more matches were found."""

    i = 0
    new_matches = True
    while new_matches:
        new_matches = False

        if matchf is not None:
            m = match(g, matchf)
        else:
            m = match(g)

        if len(m) > 0:
            i += 1

        m = match(g)
        if len(m) > 0:
            i += 1

            if not quiet:
                print(len(m), end="")

            etab, rem_verts, rem_edges, check_isolated_vertices = rewrite(g, m)
            g.add_edge_table(etab)
            g.remove_edges(rem_edges)
            g.remove_vertices(rem_verts)

            if check_isolated_vertices:
                g.remove_isolated_vertices()
            if not quiet:
                print(". ", end="")

            yield g, i
            new_matches = True

            if stats is not None:
                stats.count_rewrites(match.__name__, len(m))

    if not quiet and i > 0:
        print(" {!s} iterations".format(i))


def simp(
    g: BaseGraph[VT, ET],
    match: Callable[..., List[MatchObject]],
    rewrite: Callable[
        [BaseGraph[VT, ET], List[MatchObject]], RewriteOutputType[ET, VT]
    ],
    matchf: Optional[Union[Callable[[ET], bool], Callable[[VT], bool]]] = None,
    quiet: bool = False,
    stats: Optional[Stats] = None,
) -> int:
    """Version of :func:`simp)iter` that performs all rewrites at once,
    returns an iterator."""

    max_i = 0
    for graph, i in simp_iter(g, match, rewrite, matchf, quiet, stats):
        # for pylint not to complain
        g = graph
        max_i = i

    return max_i


def pivot_simp(
    g: BaseGraph[VT, ET],
    matchf: Optional[Callable[[ET], bool]] = None,
    quiet: bool = False,
    stats: Optional[Stats] = None,
) -> int:
    """
    simp for match
    """
    return simp(g, match_pivot, pivot, matchf=matchf, quiet=quiet, stats=stats)


def pivot_gadget_simp(
    g: BaseGraph[VT, ET],
    matchf: Optional[Callable[[ET], bool]] = None,
    quiet: bool = False,
    stats: Optional[Stats] = None,
) -> int:
    """
    simp for pivot gadget
    """
    return simp(g, match_pivot_gadget, pivot, matchf=matchf, quiet=quiet, stats=stats)


def pivot_boundary_simp(
    g: BaseGraph[VT, ET],
    matchf: Optional[Callable[[ET], bool]] = None,
    quiet: bool = False,
    stats: Optional[Stats] = None,
) -> int:
    """
    simp for pivot boundary
    """
    return simp(g, match_pivot_boundary, pivot, matchf=matchf, quiet=quiet, stats=stats)


def lcomp_simp(
    g: BaseGraph[VT, ET],
    matchf: Optional[Callable[[VT], bool]] = None,
    quiet: bool = False,
    stats: Optional[Stats] = None,
) -> int:
    """
    simp for local complementarity
    """
    return simp(g, match_lcomp, lcomp, matchf=matchf, quiet=quiet, stats=stats)


def id_simp(
    g: BaseGraph[VT, ET],
    matchf: Optional[Callable[[VT], bool]] = None,
    quiet: bool = False,
    stats: Optional[Stats] = None,
) -> int:
    """
    simp for identity
    """
    return simp(g, match_ids, remove_ids, matchf=matchf, quiet=quiet, stats=stats)


def gadget_simp(
    g: BaseGraph[VT, ET], quiet: bool = False, stats: Optional[Stats] = None
) -> int:
    """
    simp for phase gadgets
    """
    return simp(g, match_phase_gadgets, merge_phase_gadgets, quiet=quiet, stats=stats)


def supplementarity_simp(
    g: BaseGraph[VT, ET], quiet: bool = False, stats: Optional[Stats] = None
) -> int:
    """
    simp for supplementarity
    """
    return simp(
        g, match_supplementarity, apply_supplementarity, quiet=quiet, stats=stats
    )


def copy_simp(
    g: BaseGraph[VT, ET], quiet: bool = False, stats: Optional[Stats] = None
) -> int:
    """
    Copies 1-ary spiders with 0/pi phase through neighbors.
    WARNING: only use on maximally fused diagrams consisting of Z-spiders.
    """
    return simp(g, match_copy, apply_copy, quiet=quiet, stats=stats)


def phase_free_simp(
    g: BaseGraph[VT, ET], quiet: bool = False, stats: Optional[Stats] = None
) -> int:
    """
    Performs the following set of simplifications on the graph:
    spider -> bialg
    """
    i_1 = simp(g, match_spider, spider, quiet=quiet, stats=stats)
    i_2 = simp(g, match_bialg, bialg, quiet=quiet, stats=stats)
    return i_1 + i_2


def flip_spider_type(g: BaseGraph[VT, ET], v: VT) -> None:
    """
    the type of a spider is flipped
    """
    g.set_type(v, toggle_vertex(g.types()[v]))
    for e in g.incident_edges(v):
        e_t = g.edge_type(e)
        g.set_edge_type(e, toggle_edge(e_t))
