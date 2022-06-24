# PyZX - Python library for quantum circuit rewriting
#        and optimization using the ZX-calculus
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module provides methods for converting ZX-graphs into numpy tensors 
and using these tensors to test semantic equality of ZX-graphs. 
This module is not meant as an efficient quantum simulator. 
Due to the way the tensor is calculated it can only handle 
circuits of small size before running out of memory on a regular machine. 
Currently, it can reliably transform 9 qubit circuits into tensors. 
If the ZX-diagram is not circuit-like, but instead has nodes with high degree, 
it will run out of memory even sooner."""

__all__ = [
    "tensorfy",
    "compare_tensors",
    "compose_tensors",
    "adjoint",
    "is_unitary",
    "tensor_to_matrix",
    "find_scalar_correction",
]

from math import pi, sqrt

from typing import Optional

try:
    import cupy as np

    print("CUPY tensorfy!")
except:
    import numpy as np

    np.set_printoptions(suppress=True)

import time
# from memory_profiler import  profile

# typing imports
from typing import TYPE_CHECKING, List, Dict, Union
from .utils import FractionLike, FloatInt, VertexType, EdgeType

if TYPE_CHECKING:
    from .graph.base import BaseGraph, VT, ET
    from .circuit import Circuit
TensorConvertible = Union[np.ndarray, "Circuit", "BaseGraph"]


def Z_to_tensor(arity: int, phase: float) -> np.ndarray:
    m = np.zeros([2] * arity, dtype=complex)
    if arity == 0:
        m[()] = 1 + np.exp(1j * phase)
        return m
    m[(0,) * arity] = 1
    m[(1,) * arity] = np.exp(1j * phase)

    # print("Ztensor:", m)
    return m


def X_to_tensor(arity: int, phase: float) -> np.ndarray:
    m = np.ones(2 ** arity, dtype=complex)
    if arity == 0:
        m[()] = 1 + np.exp(1j * phase)
        return m
    for i in range(2 ** arity):
        if bin(i).count("1") % 2 == 0:
            m[i] += np.exp(1j * phase)
        else:
            m[i] -= np.exp(1j * phase)

    # print("Xtensor:", np.power(np.sqrt(0.5), arity) * m.reshape([2] * arity))
    return np.power(np.sqrt(0.5), arity) * m.reshape([2] * arity)


def H_to_tensor(arity: int, phase: float) -> np.ndarray:
    m = np.ones(2 ** arity, dtype=complex)
    if phase != 0:
        m[-1] = np.exp(1j * phase)
    return m.reshape([2] * arity)


def pop_and_shift_uncontracted_indices(past_verts, indices):
    past_to_contract = []
    for v in past_verts:
        past_to_contract.append(indices[v].pop())

    """
    The indices that will be contracted will disapear,
    such that indices following it (j>i) will have index lowered by the number
    of indices contracted before it 
    """
    for i in sorted(past_to_contract, reverse=True):
        # For each contracted index, the update is repeated many times
        # TODO: Very complex now, reduce complexity
        for w, l in indices.items():
            l2 = []
            for j in l:
                if j > i:
                    l2.append(j - 1)
                else:
                    l2.append(j)
            indices[w] = l2
    return past_to_contract


# @profile
def tensorfy(g: "BaseGraph[VT,ET]", preserve_scalar: bool = True) -> np.ndarray:
    """Takes in a Graph and outputs a multidimensional numpy array
    representing the linear map the ZX-diagram implements.
    Beware that quantum circuits take exponential memory to represent."""

    # tic = time.perf_counter()

    if g.is_hybrid():
        raise ValueError("Hybrid graphs are not supported.")
    rows = g.rows()
    phases = g.phases()
    types = g.types()
    depth = g.depth()
    verts_row: Dict[FloatInt, List["VT"]] = {}

    for v in g.vertices():
        curr_row = rows[v]
        if curr_row in verts_row:
            verts_row[curr_row].append(v)
        else:
            verts_row[curr_row] = [v]

    # print("### \n verts row :{} \n###".format(verts_row))

    inputs = g.inputs()
    # print("### \n inputs_0 :{} \n###".format(inputs))

    outputs = g.outputs()
    if not inputs and not outputs:
        if any(g.type(v) == VertexType.BOUNDARY for v in g.vertices()):
            raise ValueError(
                "Diagram contains BOUNDARY-type vertices, but has no inputs or outputs set. Perhaps call g.auto_detect_inputs() first?"
            )

    had = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])
    id2 = np.identity(2)

    tensor = np.array(1.0, dtype="complex128")
    qubits = len(inputs)

    for i in range(qubits):
        tensor = np.tensordot(tensor, id2, axes=0)

    # print("7777tensor\n:",tensor)
    # print("inp tensor: \n", tensor_to_matrix(tensor, len(inputs), len(outputs)))

    inputs = tuple(sorted(inputs, key=g.qubit))
    # print("### \n inputs_1 :{} \n###".format(inputs))

    indices = {}

    inputs = sorted(inputs, key=g.qubit)
    # print("### \n inputs_2 :{} \n###".format(inputs))

    uncontracted_indices = {}

    for i, v in enumerate(inputs):
        uncontracted_indices[v] = [1 + 2 * i]  # why +2*i
    # print("### \n uncontracted_indices_0 :{} \n###".format(uncontracted_indices))

    for i, curr_row in enumerate(sorted(verts_row.keys())):
        # ->row
        for v in sorted(verts_row[curr_row]):
            # we took each node from a row
            neigh = list(g.neighbors(v))  # list of neighbours

            arity = len(neigh)  # nr of neighbours

            if v in inputs:
                if types[v] != 0:
                    raise ValueError("Wrong type for input:", v, types[v])
                continue  # inputs already taken care of
            if v in outputs:
                # print("output")
                if arity != 1:
                    raise ValueError("Weird output")
                if types[v] != 0:
                    raise ValueError("Wrong type for output:", v, types[v])
                arity += 1  # why to increse the number of nr neighbours with 2 ?
                t = id2
            else:
                phase = pi * phases[v]
                if types[v] == 1:
                    t = Z_to_tensor(arity, phase)
                elif types[v] == 2:
                    t = X_to_tensor(arity, phase)
                elif types[v] == 3:
                    t = H_to_tensor(arity, phase)
                else:
                    raise ValueError(
                        "Vertex %s has non-ZXH type but is not an input or output"
                        % str(v)
                    )

            # Now I have tensor t for  a certain node v on some row curr row

            # type: ignore # TODO: allow ordering on vertex indices?
            past_vertices = list(
                filter(
                    lambda n: rows[n] < curr_row or (rows[n] == curr_row and n < v),
                    neigh,
                )
            )
            # print(
            #     "###\n current_vert={} \n past vertices: {} \n###".format(
            #         v, past_vertices
            #     )
            # )

            edge_type = {n: g.edge_type(g.edge(v, n)) for n in past_vertices}
            # print("###\n current_vert={} \n edge_types: {} \n###".format(v, edge_type))

            past_vertices.sort(key=lambda n: edge_type[n])
            # print(
            #     "###\n current_vert={} \n past_vertices_sorted: {} \n###".format(
            #         v, edge_type
            #     )
            # )

            for n in past_vertices:
                if edge_type[n] == EdgeType.HADAMARD:
                    t = np.tensordot(
                        t, had, (0, 0)
                    )  # Hadamard edges are moved to the last index of t
            # we multiply t with had if we have had gate in front if a second vertex

            # the last indices in idx_contr_past correspond to hadamard contractions
            # These are the indices in the total tensor that will be contracted

            # print(
            #     "\n###\n past vertices : {} \n current vertice:{}\n uncontracted indices:{} ".format(
            #         past_vertices, v, uncontracted_indices
            #     )
            # )

            idx_contr_past = pop_and_shift_uncontracted_indices(
                past_vertices, uncontracted_indices
            )
            # print("idx_contr_past 0:{} \n###".format(idx_contr_past))

            # The last axes in the tensor t are the one that will be contracted
            idx_contr_curr = list(
                range(len(t.shape) - len(idx_contr_past), len(t.shape))
            )
            # print(
            #     "###\nindx_contr_past 001:{} \n idx_contr_curr{} \n###".format(
            #         idx_contr_past, idx_contr_curr
            #     )
            # )

            tensor = np.tensordot(tensor, t, axes=(idx_contr_past, idx_contr_curr))

            # For the vertex v the indices that remain uncontracted are the last ones
            # The edges on witch contraction isn't done
            nr_remainining_indices = arity - len(idx_contr_past)
            uncontracted_indices[v] = list(
                range(len(tensor.shape) - nr_remainining_indices, len(tensor.shape))
            )

            if not preserve_scalar and i % 10 == 0:
                if np.abs(tensor).max() < 10 ** -6:  # Values are becoming too small
                    tensor *= 10 ** 4  # So scale all the numbers up

    if preserve_scalar:
        tensor *= g.scalar.to_number()

    # print("base tensor 0 permut: ", tensor)
    # print("base tensor_mat 0 permut: \n", tensor_to_matrix(tensor, len(inputs),len(outputs)))
    perm = []
    for o in sorted(outputs, key=g.qubit):
        assert len(uncontracted_indices[o]) == 1
        perm.append(uncontracted_indices[o][0])
    for i in range(len(inputs)):
        perm.append(i)

    # print("$$: perm:",perm)
    tensor = np.transpose(tensor, perm)

    # Test that sparse works
    # TODO : Alexandru
    # sparse_tensor = tensorfy_scipy(g, preserve_scalar)
    # assert (sparse_tensor == tensor)

    # print("@@ tensorfy @@")
    #
    # print("tesnsor shape:", tensor.shape)
    # print("tensor:", tensor)

    # toc = time.perf_counter()
    # print(f"Time in tensorfy {toc - tic:0.4f} seconds")
    return tensor


def tensor_to_matrix(t: np.ndarray, inputs: int, outputs: int) -> np.ndarray:
    """Takes a tensor generated by ``tensorfy`` and turns it into a matrix.
    The ``inputs`` and ``outputs`` arguments specify the final shape of the matrix:
    2^(outputs) x 2^(inputs)"""
    rows = []
    for r in range(2 ** outputs):
        if outputs == 0:
            o = []
        else:
            o = [int(i) for i in bin(r)[2:].zfill(outputs)]
        row = []
        if inputs == 0:
            row.append(t[tuple(o)])
        else:
            for c in range(2 ** inputs):
                a = o.copy()
                a.extend([int(i) for i in bin(c)[2:].zfill(inputs)])
                # print(a)
                # print(t[tuple(a)])
                row.append(t[tuple(a)])
        rows.append(row)
    return np.array(rows)


def compare_tensors(
        t1: TensorConvertible, t2: TensorConvertible, preserve_scalar: bool = False
) -> bool:
    """Returns true if ``t1`` and ``t2`` represent equal tensors.
    When `preserve_scalar` is False (the default), equality is checked up to nonzero rescaling.

    Example: To check whether two ZX-graphs `g1` and `g2` are semantically the same you would do::

        compare_tensors(g1,g2) # True if g1 and g2 represent the same linear map up to nonzero scalar

    """
    from .circuit import Circuit

    if not isinstance(t1, np.ndarray):
        t1 = t1.to_tensor(preserve_scalar)
    if not isinstance(t2, np.ndarray):
        t2 = t2.to_tensor(preserve_scalar)
    if np.allclose(t1, t2):
        return True
    if preserve_scalar:
        return False  # We do not check for equality up to scalar
    epsilon = 10 ** -14
    for i, a in enumerate(t1.flat):
        if abs(a) > epsilon:  # type: ignore #TODO: Figure out how numpy typing works
            if abs(t2.flat[i]) < epsilon:
                return False  # type: ignore #TODO: Figure out how numpy typing works
            break
    else:
        raise ValueError("Tensor is too close to zero")
    return np.allclose(t1 / a, t2 / t2.flat[i])


def find_scalar_correction(t1: TensorConvertible, t2: TensorConvertible) -> complex:
    """Returns the complex number ``z`` such that ``t1 = z*t2``.

    Warning:
        This function assumes that ``compare_tensors(t1,t2,preserve_scalar=False)`` is True,
        i.e. that ``t1`` and ``t2`` indeed are equal up to global scalar.
        If they aren't, this function returns garbage.

    """
    if not isinstance(t1, np.ndarray):
        t1 = t1.to_tensor(preserve_scalar=True)
    if not isinstance(t2, np.ndarray):
        t2 = t2.to_tensor(preserve_scalar=True)

    epsilon = 10 ** -14
    for i, a in enumerate(t1.flat):
        if abs(a) > epsilon:  # type: ignore #TODO: Figure out how numpy typing works
            if abs(t2.flat[i]) < epsilon:
                return 0  # type: ignore #TODO: Figure out how numpy typing works
            return a / t2.flat[i]  # type: ignore #TODO: Figure out how numpy typing works

    return 0


def compose_tensors(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Returns a tensor that is the result of composing the tensors together as if they
    were representing circuits::

        t1 = tensorfy(circ1)
        t2 = tensorfy(circ2)
        circ1.compose(circ2)
        t3 = tensorfy(circ1)
        t4 = compose_tensors(t1,t2)
        compare_tensors(t3,t4) # This is True

    """

    if len(t1.shape) != len(t2.shape):
        raise TypeError(
            "Tensors represent circuits of different amount of qubits, "
            "{!s} vs {!s}".format(len(t1.shape) // 2, len(t2.shape) // 2)
        )
    q = len(t1.shape) // 2
    contr2 = [q + i for i in range(q)]
    contr1 = [i for i in range(q)]
    t = np.tensordot(t1, t2, axes=(contr1, contr2))
    transp = []
    for i in range(q):
        transp.append(q + i)
    for i in range(q):
        transp.append(i)
    return np.transpose(t, transp)


def adjoint(t: np.ndarray) -> np.ndarray:
    """Returns the adjoint of the tensor as if it were representing
    a circuit::

        t = tensorfy(circ)
        tadj = tensorfy(circ.adjoint())
        compare_tensors(adjoint(t),tadj) # This is True

    """

    q = len(t.shape) // 2
    transp = []
    for i in range(q):
        transp.append(q + i)
    for i in range(q):
        transp.append(i)
    return np.transpose(t.conjugate(), transp)


def is_unitary(g: "BaseGraph") -> bool:
    """Returns whether the given ZX-graph is equal to a unitary (up to a number)."""
    from .generate import identity  # Imported here to prevent circularity

    adj = g.adjoint()
    adj.compose(g)
    return compare_tensors(
        adj.to_tensor(), identity(len(g.inputs()), 2).to_tensor(), False
    )
