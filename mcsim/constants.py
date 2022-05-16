from enum import Enum

class CircFormat(Enum):
    """
    Circuit formats supported by the simulator connectors
    """

    PYZX = "pyzx"


class SimulatorType(Enum):
    """
    Supported types of simulators
    """

    MCSIM = "mcsimulator"
    QSIM = "qsim"

try:
    import cirq

    PYZX_TO_CIRQ_MAPPING = {
        "one_qubit_phase": {
            "ZPhase": cirq.ZPowGate,
            "XPhase": cirq.XPowGate,
        },
        "one_qubit": {
            "Z": cirq.Z,
            "X": cirq.X,
            "NOT": cirq.X,
            "Y": cirq.Y,
            "HAD": cirq.H,
            "S": cirq.S,
            "T": cirq.T,
            "S*": (cirq.S**-1),
            "T*": (cirq.T**-1),
        },
        "two_qubit": {
            "CNOT": cirq.CNOT,
            "CZ": cirq.CZ,
            "SWAP": cirq.SWAP,
        },
        "extra_gates": {
            "CCZ": cirq.CCZ,
            "TOFFOLI": cirq.TOFFOLI,
        },
    }

except ImportError:
    PYZX_TO_CIRQ_MAPPING = {}
    pass
