from __future__ import annotations

from .expression_checks import MissingPurityCheck, NegativeYieldCheck, OutlierYieldCheck
from .flow_checks import BCellGateOutOfRangeCheck, GateSumExceeds100Check, QCFlagMismatchCheck
from .integrity_checks import AssayWithoutSequenceCheck, OrphanSamplesCheck, OrphanSequencesCheck
from .sequence_checks import DuplicateSequenceCheck, MalformedCDR3Check, MissingCDR3Check

ALL_CHECKS = [
    MissingCDR3Check,
    MalformedCDR3Check,
    DuplicateSequenceCheck,
    OutlierYieldCheck,
    NegativeYieldCheck,
    MissingPurityCheck,
    BCellGateOutOfRangeCheck,
    GateSumExceeds100Check,
    QCFlagMismatchCheck,
    OrphanSamplesCheck,
    OrphanSequencesCheck,
    AssayWithoutSequenceCheck,
]

CHECK_GROUPS: dict[str, list] = {
    "sequence": [MissingCDR3Check, MalformedCDR3Check, DuplicateSequenceCheck],
    "expression": [OutlierYieldCheck, NegativeYieldCheck, MissingPurityCheck],
    "flow": [BCellGateOutOfRangeCheck, GateSumExceeds100Check, QCFlagMismatchCheck],
    "integrity": [OrphanSamplesCheck, OrphanSequencesCheck, AssayWithoutSequenceCheck],
    "all": ALL_CHECKS,
}

__all__ = [
    "MissingCDR3Check",
    "MalformedCDR3Check",
    "DuplicateSequenceCheck",
    "OutlierYieldCheck",
    "NegativeYieldCheck",
    "MissingPurityCheck",
    "BCellGateOutOfRangeCheck",
    "GateSumExceeds100Check",
    "QCFlagMismatchCheck",
    "OrphanSamplesCheck",
    "OrphanSequencesCheck",
    "AssayWithoutSequenceCheck",
    "ALL_CHECKS",
    "CHECK_GROUPS",
]
