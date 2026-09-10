"""Analyzer layer public API (design §4.5)."""

from auditor.analyzers.base import Analyzer, AnalyzerCapability, CostTier
from auditor.analyzers.fake import CountRecordsAnalyzer

__all__ = [
    "Analyzer",
    "AnalyzerCapability",
    "CostTier",
    "CountRecordsAnalyzer",
]
