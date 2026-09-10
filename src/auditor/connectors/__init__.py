"""Connector layer public API (design §4.1)."""

from auditor.connectors.base import CapabilityDescriptor, Connector
from auditor.connectors.errors import (
    ConnectorError,
    EmptySourceError,
    MalformedRecordError,
    RecordNotFoundError,
    SourceNotFoundError,
)
from auditor.connectors.fake import InMemoryConnector
from auditor.connectors.local import LocalFilesConnector

__all__ = [
    "CapabilityDescriptor",
    "Connector",
    "ConnectorError",
    "EmptySourceError",
    "InMemoryConnector",
    "LocalFilesConnector",
    "MalformedRecordError",
    "RecordNotFoundError",
    "SourceNotFoundError",
]
