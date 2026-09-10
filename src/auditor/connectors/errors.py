"""Typed connector-layer failures — never fabricate missing information."""

from __future__ import annotations


class ConnectorError(Exception):
    """Base class for connector failures."""


class SourceNotFoundError(ConnectorError):
    """Source path does not exist or is not a readable file."""


class EmptySourceError(ConnectorError):
    """Source exists but contains no records."""


class MalformedRecordError(ConnectorError):
    """A record could not be parsed honestly from the source."""


class MalformedSourceError(ConnectorError):
    """The source as a whole is not a valid instance of the declared format."""


class RecordNotFoundError(KeyError, ConnectorError):
    """Random-access lookup failed for the given record id."""


class UnsupportedFormatError(ConnectorError, ValueError):
    """File format is not supported by this connector."""
