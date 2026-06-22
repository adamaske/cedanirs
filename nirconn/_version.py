"""Single source of truth for the package version.

Kept in its own module with only a literal assignment so that build backends
(and ``importlib.metadata`` fallbacks) can read it via static analysis without
importing the whole package and its runtime dependencies.
"""

__version__ = "0.1.0"
