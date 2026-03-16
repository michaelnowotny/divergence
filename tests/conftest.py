"""Shared test configuration and Hypothesis profiles."""

from hypothesis import settings

# Fast for local development
settings.register_profile("dev", max_examples=10)

# Thorough for CI
settings.register_profile("ci", max_examples=50)

# Exhaustive for pre-release validation
settings.register_profile("full", max_examples=200)
