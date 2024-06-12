# SPDX-License-Identifier: Apache-2.0
"""
Collection of tools for circuit building.
"""

import logging

import click

from brainbuilder.app import atlases, cells, mvd3, nrn, sonata, syn2, targets
from brainbuilder.version import __version__


@click.group("brainbuilder")
@click.version_option(__version__)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="DEBUG",
    show_default=True,
    help="Logging level",
)
def main(log_level):
    """Collection of tools for circuit building"""
    logging.basicConfig(level=log_level)


for name, app in {
    "cells": cells.app,
    "mvd3": mvd3.app,
    "nrn": nrn.app,
    "sonata": sonata.app,
    "syn2": syn2.app,
    "targets": targets.app,
    "atlases": atlases.app,
}.items():
    main.add_command(app, name)
