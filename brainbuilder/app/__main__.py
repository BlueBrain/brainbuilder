"""
Collection of tools for circuit building.
"""

import logging
import click

from brainbuilder.app import cells, mvd3, nrn, sonata, subcellular, syn2, targets
from brainbuilder.version import VERSION


def main():
    """ Collection of tools for circuit building """
    logging.basicConfig(level=logging.INFO)
    app = click.Group('brainbuilder', {
        'cells': cells.app,
        'mvd3': mvd3.app,
        'nrn': nrn.app,
        'sonata': sonata.app,
        'subcellular': subcellular.app,
        'syn2': syn2.app,
        'targets': targets.app,
    })
    app = click.version_option(VERSION)(app)
    app()


if __name__ == '__main__':
    main()
