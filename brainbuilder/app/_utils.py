"""utils"""

import click

REQUIRED_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)
REQUIRED_PATH_DIR = click.Path(
    exists=True, readable=True, file_okay=False, dir_okay=True, resolve_path=True
)
REQUIRED_PATH_DIR_OR_NONEXISTENT = click.Path(
    exists=False, readable=True, file_okay=False, dir_okay=True, resolve_path=True
)
