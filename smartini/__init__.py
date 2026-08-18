import logging
import os


def setup_logging(level=logging.INFO, force=False):
        """Configure root logging for scripts using smartini.

        By default, this only configures logging when no handlers are present.
        Set ``force=True`` to override existing handlers.
        """
        root_logger = logging.getLogger()
        if force or not root_logger.handlers:
                logging.basicConfig(
                        level=level,
                        format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s",
                        force=force,
                )
        else:
                root_logger.setLevel(level)

from smartini import solver, topology
# from ._version import __version__

# __all__ = ["solver", "topology", "partitioning", "__version__", "setup_logging"]
