"""WindFlow (Vandal et al., KDD 2022) adapter bundled with the AMV project.

The ``raft`` sub-package is vendored unchanged from
https://github.com/tjvandal/windflow (``windflow/networks/raft``), MIT licence
(see ``LICENSE_windflow``).
"""

from .inference import WindFlowTracker

__all__ = ["WindFlowTracker"]
