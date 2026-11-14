"""Re-identification baselines, registered by name for scripts/reid_run.py."""

from reid.eval.controls import DemographicModel

#: name -> zero-argument factory returning a fresh ReIDModel.
MODELS = {"control_demographics": DemographicModel}
