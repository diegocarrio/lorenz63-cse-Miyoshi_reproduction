"""Observation-subset adapter for controlled CSE experiments.

The authoritative CSE implementation generates a three-component synthetic
observation at the end of every cycle.  This module leaves that implementation
unchanged and limits what reaches the analysis update.  Keeping the original
three random draws is intentional: it preserves the common-random-number
design with the existing full-observation experiment while only the selected
components are assimilated.
"""

from pathlib import Path

import numpy as np


class ObservationSubsetDAAdapter:
    """Delegate to a DA system while assimilating a selected observation subset."""

    def __init__(self, base_das, observed_indices):
        indices = np.asarray(observed_indices, dtype=int).ravel()
        if indices.size == 0:
            raise ValueError("At least one observed state index is required.")
        if np.unique(indices).size != indices.size:
            raise ValueError("observed_indices must not contain duplicates.")
        if np.any(indices < 0) or np.any(indices >= int(base_das.xdim)):
            raise ValueError(
                f"observed_indices={indices.tolist()} are incompatible with "
                f"xdim={base_das.xdim}."
            )

        expected_h = np.eye(int(base_das.xdim))[indices]
        actual_h = np.asarray(base_das.H, dtype=float)
        if actual_h.shape != expected_h.shape or not np.allclose(actual_h, expected_h):
            raise ValueError(
                "The DA observation operator must select exactly the requested "
                f"indices. Expected H={expected_h.tolist()}, got {actual_h.tolist()}."
            )
        if np.asarray(base_das.R).shape != (indices.size, indices.size):
            raise ValueError(
                "The observation covariance shape does not match the selected "
                f"channels: R shape={np.asarray(base_das.R).shape}."
            )
        if int(base_das.ydim) != indices.size:
            raise ValueError(
                f"base_das.ydim={base_das.ydim} but {indices.size} channels were selected."
            )

        self.base_das = base_das
        self.observed_indices = tuple(int(index) for index in indices)
        self.full_observation_history = []
        self.assimilated_observation_history = []

    def __getattr__(self, name):
        return getattr(self.base_das, name)

    def compute_analysis(self, Xb, observation):
        """Select the configured channels before calling the original DA update."""
        observation = np.asarray(observation, dtype=float).ravel()
        if observation.size == int(self.base_das.xdim):
            full_observation = observation.copy()
            assimilated = observation[np.asarray(self.observed_indices)]
        elif observation.size == len(self.observed_indices):
            full_observation = None
            assimilated = observation.copy()
        else:
            raise ValueError(
                "Observation size must equal either xdim or the selected observation "
                f"dimension; got {observation.size}."
            )

        self.full_observation_history.append(full_observation)
        self.assimilated_observation_history.append(assimilated.copy())
        return self.base_das.compute_analysis(Xb, assimilated)


def run_single_cse_with_observation_subset(
    *,
    das,
    observed_indices,
    run_single_cse_core=None,
    **core_kwargs,
):
    """Run unchanged CSE dynamics while restricting the assimilated observations.

    The core still generates a full noisy state vector, so its RNG consumption
    matches the full-observation experiment.  The adapter passes only the
    selected entries to ``das.compute_analysis`` and records both representations
    in the returned result.
    """
    if run_single_cse_core is None:
        import class_cse_qiwen

        run_single_cse_core = class_cse_qiwen.run_single_cse

    if "das" in core_kwargs:
        raise TypeError("Pass das only through the named das argument.")

    cle_target_output_dir = core_kwargs.pop("cle_target_output_dir", None)
    adapter = ObservationSubsetDAAdapter(das, observed_indices)
    result = run_single_cse_core(
        das=adapter,
        cle_target_output_dir=None,
        **core_kwargs,
    )

    full_history = np.asarray(result["yo_history"], dtype=float)
    used_history = np.asarray(adapter.assimilated_observation_history, dtype=float)
    if used_history.size == 0:
        used_history = np.empty((0, len(adapter.observed_indices)), dtype=float)
    else:
        used_history = used_history.reshape(-1, len(adapter.observed_indices))

    if full_history.shape[0] != used_history.shape[0]:
        raise RuntimeError(
            "The generated and assimilated observation histories have different lengths."
        )
    if full_history.size:
        expected_used = full_history[:, adapter.observed_indices]
        if not np.allclose(used_history, expected_used):
            raise RuntimeError("Assimilated observations do not match the selected channels.")

    metadata = {
        "observation_mode": "state_component_subset",
        "observed_indices": list(adapter.observed_indices),
        "observation_operator": np.asarray(das.H, dtype=float).tolist(),
        "observation_error_covariance": np.asarray(das.R, dtype=float).tolist(),
        "common_random_numbers": (
            "full three-component observation noise is generated by the unchanged "
            "CSE core; only the selected components enter the DA update"
        ),
    }

    result["yo_history_full_generated"] = full_history.copy()
    result["yo_history"] = used_history
    result["observation_metadata"] = metadata
    result["cse_summary"] = {**result["cse_summary"], **metadata}

    if cle_target_output_dir is not None:
        from class_cse_qiwen import export_cle_target_inputs

        result["cle_target_files"] = export_cle_target_inputs(
            result,
            Path(cle_target_output_dir),
        )
    return result
