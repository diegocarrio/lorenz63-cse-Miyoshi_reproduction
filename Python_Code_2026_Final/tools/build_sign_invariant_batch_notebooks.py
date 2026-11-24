"""Create sign-invariant counterparts of the two original Figure 3 batch notebooks.

The source notebooks are retained unchanged. This is a mechanical notebook
transformation so the event selection, DA input, control mask, seeds, and
batch/grid loops stay aligned with their nature-reference predecessors.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    (
        "xyz",
        "7_run_cse_qiwen_xyz_control_nature_qiwenevents_batch.ipynb",
        "7_run_cse_qiwen_xyz_control_sign_invariant_qiwenevents_batch.ipynb",
    ),
    (
        "z",
        "7_run_cse_qiwen_z_control_nature_qiwenevents_batch.ipynb",
        "7_run_cse_qiwen_z_control_sign_invariant_qiwenevents_batch.ipynb",
    ),
)


def build_variant(channel, source_name, target_name):
    source_path = ROOT / source_name
    target_path = ROOT / target_name
    if target_path.exists():
        raise FileExistsError(f"Refusing to overwrite {target_path}")

    with source_path.open(encoding="utf-8") as stream:
        notebook = json.load(stream)

    source_text = "".join("".join(cell["source"]) for cell in notebook["cells"])
    if source_text.count("_nature_qiwenevents") < 2:
        raise ValueError(f"Unexpected source notebook: {source_path}")

    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        text = text.replace(
            "— Batch experiments",
            "— Sign-invariant batch experiments",
        )
        text = text.replace("_nature_qiwenevents", "_sign_invariant_qiwenevents")
        text = text.replace(
            "REGIME_REFERENCE_MODE = 'nature'",
            "REGIME_REFERENCE_MODE = 'sign_invariant'\n"
            "AMBIGUOUS_REGIME_POLICY = 'continue'",
        )
        text = text.replace(
            "regime_reference='nature'",
            "regime_reference='sign_invariant'",
        )
        text = text.replace(
            "REGIME_REFERENCE = 'nature'",
            "REGIME_REFERENCE = 'sign_invariant'",
        )
        text = text.replace(
            "globals().get('FIG3G_REGIME_REFERENCE', 'nature')",
            "globals().get('FIG3G_REGIME_REFERENCE', 'sign_invariant')",
        )
        text = text.replace(
            "output_suffix = '' if (REGIME_REFERENCE == 'nature' and CANDIDATE_POOL == 'full') else f'_{REGIME_REFERENCE}_{CANDIDATE_POOL}'",
            "output_suffix = f'_{REGIME_REFERENCE}_{CANDIDATE_POOL}'",
        )
        text = text.replace(
            "output_suffix = (\n"
            "        ''\n"
            "        if (FIG3G_REGIME_REFERENCE == 'nature' and FIG3G_CANDIDATE_POOL == 'full')\n"
            "        else f'_{FIG3G_REGIME_REFERENCE}_{FIG3G_CANDIDATE_POOL}'\n"
            "    )",
            "output_suffix = f'_{FIG3G_REGIME_REFERENCE}_{FIG3G_CANDIDATE_POOL}'",
        )
        text = text.replace(
            f"f'cse_qiwen_{channel}_control_40_results_D{{D_run}}{{OUTPUT_TAG}}.pkl'",
            f"f'cse_qiwen_{channel}_control_40_results_D{{D_run}}_sign_invariant_full{{OUTPUT_TAG}}.pkl'",
        )
        text = text.replace(
            "_legacy_4T0_output_file", "_fixed_horizon_4T0_output_file"
        ).replace("legacy_file", "fixed_horizon_file")
        text = text.replace(
            "        candidate_pool=candidate_pool,\n"
            "        control_indices=CONTROL_INDICES,",
            "        candidate_pool=candidate_pool,\n"
            "        ambiguous_regime_policy=AMBIGUOUS_REGIME_POLICY,\n"
            "        control_indices=CONTROL_INDICES,",
        )
        text = text.replace(
            "        candidate_pool=FIG3G_CANDIDATE_POOL,\n"
            "        control_indices=CONTROL_INDICES,",
            "        candidate_pool=FIG3G_CANDIDATE_POOL,\n"
            "        ambiguous_regime_policy=AMBIGUOUS_REGIME_POLICY,\n"
            "        control_indices=CONTROL_INDICES,",
        )
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cell["source"] = text.splitlines(keepends=True)

    intro = notebook["cells"][3]
    intro_text = "".join(intro["source"])
    intro_text += (
        "\n**Sign-invariant replication.** This notebook keeps the original "
        "xyz observation operator, Qiwen event set, control mask, seeds, "
        "D values, and T horizons. Only the regime-reference policy changes "
        "to `sign_invariant`: a past forecast is safe if its x stays strictly "
        "on either side of zero during the aligned window. Mixed-sign "
        "analysis ensembles do not stop the CSE. Results and figures go to "
        "a separate experiment directory. The 40-event and full T×D cells "
        "run expensive CSE simulations; execute them deliberately.\n"
    )
    intro["source"] = intro_text.splitlines(keepends=True)

    comparison = notebook["cells"][21]
    comparison_text = "".join(comparison["source"])
    comparison_text += (
        "\nThe Qiwen table is shown only as a historical benchmark; the "
        "sign-invariant reference policy is a different implementation.\n"
    )
    comparison["source"] = comparison_text.splitlines(keepends=True)

    all_text = "".join("".join(cell["source"]) for cell in notebook["cells"])
    for old, new in (
        (f"EXPERIMENT_NAME = '{channel}_control_nature_qiwenevents'",
         f"EXPERIMENT_NAME = '{channel}_control_sign_invariant_qiwenevents'"),
        ("REGIME_REFERENCE = 'nature'", "REGIME_REFERENCE = 'sign_invariant'"),
        ("regime_reference='nature'", "regime_reference='sign_invariant'"),
    ):
        if old in all_text or new not in all_text:
            raise ValueError(f"Incomplete transformation of {source_name}: {old}")
    if "_nature_qiwenevents" in all_text:
        raise ValueError(f"Nature-reference output name remains in {target_name}")

    with target_path.open("x", encoding="utf-8") as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
        stream.write("\n")
    print(target_path)


if __name__ == "__main__":
    for variant in VARIANTS:
        build_variant(*variant)
