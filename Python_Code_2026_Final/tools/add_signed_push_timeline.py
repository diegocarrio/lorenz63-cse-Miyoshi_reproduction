"""Add a signed-intervention timeline to the z-observation CSE notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "19_run_cse_qiwen_z_control_z_obs_sign_invariant_qiwenevents.ipynb"
MARKDOWN_ID = "signed-pushes-on-free-intro"
CODE_ID = "signed-pushes-on-free-figure"

with NOTEBOOK.open("r", encoding="utf-8") as stream:
    notebook = json.load(stream)

notebook["cells"] = [
    cell for cell in notebook["cells"]
    if cell.get("id") not in {MARKDOWN_ID, CODE_ID}
]

markdown = {
    "cell_type": "markdown",
    "id": MARKDOWN_ID,
    "metadata": {},
    "source": [
        "## Free versus Controlled Nature Runs with Signed z-Push Times\n",
        "\n",
        "This figure preserves the time-series comparison above and places "
        "orange/blue markers on the **free** $x,y,z$ curves at the times when "
        "the controller applied positive/negative $u_z$. The markers are "
        "timestamps for comparison: the free run was never perturbed. The "
        "bottom panel shows the actual signed pushes. This cell uses the "
        "saved CSE result if it is not already in memory; it does not rerun "
        "the controller.\n",
    ],
}

code_source = """# Visualize existing interventions; do not run a new CSE experiment.
import importlib
import pickle
from IPython.display import Image, display
import cse_intervention_state_plot as _intervention_state_plot

_intervention_state_plot = importlib.reload(_intervention_state_plot)
signed_push_result = globals().get('single_cse_result')
if (
    signed_push_result is None
    or int(signed_push_result['event_index']) != EVENT_INDEX
    or not np.isclose(float(signed_push_result['D']), D)
    or int(signed_push_result['t0']) != t0
):
    signed_d_tag = f'{D:g}'.replace('.', 'p')
    signed_result_path = RESULTS_DIR / (
        f'cse_z_control_z_obs_sign_invariant_event_{EVENT_INDEX}_'
        f'D{signed_d_tag}_qiwenevents.pkl'
    )
    with signed_result_path.open('rb') as stream:
        signed_push_result = pickle.load(stream)

signed_push_info = _intervention_state_plot.plot_signed_push_times_on_free(
    signed_push_result,
    x_nature=x_nature,
    Ta=Ta,
    figures_dir=FIGURES_DIR,
)
print('Positive pushes:', signed_push_info['n_positive'])
print('Negative pushes:', signed_push_info['n_negative'])
print('Figure:', signed_push_info['figure_path'])
display(Image(filename=str(signed_push_info['figure_path'])))
"""
code = {
    "cell_type": "code",
    "execution_count": None,
    "id": CODE_ID,
    "metadata": {},
    "outputs": [],
    "source": code_source.splitlines(keepends=True),
}

insert_before = next(
    index for index, cell in enumerate(notebook["cells"])
    if cell.get("id") == "preintervention-sign-figure-intro"
)
notebook["cells"][insert_before:insert_before] = [markdown, code]

with NOTEBOOK.open("w", encoding="utf-8") as stream:
    json.dump(notebook, stream, ensure_ascii=False, indent=1)
    stream.write("\n")

print(f"Updated {NOTEBOOK}")
