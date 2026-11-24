"""Add the retrospective pre-push state diagnostic to notebook 19."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "19_run_cse_qiwen_z_control_z_obs_sign_invariant_qiwenevents.ipynb"
MARKDOWN_ID = "preintervention-sign-figure-intro"
CODE_ID = "preintervention-sign-figure"

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
        "## Sign of $u_z$ versus the pre-intervention state\n",
        "\n",
        "Retrospective diagnostic only: the controller still assimilates only $z$. "
        "The figure pairs each nonzero $u_z$ with the *controlled* nature state "
        "immediately before that push, reconstructed by subtracting the saved "
        "perturbation from the post-push state. The first panel splits the same "
        "set of applied pushes separately into lower/higher $x$ and lower/higher "
        "$y$ groups, using each variable's median. These are sign frequencies, "
        "not control-success rates. The other panels show the $x$–$y$ and $x$–$z$ "
        "phase planes. "
        "Circles and triangles distinguish the current-safe and past-safe "
        "branches. The source table is saved alongside the figure.\n",
    ],
}

code_source = """# Reuse the result already in memory or load the saved event; do not rerun CSE.
import importlib
import pickle
from IPython.display import Image, display
import cse_intervention_state_plot as _intervention_state_plot

_intervention_state_plot = importlib.reload(_intervention_state_plot)
result_for_state_figure = globals().get('single_cse_result')
if (
    result_for_state_figure is None
    or int(result_for_state_figure['event_index']) != EVENT_INDEX
    or not np.isclose(float(result_for_state_figure['D']), D)
    or int(result_for_state_figure['t0']) != t0
):
    d_tag_for_state_figure = f'{D:g}'.replace('.', 'p')
    saved_result_path = RESULTS_DIR / (
        f'cse_z_control_z_obs_sign_invariant_event_{EVENT_INDEX}_'
        f'D{d_tag_for_state_figure}_qiwenevents.pkl'
    )
    with saved_result_path.open('rb') as stream:
        result_for_state_figure = pickle.load(stream)

state_figure_info = _intervention_state_plot.plot_pre_intervention_state(
    result_for_state_figure,
    Ta=Ta,
    figures_dir=FIGURES_DIR,
    results_dir=RESULTS_DIR,
)
print('Applied pushes:', state_figure_info['n_pushes'])
print('Positive / negative:', state_figure_info['n_positive'], '/',
      state_figure_info['n_negative'])
print('Per-push state table:', state_figure_info['table_path'])
print('Figure:', state_figure_info['figure_path'])
display(Image(filename=str(state_figure_info['figure_path'])))
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
    if "## Three-Dimensional Attractor Comparison" in "".join(cell["source"])
)
notebook["cells"][insert_before:insert_before] = [markdown, code]

with NOTEBOOK.open("w", encoding="utf-8") as stream:
    json.dump(notebook, stream, ensure_ascii=False, indent=1)
    stream.write("\n")

print(f"Updated {NOTEBOOK}")
