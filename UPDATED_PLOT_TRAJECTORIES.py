import os
import json
import time
import numpy as np
import torch
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from UPDATED_OPTIMIZERS_AND_LOSSES import LOSS_FUNCTIONS, RayShootingOptimizer, SGDOptimizer, AdamOptimizer

torch.manual_seed(0)
np.random.seed(0)

START_POINT = torch.tensor([-3.0, -3.0], dtype=torch.float32)
GRID_X = (-5.0, 5.0)
GRID_Y = (-5.0, 5.0)
RESOLUTION = 200
ROYGBIV = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]

def eval_grid(loss_fn, x_range=GRID_X, y_range=GRID_Y, res=RESOLUTION):
    xs = np.linspace(x_range[0], x_range[1], res)
    ys = np.linspace(y_range[0], y_range[1], res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(res):
        for j in range(res):
            pt = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
            Z[i, j] = loss_fn(pt).item()
    return xs, ys, Z

def run_optimizer_by_name(name: str, params: dict, loss_fn, start_point: torch.Tensor):
    if name == 'RayShootingOptimizer':
        p = params.copy()
        opt = RayShootingOptimizer(
            lr=p.get('lr', 0.01),
            ascent_steps=p.get('ascent_steps', 200),
            num_rays=p.get('num_rays', 16),
            ray_step_size=p.get('ray_step_size', 0.05),
            distance_threshold=p.get('distance_threshold', 0.1),
            epsilon=p.get('epsilon', 1e-4),
            max_iterations=p.get('max_iterations', 6),
            use_momentum=p.get('use_momentum', False),
            beta1=p.get('beta1', 0.9),
            beta2=p.get('beta2', 0.999),
            use_adaptive_lr=p.get('use_adaptive_lr', False),
            lr_decay=p.get('lr_decay', 0.99),
            lr_patience=p.get('lr_patience', 10),
            tol=p.get('tol', 1e-3),
            max_param_step_norm=p.get('max_param_step_norm', 0.1),
            max_grad_norm=p.get('max_grad_norm', 1.0),
            early_stop_patience=p.get('early_stop_patience', 10)
        )
        opt.optimize(loss_fn, start_point)
        return opt, opt.history
    elif name == 'SGDOptimizer':
        opt = SGDOptimizer(lr=params.get('lr', 0.01), steps=params.get('steps', 300), momentum=params.get('momentum', 0.0))
        opt.optimize(loss_fn, start_point)
        return opt, opt.history
    elif name == 'AdamOptimizer':
        opt = AdamOptimizer(lr=params.get('lr', 0.01), steps=params.get('steps', 300), beta1=params.get('beta1', 0.9), beta2=params.get('beta2', 0.999))
        opt.optimize(loss_fn, start_point)
        return opt, opt.history
    else:
        raise ValueError("Unknown optimizer")

OPT_COLORS = {
    "RayShootingOptimizer": "red",
    "SGDOptimizer": "blue",
    "AdamOptimizer": "green"
}

def plot_all_optimizers_for_loss(loss_name, loss_fn, best_params):
    xs, ys, Z = eval_grid(loss_fn)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.78, 0.22],
                        specs=[[{"type": "xy"}, {"type": "table"}]])

    fig.add_trace(go.Contour(
        z=Z, x=xs, y=ys,
        colorscale='Viridis', opacity=0.9, showscale=False,
        contours=dict(showlabels=False)),
        row=1, col=1)

    max_val = np.max(Z)
    max_pts = np.argwhere(np.isclose(Z, max_val))
    gold_legend_shown = False
    for (i, j) in max_pts:
        showleg = not gold_legend_shown
        fig.add_trace(go.Scatter(
            x=[xs[j]], y=[ys[i]],
            mode="markers",
            marker=dict(symbol="star", size=18, color="gold", line=dict(width=1, color="black")),
            name="Global max" if showleg else None,
            showlegend=showleg),
            row=1, col=1)
        gold_legend_shown = True

    table_names, table_best_vals, table_times, table_x, table_y = [], [], [], [], []

    for opt_name in ["RayShootingOptimizer", "SGDOptimizer", "AdamOptimizer"]:
        opt_entry = best_params.get(opt_name)
        if opt_entry is None:
            continue
        params = opt_entry.get("params", {})

        start_time = time.time()
        opt, history = run_optimizer_by_name(opt_name, params, loss_fn, START_POINT)
        elapsed = time.time() - start_time
        best_val = history.get("best_value")
        best_pt_raw = history.get("best_point")
        best_pt = None
        if best_pt_raw is not None:
            try:
                best_pt = best_pt_raw.detach().cpu().numpy()
            except Exception:
                best_pt = np.array(best_pt_raw)

        table_names.append(opt_name)
        table_best_vals.append(f"{best_val:.6f}" if best_val is not None else "N/A")
        table_times.append(f"{elapsed:.2f}")
        if best_pt is not None:
            table_x.append(f"{best_pt[0]:.4f}")
            table_y.append(f"{best_pt[1]:.4f}")
        else:
            table_x.append("")
            table_y.append("")

        color = OPT_COLORS.get(opt_name, "black")

        if opt_name == "RayShootingOptimizer":
            hist = history
            all_paths = hist.get('all_paths', [])
            if all_paths:
                first = True
                for p in all_paths:
                    fig.add_trace(go.Scatter(
                        x=p[:, 0], y=p[:, 1],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        name=f"{opt_name} path" if first else None,
                        showlegend=first),
                        row=1, col=1)
                    first = False

            ray_legend_shown = False
            for it in hist.get('iteration_data', []):
                ray_paths = it.get('ray_paths', [])
                ray_colors = it.get('ray_colors', [])
                for r_idx, rpath in enumerate(ray_paths):
                    rcolor = ray_colors[r_idx] if (ray_colors and r_idx < len(ray_colors)) else ROYGBIV[r_idx % len(ROYGBIV)]
                    showleg = not ray_legend_shown
                    fig.add_trace(go.Scatter(
                        x=rpath[:, 0], y=rpath[:, 1],
                        mode="lines",
                        line=dict(color=rcolor, dash='dash', width=1.6),
                        name=(f"Ray paths ({opt_name})" if showleg else None),
                        showlegend=showleg,
                        hoverinfo='skip'),
                        row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=[rpath[-1, 0]], y=[rpath[-1, 1]],
                        mode="markers",
                        marker=dict(symbol='circle-open', size=8, color=rcolor),
                        showlegend=False,
                        hoverinfo='skip'),
                        row=1, col=1)
                    ray_legend_shown = True

            if best_pt is not None:
                fig.add_trace(go.Scatter(
                    x=[best_pt[0]], y=[best_pt[1]],
                    mode="markers",
                    marker=dict(symbol="star", size=14, color=color),
                    showlegend=False),
                    row=1, col=1)
        else:
            hp = history.get('all_paths', [])
            if hp:
                first_path = True
                for p in hp:
                    fig.add_trace(go.Scatter(
                        x=p[:, 0], y=p[:, 1],
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=4),
                        name=f"{opt_name} path" if first_path else None,
                        showlegend=first_path),
                        row=1, col=1)
                    first_path = False
                last = hp[-1][-1]
                fig.add_trace(go.Scatter(
                    x=[last[0]], y=[last[1]],
                    mode="markers",
                    marker=dict(symbol="star", size=12, color=color),
                    showlegend=False),
                    row=1, col=1)

    # stats table
    if len(table_names) == 0:
        table_header = ["No optimizers ran"]
        table_cells = [["-"], ["-"], ["-"], ["-"], ["-"]]
    else:
        table_header = ["Optimizer", "Best value", "Time (s)", "Final x", "Final y"]
        table_cells = [table_names, table_best_vals, table_times, table_x, table_y]

    fig.add_trace(go.Table(
        header=dict(values=table_header, fill_color='lightgrey', align='left'),
        cells=dict(values=table_cells, align='left', height=30)
    ), row=1, col=2)

    fig.update_layout(
        title=f"{loss_name} — All Optimizers",
        width=1200, height=700,
        template="plotly_white",
        legend=dict(bgcolor="white", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

# --- Dash app ---

# Load BEST_PARAMS once at startup
if not os.path.exists('BEST_PARAMS.json'):
    raise FileNotFoundError("BEST_PARAMS.json not found. Run grid_search.py first.")
with open('BEST_PARAMS.json', 'r') as f:
    BEST_PARAMS = json.load(f)

app = dash.Dash(__name__)
app.title = "Optimizer Trajectories Viewer"

app.layout = html.Div([
    html.H1("Select Loss Function to Plot"),
    dcc.Dropdown(
        id='loss-dropdown',
        options=[{'label': name, 'value': name} for name in LOSS_FUNCTIONS.keys()],
        value=list(LOSS_FUNCTIONS.keys())[0],
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Loading(
        id="loading-plot",
        type="circle",
        children=dcc.Graph(id='loss-plot', style={'height': '750px'})
    )
])

@app.callback(
    Output('loss-plot', 'figure'),
    Input('loss-dropdown', 'value')
)
def update_plot(loss_name):
    loss_fn = LOSS_FUNCTIONS.get(loss_name)
    if loss_fn is None:
        return go.Figure()
    best_params = BEST_PARAMS.get(loss_name, {})
    fig = plot_all_optimizers_for_loss(loss_name, loss_fn, best_params)
    return fig


if __name__ == "__main__":
    app.run(debug=False)

