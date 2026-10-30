"""Save regression and residual figures without a GUI or server."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def make_figure(title, subtitle):
    figure = Figure(figsize=(10, 6), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    figure.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.78)
    figure.text(0.1, 0.94, title, fontsize=19, weight="bold", color="#15283F")
    figure.text(0.1, 0.88, subtitle, fontsize=10, color="#526275")
    figure.text(0.1, 0.045, "DATA SCIENCE PJ4 | Bundled dataset is synthetic practice data", fontsize=9, color="#526275")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(color="#E5EAF0")
    axes.set_axisbelow(True)
    return figure, axes


def save_plots(x_train, x_test, y_train, y_test, predicted, model, output: Path):
    figure, axes = make_figure("Study hours and student scores", "Linear regression fitted on training rows; test rows held out")
    axes.scatter(x_train.Hours, y_train, label="Training observations", color="#2457A7", s=45)
    axes.scatter(x_test.Hours, y_test, label="Test observations", color="#D56E25", marker="s", s=50)
    hours = np.linspace(min(x_train.Hours.min(), x_test.Hours.min()), max(x_train.Hours.max(), x_test.Hours.max()), 100)
    axes.plot(hours, model.predict(pd.DataFrame({"Hours": hours})), label="Training-fitted line", color="#138A87", linewidth=2)
    axes.set(xlabel="Study hours per day", ylabel="Score out of 100", ylim=(0, 105))
    axes.legend(frameon=False, loc="upper left", fontsize=9)
    figure.savefig(output / "regression.png", dpi=160)
    figure.clear()

    figure, axes = make_figure("Held-out prediction residuals", "Residual = actual score minus predicted score; ideal values are close to zero")
    axes.scatter(predicted, y_test.to_numpy() - predicted, color="#2457A7", s=55)
    axes.axhline(0, color="#138A87", linewidth=1.8)
    axes.set(xlabel="Predicted score", ylabel="Residual (score points)")
    figure.savefig(output / "residuals.png", dpi=160)
    figure.clear()
