from typing import List, Optional
from itertools import zip_longest

from matplotlib import pyplot as plt


def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def draw_regression(x: List[float], y: List[float], low_conf: Optional[List[float]] = None,
                    high_conf: Optional[List[float]] = None, x_label: Optional[str] = None,
                    y_label: Optional[str] = None, **plot_kwargs) -> plt.Axes:
    ax = plt.gca()
    series_to_plot = [series for series in (low_conf, high_conf) if series is not None]
    print(f'Series to plot: {series_to_plot}')
    ax.scatter(x, y, **plot_kwargs)
    for series in series_to_plot:
        ax.plot(x, series, **plot_kwargs)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    return ax
