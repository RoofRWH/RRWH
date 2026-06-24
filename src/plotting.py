from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _to_1d_array(values):
    if values is None:
        return np.asarray([])
    if hasattr(values, 'values'):
        values = values.values
    values = np.asarray(values)
    if values.ndim > 1:
        values = values.ravel()
    return values[np.isfinite(values)]


def plot_rainfall_stats(annual_rain, skewness, heaviness, output_dir):
    """Generate simple rainfall statistic plots and return file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annual = _to_1d_array(annual_rain)
    skew = _to_1d_array(skewness)
    heavy = _to_1d_array(heaviness)

    outputs = {}

    if annual.size:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(annual, bins=30, kde=True, ax=ax, color='#2a6f97')
        ax.set_title('Annual rainfall distribution')
        ax.set_xlabel('Annual rainfall')
        ax.set_ylabel('Count')
        path = output_dir / 'annual_rainfall_stats.png'
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs['annual'] = str(path)

    if skew.size:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(skew, bins=30, kde=True, ax=ax, color='#bc4749')
        ax.set_title('Rainfall skewness distribution')
        ax.set_xlabel('Skewness')
        ax.set_ylabel('Count')
        path = output_dir / 'rainfall_skewness_stats.png'
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs['skewness'] = str(path)

    if heavy.size:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(heavy, bins=30, kde=True, ax=ax, color='#6a994e')
        ax.set_title('Rainfall heaviness distribution')
        ax.set_xlabel('95th percentile rainfall')
        ax.set_ylabel('Count')
        path = output_dir / 'rainfall_heaviness_stats.png'
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs['heaviness'] = str(path)

    return outputs


def _plot_grid(grid, output_path, title, cmap='viridis'):
    grid = np.asarray(grid)
    if grid.ndim > 2:
        grid = grid.squeeze()
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(grid, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def plot_best_distribution_map(best_dist_grid, output_path):
    """Plot the map of best fitting distributions."""
    return _plot_grid(best_dist_grid, output_path, 'Best fitting distribution', cmap='tab20')


def plot_rwh_potential(rwh_grid, output_path):
    """Plot RWH potential map."""
    return _plot_grid(rwh_grid, output_path, 'RWH potential', cmap='Blues')


def create_policy_dashboard(stats, output_path):
    """Create a small summary dashboard for policy makers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    metrics = [
        ('Mean', stats.get('mean')),
        ('Median', stats.get('median')),
        ('P90', stats.get('p90')),
        ('Reliability', stats.get('reliability')),
    ]

    for ax, (label, value) in zip(axes, metrics):
        ax.bar([label], [0 if value is None else value], color='#457b9d')
        ax.set_title(label)
        ax.set_ylim(bottom=0)

    for ax in axes[len(metrics):]:
        ax.axis('off')

    fig.suptitle('RRWH Summary Dashboard')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)
