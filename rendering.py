import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_panels(panels, color_code, draw_text=False, text=None, draw_legend=True, savefig=None, remove_axis=False):
    fig, ax = plt.subplots()

    new_panels, new_colors = [], []
    for i in range(len(panels)):
        if panels[i].id_array != -1:
            new_panels.append(panels[i])
            new_colors.append(color_code[i])
    panels = new_panels
    color_code = new_colors

    # Compute limits of panels' centroid
    max_x = -np.inf
    min_x = np.inf
    max_y = -np.inf
    min_y = np.inf
    for panel in panels:
        max_x = max(max_x, panel.centroid[0]) + 1
        min_x = min(min_x, panel.centroid[0]) - 1
        max_y = max(max_y, panel.centroid[1]) + 1
        min_y = min(min_y, panel.centroid[1]) - 1

    width = 10 * (max_x - min_x) / (max_y - min_y) + 1
    if draw_legend:
        width += 1

    fig.set_size_inches(width, 10)

    # Generate color palette from indices
    palette = np.array(sns.color_palette("hls", len(np.unique(color_code))))

    for (idx, panel) in enumerate(panels):
        x = []
        y = []
        for point in panels[idx].geometry:
            x.append(point[0])
            y.append(point[1])
        if color_code is None:
            ax.plot(x, y, color='black')
        else:
            ax.plot(x, y, color=palette[color_code[idx]], linewidth=1.5)

    if color_code is not None:
        # Show legend
        for i in range(len(np.unique(color_code))):
            ax.plot([], [], color=palette[i], label=i)
        if draw_legend:
            ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if draw_text:
        for label in np.unique(color_code):
            random_idx = np.random.choice(np.where(color_code == label)[0])
            centroid = panels[random_idx].centroid
            text_label = label if text is None else text[random_idx]
            ax.text(centroid[0], centroid[1] + 4, text_label, fontsize=14, color='white', weight='bold', ha='center',
                    va='center', bbox=dict(facecolor='black', edgecolor='none', alpha=0.8))

    if remove_axis:
        plt.axis('off')
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=500, transparent=True)
    plt.show()


def plot_centroids(centroids, color_code=None, draw_text=False, plot_legend=True, text_size=12, savefig=None,
                   remove_axis=False):
    fig, ax = plt.subplots()
    max_x = centroids['x'].max() + 2
    min_x = centroids['x'].min() - 2
    max_y = centroids['y'].max() + 2
    min_y = centroids['y'].min() - 2
    fig.set_size_inches(8 * (max_x - min_x) / (max_y - min_y), 8)

    # Generate color palette from indices
    if plot_legend:
        palette = np.array(sns.color_palette("hls", len(np.unique(color_code))))
    else:
        palette = np.array(sns.color_palette("rocket", len(np.unique(color_code))))
    plt.scatter(centroids['x'], centroids['y'], c=color_code)

    if color_code is not None:
        # Show legend
        for i in range(len(np.unique(color_code))):
            ax.plot([], [], color=palette[i], label=i)
        # Place legend outside of plot
        if plot_legend:
            ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if draw_text:
        for label in np.unique(color_code):
            random_idx = np.random.choice(np.where(color_code == label)[0])
            # From pandas dataframe
            centroid = centroids.iloc[random_idx]
            ax.text(centroid[0], centroid[1], label, fontsize=text_size, color='white', weight='bold', ha='center',
                    va='center', bbox=dict(facecolor='black', edgecolor='none', alpha=0.5))

    if remove_axis:
        plt.axis('off')
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi=500)
    plt.show()


def set_up():
    # Change matplotlib style
    plt.style.use('default')

    font_mapping = {'family': 'Palatino Linotype', 'weight': 'normal', 'size': 11}
    plt.rc('font', **font_mapping)