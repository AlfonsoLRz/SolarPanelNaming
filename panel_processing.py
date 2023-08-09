import fiona
import numpy as np
import pandas as pd

import rendering
from panel import Panel
from rendering import *
from sklearn.cluster import DBSCAN


def determine_row_col(centroids, max_vertical_distance=1.5):
    if len(centroids) == 0:
        return []

    row_cluster = np.zeros(len(centroids))
    for (idx, elem) in enumerate(centroids):
        row_cluster[idx] = -1

    row = 1
    row_cluster[0] = 0

    # Determine different rows (not ordered yet)
    for (idx1, centroid) in enumerate(centroids):
        if idx1 > 0:
            min_vertical_distance = 100000
            for (idx2, centroid2) in enumerate(centroids):
                if idx1 != idx2 and row_cluster[idx2] != -1:
                    vertical_distance = abs(centroid[1] - centroid2[1])
                    if vertical_distance < min_vertical_distance and vertical_distance < max_vertical_distance:
                        min_vertical_distance = vertical_distance
                        row_cluster[idx1] = row_cluster[idx2]

            if min_vertical_distance == 100000:
                row_cluster[idx1] = row
                row += 1

    # Order rows in the x-axis
    new_labels = None
    for row_idx in range(row):
        cluster_indices = np.where(row_cluster == row_idx)[0]
        # Order cluster indices by x coordinate
        cluster_indices = [i for i in sorted(cluster_indices, key=lambda item: centroids[item][0])]

        if new_labels is None:
            new_labels = [cluster_indices]
        else:
            new_labels.append(cluster_indices)

    # Calculate aggregated centroid per row
    aggregated_centroid = np.zeros((len(new_labels), 2))
    for (idx1, cluster_indices) in enumerate(new_labels):
        for idx2 in cluster_indices:
            aggregated_centroid[idx1] += centroids[idx2]
        aggregated_centroid[idx1] /= len(cluster_indices)

    # Order in the y-axis
    new_label_order = [i[1] for i in
                       sorted(enumerate(new_labels), key=lambda item: aggregated_centroid[item[0]][1])]

    return new_label_order


def fill_panels(panels, cluster, threshold_distance=1.25):
    for key in cluster:
        row_panels = dict()

        for idx in cluster[key]:
            if panels[idx].row not in row_panels:
                row_panels[panels[idx].row] = []
            row_panels[panels[idx].row].append(idx)
            # Sort according to column
            row_panels[panels[idx].row].sort(key=lambda x: panels[x].col)

        # Grid size
        max_cols, max_rows = 0, len(row_panels)
        for row in row_panels:
            if len(row_panels[row]) > max_cols:
                max_cols = len(row_panels[row])

        # Average direction
        average_direction = np.zeros(2)
        num_panels = 0

        for row in row_panels:
            for idx in row_panels[row]:
                vec_x_1 = np.asarray(panels[idx].geometry[1]) - np.asarray(panels[idx].geometry[0])
                vec_x_2 = np.asarray(panels[idx].geometry[2]) - np.asarray(panels[idx].geometry[3])
                # Normalize
                norm_vec_x_1 = vec_x_1 / np.linalg.norm(vec_x_1)
                norm_vec_x_2 = vec_x_2 / np.linalg.norm(vec_x_2)

                average_direction = average_direction + norm_vec_x_1 + norm_vec_x_2
                num_panels += 2

        average_direction = average_direction / num_panels
        norm_x = average_direction / np.linalg.norm(average_direction)
        norm_y = np.asarray([-norm_x[1], norm_x[0]])

        # Size of array
        array_width, array_height, array_min, array_max = get_array_size(key, cluster[key], panels, norm_x)

        # Calculate average width and height of panels
        average_width = 0
        average_height = 0
        num_panels = 0

        for row in row_panels:
            for idx in row_panels[row]:
                average_width += np.linalg.norm(
                    np.asarray(panels[idx].geometry[0]) - np.asarray(panels[idx].geometry[1]))
                average_width += np.linalg.norm(
                    np.asarray(panels[idx].geometry[2]) - np.asarray(panels[idx].geometry[3]))
                average_height += np.linalg.norm(
                    np.asarray(panels[idx].geometry[1]) - np.asarray(panels[idx].geometry[2]))
                average_height += np.linalg.norm(
                    np.asarray(panels[idx].geometry[3]) - np.asarray(panels[idx].geometry[0]))
                num_panels += 2
        average_width /= num_panels
        average_height /= num_panels

        # average_distance_x = (array_width - average_width * max_cols) / (max_cols - 1)
        # average_distance_y = (array_height - average_height * max_rows) / (max_rows - 1)

        # Calculate average distance x and y in another way
        average_distance_x = 0
        num_panels_x = 0

        for row in row_panels:
            for idx in range(len(row_panels[row]) - 1):
                avg_distance = np.linalg.norm(np.asarray(panels[row_panels[row][idx]].centroid) - np.asarray(
                    panels[row_panels[row][idx + 1]].centroid))
                if avg_distance < (average_width * 1.5):
                    average_distance_x += avg_distance
                    num_panels_x += 1

        average_distance_x = average_distance_x / num_panels_x - average_width
        expected_rows = array_height / (average_height + 1 / 10 * average_height)
        average_distance_y = (array_height - average_height * expected_rows) / (expected_rows - 1)
        max_cols = np.round((array_width - average_distance_x) / (average_width + average_distance_x))

        # Minimum point of array
        min_x, min_y = 2e32, 2e32
        for row in row_panels:
            for idx in row_panels[row]:
                if panels[idx].centroid[0] < min_x:
                    min_x = panels[idx].centroid[0]
                if panels[idx].centroid[1] < min_y:
                    min_y = panels[idx].centroid[1]

        # Sort row panels according to key
        row_panels = dict(sorted(row_panels.items(), key=lambda item: item[0]))

        for row in row_panels:
            if len(row_panels[row]) < max_cols:
                # starting_point = np.asarray([min_x + average_width / 2.0, min_y + average_height / 2.0 * row_idx + (average_distance_y * (row_idx - 1) if row_idx > 0 else 0)])
                num_panels = len(row_panels[row])
                new_panels = []

                # Missing panels - start from the left
                starting_point = panels[row_panels[row][0]].centroid - norm_x * (
                            average_width + average_distance_x / 2.0) - norm_y * average_distance_y / 2.0
                while is_point_in_array(starting_point, norm_x, array_min, array_max):
                    # Include new panel
                    new_panel = Panel()
                    new_panel.id_array = key
                    new_panel.row = row
                    new_panel.col = -1
                    new_panel.id_panel = -1
                    new_panel.geometry = get_panel_geometry(starting_point, norm_x, average_width, norm_y,
                                                            average_height)
                    new_panel.centroid = starting_point
                    panels.append(new_panel)
                    new_panels.append((len(panels) - 1))
                    num_panels += 1

                    row_panels[row].insert(0, len(panels) - 1)
                    starting_point = starting_point - norm_x * (average_width + average_distance_x)

                # Find missing panels in the middle
                i = 0

                while i < (num_panels - 1) and num_panels < max_cols:
                    # Calculate distance between adjacent panels
                    centroid1 = panels[row_panels[row][i]].centroid
                    centroid2 = panels[row_panels[row][i + 1]].centroid
                    distance = np.linalg.norm(centroid1 - centroid2)

                    if distance > (
                            (average_width + average_distance_x) * threshold_distance) and distance > average_width:
                        new_centroid = centroid1 + norm_x * (
                                    average_width + average_distance_x * 1.5) - norm_y * average_distance_y / 2.0
                        new_panel = Panel()
                        new_panel.id_array = key
                        new_panel.row = row
                        new_panel.col = -1
                        new_panel.id_panel = -1
                        new_panel.geometry = get_panel_geometry(new_centroid, norm_x, average_width, norm_y,
                                                                average_height)
                        new_panel.centroid = new_centroid
                        panels.append(new_panel)
                        num_panels += 1

                        # Insert new panel in the list
                        row_panels[row].insert(i + 1, len(panels) - 1)

                    i += 1

                # Missing panels - start from the right
                starting_point = panels[row_panels[row][len(
                    row_panels[row]) - 1]].centroid + norm_x * average_distance_x - norm_y * average_distance_y / 2.0
                while num_panels < max_cols:
                    starting_point = starting_point + norm_x * (average_width + average_distance_x)
                    # Include new panel
                    new_panel = Panel()
                    new_panel.id_array = key
                    new_panel.row = row
                    new_panel.col = -1
                    new_panel.id_panel = -1
                    new_panel.geometry = get_panel_geometry(starting_point, norm_x, average_width, norm_y,
                                                            average_height)
                    new_panel.centroid = starting_point
                    panels.append(new_panel)
                    new_panels.append(len(panels) - 1)
                    num_panels += 1

                # Append new panels to row
                row_panels[row].extend(new_panels)
                # Sort by centroid x
                row_panels[row] = sorted(row_panels[row], key=lambda item: panels[item].centroid[0])

        included_new_row, iteration = True, 0
        while included_new_row or iteration == 0:
            included_new_row = False
            iteration += 1

            for i in range(len(row_panels) - 1):
                row = list(row_panels.keys())[i]
                max_row = list(row_panels.keys())[-1]
                next_row = row + 1
                while next_row not in row_panels and next_row <= max_row:
                    next_row += 1

                if next_row > max_row:
                    break

                # Calculate distance between adjacent rows
                centroid1 = panels[row_panels[row][0]].centroid
                centroid2 = panels[row_panels[next_row][0]].centroid
                distance = np.linalg.norm(centroid1 - centroid2)

                if distance > ((average_height + average_distance_y) * 1.5):
                    # Advance keys higher than next_row
                    keys = list(row_panels.keys())
                    for key_i in reversed(keys):
                        if key_i >= next_row:
                            row_panels[key_i + 1] = row_panels[key_i]
                            del row_panels[key_i]

                    row_panels[row + 1] = []

                    for col_idx in range(np.amin([len(row_panels[row]), len(row_panels[next_row + 1])])):
                        point = (panels[row_panels[row][col_idx]].centroid + panels[
                            row_panels[next_row + 1][col_idx]].centroid) / 2.0 + norm_x * average_distance_x
                        new_panel = Panel()
                        new_panel.id_array = key
                        new_panel.row = row
                        new_panel.col = -1
                        new_panel.id_panel = -1
                        new_panel.geometry = get_panel_geometry(point, norm_x, average_width, norm_y, average_height)
                        new_panel.centroid = point
                        panels.append(new_panel)
                        row_panels[row + 1].append(len(panels) - 1)

                    included_new_row = True

            # Sort row panels according to key
            row_panels = dict(sorted(row_panels.items(), key=lambda item: item[0]))

    return panels


def get_array_size(id_array, panel_indices, panel_buffer, norm_x):
    # Calculate bounding box of array
    min_x, max_x, min_y, max_y = 2e32, -2e32, 2e32, -2e32

    for idx in panel_indices:
        for coord in panel_buffer[idx].geometry:
            # Rotate geometry according to norm_x
            rotated_geometry = [coord[0] * norm_x[0] + coord[1] * norm_x[1], -coord[0] * norm_x[1] + coord[1] * norm_x[0]]
            if rotated_geometry[0] < min_x:
                min_x = rotated_geometry[0]
            if rotated_geometry[0] > max_x:
                max_x = rotated_geometry[0]
            if rotated_geometry[1] < min_y:
                min_y = rotated_geometry[1]
            if rotated_geometry[1] > max_y:
                max_y = rotated_geometry[1]

    return max_x - min_x, max_y - min_y, [min_x, min_y], [max_x, max_y]


def get_panel_geometry(point, x, width, y, height):
    max_x_max_y = point + x * width / 2 + y * height / 2
    min_x_min_y = point - x * width / 2 - y * height / 2
    max_x_min_y = point + x * width / 2 - y * height / 2
    min_x_max_y = point - x * width / 2 + y * height / 2

    return [max_x_max_y, min_x_max_y, min_x_min_y, max_x_min_y, max_x_max_y]


def load_panels(filename, rendering=False):
    filename = filename
    shp = fiona.open(filename)

    panels = []
    cluster = dict()

    for (idx, elem) in enumerate(shp):
        panel = Panel()
        panel.fill(elem)
        panels.append(panel)

        if panel.id_array not in cluster:
            cluster[panel.id_array] = []
        cluster[panel.id_array].append(idx)

    if rendering:
        plot_panels(panels, color_code=[panel.id_array for panel in panels], draw_text=True,
                    text=[str(panel.id_array) for panel in panels], draw_legend=True, savefig='results/panels.png')

    return shp, panels, cluster


def is_point_in_array(point, norm_x, array_min, array_max):
    rotated_point = [point[0] * norm_x[0] + point[1] * norm_x[1], -point[0] * norm_x[1] + point[1] * norm_x[0]]

    if array_min[0] < rotated_point[0] < array_max[0] and array_min[1] < rotated_point[1] < array_max[1]:
        return True
    else:
        return False


def name_panels(shp, clustering_distance, norm_x=None):
    # Calculate centroid
    centroid_table = []

    for (idx, elem) in enumerate(shp):
        # Calculate centroid
        centroid = [0, 0]
        for coord in elem['geometry']['coordinates'][0]:
            centroid[0] += coord[0]
            centroid[1] += coord[1]
        centroid[0] /= len(elem['geometry']['coordinates'][0])
        centroid[1] /= len(elem['geometry']['coordinates'][0])

        centroid_table.append([centroid[0], centroid[1]])

    centroid_table_pd = pd.DataFrame(centroid_table, columns=['x', 'y'])

    # Clustering
    db = DBSCAN(eps=clustering_distance, min_samples=5).fit(centroid_table_pd)
    labels = db.labels_

    # Calculate average edge vector from each cell
    if norm_x is None:
        average_direction = np.zeros(2)
        for (idx, elem) in enumerate(shp):
            average_direction = average_direction + np.asarray(
                [elem['geometry']['coordinates'][0][1][0] - elem['geometry']['coordinates'][0][0][0],
                 elem['geometry']['coordinates'][0][1][1] - elem['geometry']['coordinates'][0][0][1]])
            # average_direction = average_direction + np.asarray([elem['geometry']['coordinates'][0][3][0] - elem['geometry']['coordinates'][0][0][0],
            #             elem['geometry']['coordinates'][0][3][1] - elem['geometry']['coordinates'][0][0][1]])

        average_direction = average_direction / len(shp)

        # Normalize both vectors
        module = (average_direction[0] ** 2 + average_direction[1] ** 2) ** 0.5
        norm_x = average_direction / module

    # Rotate points to align with x-axis
    rotated_centroid_table = []

    for (idx, elem) in enumerate(shp):
        rotated_centroid_table.append([centroid_table[idx][0] * norm_x[0] + centroid_table[idx][1] * norm_x[1],
                                       -centroid_table[idx][0] * norm_x[1] + centroid_table[idx][1] * norm_x[0]])

    rotated_centroid_table_pd = pd.DataFrame(rotated_centroid_table, columns=['x', 'y'])

    # %%
    cluster_centroid = np.zeros((len(np.unique(labels)), 2))

    for cluster in np.unique(labels):
        centroids = rotated_centroid_table_pd[labels == cluster]
        # Calculate average centroid
        for centroid in centroids.values:
            cluster_centroid[cluster] += centroid
        cluster_centroid[cluster] /= len(centroids)

    cluster_centroid = cluster_centroid.tolist()
    new_label_order = determine_row_col(cluster_centroid)

    # Map new labels to old labels
    idx = 0
    new_labels_panel = labels.copy()
    for cluster_indices in new_label_order:
        for idx2 in cluster_indices:
            new_labels_panel[labels == idx2] = idx
            idx += 1

    # Sum number of cells per panel
    panel_count = np.zeros(len(np.unique(new_labels_panel)))
    for (idx, elem) in enumerate(shp):
        panel_count[new_labels_panel[idx]] += 1

    # Create vector similar to new_labels_panel
    panel_count_vector = np.zeros(len(new_labels_panel), dtype=int)
    for (idx, elem) in enumerate(new_labels_panel):
        panel_count_vector[idx] = panel_count[elem]

    # Gather centroid by panel label
    panel_centroid = [[] for i in range(len(np.unique(new_labels_panel)))]
    panel_indices = [[] for i in range(len(np.unique(new_labels_panel)))]

    # For each panel
    for (idx, elem) in enumerate(shp):
        panel_centroid[new_labels_panel[idx]].append(rotated_centroid_table[idx])
        panel_indices[new_labels_panel[idx]].append(idx)

    # Order by x-axis and y-axis
    new_labels_cell = labels.copy()
    row_indices = labels.copy()
    col_indices = labels.copy()
    idx = 0

    for (panel_idx, panel_centroids) in enumerate(panel_centroid):
        new_label_order = determine_row_col(panel_centroids, max_vertical_distance=0.25)

        # Map new labels to old labels
        for (row_idx, panel_indices_row) in enumerate(new_label_order):
            for (col_idx, idx2) in enumerate(panel_indices_row):
                new_labels_cell[panel_indices[panel_idx][idx2]] = idx
                row_indices[panel_indices[panel_idx][idx2]] = row_idx
                col_indices[panel_indices[panel_idx][idx2]] = col_idx
                idx += 1

    panels = []

    for (idx, elem) in enumerate(shp):
        panel = Panel()
        panel.id_array = new_labels_panel[idx]
        panel.id_panel = new_labels_cell[idx]
        panel.row = row_indices[idx]
        panel.col = col_indices[idx]
        panel.geometry = elem['geometry']['coordinates'][0]
        panel.compute_centroid()
        panels.append(panel)

    return panels, norm_x


def name_panels_new(panels, clusters):
    # Clustering
    centroids = [panel.centroid for panel in panels]
    centroid_table_pd = pd.DataFrame(centroids, columns=['x', 'y'])
    db = DBSCAN(eps=3, min_samples=5).fit(centroid_table_pd)
    labels = db.labels_
    for (idx, panel) in enumerate(panels):
        panel.id_array = int(labels[idx])

    # Calculate average edge vector from each cell
    average_direction = np.zeros(2)
    for panel in panels:
        average_direction = average_direction + np.asarray([panel.geometry[1][0] - panel.geometry[0][0],
                                                            panel.geometry[1][1] - panel.geometry[0][1]])
        # average_direction = average_direction + np.asarray([elem['geometry']['coordinates'][0][3][0] - elem['geometry']['coordinates'][0][0][0],
        #             elem['geometry']['coordinates'][0][3][1] - elem['geometry']['coordinates'][0][0][1]])
    average_direction = average_direction / len(panels)

    # Normalize both vectors
    module = (average_direction[0] ** 2 + average_direction[1] ** 2) ** 0.5
    norm_x = average_direction / module

    # Rotate points to align with x-axis
    rotated_centroid_table = []
    for panel in panels:
        rotated_centroid_table.append(np.asarray([panel.centroid[0] * norm_x[0] + panel.centroid[1] * norm_x[1],
                                                  -panel.centroid[0] * norm_x[1] + panel.centroid[1] * norm_x[0]]))

    # %%
    cluster_centroid = np.zeros((len(np.unique(labels)), 2))
    for cluster in np.unique(labels):
        centroids = [panel.centroid for panel in panels if panel.id_array == cluster]
        # Calculate average centroid
        for centroid in centroids:
            cluster_centroid[cluster] += centroid
        cluster_centroid[cluster] /= len(centroids)

    cluster_centroid = cluster_centroid.tolist()
    new_label_order = determine_row_col(cluster_centroid)

    # Map new labels to old labels
    idx = 0
    new_labels_panel = labels.copy()
    for cluster_indices in new_label_order:
        for idx2 in cluster_indices:
            new_labels_panel[labels == idx2] = idx
            idx += 1

    # Gather centroid by panel label
    panel_centroid = [[] for i in range(len(np.unique(new_labels_panel)))]
    panel_indices = [[] for i in range(len(np.unique(new_labels_panel)))]

    # For each panel
    for (idx, panel) in enumerate(panels):
        panel_centroid[new_labels_panel[idx]].append(rotated_centroid_table[idx])
        panel_indices[new_labels_panel[idx]].append(idx)

    # Order by x-axis and y-axis
    idx = 0

    for (panel_idx, panel_centroids) in enumerate(panel_centroid):
        new_label_order = determine_row_col(panel_centroids, max_vertical_distance=0.25)

        # Map new labels to old labels
        for (row_idx, panel_indices_row) in enumerate(new_label_order):
            for (col_idx, idx2) in enumerate(panel_indices_row):
                panels[panel_indices[panel_idx][idx2]].id_panel = idx
                panels[panel_indices[panel_idx][idx2]].row = row_idx
                panels[panel_indices[panel_idx][idx2]].col = col_idx
                idx += 1

    for (idx, panel) in enumerate(panels):
        panel.id_array = new_labels_panel[idx]

    # # Write back
    # write_panels(filename.split('.')[0] + '_new.shp', shp, panels)


def remove_repeated_panels(panels):
    # Compare every centroid to check if they are the same
    for (idx, panel1) in enumerate(panels):
        for (idx2, panel2) in enumerate(panels):
            if idx != idx2:
                if np.linalg.norm(panel1.centroid - panel2.centroid) < 0.0001:
                    print('Centroid {} and {} are the same'.format(idx, idx2))

                    # Remove the first one
                    panels.pop(idx)
                    idx -= 1
                    break

    return panels

def stress_missing_panels(panels, cluster, nrows=1, missing_panels_p=0.05):
    # Miss entire rows
    for key in cluster:
        row_panels = dict()
        rows = {}
        for idx in cluster[key]:
            if panels[idx].row not in row_panels:
                row_panels[panels[idx].row] = []
            row_panels[panels[idx].row].append(idx)
            rows[panels[idx].row] = 1

        rows = list(rows.keys())
        rows.sort()

        # Pick one random row
        for _ in range(nrows):
            if len(rows) > 2:
                random_row = np.random.choice(rows[1:-2])

                for idx in row_panels[random_row]:
                    panels[idx] = None
                    cluster[key].remove(idx)
                rows.remove(random_row)
                row_panels.pop(random_row)

    # Miss random panels
    for key in cluster:
        num_panels = len(cluster[key])
        num_missing_panels = int(num_panels * missing_panels_p)
        missing_panels = np.random.choice(cluster[key], num_missing_panels, replace=False)

        for idx in missing_panels:
            cluster[key].remove(idx)
            panels[idx] = None

    return panels


def write_panels(filename, shp, panels):
    schema = shp.schema.copy()
    input_crs = shp.crs

    # New attributes
    schema['properties']['ID_ARRAY'] = 'int'
    schema['properties']['ID_CELL'] = 'int'
    schema['properties']['ID_ROW'] = 'int'
    schema['properties']['ID_COL'] = 'int'

    with fiona.open(filename, 'w', 'ESRI Shapefile', schema, input_crs) as output:
        for panel in panels:
            if panel is None or panel.id_array == -1:
                continue

            # Build geometry buffer
            geometry_buffer = [[]]
            for point in panel.geometry:
                geometry_buffer[0].append((point[0], point[1]))

            output.write({'properties': {'No.': -1, 'Area (Ha)': -1, 'Perimetro(': panel.get_perimeter(),
                                         'Area(m2)': panel.get_area(), 'Perimetr_1': panel.get_perimeter(),
                                         'ID_ARRAY': int(panel.id_array if panel.id_array is not None else -1),
                                         'ID_CELL': int(panel.id_panel if panel.id_panel is not None else -1),
                                         'ID_ROW': int(panel.row if panel.row is not None else -1),
                                         'ID_COL': int(panel.col if panel.col is not None else -1)},
                          'geometry': {'type': 'Polygon', 'coordinates': geometry_buffer}})
