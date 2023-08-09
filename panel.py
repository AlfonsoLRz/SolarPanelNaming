import numpy as np


class Panel:
    def __init__(self):
        self.id_array = None
        self.id_panel = None
        self.row = None
        self.col = None
        self.geometry = None
        self.centroid = None

    def compute_centroid(self):
        centroid = np.asarray([0, 0], dtype=np.float64)
        for coord in self.geometry:
            centroid[0] += coord[0]
            centroid[1] += coord[1]
        centroid[0] /= len(self.geometry)
        centroid[1] /= len(self.geometry)
        self.centroid = centroid

    def fill(self, elem):
        if 'ID_ARRAY' in elem['properties']:
            self.id_array = elem['properties']['ID_ARRAY']
        if 'ID_CELL' in elem['properties']:
            self.id_panel = elem['properties']['ID_CELL']
        if 'ID_ROW' in elem['properties']:
            self.row = elem['properties']['ID_ROW']
        if 'ID_COL' in elem['properties']:
            self.col = elem['properties']['ID_COL']

        self.geometry = elem['geometry']['coordinates'][0]
        self.compute_centroid()

    def get_area(self):
        area = 0
        for i in range(len(self.geometry) - 1):
            area += self.geometry[i][0] * self.geometry[i + 1][1] - self.geometry[i + 1][0] * self.geometry[i][1]
        area += self.geometry[-1][0] * self.geometry[0][1] - self.geometry[0][0] * self.geometry[-1][1]
        area = np.abs(area) / 2

        return area

    def get_perimeter(self):
        perimeter = 0
        for i in range(len(self.geometry) - 1):
            perimeter += ((self.geometry[i][0] - self.geometry[i + 1][0]) ** 2 + (
                        self.geometry[i][1] - self.geometry[i + 1][1]) ** 2) ** 0.5
        perimeter += ((self.geometry[-1][0] - self.geometry[0][0]) ** 2 + (
                    self.geometry[-1][1] - self.geometry[0][1]) ** 2) ** 0.5

        return perimeter
