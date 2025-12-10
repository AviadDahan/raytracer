import numpy as np


class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = np.array(diffuse_color, dtype=np.float64)
        self.specular_color = np.array(specular_color, dtype=np.float64)
        self.reflection_color = np.array(reflection_color, dtype=np.float64)
        self.shininess = shininess
        self.transparency = transparency
