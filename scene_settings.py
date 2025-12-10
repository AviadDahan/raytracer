import numpy as np


class SceneSettings:
    def __init__(self, background_color, root_number_shadow_rays, max_recursions):
        self.background_color = np.array(background_color, dtype=np.float64)
        self.root_number_shadow_rays = int(root_number_shadow_rays)
        self.max_recursions = int(max_recursions)
