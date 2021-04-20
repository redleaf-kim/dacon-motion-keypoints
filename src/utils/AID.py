import numpy as np

class AID:
    """AID: Augmentation by Informantion Dropping.
    Paper ref: Huang et al. AID: Pushing the Performance Boundary of Human Pose
    Estimation with Information Dropping Augmentation ( arXiv:2008.07139 2020).
    Args:
        transforms (list): ToTensor & Normalize
        prob_cutout (float): Probability of performing cutout.
        radius_factor (float): Size factor of cutout area.
        num_patch (float): Number of patches to be cutout.
        prob_has (float): Probability of performing hide-and-seek.
        prob_has_hide (float): Probability of hiding patches.
    """

    def __init__(self,
                 prob_cutout=0.3,
                 radius_factor=0.1,
                 num_patch=1,
                 prob_has=0.3,
                 prob_has_hide=0.3):

        self.prob_cutout = prob_cutout
        self.radius_factor = radius_factor
        self.num_patch = num_patch
        self.prob_has = prob_has
        self.prob_has_hide = prob_has_hide
        assert (self.prob_cutout + self.prob_has) > 0

    def _hide_and_seek(self, img):
        # get width and height of the image
        ht, wd, _ = img.shape
        # possible grid size, 0 means no hiding
        grid_sizes = [0, 16, 32, 44, 56]

        # randomly choose one grid size
        grid_size = grid_sizes[np.random.randint(0, len(grid_sizes) - 1)]

        # hide the patches
        if grid_size != 0:
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if np.random.rand() <= self.prob_has_hide:
                        img[y:y_end, x:x_end, :] = 0
        return img

    def _cutout(self, img):
        height, width, _ = img.shape
        img = img.reshape((height * width, -1))
        feat_x_int = np.arange(0, width)
        feat_y_int = np.arange(0, height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        feat_x_int = feat_x_int.reshape((-1, ))
        feat_y_int = feat_y_int.reshape((-1, ))
        for _ in range(self.num_patch):
            center = [np.random.rand() * width, np.random.rand() * height]
            radius = self.radius_factor * (1 + np.random.rand(2)) * width
            x_offset = (center[0] - feat_x_int) / radius[0]
            y_offset = (center[1] - feat_y_int) / radius[1]
            dis = x_offset**2 + y_offset**2
            keep_pos = np.where((dis <= 1) & (dis >= 0))[0]
            img[keep_pos, :] = 0
        img = img.reshape((height, width, -1))
        return img

    def __call__(self, img):
        if np.random.rand() < self.prob_cutout:
            img = self._cutout(img)
        if np.random.rand() < self.prob_has:
            img = self._hide_and_seek(img)
        return img