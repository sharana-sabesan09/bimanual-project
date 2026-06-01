import numpy as np
import genesis.utils.geom as gu

# Transform point by quaternion
point = np.array([[1.0, 0.0, 0.0]])
quat = np.array([0.707, 0.0, 0.0, 0.707])
rotated = gu.transform_by_quat(point, quat)

# Transform point by translation and quaternion
pos = np.array([1.0, 2.0, 3.0])
transformed = gu.transform_by_trans_quat(point, pos, quat)

# Inverse transform
original = gu.inv_transform_by_trans_quat(transformed, pos, quat)