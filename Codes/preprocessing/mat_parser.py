# mat_parser.py
import numpy as np
import scipy.io as sio

def load_points_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    candidate = None

    for k in ['image_info', 'annotation', 'head']:
        if k in mat:
            candidate = mat[k]
            break

    # Nested extractor
    def extract(obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 2 and obj.shape[1] == 2:
                return obj.astype(np.float32)

            for e in obj.ravel():
                r = extract(e)
                if r is not None:
                    return r
        return None

    if 'image_info' in mat:
        pts = extract(mat['image_info'])
        if pts is not None:
            return pts

    # Other common keys
    for key in ['annPoints', 'points', 'locations']:
        if key in mat:
            arr = mat[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                return arr.astype(np.float32)

    # Fallback
    def search_all(obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 2 and obj.shape[1] == 2:
                return obj
            for x in obj.ravel():
                r = search_all(x)
                if r is not None:
                    return r
        return None

    pts = search_all(mat)
    if pts is not None:
        return pts.astype(np.float32)

    return np.zeros((0, 2), np.float32)