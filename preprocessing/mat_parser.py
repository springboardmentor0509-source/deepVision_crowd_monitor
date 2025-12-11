# mat_parser.py
import numpy as np
import scipy.io as sio

def load_points_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    
    # For ShanghaiTech format: image_info[0,0]['location'][0,0]
    if 'image_info' in mat:
        img_info = mat['image_info']
        if isinstance(img_info, np.ndarray) and img_info.shape == (1, 1):
            obj = img_info[0, 0]
            if isinstance(obj, np.ndarray) and hasattr(obj.dtype, 'names'):
                # It's a structured array (MATLAB struct)
                if 'location' in obj.dtype.names:
                    location = obj['location']
                    if isinstance(location, np.ndarray) and location.shape == (1, 1):
                        pts = location[0, 0]
                        if isinstance(pts, np.ndarray) and pts.ndim == 2:
                            return pts.astype(np.float32)
    
    # Fallback: nested extraction
    def extract(obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 2 and obj.shape[1] == 2:
                return obj.astype(np.float32)
            for e in obj.ravel():
                r = extract(e)
                if r is not None:
                    return r
        return None

    result = extract(mat.get('image_info'))
    if result is not None:
        return result
    
    # Other common keys
    for key in ['annPoints', 'points', 'locations']:
        if key in mat:
            arr = mat[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                return arr.astype(np.float32)

    # Final fallback
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
