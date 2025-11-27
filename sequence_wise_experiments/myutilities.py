def save_activations(path: str, arr: np.ndarray):
    arr16 = arr.astype(np.float16)     # or np.float32 -> np.float16
    np.savez_compressed(path, arr=arr16)

def load_activations(path: str) -> np.ndarray:
    data = np.load(path)
    arr16 = data["arr"]
    return arr16.astype(np.float32)    # back to float32 if you want
