import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List

try:
    # Comment: Pillow is only used for image loading/resizing; core math stays in NumPy.
    from PIL import Image
except ImportError as exc:
    raise ImportError("Pillow (PIL) is required for image loading. Please install it via `pip install pillow`.") from exc


CLASS_NAMES: Tuple[str, ...] = ("bicycle", "cat")


# Helper to normalize images and convert them to NCHW float32 tensors.
def _prepare_inputs(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"Expected input of shape (N, H, W, C) or (N, C, H, W); got {images.shape}")
    if images.shape[1] in (1, 3):
        # Already NCHW; just ensure float32 normalization.
        nchw = images.astype(np.float32)
    else:
        # Convert NHWC to NCHW.
        nchw = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)
    return nchw / 255.0

def load_image_folder(root: Path, target_size: int = 64, label_map: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if label_map is None:
        label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    data: List[np.ndarray] = []
    labels: List[int] = []
    for class_name, idx in label_map.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg"}:
                continue
            with Image.open(img_path) as img:
                # Ensure 3-channel RGB then resize for faster training.
                rgb = img.convert("RGB").resize((target_size, target_size))
                data.append(np.array(rgb, dtype=np.float32))
                labels.append(idx)
    if not data:
        raise ValueError(f"No images found under {root}")
    X = np.stack(data, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y, label_map

def print_confusion_matrix(y_true, y_pred, class_names):
    num_classes = len(class_names)
    # K x K size zero matrix generate
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Fill the matrix for each prediction
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    print("\n--- Confusion Matrix ---")
    # Print headers
    print(f"{'':>10} | " + " | ".join(f"{name:>10}" for name in class_names))
    print("-" * (13 + 13 * num_classes))
    
    # Print rows
    for i, true_name in enumerate(class_names):
        row_str = " | ".join(f"{cm[i, j]:>10}" for j in range(num_classes))
        print(f"{true_name:>10} | {row_str}")
        
    return cm

class BaseModel:
 
    def __init__(self):
        self.weights = {}

    def train(self, X_train, y_train, epochs=10, learning_rate=0.01):
        """
        """
        
    def predict(self, X_test):
        """
        """
        
    def save(self, path):
        """
        Saves the trained model instance using pickle.
        Args:
            path: Destination file path (e.g., 'model.pkl')
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved successfully to {path}")

    @classmethod
    def load(cls, path):
        """
        Loads a trained model instance from a file.
        Args:
            path: Source file path
        Returns:
            The loaded model instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {path}")
        return model


class MyModel(BaseModel):
    # Initialize hyperparameters and weights for a small CNN.
    def __init__(self, image_size: int = 64, num_classes: int = len(CLASS_NAMES), batch_size: int = 32, seed: int = 42):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self._build_parameters()

    # Sets up convolutional and fully-connected layer weights.
    def _build_parameters(self):
        c1_out = 8
        c2_out = 16
        flatten_dim = c2_out * (self.image_size // 4) * (self.image_size // 4)
        fc_hidden = 64

        self.W1 = self.rng.normal(0, np.sqrt(2 / (3 * 3 * 3)), size=(c1_out, 3, 3, 3)).astype(np.float32)
        self.b1 = np.zeros(c1_out, dtype=np.float32)
        self.W2 = self.rng.normal(0, np.sqrt(2 / (c1_out * 3 * 3)), size=(c2_out, c1_out, 3, 3)).astype(np.float32)
        self.b2 = np.zeros(c2_out, dtype=np.float32)
        self.W3 = self.rng.normal(0, np.sqrt(2 / flatten_dim), size=(flatten_dim, fc_hidden)).astype(np.float32)
        self.b3 = np.zeros(fc_hidden, dtype=np.float32)
        self.W4 = self.rng.normal(0, np.sqrt(2 / fc_hidden), size=(fc_hidden, self.num_classes)).astype(np.float32)
        self.b4 = np.zeros(self.num_classes, dtype=np.float32)
        self._sync_weights()

    # Keep BaseModel.weights aligned for serialization/debugging.
    def _sync_weights(self):
        self.weights = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
            "W4": self.W4,
            "b4": self.b4,
        }

    # Forward pass for convolutional layer with padding/stride.
    def _conv_forward(self, X: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int = 1, pad: int = 1) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        N, C, H, W_in = X.shape
        F, _, KH, KW = W.shape
        H_out = (H + 2 * pad - KH) // stride + 1
        W_out = (W_in + 2 * pad - KW) // stride + 1
        X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
        out = np.zeros((N, F, H_out, W_out), dtype=np.float32)

        for i in range(H_out):
            h_start = i * stride
            h_end = h_start + KH
            for j in range(W_out):
                w_start = j * stride
                w_end = w_start + KW
                window = X_padded[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.tensordot(window, W, axes=([1, 2, 3], [1, 2, 3]))

        out += b[None, :, None, None]
        cache = {"X_padded": X_padded, "W": W, "b": b, "stride": stride, "pad": pad, "out_shape": out.shape}
        return out, cache

    # Backward pass for convolutional layer.
    def _conv_backward(self, dout: np.ndarray, cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_padded, W, stride, pad = cache["X_padded"], cache["W"], cache["stride"], cache["pad"]
        N, C, H_padded, W_padded = X_padded.shape
        F, _, KH, KW = W.shape
        _, _, H_out, W_out = dout.shape

        dX_padded = np.zeros_like(X_padded, dtype=np.float32)
        dW = np.zeros_like(W, dtype=np.float32)
        db = np.sum(dout, axis=(0, 2, 3))

        for i in range(H_out):
            h_start = i * stride
            h_end = h_start + KH
            for j in range(W_out):
                w_start = j * stride
                w_end = w_start + KW
                window = X_padded[:, :, h_start:h_end, w_start:w_end]
                dout_slice = dout[:, :, i, j]
                dW += np.tensordot(dout_slice, window, axes=([0], [0]))
                dX_padded[:, :, h_start:h_end, w_start:w_end] += np.tensordot(dout_slice, W, axes=([1], [0]))

        if pad > 0:
            dX = dX_padded[:, :, pad:-pad, pad:-pad]
        else:
            dX = dX_padded
        return dX, dW, db

    # Max-pooling forward pass.
    def _max_pool_forward(self, X: np.ndarray, pool: int = 2, stride: int = 2) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        N, C, H, W = X.shape
        H_out = (H - pool) // stride + 1
        W_out = (W - pool) // stride + 1
        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        for i in range(H_out):
            h_start = i * stride
            h_end = h_start + pool
            for j in range(W_out):
                w_start = j * stride
                w_end = w_start + pool
                window = X[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(window, axis=(2, 3))

        cache = {"X": X, "pool": pool, "stride": stride, "out": out}
        return out, cache

    # Max-pooling backward pass.
    def _max_pool_backward(self, dout: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
        X, pool, stride = cache["X"], cache["pool"], cache["stride"]
        N, C, H, W = X.shape
        _, _, H_out, W_out = dout.shape
        dX = np.zeros_like(X, dtype=np.float32)

        for i in range(H_out):
            h_start = i * stride
            h_end = h_start + pool
            for j in range(W_out):
                w_start = j * stride
                w_end = w_start + pool
                window = X[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(window, axis=(2, 3), keepdims=True)
                mask = (window == max_vals)
                dX[:, :, h_start:h_end, w_start:w_end] += mask * dout[:, :, i, j][:, :, None, None]
        return dX

    # Numerically stable softmax.
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    # Cross-entropy loss for integer labels.
    def _cross_entropy(self, probs: np.ndarray, y: np.ndarray) -> float:
        N = y.shape[0]
        clipped = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(np.log(clipped[np.arange(N), y])) / N)

    # Complete forward pass through the CNN.
    def forward(self, X: np.ndarray, return_cache: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        conv1, cache1 = self._conv_forward(X, self.W1, self.b1, stride=1, pad=1)
        relu1 = np.maximum(conv1, 0)
        pool1, pool_cache1 = self._max_pool_forward(relu1)

        conv2, cache2 = self._conv_forward(pool1, self.W2, self.b2, stride=1, pad=1)
        relu2 = np.maximum(conv2, 0)
        pool2, pool_cache2 = self._max_pool_forward(relu2)

        flat = pool2.reshape(pool2.shape[0], -1)
        fc1 = flat @ self.W3 + self.b3
        relu3 = np.maximum(fc1, 0)
        logits = relu3 @ self.W4 + self.b4

        if not return_cache:
            return logits, None

        cache = {
            "X": X,
            "conv1": conv1,
            "cache1": cache1,
            "pool_cache1": pool_cache1,
            "conv2": conv2,
            "cache2": cache2,
            "pool_cache2": pool_cache2,
            "flat": flat,
            "fc1": fc1,
            "relu3": relu3,
            "pool2_shape": pool2.shape,
        }
        return logits, cache

    # Backward pass with weight updates applied in-place.
    def backward(self, cache: Dict[str, np.ndarray], y: np.ndarray, probs: np.ndarray, learning_rate: float):
        N = y.shape[0]
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1
        dlogits /= N

        dW4 = cache["relu3"].T @ dlogits
        db4 = np.sum(dlogits, axis=0)
        drelu3 = dlogits @ self.W4.T
        dfc1 = drelu3 * (cache["fc1"] > 0)

        dW3 = cache["flat"].T @ dfc1
        db3 = np.sum(dfc1, axis=0)
        dflat = dfc1 @ self.W3.T
        dpool2 = dflat.reshape(cache["pool2_shape"])

        drelu2 = self._max_pool_backward(dpool2, cache["pool_cache2"])
        dconv2 = drelu2 * (cache["conv2"] > 0)
        dpool1, dW2, db2 = self._conv_backward(dconv2, cache["cache2"])

        drelu1 = self._max_pool_backward(dpool1, cache["pool_cache1"])
        dconv1 = drelu1 * (cache["conv1"] > 0)
        _, dW1, db1 = self._conv_backward(dconv1, cache["cache1"])

        self.W4 -= learning_rate * dW4
        self.b4 -= learning_rate * db4
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self._sync_weights()

    # Full training loop with mini-batching and shuffling.
    def train(self, X_train, y_train, epochs=10, learning_rate=0.01):
        print(f"Starting training for {epochs} epochs...")
        X = _prepare_inputs(np.array(X_train))
        y = np.array(y_train, dtype=np.int64)
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = self.rng.permutation(n_samples)
            total_loss = 0.0
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]

                logits, cache = self.forward(X_batch, return_cache=True)
                probs = self._softmax(logits)
                loss = self._cross_entropy(probs, y_batch)
                total_loss += loss * X_batch.shape[0]
                self.backward(cache, y_batch, probs, learning_rate)

            avg_loss = total_loss / n_samples
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

        print("Training completed.")
        return self

    # Prediction step without storing caches to save memory.
    def predict(self, X_test):
        X = _prepare_inputs(np.array(X_test))
        logits, _ = self.forward(X, return_cache=False)
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)

if __name__ == "__main__":
    # Comment: Simple CLI entrypoint to train on MP3_Dataset and save a pickle.
    project_root = Path(__file__).parent
    data_root = project_root / "MP3_Dataset"
    train_root = data_root / "train"
    val_root = data_root / "val"

    # Comment: Resize to 64x64 to keep the from-scratch CNN computationally manageable.
    target_size = 64
    X_train, y_train, lbl_map = load_image_folder(train_root, target_size=target_size)
    X_val, y_val, _ = load_image_folder(val_root, target_size=target_size, label_map=lbl_map)

    model = MyModel(image_size=target_size, num_classes=len(lbl_map), batch_size=32, seed=42)
    model.train(X_train, y_train, epochs=55, learning_rate=0.01)
    val_preds = model.predict(X_val)
    val_acc = float(np.mean(val_preds == y_val))
    print(f"Validation accuracy: {val_acc:.4f}")
    print_confusion_matrix(y_val, val_preds, CLASS_NAMES)
    model.save(project_root / "model.pkl")