"""Train the Digit Reader network.

Data: 10,000 MNIST handwritten digits (28x28) from the MIT-licensed `mnist` npm package.
Model: 784 -> 128 -> 64 -> 10 multilayer perceptron with ReLU, trained with Adam.
Augmentation: small random shifts, rotations and scaling, so the model copes with
digits drawn by mouse or finger instead of only scanned ones.
Output: int8-quantised weights in ../model.js
"""
import json, sys, base64
import numpy as np
from scipy import ndimage
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

src = sys.argv[1]
X, y = [], []
for d in range(10):
    a = np.array(json.load(open(f"{src}/{d}.json"))["data"], dtype=np.float32).reshape(-1, 28, 28)
    X.append(a); y += [d] * len(a)
X = np.concatenate(X); y = np.array(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=7)

def recentre(img):
    """Same preprocessing the browser applies to a drawing: fit to 20x20 box, centre by mass."""
    ys, xs = np.nonzero(img > 0.1)
    if len(xs) == 0: return img
    crop = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
    s = 20 / max(crop.shape)
    crop = np.clip(ndimage.zoom(crop, s, order=1), 0, 1)
    out = np.zeros((28, 28), np.float32)
    h, w = crop.shape
    out[(28-h)//2:(28-h)//2+h, (28-w)//2:(28-w)//2+w] = crop[:28, :28]
    cy, cx = ndimage.center_of_mass(out)
    return np.clip(ndimage.shift(out, (14 - cy, 14 - cx), order=1), 0, 1)

rng = np.random.default_rng(7)
def augment(img):
    img = ndimage.rotate(img, rng.uniform(-14, 14), reshape=False, order=1)
    z = rng.uniform(0.85, 1.12)
    img = ndimage.affine_transform(img, np.eye(2) / z, offset=14 - 14 / z, order=1)
    if rng.random() < 0.5:   # thicker strokes, like a mouse brush
        img = ndimage.grey_dilation(img, size=(2, 2))
    return recentre(np.clip(img, 0, 1))

Xtr_p = np.array([recentre(i) for i in Xtr])
Xaug = np.array([augment(i) for i in Xtr for _ in range(3)])
Xfit = np.concatenate([Xtr_p, Xaug]).reshape(-1, 784)
yfit = np.concatenate([ytr, np.repeat(ytr, 3)])
Xte_p = np.array([recentre(i) for i in Xte]).reshape(-1, 784)

clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=128,
                    learning_rate_init=1e-3, max_iter=40, early_stopping=True,
                    n_iter_no_change=5, random_state=7)
clf.fit(Xfit, yfit)

def forward(W, B, x):
    for i, (w, b) in enumerate(zip(W, B)):
        x = x @ w + b
        if i < len(W) - 1: x = np.maximum(x, 0)
    return x.argmax(1)

acc = accuracy_score(yte, clf.predict(Xte_p))
# int8 quantisation, per layer
layers, Wq = [], []
for w, b in zip(clf.coefs_, clf.intercepts_):
    s = np.abs(w).max() / 127
    q = np.round(w / s).astype(np.int8)
    Wq.append(q.astype(np.float32) * s)
    layers.append({"shape": list(w.shape), "scale": float(s),
                   "w": base64.b64encode(q.tobytes()).decode(),
                   "b": [round(float(v), 5) for v in b]})
accq = accuracy_score(yte, forward(Wq, clf.intercepts_, Xte_p))
cm = confusion_matrix(yte, forward(Wq, clf.intercepts_, Xte_p))
per = (cm.diagonal() / cm.sum(1)).round(4).tolist()
meta = {"train": int(len(Xtr)), "train_with_aug": int(len(Xfit)), "test": int(len(Xte)),
        "acc": round(float(acc), 4), "acc_int8": round(float(accq), 4), "per_class": per,
        "epochs": int(clf.n_iter_)}
print(json.dumps(meta))
with open("../model.js", "w") as f:
    f.write("window.DIGIT_MODEL = " + json.dumps({"layers": layers, "meta": meta}) + ";\n")
