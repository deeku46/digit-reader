"""Export one held-out test digit per class for the Digit Reader's sample buttons (same split as training)."""
import json, sys, base64
import numpy as np
from sklearn.model_selection import train_test_split
src = sys.argv[1]
X, y = [], []
for d in range(10):
    a = np.array(json.load(open(f"{src}/{d}.json"))["data"], dtype=np.float32).reshape(-1, 784)
    X.append(a); y += [d] * len(a)
X = np.concatenate(X); y = np.array(y)
_, Xte, _, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=7)
out = []
for d in range(10):
    i = np.where(yte == d)[0][0]
    out.append({"label": d, "px": base64.b64encode((Xte[i] * 255).round().astype(np.uint8).tobytes()).decode()})
open("../samples.js", "w").write("window.DIGIT_SAMPLES = " + json.dumps(out) + ";\n")
print("ok")
