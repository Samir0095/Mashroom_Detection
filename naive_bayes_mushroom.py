"""
CSE 3207 - Artificial Intelligence
Assignment: Understanding Bayes Theorem and Naive Bayes Classifier

Dataset : Mushroom (UCI Machine Learning Repository), 8124 records, 22 categorical features
Model   : Categorical Naive Bayes with Laplace smoothing, written from scratch
          using NumPy and pandas only.
          Matplotlib / seaborn are used for plots. scikit-learn is imported ONLY at the
          end to cross-check my answers. It is NOT used to build the classifier.
          ROC curve, precision-recall curve, AUC and all metrics are also coded by hand.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------
# keep_default_na=False so that pandas does not hide any value by itself
raw = pd.read_csv("data/mushrooms.csv", keep_default_na=False)
print("Shape:", raw.shape)
n_raw = len(raw)
class_counts = raw["class"].value_counts().to_dict()          # e = edible, p = poisonous

# ---------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------
# 2a. Missing values: in this dataset a missing value is written as "?".
miss_per_col = {c: int((raw[c] == "?").sum()) for c in raw.columns if (raw[c] == "?").any()}
print("Missing ('?') values per column:", miss_per_col)
n_missing = int(sum(miss_per_col.values()))
n_dup = int(raw.duplicated().sum())

df = raw.copy()
# 'stalk-root' has 2480 missing values (30.5 percent), too many to delete the rows. And I
# do not want to guess them. So "?" is kept as its own category named "missing".
# (Later I also test the other way: fill with the most common value.)
df = df.replace("?", "missing")

# 2b. Remove a useless column: 'veil-type' has only ONE value for every mushroom,
#     so it gives no information to separate the classes.
const_cols = [c for c in df.columns if c != "class" and df[c].nunique() == 1]
df = df.drop(columns=const_cols)
feature_names = [c for c in df.columns if c != "class"]
print("Dropped constant columns:", const_cols, "| features left:", len(feature_names))

# 2c. Encode categorical variables.
#     Target: edible -> 0, poisonous -> 1  (poisonous = positive class, because
#     eating a poisonous mushroom is the dangerous mistake).
y_all = df["class"].map({"e": 0, "p": 1}).values.astype(int)

#     Features: every category of a feature gets an integer code 0,1,2,...
categories = {c: sorted(df[c].unique()) for c in feature_names}      # list of possible values
X_all = np.zeros((len(df), len(feature_names)), dtype=int)
for j, c in enumerate(feature_names):
    code = {v: i for i, v in enumerate(categories[c])}
    X_all[:, j] = df[c].map(code).values
n_cats = np.array([len(categories[c]) for c in feature_names])      # K_j for every feature
print("Total categories (one-hot size would be):", int(n_cats.sum()))

# 2d. Standardization: NOT used. All features are categories (codes are only labels,
#     not real numbers), so mean/std have no meaning here.


# 2e. Train / test split 80/20 (by hand, stratified, fixed seed)
def stratified_split(y, test_frac, seed):
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = int(round(test_frac * len(idx)))
        te += list(idx[:n_te]); tr += list(idx[n_te:])
    rng.shuffle(tr); rng.shuffle(te)
    return np.array(tr), np.array(te)

tr_idx, te_idx = stratified_split(y_all, 0.20, SEED)
X_train, y_train = X_all[tr_idx], y_all[tr_idx]
X_test, y_test = X_all[te_idx], y_all[te_idx]
print("Train:", len(y_train), " Test:", len(y_test))


# ---------------------------------------------------------------
# 3. CATEGORICAL NAIVE BAYES FROM SCRATCH
# ---------------------------------------------------------------
class CategoricalNaiveBayes:
    """
    Naive Bayes for categorical features, with Laplace (add-alpha) smoothing.

    Training:
        P(class c)          = (rows of class c) / (all rows)
        P(feature j = v | c) = (rows of class c with feature j = v + alpha)
                               / (rows of class c + alpha * K_j)      K_j = number of values of feature j
    Prediction (log form, so numbers do not get too small):
        log P(c | x) ~ log P(c) + sum_j log P(x_j | c)
    """

    def __init__(self, n_cats, alpha=1.0):
        self.n_cats = n_cats
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.log_prior_ = np.zeros(len(self.classes_))
        # one table for every feature: rows = classes, columns = categories
        self.log_lik_ = [np.zeros((len(self.classes_), k)) for k in self.n_cats]
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]                                   # rows of this class only
            self.log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            for j, k in enumerate(self.n_cats):
                counts = np.bincount(X_c[:, j], minlength=k) + self.alpha   # count + alpha
                self.log_lik_[j][i] = np.log(counts / counts.sum())
        return self

    def log_posterior(self, X):
        """log P(c) + sum_j log P(x_j | c), for every row and every class."""
        out = np.tile(self.log_prior_, (X.shape[0], 1))
        for j in range(X.shape[1]):
            out += self.log_lik_[j][:, X[:, j]].T             # look up the probability table
        return out

    def predict_proba(self, X):
        lp = self.log_posterior(X)
        lp = lp - lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)               # divide by evidence

    def decision_score(self, X):
        """log-odds of 'poisonous' vs 'edible'. Bigger = more likely poisonous."""
        lp = self.log_posterior(X)
        return lp[:, 1] - lp[:, 0]

    def predict(self, X):
        return self.classes_[np.argmax(self.log_posterior(X), axis=1)]


# ---------------------------------------------------------------
# 4. METRICS (all written by hand)
# ---------------------------------------------------------------
def confusion_counts(y_true, y_hat):
    TP = int(np.sum((y_true == 1) & (y_hat == 1)))
    FP = int(np.sum((y_true == 0) & (y_hat == 1)))
    FN = int(np.sum((y_true == 1) & (y_hat == 0)))
    TN = int(np.sum((y_true == 0) & (y_hat == 0)))
    return TP, FP, FN, TN

def safe_div(a, b):
    return a / b if b else 0.0

def basic_metrics(y_true, y_hat):
    TP, FP, FN, TN = confusion_counts(y_true, y_hat)
    acc = (TP + TN) / (TP + TN + FP + FN)
    p = safe_div(TP, TP + FP)
    r = safe_div(TP, TP + FN)
    f = safe_div(2 * p * r, p + r)
    return acc, p, r, f

def roc_points(y_true, score):
    """ROC curve: false positive rate and true positive rate for every threshold."""
    order = np.argsort(-score, kind="mergesort")
    y_s, s_s = y_true[order], score[order]
    tps = np.cumsum(y_s == 1)
    fps = np.cumsum(y_s == 0)
    last = np.r_[np.where(np.diff(s_s))[0], len(s_s) - 1]      # last index of each distinct score
    tpr = np.r_[0, tps[last] / tps[-1]]
    fpr = np.r_[0, fps[last] / fps[-1]]
    return fpr, tpr

def auc_trapezoid(x, y):
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))

def pr_points(y_true, score):
    """Precision-recall curve and average precision (AP)."""
    order = np.argsort(-score, kind="mergesort")
    y_s, s_s = y_true[order], score[order]
    tps = np.cumsum(y_s == 1)
    fps = np.cumsum(y_s == 0)
    last = np.r_[np.where(np.diff(s_s))[0], len(s_s) - 1]
    prec = tps[last] / (tps[last] + fps[last])
    rec = tps[last] / tps[-1]
    ap = float(np.sum(np.diff(np.r_[0, rec]) * prec))
    return np.r_[1.0, prec], np.r_[0, rec], ap, s_s[last]


# ---------------------------------------------------------------
# 5. TRAIN AND TEST
# ---------------------------------------------------------------
model = CategoricalNaiveBayes(n_cats, alpha=1.0).fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
score_test = model.decision_score(X_test)

TP, FP, FN, TN = confusion_counts(y_test, y_pred)
cm = np.array([[TN, FP], [FN, TP]])               # rows = actual (edible, poisonous)
accuracy, precision, recall, f1 = basic_metrics(y_test, y_pred)
train_acc = float(np.mean(y_pred_train == y_train))
print("Confusion matrix:\n", cm)
print(f"Acc={accuracy:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f} | train acc={train_acc:.4f}")

# metrics for the edible class too (edible treated as positive)
tp_e, fp_e, fn_e = TN, FN, FP
p_e = safe_div(tp_e, tp_e + fp_e); r_e = safe_div(tp_e, tp_e + fn_e)
f_e = safe_div(2 * p_e * r_e, p_e + r_e)
per_class = {"Edible": (p_e, r_e, f_e, int((y_test == 0).sum())),
             "Poisonous": (precision, recall, f1, int((y_test == 1).sum()))}
macro = [float(np.mean([per_class[k][i] for k in per_class])) for i in range(3)]

fpr, tpr = roc_points(y_test, score_test)
roc_auc = auc_trapezoid(fpr, tpr)
prec_c, rec_c, ap, thr_c = pr_points(y_test, score_test)
print(f"ROC AUC={roc_auc:.4f}  Average precision={ap:.4f}")

# Always-say-edible baseline (a "silly" model)
baseline_acc = float(np.mean(y_test == 0))

# A safer decision threshold. It is chosen from TRAINING data only (the lowest score of any
# poisonous mushroom in training), then tested on the unseen test data. No test labels used.
thr_safe = float(model.decision_score(X_train)[y_train == 1].min())
yhat_safe = (score_test >= thr_safe).astype(int)
acc_s, p_s, r_s, f_s = basic_metrics(y_test, yhat_safe)
TPs, FPs, FNs, TNs = confusion_counts(y_test, yhat_safe)
print(f"Safe threshold {thr_safe:.2f}: acc={acc_s:.4f} P={p_s:.4f} R={r_s:.4f} (FP={FPs}, FN={FNs})")

# ---------------------------------------------------------------
# 6. EXTRA EXPERIMENTS
# ---------------------------------------------------------------
# 6a. Two ways to handle the missing 'stalk-root' values
#     (A) own category "missing" (used above)   (B) fill with most common training value
j_sr = feature_names.index("stalk-root")
miss_code = categories["stalk-root"].index("missing")
known_train = X_train[X_train[:, j_sr] != miss_code, j_sr]
mode_code = int(np.bincount(known_train).argmax())
mode_name = categories["stalk-root"][mode_code]
Xtr_b, Xte_b = X_train.copy(), X_test.copy()
Xtr_b[Xtr_b[:, j_sr] == miss_code, j_sr] = mode_code
Xte_b[Xte_b[:, j_sr] == miss_code, j_sr] = mode_code
mb = CategoricalNaiveBayes(n_cats).fit(Xtr_b, y_train)
met_b = basic_metrics(y_test, mb.predict(Xte_b))
print("Mode-fill variant (acc, P, R, F1):", np.round(met_b, 4), "| mode =", mode_name)

# 6b. Effect of the smoothing value alpha - checked with 5-fold CV on TRAINING data only
def kfold_indices(y, k, seed):
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i] += list(part)
    return [np.array(f) for f in folds]

# 6c. 5-fold cross-validation on the whole dataset (checks the result is stable)
folds = kfold_indices(y_all, 5, SEED)
cv_acc, cv_rec, cv_prec = [], [], []
for i in range(5):
    te = folds[i]
    tr = np.concatenate([folds[k] for k in range(5) if k != i])
    m = CategoricalNaiveBayes(n_cats).fit(X_all[tr], y_all[tr])
    a, p, r, f = basic_metrics(y_all[te], m.predict(X_all[te]))
    cv_acc.append(a); cv_prec.append(p); cv_rec.append(r)
print("CV accuracy per fold:", np.round(cv_acc, 4), "mean", np.mean(cv_acc).round(4), "std", np.std(cv_acc).round(4))

# ---------------------------------------------------------------
# 7. CROSS-CHECK with scikit-learn (verification only)
# ---------------------------------------------------------------
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_auc_score, average_precision_score
chk = CategoricalNB(alpha=1.0, min_categories=n_cats).fit(X_train, y_train)
agree = float(np.mean(chk.predict(X_test) == y_pred))
auc_chk = float(roc_auc_score(y_test, score_test))
ap_chk = float(average_precision_score(y_test, score_test))
print("Agreement with sklearn CategoricalNB:", agree, "| sklearn AUC:", round(auc_chk, 4), "AP:", round(ap_chk, 4))

# ---------------------------------------------------------------
# 8. MOST INFLUENTIAL FEATURE VALUES
# ---------------------------------------------------------------
NAMES = {
    "odor": dict(a="almond", l="anise", c="creosote", y="fishy", f="foul", m="musty", n="none", p="pungent", s="spicy"),
    "spore-print-color": dict(k="black", n="brown", b="buff", h="chocolate", r="green", o="orange", u="purple", w="white", y="yellow"),
    "gill-size": dict(b="broad", n="narrow"),
    "gill-color": dict(k="black", n="brown", b="buff", h="chocolate", g="gray", r="green", o="orange", p="pink", u="purple", e="red", w="white", y="yellow"),
    "ring-type": dict(c="cobwebby", e="evanescent", f="flaring", l="large", n="none", p="pendant", s="sheathing", z="zone"),
    "bruises": dict(t="yes", f="no"),
    "population": dict(a="abundant", c="clustered", n="numerous", s="scattered", v="several", y="solitary"),
    "habitat": dict(g="grasses", l="leaves", m="meadows", p="paths", u="urban", w="waste", d="woods"),
    "stalk-root": dict(b="bulbous", c="club", u="cup", e="equal", z="rhizomorphs", r="rooted"),
    "stalk-surface-above-ring": dict(f="fibrous", y="scaly", k="silky", s="smooth"),
    "stalk-surface-below-ring": dict(f="fibrous", y="scaly", k="silky", s="smooth"),
    "stalk-color-above-ring": dict(n="brown", b="buff", c="cinnamon", g="gray", o="orange", p="pink", e="red", w="white", y="yellow"),
    "stalk-color-below-ring": dict(n="brown", b="buff", c="cinnamon", g="gray", o="orange", p="pink", e="red", w="white", y="yellow"),
    "cap-shape": dict(b="bell", c="conical", x="convex", f="flat", k="knobbed", s="sunken"),
    "cap-surface": dict(f="fibrous", g="grooves", y="scaly", s="smooth"),
    "cap-color": dict(n="brown", b="buff", c="cinnamon", g="gray", r="green", p="pink", u="purple", e="red", w="white", y="yellow"),
    "gill-attachment": dict(a="attached", d="descending", f="free", n="notched"),
    "gill-spacing": dict(c="close", w="crowded", d="distant"),
    "stalk-shape": dict(e="enlarging", t="tapering"),
    "veil-color": dict(n="brown", o="orange", w="white", y="yellow"),
    "ring-number": dict(n="none", o="one", t="two"),
}
def pretty(feat, val):
    return f"{feat} = {NAMES.get(feat, {}).get(val, val)}"

rows = []
for j, feat in enumerate(feature_names):
    for v_i, v in enumerate(categories[feat]):
        lo = model.log_lik_[j][1, v_i] - model.log_lik_[j][0, v_i]   # log P(v|poison) - log P(v|edible)
        rows.append((pretty(feat, v), float(lo)))
rows.sort(key=lambda t: t[1])
top_edible = rows[:7]
top_poison = rows[-7:][::-1]
print("Top poisonous-leaning:", top_poison)
print("Top edible-leaning:", top_edible)

# ---------------------------------------------------------------
# 9. GRAPHS
# ---------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "font.size": 11})
BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

# 9a. Dataset overview: class balance + missing values
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
vals = [class_counts["e"], class_counts["p"]]
b = ax[0].bar(["Edible", "Poisonous"], vals, color=[GREEN, RED])
for r in b:
    ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 60, str(int(r.get_height())), ha="center")
ax[0].set_ylabel("Number of mushrooms"); ax[0].set_title("Class distribution"); ax[0].set_ylim(0, 4900)
known = n_raw - miss_per_col["stalk-root"]
b = ax[1].bar(["Known", "Missing (?)"], [known, miss_per_col["stalk-root"]], color=[BLUE, ORANGE])
for r in b:
    ax[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 100, str(int(r.get_height())), ha="center")
ax[1].set_title("'stalk-root' values"); ax[1].set_ylim(0, 9200)
plt.tight_layout(); plt.savefig("fig_dataset.png", dpi=200); plt.close()

# 9b. Confusion matrix
names = ["Edible", "Poisonous"]
fig, ax = plt.subplots(figsize=(4.4, 3.7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names,
            annot_kws={"size": 15}, ax=ax)
ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label"); ax.set_title("Confusion Matrix (Test Data)")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=200); plt.close()

# 9c. Actual vs predicted labels
act = [int((y_test == c).sum()) for c in (0, 1)]
prd = [int((y_pred == c).sum()) for c in (0, 1)]
x = np.arange(2); w = 0.36
fig, ax = plt.subplots(figsize=(4.8, 3.7))
b1 = ax.bar(x - w / 2, act, w, label="Actual", color=BLUE)
b2 = ax.bar(x + w / 2, prd, w, label="Predicted", color=ORANGE)
for r in list(b1) + list(b2):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 8, str(int(r.get_height())), ha="center")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Number of mushrooms")
ax.set_title("Actual vs Predicted Labels (Test Data)"); ax.set_ylim(0, max(act + prd) * 1.15); ax.legend()
plt.tight_layout(); plt.savefig("label_distribution.png", dpi=200); plt.close()

# 9d. Performance summary
fig, ax = plt.subplots(figsize=(4.9, 3.9))
labels = ["Precision", "Recall", "F1-score"]
e_vals, p_vals = list(per_class["Edible"][:3]), list(per_class["Poisonous"][:3])
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w / 2, e_vals, w, label="Edible", color=GREEN)
b2 = ax.bar(x + w / 2, p_vals, w, label="Poisonous", color=RED)
for r in list(b1) + list(b2):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004, f"{r.get_height():.3f}", ha="center", fontsize=8)
ax.axhline(accuracy, color="black", ls="--", lw=1, label=f"Accuracy = {accuracy:.3f}")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.85, 1.03)
ax.set_ylabel("Score"); ax.set_title("Performance Summary on Test Data")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
plt.tight_layout(); plt.savefig("metrics_summary.png", dpi=200); plt.close()

# 9e. ROC curve
fig, ax = plt.subplots(figsize=(4.4, 3.8))
ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"Naive Bayes (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Random guess (AUC = 0.5)")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
ax.set_title("ROC Curve"); ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("roc_curve.png", dpi=200); plt.close()

# 9f. Precision-recall curve
fig, ax = plt.subplots(figsize=(4.4, 3.8))
ax.plot(rec_c, prec_c, color=ORANGE, lw=2, label=f"Naive Bayes (AP = {ap:.4f})")
ax.axhline(float(np.mean(y_test == 1)), ls="--", color="gray", label="Random guess (precision = 0.48)")
ax.plot(recall, precision, "o", color="black", ms=6, label="Default threshold")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve")
ax.set_ylim(0.4, 1.03); ax.legend(loc="lower left", fontsize=8.5); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("pr_curve.png", dpi=200); plt.close()

# 9g. Score distribution (log-odds)
clip = lambda s: np.clip(s, -60, 60)
fig, ax = plt.subplots(figsize=(4.8, 3.7))
bins = np.linspace(-60, 60, 41)
ax.hist(clip(score_test[y_test == 0]), bins=bins, alpha=0.8, color=GREEN, label="Actual edible")
ax.hist(clip(score_test[y_test == 1]), bins=bins, alpha=0.8, color=RED, label="Actual poisonous")
ax.axvline(0, color="black", ls="--", lw=1)
ax.set_yscale("log")
ax.set_xlabel("Log-odds score (positive = predicted poisonous)"); ax.set_ylabel("Number of mushrooms (log scale)")
ax.set_title("Model Score on Test Data"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig("score_hist.png", dpi=200); plt.close()

# 9h. Most influential feature values
fig, ax = plt.subplots(figsize=(7.2, 3.8))
items = top_poison[::-1] + top_edible[::-1]
ax.barh([i[0] for i in items], [i[1] for i in items], color=[RED] * 7 + [GREEN] * 7)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("log P(value | poisonous) - log P(value | edible)")
ax.set_title("Feature Values That Push the Prediction Most")
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout(); plt.savefig("top_features.png", dpi=200); plt.close()

# ---------------------------------------------------------------
# 10. SAVE ALL NUMBERS FOR THE REPORT
# ---------------------------------------------------------------
res = dict(
    n_raw=n_raw, n_edible=int(class_counts["e"]), n_poison=int(class_counts["p"]),
    n_missing=n_missing, missing_cols=miss_per_col, n_dup=n_dup, const_cols=const_cols,
    n_features_used=len(feature_names), total_categories=int(n_cats.sum()),
    n_train=int(len(y_train)), n_test=int(len(y_test)),
    train_edible=int((y_train == 0).sum()), train_poison=int((y_train == 1).sum()),
    test_edible=int((y_test == 0).sum()), test_poison=int((y_test == 1).sum()),
    prior_edible=float(np.exp(model.log_prior_[0])), prior_poison=float(np.exp(model.log_prior_[1])),
    TP=TP, FP=FP, FN=FN, TN=TN, accuracy=accuracy, precision=precision, recall=recall, f1=f1,
    train_acc=train_acc, baseline_acc=baseline_acc, per_class=per_class, macro=macro,
    roc_auc=roc_auc, avg_precision=ap, sklearn_agreement=agree, sklearn_auc=auc_chk, sklearn_ap=ap_chk,
    safe=dict(thr=thr_safe, acc=acc_s, precision=p_s, recall=r_s, FP=FPs, FN=FNs, TP=TPs, TN=TNs),
    modefill=dict(mode=mode_name, acc=met_b[0], precision=met_b[1], recall=met_b[2], f1=met_b[3]),
    cv_acc=[float(a) for a in cv_acc], cv_mean=float(np.mean(cv_acc)), cv_std=float(np.std(cv_acc)),
    cv_recall=[float(a) for a in cv_rec], cv_prec=[float(a) for a in cv_prec],
    top_poison=top_poison, top_edible=top_edible,
)
json.dump(res, open("results.json", "w"), indent=2)
