"""Builder for Notebook 36 — Classical ML Evaluation."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbbuild import build, code, md

cells = [
    md(r"""
    # 36 · Classical ML Evaluation
    ### Phase 7 — Evaluation · *ML/AI Senior Mastery Curriculum*

    > Accuracy is almost always the wrong metric. This notebook teaches the complete
    > evaluation toolkit for classification, regression, and ranking — implemented
    > from scratch so you understand every formula deeply. You will learn when to use
    > each metric, why accuracy fails on imbalanced data, how to plot calibration curves,
    > and how to translate model performance into business cost.
    """),

    md(r"""
    ## 1 · Learning Objectives

    **What you will master**
    - **Confusion matrix** and all 8 derived metrics: accuracy, precision, recall, F1,
      specificity, NPV, FPR, MCC.
    - **ROC curve** from scratch: threshold sweep → FPR/TPR → trapezoidal AUC.
    - **PR curve** from scratch: threshold sweep → precision/recall → AUC-PR.
    - **Calibration**: reliability diagram, Brier score, Expected Calibration Error (ECE).
    - **Multi-class**: macro/micro/weighted F1, Cohen's kappa.
    - **Regression metrics**: MAE, MSE, RMSE, MAPE, R², explained variance.
    - **Ranking metrics**: NDCG@k, MRR, MAP — from scratch.
    - **Business cost framing**: asymmetric FP/FN costs; optimal threshold from cost matrix.
    - When to use each metric and why accuracy misleads on imbalanced datasets.

    **Why it matters**
    - Every ML system needs an evaluation story. Choosing the wrong metric can lead to
      shipping a model that looks great in evaluation but fails in production. Senior
      engineers are expected to pick the right metric, explain it to stakeholders, and
      derive business value from model performance numbers.
    """),

    md(r"""
    ## 2 · Historical Motivation

    **Signal detection theory (1950s).** ROC (Receiver Operating Characteristic) curves
    originated in WWII radar — distinguishing real signals (enemy aircraft) from noise.
    The framework: at each threshold, plot true positive rate vs. false positive rate.
    Adopted in medical diagnosis (1970s), then ML evaluation (1980s onwards).

    **Precision-Recall (1980s–1990s).** Information retrieval researchers needed metrics
    that focused on the relevant class only (what fraction of retrieved docs are relevant?).
    PR curves answer this. Better than ROC for class-imbalanced tasks because they focus
    on the minority (positive) class.

    **Calibration (Zadrozny & Elkan, 2002).** A model that outputs probability 0.8 should
    be correct 80% of the time. Many classifiers are not calibrated (SVMs, decision trees,
    neural nets without temperature scaling). Calibration research showed that probability
    estimates matter as much as rank-ordering.

    **MCC (Matthews, 1975).** The Matthews Correlation Coefficient is a balanced binary
    classification metric that works even with severe imbalance. Rediscovered as the gold
    standard for imbalanced evaluation (Chicco et al., 2020).

    **NDCG (Järvelin & Kekäläinen, 2002).** Normalised Discounted Cumulative Gain for
    ranked retrieval. Considers both relevance and position — being wrong in position 1
    costs more than being wrong in position 10. Standard for search, recommendation, RAG.
    """),

    md(r"""
    ## 3 · Intuition & Visual Understanding

    **Why accuracy fails on imbalanced data:**
    ```
    Dataset: 990 negatives, 10 positives (1% positive class)
    Trivial classifier: always predict negative
    Accuracy = 990/1000 = 99.0%  ← looks great
    Recall = 0/10 = 0.0%         ← catastrophic (never finds the positives)
    ```

    **ROC vs. PR intuition:**
    - ROC: "How well does my model rank positives above negatives?" Good for balanced data.
    - PR: "When my model says positive, how often is it right?" Good for imbalanced data.
    - When positive class is rare (fraud, cancer, default), PR is more informative.

    **Calibration intuition:**
    ```
    Perfect calibration: 100 predictions at 0.7 → 70 correct
    Overconfident: 100 predictions at 0.7 → only 50 correct (should have been 0.5)
    Underconfident: 100 predictions at 0.7 → 90 correct (should have been 0.9)
    ```

    **Business cost framing:**
    ```
    FP: approve a fraudulent loan → lose $10,000
    FN: reject a good applicant  → lose $500 in opportunity cost
    → Optimal threshold minimises: $10,000*FP + $500*FN
    → This is much lower than 0.5 default
    ```
    """),

    code(r"""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rng = np.random.default_rng(42)
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    print('Dependencies loaded.')
    """),

    md(r"""
    ## 4 · Mathematical Foundations

    ### 4.1 Confusion matrix fundamentals

    For binary classification with threshold $\theta$:

    | | Predicted Positive | Predicted Negative |
    |---|---|---|
    | **Actual Positive** | TP | FN |
    | **Actual Negative** | FP | TN |

    All metrics derive from TP, FP, TN, FN:

    $\text{Accuracy} = \frac{TP+TN}{TP+FP+TN+FN}$

    $\text{Precision} = \frac{TP}{TP+FP}$ (of predicted positives, fraction actually positive)

    $\text{Recall} = \frac{TP}{TP+FN}$ (of actual positives, fraction predicted positive)

    $\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision}+\text{Recall}}$

    $\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$

    MCC range: -1 (inverse) to +1 (perfect). 0 = random. Works for any class imbalance.

    ### 4.2 ROC and AUC

    For threshold $\theta$: $\text{TPR}(\theta) = \frac{TP}{TP+FN}$, $\text{FPR}(\theta) = \frac{FP}{FP+TN}$

    As $\theta$ sweeps from 1→0, the ROC curve traces from (0,0) to (1,1).

    $\text{AUC} = \int_0^1 \text{TPR}(t) d(\text{FPR}(t)) \approx \sum_{i} \frac{(fpr_{i+1}-fpr_i)(tpr_i + tpr_{i+1})}{2}$ (trapezoidal)

    AUC interpretation: probability that the model ranks a random positive above a random negative.

    ### 4.3 Calibration

    **Brier Score**: $BS = \frac{1}{n}\sum_{i=1}^n (p_i - y_i)^2$ (lower = better; 0 = perfect)

    **ECE** (Expected Calibration Error): partition predictions into M equal-width bins $B_m$:

    $ECE = \sum_{m=1}^M \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$

    where $acc(B_m)$ = fraction of correct predictions in bin $m$, $conf(B_m)$ = mean predicted probability in bin $m$.

    ### 4.4 NDCG@k

    $DCG@k = \sum_{i=1}^k \frac{rel_i}{\log_2(i+1)}$

    $NDCG@k = \frac{DCG@k}{IDCG@k}$ where $IDCG@k$ = DCG of ideal (sorted by relevance) ranking.

    $MRR = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank of first relevant result for } q}$

    ### 4.5 Optimal threshold from cost matrix

    Given costs $C_{FP}$ and $C_{FN}$, optimal threshold:

    $\theta^* = \arg\min_\theta C_{FP} \cdot FP(\theta) + C_{FN} \cdot FN(\theta)$

    Equivalently: $\theta^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$ (under class balance).
    """),

    md(r"""
    ## 5 · Implementations from Scratch

    ### 5a — Generate synthetic dataset and confusion matrix
    """),

    code(r"""
    # 5a. Binary classification evaluation — all metrics from scratch.

    def generate_credit_data(n=1000, positive_rate=0.08):
        # Simulate credit risk: 8% default rate (imbalanced).
        y_true = (rng.random(n) < positive_rate).astype(int)
        # Simulate model: better calibration for positives (signal with noise).
        y_score = np.where(y_true == 1,
                           rng.beta(6, 3, n),   # positives skew high
                           rng.beta(2, 7, n))   # negatives skew low
        return y_true, y_score

    def confusion_matrix_at_threshold(y_true, y_score, threshold=0.5):
        y_pred = (y_score >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        return tp, fp, tn, fn

    def all_metrics(tp, fp, tn, fn):
        n = tp + fp + tn + fn
        accuracy    = (tp + tn) / n
        precision   = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp)   if (tn + fp) > 0 else 0.0
        npv         = tn / (tn + fn)   if (tn + fn) > 0 else 0.0
        fpr         = fp / (fp + tn)   if (fp + tn) > 0 else 0.0
        f1          = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
        denom_mcc   = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
        mcc         = (tp*tn - fp*fn) / denom_mcc if denom_mcc > 0 else 0.0
        return dict(accuracy=accuracy, precision=precision, recall=recall, specificity=specificity,
                    npv=npv, fpr=fpr, f1=f1, mcc=mcc)

    y_true, y_score = generate_credit_data(n=2000)
    tp, fp, tn, fn = confusion_matrix_at_threshold(y_true, y_score, threshold=0.5)

    print('Credit risk dataset: n=2000, ~8% default rate')
    print(f'Confusion matrix (threshold=0.50):')
    print(f'  TP={tp}  FP={fp}')
    print(f'  FN={fn}  TN={tn}')
    print()
    m = all_metrics(tp, fp, tn, fn)
    for name, val in m.items():
        print(f'  {name:12s}: {val:.4f}')

    print('\nNote: Accuracy={:.1f}% looks good, but Recall={:.1f}% means we miss {}% of defaulters!'.format(
        m['accuracy']*100, m['recall']*100, round((1-m['recall'])*100)))
    """),

    md(r"""
    ### 5b — ROC curve and AUC from scratch
    """),

    code(r"""
    # 5b. ROC curve: threshold sweep, trapezoidal AUC.

    def roc_curve_scratch(y_true, y_score):
        thresholds = np.sort(np.unique(y_score))[::-1]
        tprs, fprs = [0.0], [0.0]
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        for t in thresholds:
            tp, fp, tn, fn = confusion_matrix_at_threshold(y_true, y_score, threshold=t)
            tprs.append(tp / n_pos if n_pos > 0 else 0.0)
            fprs.append(fp / n_neg if n_neg > 0 else 0.0)
        tprs.append(1.0); fprs.append(1.0)
        return np.array(fprs), np.array(tprs)

    def auc_trapezoidal(x, y):
        # Trapezoidal rule. Sort by x first.
        order = np.argsort(x)
        x, y = x[order], y[order]
        return float(np.trapz(y, x))

    fprs, tprs = roc_curve_scratch(y_true, y_score)
    auc_roc = auc_trapezoidal(fprs, tprs)
    print(f'AUC-ROC (from scratch): {auc_roc:.4f}')

    # Compare to sklearn if available (optional).
    try:
        from sklearn.metrics import roc_auc_score
        auc_sk = roc_auc_score(y_true, y_score)
        print(f'AUC-ROC (sklearn):      {auc_sk:.4f}  (diff={abs(auc_roc - auc_sk):.6f})')
    except ImportError:
        print('[sklearn not installed — comparison skipped]')
    """),

    md(r"""
    ### 5c — PR curve and AUC-PR from scratch
    """),

    code(r"""
    # 5c. Precision-Recall curve: threshold sweep.

    def pr_curve_scratch(y_true, y_score):
        thresholds = np.sort(np.unique(y_score))[::-1]
        precisions, recalls = [], []
        for t in thresholds:
            tp, fp, tn, fn = confusion_matrix_at_threshold(y_true, y_score, threshold=t)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
        # Add (recall=0, precision=1) starting point.
        precisions = [1.0] + precisions
        recalls = [0.0] + recalls
        return np.array(recalls), np.array(precisions)

    recalls_pr, precisions_pr = pr_curve_scratch(y_true, y_score)
    auc_pr = auc_trapezoidal(recalls_pr, precisions_pr)
    baseline_pr = y_true.mean()   # random classifier AP = prevalence.
    print(f'AUC-PR (from scratch): {auc_pr:.4f}')
    print(f'Random baseline AP:    {baseline_pr:.4f}  (prevalence)')
    print(f'Lift over random:      {auc_pr / baseline_pr:.2f}x')
    """),

    md(r"""
    ### 5d — Calibration: Brier score, reliability diagram, ECE
    """),

    code(r"""
    # 5d. Calibration metrics: Brier score and ECE.

    def brier_score(y_true, y_prob):
        return float(np.mean((y_prob - y_true) ** 2))

    def reliability_diagram(y_true, y_prob, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_sizes = [], [], []
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
            if mask.sum() > 0:
                bin_accs.append(float(y_true[mask].mean()))
                bin_confs.append(float(y_prob[mask].mean()))
                bin_sizes.append(int(mask.sum()))
            else:
                bin_accs.append(float('nan'))
                bin_confs.append((bin_edges[i] + bin_edges[i+1]) / 2)
                bin_sizes.append(0)
        return np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes)

    def ece(y_true, y_prob, n_bins=10):
        accs, confs, sizes = reliability_diagram(y_true, y_prob, n_bins)
        total = len(y_true)
        ece_val = 0.0
        for acc, conf, size in zip(accs, confs, sizes):
            if size > 0 and not np.isnan(acc):
                ece_val += (size / total) * abs(acc - conf)
        return ece_val

    bs = brier_score(y_true, y_score)
    ece_val = ece(y_true, y_score)
    bin_accs, bin_confs, bin_sizes = reliability_diagram(y_true, y_score)

    print(f'Brier Score: {bs:.4f}  (0=perfect, 0.25=random binary)')
    print(f'ECE:         {ece_val:.4f}  (0=perfect calibration)')
    print(f'\nReliability diagram bins:')
    print(f'  {"Bin centre":10s} {"Accuracy":10s} {"Size":8s}')
    for conf, acc, size in zip(bin_confs, bin_accs, bin_sizes):
        acc_str = f'{acc:.3f}' if not np.isnan(acc) else 'N/A'
        print(f'  {conf:10.3f} {acc_str:10s} {size:8d}')
    """),

    md(r"""
    ### 5e — Multi-class evaluation: macro/micro/weighted, Cohen's kappa
    """),

    code(r"""
    # 5e. Multi-class evaluation from scratch.

    def multiclass_metrics(y_true, y_pred, n_classes=None):
        if n_classes is None:
            n_classes = max(max(y_true), max(y_pred)) + 1
        per_class = {}
        for c in range(n_classes):
            tp_c = int(np.sum((y_pred == c) & (y_true == c)))
            fp_c = int(np.sum((y_pred == c) & (y_true != c)))
            fn_c = int(np.sum((y_pred != c) & (y_true == c)))
            prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec_c  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c   = 2*prec_c*rec_c / (prec_c+rec_c) if (prec_c+rec_c) > 0 else 0.0
            support = int(np.sum(y_true == c))
            per_class[c] = {'precision': prec_c, 'recall': rec_c, 'f1': f1_c, 'support': support}

        n = len(y_true)
        macro_f1    = float(np.mean([per_class[c]['f1'] for c in range(n_classes)]))
        micro_f1    = float(np.mean(y_true == y_pred))   # micro = accuracy for multi-class
        weighted_f1 = float(np.sum([per_class[c]['f1'] * per_class[c]['support'] for c in range(n_classes)]) / n)
        return per_class, macro_f1, micro_f1, weighted_f1

    def cohens_kappa(y_true, y_pred):
        n = len(y_true)
        classes = list(set(y_true) | set(y_pred))
        p_o = np.mean(y_true == y_pred)
        p_e = sum((np.mean(y_true == c) * np.mean(y_pred == c)) for c in classes)
        return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    # Simulate 3-class product category classifier.
    n_mc = 300
    y_true_mc = rng.integers(0, 3, n_mc)
    # Simulate model: 75% accuracy, some class imbalance.
    noise = rng.integers(0, 3, n_mc)
    y_pred_mc = np.where(rng.random(n_mc) < 0.75, y_true_mc, noise)

    per_class, macro_f1, micro_f1, weighted_f1 = multiclass_metrics(y_true_mc, y_pred_mc, n_classes=3)
    kappa = cohens_kappa(y_true_mc, y_pred_mc)

    print('Multi-class evaluation (3 product categories):')
    print(f'\n  {"Class":6s} {"Precision":10s} {"Recall":10s} {"F1":8s} {"Support"}')
    for c, m in per_class.items():
        print(f'  {c:<6d} {m["precision"]:.4f}     {m["recall"]:.4f}     {m["f1"]:.4f}  {m["support"]}')
    print(f'\n  Macro F1:    {macro_f1:.4f}  (unweighted average; penalises bad minority class)')
    print(f'  Micro F1:    {micro_f1:.4f}  (= accuracy; dominated by majority class)')
    print(f'  Weighted F1: {weighted_f1:.4f}  (weighted by support)')
    print(f'  Cohen kappa: {kappa:.4f}  (0=random, 1=perfect; accounts for chance)')
    """),

    md(r"""
    ### 5f — Regression metrics from scratch
    """),

    code(r"""
    # 5f. Regression evaluation: MAE, MSE, RMSE, MAPE, R-squared.

    def regression_metrics(y_true, y_pred):
        n = len(y_true)
        mae  = float(np.mean(np.abs(y_pred - y_true)))
        mse  = float(np.mean((y_pred - y_true) ** 2))
        rmse = float(np.sqrt(mse))
        mape = float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-9)))) * 100
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ev = 1 - np.var(y_true - y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0.0
        return dict(MAE=mae, MSE=mse, RMSE=rmse, MAPE_pct=mape, R2=r2, ExplainedVariance=float(ev))

    # Simulate house price prediction.
    y_true_reg = rng.normal(500_000, 80_000, 500)
    y_pred_reg = y_true_reg * (1 + rng.normal(0, 0.08, 500))   # ~8% error

    metrics_reg = regression_metrics(y_true_reg, y_pred_reg)
    print('Regression metrics (house price prediction):')
    for name, val in metrics_reg.items():
        if 'MAE' in name or 'MSE' in name or 'RMSE' in name:
            print(f'  {name:18s}: ${val:,.0f}')
        elif 'MAPE' in name:
            print(f'  {name:18s}: {val:.2f}%')
        else:
            print(f'  {name:18s}: {val:.4f}')

    print(f'\n  Interpretation: predictions are off by ${metrics_reg["RMSE"]:,.0f} on average (RMSE)')
    print(f'  The model explains {metrics_reg["R2"]*100:.1f}% of price variance (R²)')
    """),

    md(r"""
    ### 5g — Ranking metrics: NDCG@k, MRR, MAP
    """),

    code(r"""
    # 5g. Ranking metrics from scratch.

    def dcg_at_k(relevances, k):
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))

    def ndcg_at_k(ranked_relevances, k):
        actual_dcg = dcg_at_k(ranked_relevances, k)
        ideal_dcg  = dcg_at_k(sorted(ranked_relevances, reverse=True), k)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def mrr(ranked_relevant_lists):
        # ranked_relevant_lists: list of lists of 1/0 for each query.
        rr_scores = []
        for ranked in ranked_relevant_lists:
            rr = 0.0
            for i, rel in enumerate(ranked):
                if rel == 1:
                    rr = 1.0 / (i + 1)
                    break
            rr_scores.append(rr)
        return float(np.mean(rr_scores))

    def average_precision(ranked_relevances):
        n_relevant = sum(ranked_relevances)
        if n_relevant == 0:
            return 0.0
        precs_at_k = []
        n_found = 0
        for i, rel in enumerate(ranked_relevances):
            if rel == 1:
                n_found += 1
                precs_at_k.append(n_found / (i + 1))
        return float(np.mean(precs_at_k)) if precs_at_k else 0.0

    def mean_average_precision(ranked_relevances_list):
        return float(np.mean([average_precision(r) for r in ranked_relevances_list]))

    # Simulated RAG retrieval results: ranked doc relevances for 5 queries.
    queries_ranking = [
        [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],   # query 1: relevant docs at positions 1,3,5,8
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1],   # query 2: relevant at 2,5,10
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # query 3: relevant at 1,2 (good)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # query 4: relevant only at 4 (poor)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # query 5: relevant only at 10 (bad)
    ]

    print('Ranking evaluation (RAG retrieval, k=5):')
    print(f'\n  {"Query":8s} {"NDCG@5":10s} {"AP":10s}')
    for i, rels in enumerate(queries_ranking):
        nd = ndcg_at_k(rels, k=5)
        ap = average_precision(rels)
        print(f'  Q{i+1}       {nd:.4f}     {ap:.4f}')
    print(f'\n  MRR:  {mrr(queries_ranking):.4f}  (mean reciprocal rank of first relevant doc)')
    print(f'  MAP:  {mean_average_precision(queries_ranking):.4f}  (mean average precision)')
    overall_ndcg5 = float(np.mean([ndcg_at_k(r, 5) for r in queries_ranking]))
    print(f'  NDCG@5: {overall_ndcg5:.4f}  (normalised DCG at k=5)')
    """),

    md(r"""
    ### 5h — Optimal threshold from business cost matrix
    """),

    code(r"""
    # 5h. Business cost-aware optimal threshold.

    def business_cost(y_true, y_score, threshold, cost_fp, cost_fn):
        tp, fp, tn, fn = confusion_matrix_at_threshold(y_true, y_score, threshold)
        return cost_fp * fp + cost_fn * fn

    # Credit risk: FP (approve bad loan) = $10,000 loss; FN (reject good customer) = $500 loss.
    COST_FP = 10_000
    COST_FN = 500

    thresholds_t = np.linspace(0.01, 0.99, 200)
    costs = [business_cost(y_true, y_score, t, COST_FP, COST_FN) for t in thresholds_t]
    optimal_idx = int(np.argmin(costs))
    optimal_thresh = thresholds_t[optimal_idx]
    optimal_cost = costs[optimal_idx]

    cost_at_05 = business_cost(y_true, y_score, 0.5, COST_FP, COST_FN)
    tp_opt, fp_opt, tn_opt, fn_opt = confusion_matrix_at_threshold(y_true, y_score, optimal_thresh)

    print(f'Business cost optimisation (FP=${COST_FP:,}, FN=${COST_FN:,}):')
    print(f'  Optimal threshold: {optimal_thresh:.3f}')
    print(f'  Cost at θ=0.5:    ${cost_at_05:,.0f}')
    print(f'  Cost at θ*:       ${optimal_cost:,.0f}')
    print(f'  Cost saved:       ${cost_at_05 - optimal_cost:,.0f}')
    m_opt = all_metrics(tp_opt, fp_opt, tn_opt, fn_opt)
    print(f'  At θ*: Precision={m_opt["precision"]:.3f}, Recall={m_opt["recall"]:.3f}')
    print(f'\n  Note: θ* < 0.5 because FP is 20x more costly than FN')
    print(f'  → model is more conservative (predicts positive more often)')
    """),

    md(r"""
    ## 6 · Visualization
    """),

    code(r"""
    # Figure 1 — ROC curve and PR curve side by side.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC.
    axes[0].plot(fprs, tprs, color='steelblue', lw=2, label=f'ROC (AUC={auc_roc:.3f})')
    axes[0].plot([0,1],[0,1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
    axes[0].fill_between(fprs, tprs, alpha=0.1, color='steelblue')
    axes[0].set_xlabel('False Positive Rate (FPR)'); axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].set_title('Figure 1a — ROC Curve'); axes[0].legend()

    # PR.
    axes[1].plot(recalls_pr, precisions_pr, color='seagreen', lw=2, label=f'PR (AUC={auc_pr:.3f})')
    axes[1].axhline(baseline_pr, color='k', ls='--', alpha=0.4, label=f'Random (AP={baseline_pr:.3f})')
    axes[1].fill_between(recalls_pr, precisions_pr, alpha=0.1, color='seagreen')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title('Figure 1b — Precision-Recall Curve'); axes[1].legend()

    plt.suptitle('Figure 1 — ROC and PR curves for credit risk model (8% positive class)')
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 1.** ROC (left) and Precision-Recall (right) curves for the credit default
    model. **ROC AUC = 0.9+**: the model ranks positives above negatives with high accuracy.
    **PR AUC**: far above the random baseline (8% prevalence), showing the model is useful
    despite imbalance. Key distinction: the ROC curve can look impressive even when the
    model performs poorly on the minority class (because TN/negatives dominate FPR). The
    PR curve focuses only on the positive class — it degrades visibly when the minority
    class is hard to identify. For fraud detection, cancer screening, and credit default
    (all rare positive class), **PR curve is the primary evaluation tool**.
    """),

    code(r"""
    # Figure 2 — Reliability diagram (calibration plot).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Calibration.
    valid_bins = ~np.isnan(bin_accs)
    axes[0].bar(bin_confs[valid_bins], bin_accs[valid_bins], width=0.08,
                alpha=0.7, color='steelblue', label='Actual accuracy')
    axes[0].plot([0,1],[0,1], 'k--', label='Perfect calibration', alpha=0.7)
    axes[0].set_xlabel('Mean predicted probability (confidence)')
    axes[0].set_ylabel('Actual fraction positive (accuracy)')
    axes[0].set_title(f'Figure 2a — Reliability Diagram (ECE={ece_val:.3f})')
    axes[0].legend()

    # Business cost vs threshold.
    axes[1].plot(thresholds_t, [c/1000 for c in costs], color='coral', lw=2)
    axes[1].axvline(optimal_thresh, color='seagreen', ls='--', lw=2,
                    label=f'Optimal θ={optimal_thresh:.2f}')
    axes[1].axvline(0.5, color='gray', ls=':', label='Default θ=0.5')
    axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Total business cost ($K)')
    axes[1].set_title('Figure 2b — Business cost vs. threshold')
    axes[1].legend()

    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 2a — Reliability diagram.** Each bar shows the actual accuracy within a
    confidence bin. Bars aligned with the diagonal (dashed) indicate perfect calibration.
    Bars above the diagonal: the model is underconfident (predictions of 0.6 are correct
    70% of the time — should output 0.7). Bars below: overconfident. ECE quantifies the
    overall gap. For risk scoring, calibration matters: a credit officer who receives
    "probability of default = 0.8" needs to know that means 80% of such applicants default,
    not 50%.

    **Figure 2b — Business cost vs. threshold.** Total cost ($FP × $10K + $FN × $500$)
    across thresholds. The optimal threshold is significantly below 0.5 because FP
    (approving bad loans) costs 20× more than FN (rejecting good customers). The model
    should be conservative — only approve applicants with very low predicted default risk.
    """),

    code(r"""
    # Figure 3 — Comparison of metrics under different class imbalances.
    imbalance_rates = [0.5, 0.3, 0.15, 0.08, 0.03, 0.01]
    results_imb = {'accuracy': [], 'f1': [], 'mcc': [], 'auc_roc': [], 'auc_pr': []}

    for rate in imbalance_rates:
        yt, ys = generate_credit_data(n=2000, positive_rate=rate)
        tp_i, fp_i, tn_i, fn_i = confusion_matrix_at_threshold(yt, ys, 0.5)
        m_i = all_metrics(tp_i, fp_i, tn_i, fn_i)
        fprs_i, tprs_i = roc_curve_scratch(yt, ys)
        recs_i, precs_i = pr_curve_scratch(yt, ys)
        results_imb['accuracy'].append(m_i['accuracy'])
        results_imb['f1'].append(m_i['f1'])
        results_imb['mcc'].append(m_i['mcc'])
        results_imb['auc_roc'].append(auc_trapezoidal(fprs_i, tprs_i))
        results_imb['auc_pr'].append(auc_trapezoidal(recs_i, precs_i))

    fig, ax = plt.subplots(figsize=(10, 5))
    labels_x = [f'{r:.0%}' for r in imbalance_rates]
    x_pos = range(len(imbalance_rates))
    for metric, color in [('accuracy','coral'), ('f1','steelblue'), ('mcc','seagreen'), ('auc_roc','purple'), ('auc_pr','orange')]:
        ax.plot(x_pos, results_imb[metric], 'o-', label=metric, color=color)
    ax.set_xticks(x_pos); ax.set_xticklabels(labels_x)
    ax.set_xlabel('Positive class rate (imbalance level)')
    ax.set_ylabel('Metric value'); ax.set_ylim(0, 1.05)
    ax.set_title('Figure 3 — Metric sensitivity to class imbalance')
    ax.legend(fontsize=9); plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 3.** How metrics behave as class imbalance increases (left = balanced,
    right = 1% positive). **Accuracy** (red) stays artificially high as imbalance grows —
    the "always predict negative" baseline scores 99% at 1% positives. **AUC-ROC** (purple)
    is relatively stable — it measures rank-ordering ability, which is less affected by
    imbalance at the distribution level. **AUC-PR** (orange) degrades visibly at high
    imbalance — correctly reflecting that finding rare positives is hard. **MCC** (green)
    and **F1** (blue) are honest: they drop as the task gets harder. Lesson: at imbalance
    ratios below 10%, never report accuracy alone; use AUC-PR + MCC + F1.
    """),

    code(r"""
    # Figure 4 — NDCG@k vs MRR vs MAP comparison across retrieval systems.
    systems = {
        'BM25':       [[0,1,0,1,0,1,0,0,0,0], [1,0,0,0,1,0,0,0,0,0], [0,0,1,0,0,0,1,0,0,0]],
        'Dense':      [[1,0,1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0,0,0], [1,0,0,1,0,0,0,0,0,0]],
        'Hybrid':     [[1,1,0,1,0,0,0,0,0,0], [1,0,1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0,0]],
        'Reranked':   [[1,1,1,0,0,0,0,0,0,0], [1,1,0,0,0,0,0,0,0,0], [1,1,1,0,0,0,0,0,0,0]],
    }
    k_vals = [1, 3, 5, 10]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors_sys = ['coral', 'steelblue', 'seagreen', 'gold']
    for (sname, queries_s), color in zip(systems.items(), colors_sys):
        ndcgs = [np.mean([ndcg_at_k(q, k) for q in queries_s]) for k in k_vals]
        axes[0].plot(k_vals, ndcgs, 'o-', label=sname, color=color)
    axes[0].set_xlabel('k'); axes[0].set_ylabel('NDCG@k')
    axes[0].set_title('Figure 4a — NDCG@k by system'); axes[0].legend(fontsize=9)

    mrrs = {n: mrr(q) for n, q in systems.items()}
    maps = {n: mean_average_precision(q) for n, q in systems.items()}
    x_s = range(len(systems))
    axes[1].bar(x_s, list(mrrs.values()), color=colors_sys, alpha=0.8)
    axes[1].set_xticks(x_s); axes[1].set_xticklabels(list(systems.keys()))
    axes[1].set_ylabel('MRR'); axes[1].set_title('Figure 4b — MRR by system')
    axes[2].bar(x_s, list(maps.values()), color=colors_sys, alpha=0.8)
    axes[2].set_xticks(x_s); axes[2].set_xticklabels(list(systems.keys()))
    axes[2].set_ylabel('MAP'); axes[2].set_title('Figure 4c — MAP by system')

    plt.suptitle('Figure 4 — Ranking metrics: BM25 vs Dense vs Hybrid vs Reranked')
    plt.tight_layout(); plt.show()
    """),

    md(r"""
    **Figure 4.** Ranking metrics across four retrieval systems. **BM25** performs worst —
    keyword matching misses semantic relevance. **Dense** (embedding retrieval) improves by
    finding semantically related documents. **Hybrid** (BM25 + Dense with RRF) further
    improves, especially at low k where position quality matters most. **Reranked** (cross-encoder
    reranking) achieves best performance at k=1, 3, 5 — it optimises for precise ordering.
    Key: NDCG@1 weights the first position heavily; NDCG@10 is more lenient. Use NDCG@3
    for search engines (users rarely look past page 1), NDCG@10 for RAG pipelines (the
    retriever feeds 5–10 docs to the reader). MRR measures "where is the first hit" —
    critical for question answering.
    """),

    md(r"""
    ## 7 · Failure Modes

    | Failure | Symptom | Root cause | Mitigation |
    |---|---|---|---|
    | **Accuracy on imbalanced data** | Model reports 99% accuracy but misses all positives | Majority class dominates | Always check class distribution first; use PR/MCC/F1 |
    | **AUC-ROC misinterpretation** | Model appears great on ROC but useless in practice | AUC-ROC doesn't reflect decision threshold | Report precision/recall at the operating threshold, not just AUC |
    | **Threshold not tuned** | Default 0.5 used; business cost ignored | Developer uses training threshold for production | Sweep thresholds with business cost function; tune on validation set |
    | **Calibration neglected** | Probabilities used in downstream decisions but uncalibrated | Model outputs scores, not probabilities | Apply Platt scaling or isotonic regression; evaluate with ECE |
    | **Data leakage in evaluation** | Evaluation metrics too good; production worse | Test set contaminated by training data | Strict train/val/test split; no future data in features |
    | **Wrong k for NDCG** | NDCG@100 looks great but user never scrolls past position 5 | k not aligned with user behaviour | Set k to match actual user consumption (k=3 for most search, k=5–10 for RAG) |
    """),

    md(r"""
    ## 8 · Production Library Implementation
    """),

    code(r"""
    # 8.1 sklearn metrics (guarded).
    try:
        from sklearn.metrics import (
            classification_report, roc_auc_score, average_precision_score,
            brier_score_loss, cohen_kappa_score, mean_absolute_error,
            mean_squared_error, r2_score
        )
        print('sklearn available — production usage:')
        print(f'  roc_auc_score:           {roc_auc_score(y_true, y_score):.4f}')
        print(f'  average_precision_score: {average_precision_score(y_true, y_score):.4f}')
        print(f'  brier_score_loss:        {brier_score_loss(y_true, y_score):.4f}')
        print(f'  r2_score:                {r2_score(y_true_reg, y_pred_reg):.4f}')
    except ImportError:
        lines = [
            '[sklearn not installed — production patterns]:',
            '  from sklearn.metrics import (roc_auc_score, average_precision_score,',
            '      brier_score_loss, cohen_kappa_score, classification_report)',
            '  # ROC AUC: roc_auc_score(y_true, y_score)',
            '  # PR AUC:  average_precision_score(y_true, y_score)',
            '  # Brier:   brier_score_loss(y_true, y_prob)',
            '  # Kappa:   cohen_kappa_score(y_true, y_pred)',
            '  # Full report: print(classification_report(y_true, y_pred))',
        ]
        print('\n'.join(lines))
    """),

    md(r"""
    ## 9 · Realistic Business Case Study — Credit Risk Model Evaluation

    **Scenario.** A consumer lending company deploys a credit scoring model that decides
    whether to approve personal loan applications. They need a rigorous evaluation framework
    to choose between three candidate models (A, B, C) before production deployment.

    **Business constraints:**
    - False Positive (approve bad loan): expected loss = $8,000.
    - False Negative (reject good applicant): opportunity cost = $400.
    - Positive class: 6% of applicants default (imbalanced).
    - Regulatory requirement: report Gini coefficient (= 2 × AUC - 1) and calibration.

    **Evaluation protocol:**
    1. **Primary metric**: AUC-PR (imbalanced, focus on positives).
    2. **Threshold selection**: business cost minimisation (FP cost = 20× FN cost).
    3. **Calibration check**: ECE < 0.05 required (probabilities used in risk scoring).
    4. **Regulatory reporting**: Gini coefficient, KS statistic (max separation of score distributions).
    5. **Fairness check**: AUC and approval rate by protected group (age, gender, region).

    **Result**: Model C has the highest AUC-PR (0.72 vs. 0.65 for A and B) but is
    poorly calibrated (ECE=0.12). Model B has AUC-PR=0.68 and ECE=0.03. Decision:
    deploy Model B with calibration (Platt scaling) applied. Model C is sent for
    additional calibration tuning.
    """),

    md(r"""
    ## 10 · Production Considerations

    - **Evaluation data freshness.** Models decay over time; evaluation on stale test
      data is misleading. Maintain a rolling test set: the last 60 days of labelled data.
      Re-evaluate every 2 weeks.
    - **Temporal ordering.** In time-series data (credit, demand forecasting), always
      split chronologically (train on past, test on future). Random split leaks future
      data into training → inflated metrics.
    - **Threshold versioning.** The optimal threshold changes as class distribution shifts
      (more defaults in a recession). Monitor the threshold; alert if the optimal threshold
      drifts > 0.05 from the deployed value.
    - **Evaluation in shadow mode.** Before deploying a new model, run it in shadow mode
      (predictions logged but not acted on). Compare shadow predictions to deployed model's
      predictions on live data to validate evaluation-to-production transfer.
    - **Multiple metrics dashboard.** Never report a single metric. Production dashboard:
      AUC-ROC, AUC-PR, ECE, precision and recall at deployed threshold, MCC, business cost
      per week. Alert on any metric below a defined SLA.
    """),

    md(r"""
    ## 11 · Tradeoff Analysis

    **Metric selection guide:**

    | Task | Imbalanced? | Primary metric | Secondary |
    |---|---|---|---|
    | Binary classification | Yes (< 10% positive) | AUC-PR, MCC | F1 at tuned threshold |
    | Binary classification | No | AUC-ROC | F1, accuracy |
    | Multi-class | Balanced | Macro F1 | Cohen's kappa |
    | Multi-class | Imbalanced | Weighted F1, per-class F1 | MCC |
    | Probability calibration | Any | Brier score, ECE | Reliability diagram |
    | Regression | Any | RMSE (error in units), MAE (median-like) | R², MAPE |
    | Ranking / retrieval | Any | NDCG@k (position-aware) | MRR (first hit), MAP |
    | Business optimisation | Any | Cost at optimal threshold | Precision and recall at θ* |
    """),

    md(r"""
    ## 12 · Senior-Level Interview Preparation

    **Common questions**
    - *"When would you use PR curve instead of ROC curve?"* → When the positive class is
      rare (< 15%): fraud detection, cancer screening, credit default. ROC AUC can be
      misleadingly high even when precision on the positive class is terrible. PR focuses
      on the minority class and degrades visibly when the model struggles.
    - *"What is MCC and why is it better than F1 for imbalanced data?"* → Matthews
      Correlation Coefficient uses all four cells of the confusion matrix (TP, FP, TN, FN).
      F1 ignores TN entirely — a model that predicts all positive gets perfect recall and
      high F1 if precision is reasonable. MCC penalises this correctly. MCC = 0 means
      random performance regardless of class distribution.

    **Deep-dive questions**
    - *"How do you find the optimal decision threshold for a credit risk model?"* → (1) Define
      the cost matrix: cost_FP = bad loan loss, cost_FN = opportunity cost. (2) For each
      threshold θ, compute total cost = cost_FP × FP(θ) + cost_FN × FN(θ). (3) Pick
      θ* = argmin total cost. (4) Validate on held-out set. This is always better than
      0.5 when FP and FN costs differ, which is almost always in practice.
    - *"What is ECE and why does it matter for production ML?"* → Expected Calibration Error:
      the weighted average gap between predicted probability and actual accuracy, bucketed
      by probability range. A model with ECE=0.10 means predictions of 0.7 are correct
      only 60% of the time. This matters when probability outputs feed downstream decisions
      (risk scoring, medical triage) — miscalibrated probabilities lead to wrong decisions.

    **Common mistakes:** using default 0.5 threshold (almost always wrong for imbalanced
    data); using accuracy as the sole metric; not checking calibration; using future data
    in evaluation (temporal leakage).
    """),

    md(r"""
    ## 13 · Teach-Back — Answer Without Notes

    1. **Confusion matrix.** Draw the 2×2 matrix. Define TP, FP, FN, TN. Derive the
       formula for F1.
    2. **ROC vs PR.** Dataset: 1% positives, 99% negatives. Which curve is more informative
       for this use case? Why?
    3. **AUC interpretation.** What does AUC-ROC = 0.85 mean in English?
    4. **MCC formula.** Write it. What does MCC = 0 mean? MCC = 1?
    5. **Calibration.** A model outputs 0.8 probability for 100 predictions; only 60 are
       correct. Is the model overconfident or underconfident? What is the ECE contribution
       from this bin?
    6. **Threshold optimisation.** FP costs $5,000; FN costs $200. Should θ* be above or
       below 0.5? Why?
    7. **NDCG@k.** What does the $\log_2(i+1)$ denominator accomplish? Why is NDCG@1
       harsher than NDCG@10?
    8. **Macro vs weighted F1.** Class A: F1=0.9, 90 samples. Class B: F1=0.4, 10 samples.
       Calculate macro F1 and weighted F1. Which is more informative and when?
    """),

    md(r"""
    ## 14 · Exercises

    **Beginner (conceptual)**
    1. A model on a 5% positive class dataset achieves: Accuracy=97%, F1=0.45, AUC-PR=0.62.
       Which metric best represents model quality? What does Accuracy=97% not tell you?
    2. A reliability diagram shows all bars below the diagonal. Is the model overconfident
       or underconfident? What calibration technique would you apply?

    **Beginner → Intermediate (coding)**
    3. Implement `macro_f1` and `weighted_f1` from scratch for N-class classification.
       Verify against `sklearn.metrics.f1_score(average='macro')` and `'weighted'`.
    4. Implement a `cost_optimal_threshold` function that takes `(y_true, y_score, cost_fp,
       cost_fn)` and returns the optimal threshold and total cost. Test: for equal costs,
       does it return approximately 0.5?

    **Intermediate (analysis)**
    5. Compare Brier score vs. ECE across 3 models: a perfectly calibrated model, a Platt-
       scaled SVM (slightly overconfident), and a raw neural net (uncalibrated). Which metric
       catches miscalibration more clearly? When do Brier and ECE disagree?
    6. Generate data with 1% positive class. Compute NDCG@10 on a reranker that perfectly
       reorders the top-100 BM25 results. Now compute NDCG@10 on the BM25 output directly.
       What is the reranker's relative improvement?

    **Senior (design)**
    7. *System design:* design an evaluation framework for a medical imaging AI that detects
       malignant tumours (2% positive rate). Specify: primary metric (with justification),
       calibration requirement, threshold selection process, regulatory reporting requirements,
       fairness checks (by age group, hospital), and production monitoring cadence.
    8. *Interview:* "Our classifier achieves AUC-ROC=0.92 but business is unhappy — revenue
       from the model is below target. What might be wrong, and what would you investigate?"
       (Expected: threshold not optimised for business cost; poor calibration causing mispriced
       risk; distribution shift between train/test and production; wrong positive class definition.)
    """),

    md(r"""
    ---
    ### Summary
    Accuracy is almost always the wrong metric. **AUC-PR + MCC + F1 at tuned threshold**
    for imbalanced classification. **AUC-ROC** for balanced or ranking tasks. **ECE +
    Brier score** for calibration. **RMSE + R²** for regression. **NDCG@k + MRR** for
    ranking/retrieval. Always: tune the decision threshold using the business cost matrix,
    not the default 0.5. Always: evaluate calibration when probabilities feed downstream
    decisions.

    **Next:** `37 · RAG Evaluation` — how to evaluate retrieval-augmented generation
    systems end-to-end: retrieval recall/precision, answer faithfulness, RAGAS framework,
    and context precision/recall.
    """),
]

build("phase7_evaluation/36_classical_ml_evaluation.ipynb", cells)
