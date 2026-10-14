#@title ▶ Probe helper (written in part 2) { display-mode: "form" }
def probe_auc(E_train, y_train, E_test, y_test, C=0.1):
    """Fit a linear probe (logistic regression) on the training vectors; AUROC on the test vectors.
    Smaller C = stronger regularisation (smaller weights)."""
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
    probe.fit(E_train, y_train)
    return roc_auc_score(y_test, probe.predict_proba(E_test)[:, 1]), probe
