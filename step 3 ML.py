models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGB': XGBClassifier(),
    'Neural Network': MLPClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Extra Tree': ExtraTreesClassifier()
}


# . Prepare plots and results
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()
summary_results = []

# . Train and evaluate models
for i, (title, sampler) in enumerate(samplers.items()):
    ax = axs[i]
    if sampler:
        X_resampled, y_resampled = sampler.fit_resample(x_train_scaled, y_train)
    else:
        X_resampled, y_resampled = x_train_scaled, y_train

    ax.set_title(f'({chr(97+i)}) {title}')
    for name, model in models.items():
        clf = model
        clf = clf.set_params(class_weight='balanced') if title == 'Cost-sensitive learning' and hasattr(clf, 'class_weight') else clf
        if title == 'Cost-sensitive learning' and isinstance(clf, XGBClassifier):
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            clf.set_params(scale_pos_weight=class_weights[1] / class_weights[0])

        clf.fit(X_resampled, y_resampled)
        y_pred = clf.predict(x_test_scaled)
        y_proba = clf.predict_proba(x_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(x_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        summary_results.append({
            'Sampling': title,
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })

    ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True)

plt.tight_layout()
plt.suptitle('ROC curve for different sampling methods', fontsize=16, y=1.03)
plt.subplots_adjust(top=0.92)
plt.savefig('roc.png')
plt.show()

# . Create summary DataFrame
performance_df = pd.DataFrame(summary_results)
performance_df.head(20)
