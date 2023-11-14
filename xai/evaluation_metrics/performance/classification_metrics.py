from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# TODO GJ: Tidy up variable names
def calculate_evaluation_metrics(test_labels, predicted_classes, output_probs):
    evaluation_results = {
        'accuracy': accuracy_score(test_labels, predicted_classes),
        'f1': f1_score(test_labels, predicted_classes, average='micro'),
        'auc': roc_auc_score(test_labels, output_probs.detach(), multi_class='ovr')
    }
    return evaluation_results
