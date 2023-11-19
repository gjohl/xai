from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def calculate_accuracy_metrics(test_labels, predicted_classes, output_probs):
    evaluation_results = {
        'accuracy': accuracy_score(test_labels, predicted_classes),
        'f1': f1_score(test_labels, predicted_classes, average='micro'),
        'auc': roc_auc_score(test_labels, output_probs.detach(), multi_class='ovr'),
        # 'mean_probability': output_probs
    }
    return evaluation_results


def classwise_probability(output_probs, predicted_classes):
    """Calculate the mean and standard deviation of probabilities relative to the predicted classification."""
    pass
