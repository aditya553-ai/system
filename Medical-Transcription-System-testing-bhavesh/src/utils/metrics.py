import numpy as np

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    """
    reference = reference.split()
    hypothesis = hypothesis.split()
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=int)

    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,      # deletion
                           d[i][j - 1] + 1,      # insertion
                           d[i - 1][j - 1] + cost)  # substitution

    return d[len(reference)][len(hypothesis)] / len(reference) if len(reference) > 0 else float('inf')

def calculate_entity_f1(precision, recall):
    """
    Calculate F1 score given precision and recall.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def track_metrics(reference_transcripts, generated_transcripts):
    """
    Track and calculate end-to-end accuracy metrics.
    """
    wer_scores = [calculate_wer(ref, hyp) for ref, hyp in zip(reference_transcripts, generated_transcripts)]
    avg_wer = np.mean(wer_scores)

    # Placeholder for entity precision and recall
    entity_precision = 0.0
    entity_recall = 0.0
    avg_f1 = calculate_entity_f1(entity_precision, entity_recall)

    return {
        "average_wer": avg_wer,
        "average_f1": avg_f1
    }