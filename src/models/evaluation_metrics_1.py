
from import_libraries import *

def compute_metrics(y_true, y_pred, label_names):
    metrics_df = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'False Positive Rate', 'False Negative Rate'])
    
    for class_label in label_names:
        TP = np.sum((y_true == class_label) & (y_pred == class_label))
        FP = np.sum((y_true != class_label) & (y_pred == class_label))
        FN = np.sum((y_true == class_label) & (y_pred != class_label))
        TN = np.sum((y_true != class_label) & (y_pred != class_label))
        
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
        specificity = TN / (FP + TN)
        false_positive_rate = FP / (FP + TN)
        false_negative_rate = FN / (TP + FN)
        
        metrics_df = metrics_df.append({
            'Class': class_label,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Specificity': specificity,
            'False Positive Rate': false_positive_rate,
            'False Negative Rate': false_negative_rate
        }, ignore_index=True)
    
    metrics_df = metrics_df.fillna(0)
    return metrics_df

