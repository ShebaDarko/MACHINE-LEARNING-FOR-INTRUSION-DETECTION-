from import_libraries import *


# Define functions related to evaluation metrics here
def calculate_metrics(confusion_matrix_df, label_names):
    metrics_df = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'False Positive Rate', 'False Negative Rate'])

    for class_label in label_names:
        TP = confusion_matrix_df.loc[class_label, class_label]
        FP = confusion_matrix_df.loc[label_names[label_names != class_label], class_label].sum()
        FN = confusion_matrix_df.loc[class_label, label_names[label_names != class_label]].sum()
        TN = confusion_matrix_df.loc[label_names[label_names != class_label], label_names[label_names != class_label]].sum().sum()

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

    # Fill NaN values with 0
    metrics_df = metrics_df.fillna(0)

    return metrics_df

def plot_comprehensive_metrics(metrics_df):
    # Plotting Comprehensive Evaluation Metrics by Class
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Class', y='Accuracy', data=metrics_df, color='blue', alpha=0.6, label='Accuracy')
    sns.barplot(x='Class', y='Precision', data=metrics_df, color='red', alpha=0.6, label='Precision')
    sns.barplot(x='Class', y='Recall', data=metrics_df, color='green', alpha=0.6, label='Recall')
    sns.barplot(x='Class', y='F1-Score', data=metrics_df, color='purple', alpha=0.6, label='F1-Score')
    plt.xticks(rotation=45)
    plt.title('Comprehensive Evaluation Metrics by Class')
    plt.legend()
    plt.tight_layout()
    plt.show()

