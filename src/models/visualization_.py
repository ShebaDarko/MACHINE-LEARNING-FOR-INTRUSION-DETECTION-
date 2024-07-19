from import_libraries import *

# Define your visualization functions here
def plot_evaluation_metrics_by_class(class_names, accuracy, precision, recall, f1_score):
    # Example of a visualization function
    x = np.arange(len(class_names))
    plt.figure(figsize=(12, 8))

    plt.plot(x, accuracy, marker='o', label='Accuracy')
    plt.plot(x, precision, marker='o', label='Precision')
    plt.plot(x, recall, marker='o', label='Recall')
    plt.plot(x, f1_score, marker='o', label='F1-Score')

    # Adding labels, title, and legend
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics by Class')
    plt.xticks(x, class_names, rotation=90)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Other visualization functions can follow here

