from import_libraries import *

# Assuming you are using a neural network model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, activation='relu', solver='adam')

# Train the model
model.fit(x_train, y_train)

# Make predictions
pred = model.predict(x_test)

# Calculate and print the ROC-AUC score
def multiclass_roc_auc_score(y_test, pred, average="macro"):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test_bin = lb.transform(y_test)
    pred_bin = lb.transform(pred)
    return roc_auc_score(y_test_bin, pred_bin, average=average)

roc_auc = multiclass_roc_auc_score(y_test, pred, average="macro")
print(f'ROC-AUC Score: {roc_auc}')

# Now, create a confusion matrix
label_names = ['smurf', 'neptune', 'normal', 'back', 'satan', 'ipsweep', 'portsweep', 'warezclient', 'teardrop', 'pod', 'nmap', 'guess_passwd', 'butter_overflow', 'land', 'warezmaster', 'imap', 'rootkit', 'loadmodule', 'ftp_write', 'multihop', 'phf', 'perl', 'spy']
y_test_labels = [label_names[int(label)] for label in y_test]
pred_labels = [label_names[int(label)] for label in pred]

def confusion_matrix_func(y_true, y_pred, labels):
    C = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(C, index=labels, columns=labels)

    plt.figure(figsize=(20, 15))
    sns.set(font_scale=1.4)
    sns.heatmap(cm_df, annot=True, annot_kws={"size": 12}, fmt='g', cmap='Blues')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.show()

# Call the function to plot the confusion matrix
confusion_matrix_func(y_test_labels, pred_labels, label_names)

