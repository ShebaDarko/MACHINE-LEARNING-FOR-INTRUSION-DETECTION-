def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(true_labels, pred_labels)
    cr = classification_report(true_labels, pred_labels)
    
    return cm, cr

