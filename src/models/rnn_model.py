
# Define your RNN model training and evaluation here
def train_rnn_model(x_train, y_train, x_test, y_test, label_encoder):
    # Reshape the input data for LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    # Create recurrent neural network model
    model = Sequential()
    model.add(LSTM(60, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Number of units matches the number of classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor], verbose=2, epochs=19)

    # Make predictions
    pred_probabilities = model.predict(x_test)

    # Convert predicted probabilities to class labels
    pred_labels = np.argmax(pred_probabilities, axis=1)

    # Now, create a confusion matrix
    label_names = label_encoder.classes_
    y_test_labels = label_encoder.inverse_transform(y_test.astype(int))
    pred_labels = label_encoder.inverse_transform(pred_labels)

    def confusion_matrix_func(y_true, y_pred, labels):
        C = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(C, index=labels, columns=labels)

        plt.figure(figsize=(20, 15))
        sns.set(font_scale=1.4)
        sns.heatmap(cm_df, annot=True, annot_kws={"size": 12}, fmt='g', cmap='Blues')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.title('')
        plt.show()

    # Call the function to plot the confusion matrix
    confusion_matrix_func(y_test_labels, pred_labels, label_names)

