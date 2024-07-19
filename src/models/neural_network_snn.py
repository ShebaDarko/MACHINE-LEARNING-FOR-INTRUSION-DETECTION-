from import_libraries import *


# Load the pre-trained neural network model
model = Sequential()
model.add(Dense(10, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Number of units matches the number of classes
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with systematic epochs
epochs = 19
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Make predictions using the trained model
pred_probabilities = model.predict(x_test)

# Convert predicted probabilities to class labels
pred_labels = label_encoder.inverse_transform(pred_probabilities.argmax(axis=1))

# Create a confusion matrix
confusion_matrix_data = confusion_matrix(y_test_labels, pred_labels, labels=label_names)
confusion_matrix_df = pd.DataFrame(confusion_matrix_data, index=label_names, columns=label_names)

