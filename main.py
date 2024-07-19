from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.models.snn_model import create_snn_model
from src.models.rnn_model import create_rnn_model
from src.utils.optimizers import get_optimizer
from import_libraries import *


# Load and preprocess data
data_filepath = 'path_to_your_data.csv'
df = load_data(data_filepath)
X = preprocess_data(df.drop('target_column', axis=1))
y = df['target_column']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
input_shape = (X_train.shape[1],)
num_classes = len(y.unique())
model = create_snn_model(input_shape, num_classes)

# Compile model
optimizer = get_optimizer(name='adam', learning_rate=0.001)
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate model
y_pred = np.argmax(model.predict(X_test), axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, cmap='coolwarm')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
plt.show()

