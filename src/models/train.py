from import_libraries import *
from data.load_data import load_data
from data.preprocess import preprocess_data
from models.mlp_model import create_mlp_model
from models.snn_model import create_snn_model
from utils import evaluate_model

# Load and preprocess data
filepath = 'path_to_your_dataset.csv'
df = load_data(filepath)

# Preprocess data
x = preprocess_data(df.drop('label', axis=1))
y = LabelEncoder().fit_transform(df['label'])
y = to_categorical(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Create model
input_shape = (x_train.shape[1],)
num_classes = y_train.shape[1]

model = create_mlp_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=19, batch_size=32, validation_split=0.2)

# Evaluate model
cm, cr = evaluate_model(model, x_test, y_test)
print(cm)
print(cr)

