import unittest
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from models.mlp_model import create_mlp_model

class TestModels(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(100, 42)
        self.y = LabelEncoder().fit_transform(np.random.choice(['normal', 'attack'], 100))
        self.y = to_categorical(self.y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.25)

    def test_mlp_model(self):
        input_shape = (self.x_train.shape[1],)
        num_classes = self.y_train.shape[1]
        model = create_mlp_model(input_shape, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        loss, accuracy = model.evaluate(self.x_test, self.y_test)
        self.assertGreater(accuracy, 0.5)

if __name__ == '__main__':
    unittest.main()

