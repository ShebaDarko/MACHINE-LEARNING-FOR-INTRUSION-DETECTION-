import unittest
from src.models.cnn_model import create_cnn_model

class TestCNNModel(unittest.TestCase):
    def test_create_cnn_model(self):
        input_shape = (28, 28, 1)
        num_classes = 10
        model = create_cnn_model(input_shape, num_classes)
        self.assertEqual(model.layers[0].input_shape[1:], input_shape)
        self.assertEqual(model.layers[-1].output_shape[-1], num_classes)

if __name__ == '__main__':
    unittest.main()

