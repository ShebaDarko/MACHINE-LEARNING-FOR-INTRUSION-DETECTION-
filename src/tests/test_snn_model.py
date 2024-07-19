import unittest
from src.models.snn_model import create_snn_model

class TestSNNModel(unittest.TestCase):
    def test_create_snn_model(self):
        input_shape = (784,)
        num_classes = 10
        model = create_snn_model(input_shape, num_classes)
        self.assertEqual(model.layers[0].input_shape[1:], input_shape)
        self.assertEqual(model.layers[-1].output_shape[-1], num_classes)

if __name__ == '__main__':
    unittest.main()

