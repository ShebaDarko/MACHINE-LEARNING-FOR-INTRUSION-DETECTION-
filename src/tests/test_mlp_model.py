
from src.models.mlp_model import create_mlp_model

class TestMLPModel(unittest.TestCase):
    def test_create_mlp_model(self):
        input_shape = (784,)
        num_classes = 10
        model = create_mlp_model(input_shape, num_classes)
        self.assertEqual(model.layers[0].input_shape[1:], input_shape)
        self.assertEqual(model.layers[-1].output_shape[-1], num_classes)

if __name__ == '__main__':
    unittest.main()

