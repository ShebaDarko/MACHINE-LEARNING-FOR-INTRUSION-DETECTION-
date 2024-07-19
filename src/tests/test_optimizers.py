from src.utils.optimizers import get_optimizer

class TestOptimizers(unittest.TestCase):
    def test_get_optimizer_adam(self):
        optimizer = get_optimizer(name='adam', learning_rate=0.001)
        self.assertEqual(optimizer._name, 'Adam')

    def test_get_optimizer_sgd(self):
        optimizer = get_optimizer(name='sgd', learning_rate=0.001)
        self.assertEqual(optimizer._name, 'SGD')

if __name__ == '__main__':
    unittest.main()

