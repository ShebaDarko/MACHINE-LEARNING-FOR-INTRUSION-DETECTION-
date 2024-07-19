import unittest
from src.utils.regularizers import l2_regularizer
from tensorflow.keras.regularizers import L2

class TestRegularizers(unittest.TestCase):
    def test_l2_regularizer(self):
        regularizer = l2_regularizer(0.01)
        self.assertIsInstance(regularizer, L2)

if __name__ == '__main__':
    unittest.main()

