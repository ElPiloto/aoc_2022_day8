from absl.testing import absltest

import dataset
import ml_collections

class DatasetTest(absltest.TestCase):

  def setUp(self):
    self.config = ml_collections.ConfigDict()
    self.config.size = 4
    self.config.max_height = 5
    self.rng_seed = 787
    

  def test_runs(self):
    gen = dataset.build_train_data(self.config, self.rng_seed, batch_size=1)
    height, visibility = next(gen)


if __name__ == '__main__':
  absltest.main()

