import unittest
from stl_auto_encoder.dataset import STLDataset
from torch.utils.data import DataLoader


class TestSTLDataset(unittest.TestCase):
    def test_dataset(self):
        datafolder = 'data'
        dataset = STLDataset(datafolder)
        print('length:', len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=10,
            pin_memory=True)
        for x in dataloader:
            print(x.shape)


if __name__ == "__main__":
    unittest.main()
