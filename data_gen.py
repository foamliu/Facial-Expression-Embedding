import pickle
import random

import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


def swap(img_1, img_2, img_3, annotation):
    if annotation == 1:
        return img_2, img_3, img_1
    elif annotation == 2:
        return img_1, img_3, img_2
    else:  # annotation == 3
        return img_1, img_2, img_3


class FECDataset(Dataset):
    def __init__(self, split):
        filename = 'data/{}.pkl'.format(split)
        with open(filename, 'rb') as file:
            samples = pickle.load(file)

        self.samples = samples

        self.transformer = data_transforms[split]

    def get_image(self, image_name):
        # full_path = os.path.join(image_folder, image_name)
        img = cv.imread(image_name)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        return img

    def __getitem__(self, i):
        sample = self.samples[i]
        img_1 = self.get_image(sample['image_1'])
        img_2 = self.get_image(sample['image_2'])
        img_3 = self.get_image(sample['image_3'])
        triplet_type = sample['triplet_type']
        assert (triplet_type in ['ONE_CLASS_TRIPLET', 'TWO_CLASS_TRIPLET', 'THREE_CLASS_TRIPLET'])
        # annotation = int(round(sample['annotation'].mean()))
        annotation = sample['annotation']
        assert (annotation in [1, 2, 3])

        img_1, img_2, img_3 = swap(img_1, img_2, img_3, annotation)

        if random.random() > 0.5:
            return img_1, img_2, img_3, 0.
        else:
            return img_2, img_1, img_3, 0.

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train = FECDataset('train')
    print('num_train: ' + str(len(train)))
    valid = FECDataset('valid')
    print('num_valid: ' + str(len(valid)))

    print(train[0])
    print(valid[0])
