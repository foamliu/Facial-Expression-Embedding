import pickle

import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomCrop(im_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(im_size),
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
        # annotation = int(round(sample['annotation'].mean()))
        annotation = sample['annotation']
        assert (annotation in [1, 2, 3])
        if triplet_type == 'ONE_CLASS_TRIPLET':
            margin = 0.1
        elif triplet_type in ['TWO_CLASS_TRIPLET', 'THREE_CLASS_TRIPLET']:
            margin = 0.2
        else:
            margin = 0.0
        # for a triplet (I1,I2,I3) with the most similar pair (I1,I2).
        anchor, positive, negative = swap(img_1, img_2, img_3, annotation)

        return anchor, positive, negative, margin

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    train = FECDataset('train')
    print('num_train: ' + str(len(train)))
    valid = FECDataset('valid')
    print('num_valid: ' + str(len(valid)))

    print(train[0])
    print(valid[0])
