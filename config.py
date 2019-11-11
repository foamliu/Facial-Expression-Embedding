import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 224

train_anno_file = 'data/FEC_dataset/faceexp-comparison-data-train-public.csv'
test_anno_file = 'data/FEC_dataset/faceexp-comparison-data-test-public.csv'

download_folder = 'data/FEC_dataset/download'
image_folder = 'data/FEC_dataset/images'
