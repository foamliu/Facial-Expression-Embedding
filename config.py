import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

im_size = 224

train_anno_file = 'data/FEC_dataset/faceexp-comparison-data-train-public.csv'
test_anno_file = 'data/FEC_dataset/faceexp-comparison-data-test-public.csv'

download_folder = 'data/FEC_dataset/download'
image_folder = 'data/FEC_dataset/images'

# Training parameters
num_workers = 8  # for data-loading
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
