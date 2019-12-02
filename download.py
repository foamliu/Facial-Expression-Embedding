import os
import pickle
from subprocess import Popen, PIPE

import cv2 as cv
from tqdm import tqdm

from config import download_folder, image_folder, im_size
from utils import ensure_folder


def download(tokens, idx, num):
    url = tokens[0].replace('"', '').strip()
    left = float(tokens[1].strip())
    right = float(tokens[2].strip())
    top = float(tokens[3].strip())
    bottom = float(tokens[4].strip())

    filename = url[url.rfind("/") + 1:].strip()
    fullname = os.path.join(download_folder, filename)
    # if not os.path.isfile(fullname):
    #     process = Popen(["wget", '-N', url, "-P", download_folder], stdout=PIPE)
    #     (output, err) = process.communicate()
    #     exit_code = process.wait()

    filename = '{}_{}.jpg'.format(idx, num)
    filename = os.path.join(image_folder, filename)
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        return filename

    if os.path.isfile(fullname) and os.path.getsize(fullname) > 0:
        img = cv.imread(fullname)
        height, width = img.shape[:2]
        left, right = int(round(left * width)), int(round(right * width))
        top, bottom = int(round(top * height)), int(round(bottom * height))
        img = img[top:bottom, left:right, :]
        img = cv.resize(img, (im_size, im_size))
        cv.imwrite(filename, img)
        return filename

    return None


def get_samples(image_1, image_2, image_3, triplet_type, tokens):
    annotations = []
    for i in range(0, len(tokens), 2):
        # annotator_id = tokens[i]
        annotation = int(tokens[i + 1])
        assert (annotation in [1, 2, 3])
        annotations.append(annotation)

    annotation = int(round(sum(annotations) / len(annotations)))
    assert (annotation in [1, 2, 3])
    sample_list = [{'image_1': image_1, 'image_2': image_2, 'image_3': image_3, 'triplet_type': triplet_type,
                    'annotation': annotation}]
    return sample_list


def get_data(split):
    print('downloading {} data...'.format(split))
    anno_file = 'data/FEC_dataset/faceexp-comparison-data-{}-public.csv'.format(split)
    with open(anno_file, 'r') as f:
        lines = f.readlines()
    samples = []
    for i in tqdm(range(len(lines))):
        line = lines[i]
        tokens = line.split(',')
        image_1 = download(tokens[:5], i, 1)
        image_2 = download(tokens[5:10], i, 2)
        image_3 = download(tokens[10:15], i, 3)
        if image_1 is not None and image_2 is not None and image_3 is not None:
            triplet_type = tokens[15]
            sample_list = get_samples(image_1, image_2, image_3, triplet_type, tokens[16:])
            samples = samples + sample_list
    return samples


def main():
    ensure_folder(download_folder)
    ensure_folder(image_folder)

    train = get_data('train')
    test = get_data('test')

    print('num_train: ' + str(len(train)))
    print('num_test: ' + str(len(test)))

    print('train[:10]: ' + str(train[:10]))
    print('test[:10]: ' + str(test[:10]))

    with open('data/train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open('data/test.pkl', 'wb') as f:
        pickle.dump(test, f)


if __name__ == '__main__':
    main()
