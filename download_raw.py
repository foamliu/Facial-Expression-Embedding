import os
from subprocess import Popen, PIPE

from tqdm import tqdm

download_folder = 'data/FEC_dataset/download'


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def download(tokens, idx, num):
    url = tokens[0].replace('"', '').strip()
    filename = url[url.rfind("/") + 1:].strip()
    fullname = os.path.join(download_folder, filename)
    if not os.path.isfile(fullname):
        process = Popen(["wget", '-N', url, "-P", download_folder], stdout=PIPE)
        (output, err) = process.communicate()
        exit_code = process.wait()


def get_data(split):
    print('downloading {} data...'.format(split))
    anno_file = 'data/FEC_dataset/faceexp-comparison-data-{}-public.csv'.format(split)
    with open(anno_file, 'r') as f:
        lines = f.readlines()
    samples = []
    for i in tqdm(range(len(lines))):
        line = lines[i]
        tokens = line.split(',')
        download(tokens[:5], i, 1)
        download(tokens[5:10], i, 2)
        download(tokens[10:15], i, 3)

    return samples


def main():
    ensure_folder(download_folder)

    get_data('train')
    get_data('test')


if __name__ == '__main__':
    main()
