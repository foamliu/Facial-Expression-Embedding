import pickle

if __name__ == "__main__":
    with open('data/train.pkl', 'wb') as f:
        train = pickle.load(f)
    with open('data/test.pkl', 'wb') as f:
        test = pickle.load(f)
    print('num_train: ' + str(len(train)))
    print('num_test: ' + str(len(test)))

    print('train[:10]: ' + str(train[:10]))
    print('test[:10]: ' + str(test[:10]))
