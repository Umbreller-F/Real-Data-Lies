from data.dataset_split import GENVIDEO_PIKA, GENVIDEO_SEINE, MYVIDEOS, MYVIDEOS_COMPRESSED
from data.preprocess import setup_dataset


if __name__ == "__main__":
    setup_dataset(data_path='../Data/myvideos', generation_model=MYVIDEOS['real']['test'][0], label='real', mode='test')
    for gen_model in MYVIDEOS['fake']['test']:
        setup_dataset(data_path='../Data/myvideos', generation_model=gen_model, label='fake', mode='test')
    setup_dataset(data_path='../Data/myvideos_compressed', generation_model=MYVIDEOS_COMPRESSED['real']['test'][0], label='real', mode='test')
    for gen_model in MYVIDEOS_COMPRESSED['fake']['test']:
        setup_dataset(data_path='../Data/myvideos_compressed', generation_model=gen_model, label='fake', mode='test')