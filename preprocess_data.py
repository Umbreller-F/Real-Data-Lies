from data.dataset_split import GENVIDEO_PIKA, GENVIDEO_SEINE, MYVIDEOS, MYVIDEOS_COMPRESSED
from data.preprocess import setup_dataset, setup_dataset_consecutive


if __name__ == "__main__":
    for mode in ['train', 'val', 'test']:
        setup_dataset_consecutive(data_path='../Data/GenVideo', generation_model=GENVIDEO_PIKA['real'][mode][0], label='real', mode=mode)
        for gen_model in GENVIDEO_PIKA['fake'][mode]:
            setup_dataset_consecutive(data_path='../Data/GenVideo', generation_model=gen_model, label='fake', mode=mode)
    
    setup_dataset_consecutive(data_path='../Data/myvideos', generation_model=MYVIDEOS['real']['test'][0], label='real', mode='test', len_load=100)
    for gen_model in MYVIDEOS['fake']['test']:
        setup_dataset_consecutive(data_path='../Data/myvideos', generation_model=gen_model, label='fake', mode='test')
    
    setup_dataset_consecutive(data_path='../Data/myvideos_compressed', generation_model=MYVIDEOS_COMPRESSED['real']['test'][0], label='real', mode='test', len_load=100)
    for gen_model in MYVIDEOS_COMPRESSED['fake']['test']:
        setup_dataset_consecutive(data_path='../Data/myvideos_compressed', generation_model=gen_model, label='fake', mode='test')