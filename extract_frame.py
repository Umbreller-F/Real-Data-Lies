from data.dataset_split import GENVIDEO_PIKA, GENVIDEO_SEINE, MYVIDEOS, MYVIDEOS_COMPRESSED, TEST100, REALDIST_PIKA, GENVIDEO_Y_PIKA, REALDIST_I_PIKA, REALDIST_U_PIKA
from data.preprocess import dataset_frame_extract
from loguru import logger


if __name__ == "__main__":
    # for mode in ['train', 'val', 'test']:
    #     # dataset_frame_extract(data_path='../Data/GenVideo', generation_model=GENVIDEO_PIKA['real'][mode][0], label='real', mode=mode)
    #     for gen_model in GENVIDEO_PIKA['fake'][mode]:
    #         dataset_frame_extract(data_path='../Data/GenVideo', generation_model=gen_model, label='fake', mode=mode)
    # for mode in ['train', 'val', 'test']:
    #     dataset_frame_extract(data_path='../Data/GenVideo', generation_model=GENVIDEO_SEINE['real'][mode][0], label='real', mode=mode)
    #     for gen_model in GENVIDEO_SEINE['fake'][mode]:
    #         dataset_frame_extract(data_path='../Data/GenVideo', generation_model=gen_model, label='fake', mode=mode)
    
    # # dataset_frame_extract(data_path='../Data/myvideos', generation_model=MYVIDEOS['real']['test'][0], label='real', mode='test', len_load=100)
    # for gen_model in MYVIDEOS['fake']['test']:
    #     dataset_frame_extract(data_path='../Data/myvideos', generation_model=gen_model, label='fake', mode='test')
    
    # dataset_frame_extract(data_path='../Data/myvideos_compressed', generation_model=MYVIDEOS_COMPRESSED['real']['test'][0], label='real', mode='test')
    # for gen_model in MYVIDEOS_COMPRESSED['fake']['test']:
    #     dataset_frame_extract(data_path='../Data/myvideos_compressed', generation_model=gen_model, label='fake', mode='test')

    # for gen_model in TEST100['fake']['test']:
    #     dataset_frame_extract(data_path='../Data/test100', generation_model=gen_model, label='fake', mode='test')
    # dataset_frame_extract(data_path='../Data/test100', generation_model='vsr', label='real', mode='test')

    # for mode in ['train', 'val', 'test']:
    #     for real_model in REALDIST_PIKA['real'][mode]:
    #         dataset_frame_extract(data_path='../Data/RealDist', generation_model=real_model, label='real', mode=mode)
        # for gen_model in REALDIST_PIKA['fake'][mode]:
        #     dataset_frame_extract(data_path='../Data/RealDist', generation_model=gen_model, label='fake', mode=mode)

    # for mode in ['train', 'val', 'test']:
    #     for real_model in GENVIDEO_PIKA['real'][mode]:
    #         dataset_frame_extract(data_path='../Data/GenVideo', generation_model=real_model, label='real', mode=mode)

    # for mode in ['train', 'val', 'test']:
    #     for real_model in GENVIDEO_Y_PIKA['real'][mode]:
    #         dataset_frame_extract(data_path='../Data/GenVideo', generation_model=real_model, label='real', mode=mode)

    # for mode in ['train', 'val', 'test']:
    #     for real_model in REALDIST_I_PIKA['real'][mode]:
    #         logger.info(f'Processing real model: {real_model} for mode: {mode}')
    #         dataset_frame_extract(data_path='../Data/RealDist', generation_model=real_model, label='real', mode=mode)
    #     for gen_model in REALDIST_I_PIKA['fake'][mode]:
    #         logger.info(f'Processing fake model: {gen_model} for mode: {mode}')
    #         dataset_frame_extract(data_path='../Data/RealDist', generation_model=gen_model, label='fake', mode=mode)

    for mode in ['train', 'val', 'test']:
        for real_model in REALDIST_U_PIKA['real'][mode]:
            logger.info(f'Processing real model: {real_model} for mode: {mode}')
            dataset_frame_extract(data_path='../Data/RealDist', generation_model=real_model, label='real', mode=mode)
        for gen_model in REALDIST_U_PIKA['fake'][mode]:
            logger.info(f'Processing fake model: {gen_model} for mode: {mode}')
            dataset_frame_extract(data_path='../Data/RealDist', generation_model=gen_model, label='fake', mode=mode)