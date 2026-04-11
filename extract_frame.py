from data.preprocess import dataset_frame_extract


if __name__ == "__main__":
    RDL = {
        "real": {
            "train": [
                "Kinetics-400",
                "InternVid-AES",
                "Youku",
                "OpenVidHD",
                "Vript"
            ],
            "val":   [
                "Kinetics-400",
                "InternVid-AES",
                "Youku",
                "OpenVidHD"
            ],
            "test":  [
                "InternVid-AES",
                "RealVSR",
                "MSR-VTT",
                "Youku",
                "Kinetics-400",
                "Vript",
                "HD-VG-130M",
                "OpenVidHD",
                "UltraVideo",
            ]
        },
        "fake": {
            "train": [
                "Pika",
                "OpenSora",
                "DynamicCrafter"
            ],
            "val":   [
                "SEINE",
                # VQA only:
                "Pika",
                # candidate:
                "OpenSora", 
                "SD",
                "SVD",
                "I2VGEN_XL",
                "DynamicCrafter",
                "Latte",
                "VideoCrafter"
            ],
            "test":  [
                "ModelScope", 
                "MorphStudio",  
                "MoonValley",
                "Show_1",
                "Gen2", 
                "Crafter",
                "Lavie", 
                "Sora", 
                "WildScrape",
            ]
        },
    }

    for label in ['real', 'fake']:
        for mode in ['train', 'val', 'test']:
            for data_model in RDL[label][mode]:
                dataset_frame_extract(data_path='../Data/RDL', generation_model=data_model, label=label, mode=mode)