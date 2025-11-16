GENVIDEO_PIKA = {
    "real": {
        "train": ["Kinetics-400"],
        "val":   ["Kinetics-400"],
        "test":  ["MSR-VTT"]
    },
    "fake": {
        "train": ["Pika"],
        "val":   ["Pika"],
        "test":  [
            "ModelScope", 
            "MorphStudio",  
            "MoonValley", 
            "HotShot",
            "Show_1",
            "Gen2", 
            "Crafter",
            "Lavie", 
            "Sora", 
            "WildScrape"
        ]
    },
}

GENVIDEO_SEINE = {
    "real": {
        "train": ["Kinetics-400"],
        "val":   ["Kinetics-400"],
        "test":  ["MSR-VTT"]
    },
    "fake": {
        "train": ["SEINE"],
        "val":   ["SEINE"],
        "test":  [
            "ModelScope", 
            "MorphStudio",  
            "MoonValley", 
            "HotShot",
            "Show_1",
            "Gen2", 
            "Crafter",
            "Lavie", 
            "Sora", 
            "WildScrape"
        ]
    },
}

# test only
MYVIDEOS = {
    "real": {
        "test":  ["MSR-VTT"]
    },
    "fake": {
        "test":  [
            # 'AnimateDiff', 
            # 'CogVideoX', 
            # 'FramePack', 
            # 'HunyuanVideo', 
            # 'MAGI-1', 
            'sora2', 
            # 'Veo3.1',
            'Kling',
            'Ray3',
            'Hailuo02',
            # 'Wan2.1'
        ]
    },
}

MYVIDEOS_COMPRESSED = {
    "real": {
        "test":  ["MSR-VTT"]
    },
    "fake": {
        "test":  [
            # 'AnimateDiff', 
            # 'CogVideoX', 
            # 'FramePack', 
            # 'HunyuanVideo', 
            # 'MAGI-1', 
            'sora2', 
            'Kling',
            'Ray3',
            'Hailuo02',
            # 'Wan2.1'
        ]
    },
}