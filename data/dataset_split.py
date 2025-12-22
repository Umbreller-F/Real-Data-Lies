GENVIDEO_PIKA = {
    "real": {
        "train": ["Kinetics-400"],
        "val":   ["Kinetics-400"],
        "test":  [
            "MSR-VTT",
            "Youku"
        ]
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

MYVIDEOS_CROPPED = {
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

TEST100 = {
    "real": {
        # "test":  ["MSR-VTT"]
        # "test":  ["Youku"]
        "test":  ["vsr"]
    },
    "fake": {
        "test":  [
            'Hailuo',
            'PixVerse'
        ]
    },
}

TEST50 = {
    "real": {
        "test":  ["vsr"]
    },
    "fake": {
        "test":  [
            'Hailuo',
            'PixVerse'
        ]
    },
}

VAE = {
    "real": {
        "test":  ["MSR-VTT"]
    },
    "fake": {
        "test":  [
            'Wan2.2',
            # 'Hunyuan'
        ]
    },
}

# real data distribution
REALDIST_PIKA = {
    "real": {
        "train": ["LSVQ"],
        "val":   ["LSVQ"],
        "test":  [
            "LSVQ",
            "LSVQ_1080p",
            "InternVid-AES",
            "RealVSR",
            "MSR-VTT",
            "Youku",
            "Kinetics-400"
        ]
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

# real data distribution
REALDIST_I_PIKA = {
    "real": {
        "train": ["InternVid-AES"],
        "val":   ["InternVid-AES"],
        "test":  [
            "LSVQ",
            "LSVQ_1080p",
            "InternVid-AES",
            "RealVSR",
            "MSR-VTT",
            "Youku",
            "Kinetics-400"
        ]
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

# real data distribution
REALDIST_U_PIKA = {
    "real": {
        "train": ["Uniform"],
        "val":   ["Uniform"],
        "test":  [
            "LSVQ",
            "LSVQ_1080p",
            "InternVid-AES",
            "RealVSR",
            "MSR-VTT",
            "Youku",
            "Kinetics-400"
        ]
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

GENVIDEO_Y_PIKA = {
    "real": {
        "train": ["Youku"],
        "val":   ["Youku"],
        "test":  [
            "MSR-VTT",
            "Kinetics-400"
        ]
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