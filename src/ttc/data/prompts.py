CALTECH101_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'a painting of a {name}.',
    lambda name: f'a plastic {name}.',
    lambda name: f'a sculpture of a {name}.',
    lambda name: f'a sketch of a {name}.',
    lambda name: f'a tattoo of a {name}.',
    lambda name: f'a toy {name}.',
    lambda name: f'a rendition of a {name}.',
    lambda name: f'a embroidered {name}.',
    lambda name: f'a cartoon {name}.',
    lambda name: f'a {name} in a video game.',
    lambda name: f'a plushie {name}.',
    lambda name: f'a origami {name}.',
    lambda name: f'art of a {name}.',
    lambda name: f'graffiti of a {name}.',
    lambda name: f'a drawing of a {name}.',
    lambda name: f'a doodle of a {name}.',
    lambda name: f'a photo of the {name}.',
    lambda name: f'a painting of the {name}.',
    lambda name: f'the plastic {name}.',
    lambda name: f'a sculpture of the {name}.',
    lambda name: f'a sketch of the {name}.',
    lambda name: f'a tattoo of the {name}.',
    lambda name: f'the toy {name}.',
    lambda name: f'a rendition of the {name}.',
    lambda name: f'the embroidered {name}.',
    lambda name: f'the cartoon {name}.',
    lambda name: f'the {name} in a video game.',
    lambda name: f'the plushie {name}.',
    lambda name: f'the origami {name}.',
    lambda name: f'art of the {name}.',
    lambda name: f'graffiti of the {name}.',
    lambda name: f'a drawing of the {name}.',
    lambda name: f'a doodle of the {name}.',
]

CIFAR_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'a blurry photo of a {name}.',
    lambda name: f'a black and white photo of a {name}.',
    lambda name: f'a low contrast photo of a {name}.',
    lambda name: f'a high contrast photo of a {name}.',
    lambda name: f'a bad photo of a {name}.',
    lambda name: f'a good photo of a {name}.',
    lambda name: f'a photo of a small {name}.',
    lambda name: f'a photo of a big {name}.',
    lambda name: f'a photo of the {name}.',
    lambda name: f'a blurry photo of the {name}.',
    lambda name: f'a black and white photo of the {name}.',
    lambda name: f'a low contrast photo of the {name}.',
    lambda name: f'a high contrast photo of the {name}.',
    lambda name: f'a bad photo of the {name}.',
    lambda name: f'a good photo of the {name}.',
    lambda name: f'a photo of the small {name}.',
    lambda name: f'a photo of the big {name}.',
]

COUNTRY211_PROMPT = [
    lambda name: f'a photo i took in {name}.',
    lambda name: f'a photo i took while visiting {name}.',
    lambda name: f'a photo from my home country of {name}.',
    lambda name: f'a photo from my visit to {name}.',
    lambda name: f'a photo showing the country of {name}.',
]

DESCRIBABLE_TEXTURES_PROMPT = [
    lambda name: f'a photo of a {name} texture.',
    lambda name: f'a photo of a {name} pattern.',
    lambda name: f'a photo of a {name} thing.',
    lambda name: f'a photo of a {name} object.',
    lambda name: f'a photo of the {name} texture.',
    lambda name: f'a photo of the {name} pattern.',
    lambda name: f'a photo of the {name} thing.',
    lambda name: f'a photo of the {name} object.',
]

EUROSAT_PROMPT = [
    lambda name: f'a centered satellite photo of {name}.',
    lambda name: f'a centered satellite photo of a {name}.',
    lambda name: f'a centered satellite photo of the {name}.',
]

FGVC_PROMPT = [
    lambda name: f'a photo of a {name}, a type of aircraft.',
    lambda name: f'a photo of the {name}, a type of aircraft.',
]

FLOWERS102_PROMPT = [
    lambda name: f'a photo of a {name}, a type of flower.'
]

FMOW_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'{name} in the wild.',
]

FOOD101_PROMPT = [
    lambda name: f'a photo of a {name}, a type of food.'
]

IMAGENET_PROMPT = [
    lambda name: f'a bad photo of a {name}.',
    lambda name: f'a photo of many {name}.',
    lambda name: f'a sculpture of a {name}.',
    lambda name: f'a photo of the hard to see {name}.',
    lambda name: f'a low resolution photo of the {name}.',
    lambda name: f'a rendering of a {name}.',
    lambda name: f'graffiti of a {name}.',
    lambda name: f'a bad photo of the {name}.',
    lambda name: f'a cropped photo of the {name}.',
    lambda name: f'a tattoo of a {name}.',
    lambda name: f'the embroidered {name}.',
    lambda name: f'a photo of a hard to see {name}.',
    lambda name: f'a bright photo of a {name}.',
    lambda name: f'a photo of a clean {name}.',
    lambda name: f'a photo of a dirty {name}.',
    lambda name: f'a dark photo of the {name}.',
    lambda name: f'a drawing of a {name}.',
    lambda name: f'a photo of my {name}.',
    lambda name: f'the plastic {name}.',
    lambda name: f'a photo of the cool {name}.',
    lambda name: f'a close-up photo of a {name}.',
    lambda name: f'a black and white photo of the {name}.',
    lambda name: f'a painting of the {name}.',
    lambda name: f'a painting of a {name}.',
    lambda name: f'a pixelated photo of the {name}.',
    lambda name: f'a sculpture of the {name}.',
    lambda name: f'a bright photo of the {name}.',
    lambda name: f'a cropped photo of a {name}.',
    lambda name: f'a plastic {name}.',
    lambda name: f'a photo of the dirty {name}.',
    lambda name: f'a jpeg corrupted photo of a {name}.',
    lambda name: f'a blurry photo of the {name}.',
    lambda name: f'a photo of the {name}.',
    lambda name: f'a good photo of the {name}.',
    lambda name: f'a rendering of the {name}.',
    lambda name: f'a {name} in a video game.',
    lambda name: f'a photo of one {name}.',
    lambda name: f'a doodle of a {name}.',
    lambda name: f'a close-up photo of the {name}.',
    lambda name: f'a photo of a {name}.',
    lambda name: f'the origami {name}.',
    lambda name: f'the {name} in a video game.',
    lambda name: f'a sketch of a {name}.',
    lambda name: f'a doodle of the {name}.',
    lambda name: f'a origami {name}.',
    lambda name: f'a low resolution photo of a {name}.',
    lambda name: f'the toy {name}.',
    lambda name: f'a rendition of the {name}.',
    lambda name: f'a photo of the clean {name}.',
    lambda name: f'a photo of a large {name}.',
    lambda name: f'a rendition of a {name}.',
    lambda name: f'a photo of a nice {name}.',
    lambda name: f'a photo of a weird {name}.',
    lambda name: f'a blurry photo of a {name}.',
    lambda name: f'a cartoon {name}.',
    lambda name: f'art of a {name}.',
    lambda name: f'a sketch of the {name}.',
    lambda name: f'a embroidered {name}.',
    lambda name: f'a pixelated photo of a {name}.',
    lambda name: f'itap of the {name}.',
    lambda name: f'a jpeg corrupted photo of the {name}.',
    lambda name: f'a good photo of a {name}.',
    lambda name: f'a plushie {name}.',
    lambda name: f'a photo of the nice {name}.',
    lambda name: f'a photo of the small {name}.',
    lambda name: f'a photo of the weird {name}.',
    lambda name: f'the cartoon {name}.',
    lambda name: f'art of the {name}.',
    lambda name: f'a drawing of the {name}.',
    lambda name: f'a photo of the large {name}.',
    lambda name: f'a black and white photo of a {name}.',
    lambda name: f'the plushie {name}.',
    lambda name: f'a dark photo of a {name}.',
    lambda name: f'itap of a {name}.',
    lambda name: f'graffiti of the {name}.',
    lambda name: f'a toy {name}.',
    lambda name: f'itap of my {name}.',
    lambda name: f'a photo of a cool {name}.',
    lambda name: f'a photo of a small {name}.',
    lambda name: f'a tattoo of the {name}.',
]

SIMPLE_IMAGENET_PROMPT = [
    lambda name: f'itap of a {name}.',
    lambda name: f'a bad photo of the {name}.',
    lambda name: f'a origami {name}.',
    lambda name: f'a photo of the large {name}.',
    lambda name: f'a {name} in a video game.',
    lambda name: f'art of the {name}.',
    lambda name: f'a photo of the small {name}.'
]

IWILDCAM_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'{name} in the wild.',
]

OXFORD_IIIT_PETS_PROMPT = [
    lambda name: f'a photo of a {name}, a type of pet.'
]

PCAM_PROMPT = [
    lambda name: f'this is a photo of {name}',
    lambda name: f'a histopathology slide showing {name}',
    lambda name: f'histopathology image of {name}',
]

STANFORDCARS_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'a photo of the {name}.',
    lambda name: f'a photo of my {name}.',
    lambda name: f'i love my {name}!',
    lambda name: f'a photo of my dirty {name}.',
    lambda name: f'a photo of my clean {name}.',
    lambda name: f'a photo of my new {name}.',
    lambda name: f'a photo of my old {name}.',
]

SUN397_PROMPT = [
    lambda name: f'a photo of a {name}.',
    lambda name: f'a photo of the {name}.',
]

UCF101_PROMPT = [
    lambda name: f'a photo of a person {name}.',
    lambda name: f'a photo of a person using {name}.',
    lambda name: f'a photo of a person doing {name}.',
    lambda name: f'a photo of a person during {name}.',
    lambda name: f'a photo of a person performing {name}.',
    lambda name: f'a photo of a person practicing {name}.',
]
