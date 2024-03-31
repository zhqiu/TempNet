import json
import os
import random
import numpy as np

from torch.utils.data import Dataset
import torchvision

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption

from torchvision.datasets import ImageFolder


# from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


class ImageNet100Dataset(ImageFolder):
    def __init__(self, root, transform=None, noise_level=0.0):
        image_path = os.path.join(root, 'train')
        super().__init__(image_path, transform)

        print("Noise level for Imagenet100:", noise_level)
        self.noise_level = noise_level
        self._gen_noise_idx()

        label_path = os.path.join(root, 'Labels.json')
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

    def _gen_noise_idx(self):
        file_name = "noise_sample_idx_"+str(self.noise_level)+".txt"

        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                self.noise_sample_idx = np.loadtxt(f)

        else:
            sample_num = self.__len__()
            self.noise_sample_idx = np.random.choice(range(sample_num), int(sample_num*self.noise_level), replace=False)

            with open(file_name, "w") as f:
                np.savetxt(f, self.noise_sample_idx)

        
    def __getitem__(self, index):
        # Get image and label
        image, label_idx = super().__getitem__(index)

        if index in self.noise_sample_idx:
            class_name = random.choice(self.labels.values().remove(self.labels[self.classes[label_idx]]))
        else:
            class_name = self.labels[self.classes[label_idx]]
        
        # Generate text label
        text = random.choice(imagenet_templates).format(class_name)
        
        return image, text, index, index



class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index, enable_transform=True):    
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['image'])

        image = Image.open(image_path).convert('RGB')   

        if enable_transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']], index
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []

            if type(ann['caption']) == list:  # for coco and flickr datasets
                for i, caption in enumerate(ann['caption']):
                    self.text.append(pre_caption(caption, self.max_words))
                    self.img2txt[img_id].append(txt_id)
                    self.txt2img[txt_id] = img_id
                    txt_id += 1

            elif type(ann['caption']) == str: # for sbu dataset
                self.text.append(pre_caption(ann['caption'], self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

            else:
                assert 0
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index, enable_transform=True):    
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        
        if enable_transform:
            image = self.transform(image)
        else:
            image = torchvision.transforms.ToTensor()(image)

        return image, index
      
