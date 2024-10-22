import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from ldm.data.base import ImagePaths
# from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

from PIL import Image   
import json 

class TerrainBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None #dict()
        self.keys = None #dict()
        self.augment = kwargs.get('augment', False)
        self.transform = self.getAugmentTransform() if self.augment else None
            

    def getAugmentTransform(self):
        transform = albumentations.Compose([
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=1),
                albumentations.VerticalFlip(p=1),
            ], p=0.5),
            albumentations.RandomRotate90(p=0.5),
            ]
        )
        return transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        if self.transform:
            example['image'] = self.transform(image=example['image'])['image']  #CHECK IF NEW EXAMPLE IMAGE IS DIFFERENT THAN ORIGINAL
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex



class TerrainRGBATrain(TerrainBase):
    def __init__(self, size,augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/textured-terrain"
        with open("data/RGBA_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)
        self.keys = keys

class TerrainRGBAValidation(TerrainBase):
    def __init__(self, size, augment= None,keys=None):
        super().__init__(augment=augment)
        root = "data/textured-terrain"
        with open("data/RGBA_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)
        self.keys = keys

class TerrainRGBATest(TerrainBase):
    def __init__(self, size, augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/textured-terrain"
        with open("data/RGBA_test.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)
        self.keys = keys



######################################################################
#                         Segmentation Set
######################################################################

class TerrainSegmentationBase(Dataset):
    def __init__(self, *args, **kwargs):
        self.data = dict()
        self.keys = dict()
        self.augment = kwargs.get('augment', False)
        self.rgba = kwargs.get('rgba', False)
        self.grayscale = kwargs.get('grayscale', False)
        self.transform = self.getAugmentTransform() if self.augment else None
            

    def getAugmentTransform(self):
        transform = albumentations.Compose([
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=1),
                albumentations.VerticalFlip(p=1),
            ], p=0.5),
            albumentations.RandomRotate90(p=0.5),
            ],
            additional_targets={'segmentation' : 'image'}
        )
        return transform

    def __len__(self):
        return len(self.data['image'])

    def preprocess_images(self, image_path, segment_path):
        image = Image.open(image_path)
        if self.grayscale == True:
            if not image.mode == "L":
                image = image.convert("L")
        elif self.rgba == True:
            if not image.mode == "RGBA":
                image = image.convert("RGBA")
        elif not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)


        #WARNING: Segmentation img presumed to be RGB 
        segmentImg = Image.open(segment_path)
        if not segmentImg.mode == "RGB":
            segmentImg = segmentImg.convert("RGB")
        segmentation = np.array(segmentImg).astype(np.uint8)

        if self.transform:
            processed = self.transform(image=image, segmentation=segmentation)
            image, segmentation = processed["image"], processed["segmentation"]
        image = (image/127.5 - 1.0).astype(np.float32)
        segmentation = (segmentation / 127.5 - 1.0).astype(np.float32)

        return image, segmentation
    
    def __getitem__(self, i):
        img_path = self.data['image'][i]
        seg_path = self.data['segmentation'][i]
        image, segmentation = self.preprocess_images(img_path,seg_path)
        example = {
            "image": image,
            "segmentation": segmentation
        }
        return example


class TerrainSegmentationTrain(TerrainSegmentationBase):
    def __init__(self, size, augment= None, rgba=False, grayscale=False, keys=None):
        super().__init__(augment=augment,rgba=rgba,grayscale=grayscale)
        root = "data/textured-terrain"
        segmentationRoot = "data/RGBA-Mask"
        with open("data/mask_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        segPaths = [os.path.join(segmentationRoot, relpath) for relpath in relpaths]
        self.data['image'] = paths 
        self.data['segmentation'] = segPaths
        self.keys = ['image','segmentation']

    

class TerrainSegmentationValidation(TerrainSegmentationBase):
    def __init__(self, size, augment= None, rgba=False, grayscale=False,keys=None):
        super().__init__(augment=augment,rgba=rgba,grayscale=grayscale)
        root = "data/textured-terrain"
        segmentationRoot = "data/RGBA-Mask"
        with open("data/mask_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        segPaths = [os.path.join(segmentationRoot, relpath) for relpath in relpaths]
        self.data['image'] = paths 
        self.data['segmentation'] = segPaths
        self.keys = ['image','segmentation']


class TerrainSegmentationTest(TerrainSegmentationBase):
    def __init__(self, size, augment= None, rgba=False, grayscale=False, keys=None):
        super().__init__(augment=augment,rgba=rgba,grayscale=grayscale)
        root = "data/textured-terrain"
        segmentationRoot = "data/RGBA-Mask"
        with open("data/mask_test.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        segPaths = [os.path.join(segmentationRoot, relpath) for relpath in relpaths]
        self.data['image'] = paths 
        self.data['segmentation'] = segPaths
        self.keys = ['image','segmentation']




######################################################################
#                         Test Set
######################################################################

        
        
class TestRGBATrain(TerrainBase):
    def __init__(self, size,augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/terrainTinyRGBA"
        with open("data/tinyRGBA_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)
        self.keys = keys

class TestRGBAValidation(TerrainBase):
    def __init__(self, size, augment= None,keys=None):
        super().__init__(augment=augment)
        root = "data/terrainTinyRGBA"
        with open("data/tinyRGBA_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)
        self.keys = keys

class TestRGBATest(TerrainBase):
    def __init__(self, size, augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/terrainTinyRGBA"
        with open("data/tinyRGBA_test.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)
        self.keys = keys




class TerrainGSTrain(TerrainBase):
    def __init__(self, size, augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/terrainGS"
        with open("data/terrainGS_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, grayscale=True)
        self.keys = keys


class TerrainGSValidation(TerrainBase):
    def __init__(self, size, augment= None, keys=None):
        super().__init__(augment=augment)
        root = "data/terrainGS"
        with open("data/terrainGS_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False,grayscale=True)
        self.keys = keys




######################################################################
#                         Caption Set
######################################################################


# from makeCaptions import ShuffleCaptions

class TerrainCaptionBase(TerrainBase):
    def __init__(self, *args, **kwargs):
        super().__init__(kwargs=kwargs)
        self.rgba = kwargs.get('rgba', False)
        self.grayscale = kwargs.get('grayscale', False)
        self.captionFilePath = 'data/mask-caption-v2.json'
        with open(self.captionFilePath) as json_file:
            self.captionDict = json.load(json_file)
            
        self.augment = kwargs.get('augment', False)
        self.transform = self.getAugmentTransform() if self.augment else None
            

    def getAugmentTransform(self):
        transform = albumentations.Compose([
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=1),
                albumentations.VerticalFlip(p=1),
            ], p=0.5),
            albumentations.RandomRotate90(p=0.5),
            ]
        )
        return transform

    
    def __getitem__(self, i):
        example = self.data[i]
        caption = self.captionDict[os.path.basename(example['file_path_'])]
        if self.transform:
            example['image'] = self.transform(image=example['image'])['image']
            
        example['caption'] = caption # ShuffleCaptions.shuffleCaptionLabelOrder(str(caption))
        example["human_label"] = example['caption']
        return example




class TerrainRGBACaptionTrain(TerrainCaptionBase):
    def __init__(self, size,augment= None, keys=None,**kwargs):
        super().__init__(augment=augment, kwargs=kwargs)
        root = "data/textured-terrain"
        with open("data/mask_train.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)
        self.keys = keys

class TerrainRGBACaptionValidation(TerrainCaptionBase):
    def __init__(self, size,augment= None, keys=None,**kwargs):
        super().__init__(augment=augment, kwargs=kwargs)
        root = "data/textured-terrain"
        with open("data/mask_validation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)
        self.keys = keys

class TerrainRGBACaptionTest(TerrainCaptionBase):
    def __init__(self, size,augment= None, keys=None,**kwargs):
        super().__init__(augment=augment, kwargs=kwargs)
        root = "data/textured-terrain"
        with open("data/mask_test.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, rgba=True)
        self.keys = keys
