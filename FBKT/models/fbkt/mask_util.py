import torch
import numpy as np
import torchvision.transforms as transforms

class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=16, model_patch_size=4, mask_ratio=0.75):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        assert self.input_size % self.mask_patch_size == 0
        # assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))


    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.ones(self.token_count, dtype=int)
        un_mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 0
        un_mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.mask_patch_size, axis=0).repeat(self.mask_patch_size, axis=1)
        un_mask = un_mask.reshape((self.rand_size, self.rand_size))
        un_mask = un_mask.repeat(self.mask_patch_size, axis=0).repeat(self.mask_patch_size, axis=1)
        return mask, un_mask

class NewMaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=16, small_input_size=96, small_patch_size=8, model_patch_size=4, mask_ratio=0.75):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size

        self.small_input_size = small_input_size
        self.small_patch_size = small_patch_size

        self.mask_ratio = mask_ratio
        assert self.input_size % self.mask_patch_size == 0
        # assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.small_rand_size = self.small_input_size // self.small_patch_size

        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.small_token_count = self.small_rand_size ** 2

        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        self.small_mask_count = int(np.ceil(self.small_token_count * self.mask_ratio))


    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        small_idx = np.random.permutation(self.small_token_count)[:self.small_mask_count]

        mask = np.ones(self.token_count, dtype=int)
        un_mask = np.zeros(self.token_count, dtype=int)
        small_un_mask = np.zeros(self.small_token_count, dtype=int)

        mask[mask_idx] = 0
        un_mask[mask_idx] = 1
        small_un_mask[small_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.mask_patch_size, axis=0).repeat(self.mask_patch_size, axis=1)
        un_mask = un_mask.reshape((self.rand_size, self.rand_size))
        un_mask = un_mask.repeat(self.mask_patch_size, axis=0).repeat(self.mask_patch_size, axis=1)

        small_un_mask = small_un_mask.reshape((self.small_rand_size, self.small_rand_size))
        small_un_mask = small_un_mask.repeat(self.small_patch_size, axis=0).repeat(self.small_patch_size, axis=1)
        return mask, un_mask, small_un_mask

def mask_transform(images):
    size = images.shape[1:]
    im_size = images.shape[2]
    if im_size == 224:
        mask_patch_size = 16
        mask_ratio = 0.70
    elif im_size == 32:
        mask_patch_size = 2
        mask_ratio = 0.70
    else:
        mask_patch_size = 6
        mask_ratio = 0.70
    mask_generator = MaskGenerator(input_size=im_size, mask_patch_size=mask_patch_size, model_patch_size=4, mask_ratio=mask_ratio)
    mask, unmask = mask_generator()
    # mask = torch.tensor(mask).cuda()
    mask_array = np.array(mask)
    mask_tensor = torch.from_numpy(mask_array)
    mask = mask_tensor.cuda()

    unmask_array = np.array(unmask)
    unmask_tensor = torch.from_numpy(unmask_array)
    unmask = unmask_tensor.cuda()


    expanded_mask = mask.unsqueeze(0).unsqueeze(1).expand_as(images)
    expanded_unmask = unmask.unsqueeze(0).unsqueeze(1).expand_as(images)

    masked_batch_image = expanded_mask * images
    unmasked_batch_image = expanded_unmask * images
    unmasked_batch_image = torch.rot90(unmasked_batch_image, 2, (2, 3))

    return masked_batch_image, unmasked_batch_image

def mask_qk_transform(images_classify, images_q, images_k):
    size = images_q.shape[1:]
    im_size = images_q.shape[2]
    if im_size == 224:
        mask_patch_size = 16
        mask_ratio = 0.70
    elif im_size == 32:
        mask_patch_size = 2
        mask_ratio = 0.70
    else:
        mask_patch_size = 6
        mask_ratio = 0.70
    mask_generator = MaskGenerator(input_size=im_size, mask_patch_size=mask_patch_size, model_patch_size=4, mask_ratio=mask_ratio)
    mask, unmask = mask_generator()
    # mask = torch.tensor(mask).cuda()
    mask_array = np.array(mask)
    mask_tensor = torch.from_numpy(mask_array)
    mask = mask_tensor.cuda()

    unmask_array = np.array(unmask)
    unmask_tensor = torch.from_numpy(unmask_array)
    unmask = unmask_tensor.cuda()

    expanded_mask = mask.unsqueeze(0).unsqueeze(1).expand_as(images_classify)
    expanded_unmask = unmask.unsqueeze(0).unsqueeze(1).expand_as(images_q)

    masked_batch_image = expanded_mask * images_classify

    unmasked_q = expanded_unmask * images_q
    unmasked_k = expanded_unmask * images_k
    unmasked_q = torch.rot90(unmasked_q, 2, (2, 3))
    unmasked_k = torch.rot90(unmasked_k, 2, (2, 3))

    return masked_batch_image, unmasked_q, unmasked_k

def color_jitter(batch_iamges):
    size = batch_iamges.shape[1:]
    im_size = batch_iamges.shape[2]
    jitter = transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2
    )

    weak_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=im_size, scale=(0.8, 1.2)),
    ])

    best_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.6, saturation=0.4, hue=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=im_size, scale=(0.7, 1.3)),
        transforms.RandomGrayscale(p=0.2),
    ])

    strong_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.6, saturation=0.4, hue=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=im_size, scale=(0.7, 1.3)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    jittered_images = torch.stack([best_transform(image) for image in batch_iamges], 1).view(-1, *size).contiguous()

    return jittered_images




