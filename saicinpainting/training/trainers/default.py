import logging
import torch.nn as nn
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import cv2
from torch.optim import Adam, SGD 

from saicinpainting.training.data.datasets import make_constant_area_crop_params
from saicinpainting.training.losses.distance_weighting import make_mask_distance_weighter
from saicinpainting.training.losses.feature_matching import feature_matching_loss, masked_l1_loss
from saicinpainting.training.modules.fake_fakes import FakeFakesGenerator
from saicinpainting.training.trainers.base import BaseInpaintingTrainingModule, make_multiscale_noise
from saicinpainting.utils import add_prefix_to_keys, get_ramp
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.refinement import  _get_image_mask_pyramid, _pyrdown, _pyrdown_mask, _erode_mask, _l1_loss
from saicinpainting.evaluation.utils import move_to_device
from tqdm import tqdm
import pdb

LOGGER = logging.getLogger(__name__)


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(img_height=batch['image'].shape[2],
                                                                             img_width=batch['image'].shape[3],
                                                                             **kwargs)
    batch['image'] = batch['image'][:, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    batch['mask'] = batch['mask'][:, :, crop_y: crop_y + crop_height, crop_x: crop_x + crop_width]
    return batch


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, concat_mask=True, rescale_scheduler_kwargs=None, image_to_discriminator='predicted_image',
                 add_noise_kwargs=None, noise_fill_hole=False, const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0, fake_fakes_generator_kwargs=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask
        self.rescale_size_getter = get_ramp(**rescale_scheduler_kwargs) if rescale_scheduler_kwargs is not None else None
        self.image_to_discriminator = image_to_discriminator
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole
        self.const_area_crop_kwargs = const_area_crop_kwargs
        self.refine_mask_for_losses = make_mask_distance_weighter(**distance_weighter_kwargs) \
            if distance_weighter_kwargs is not None else None
        self.distance_weighted_mask_for_discr = distance_weighted_mask_for_discr

        self.fake_fakes_proba = fake_fakes_proba
        if self.fake_fakes_proba > 1e-3:
            self.fake_fakes_gen = FakeFakesGenerator(**(fake_fakes_generator_kwargs or {}))

    def forward(self, batch):
        if self.training and self.rescale_size_getter is not None:
            cur_size = self.rescale_size_getter(self.global_step)
            batch['image'] = F.interpolate(batch['image'], size=cur_size, mode='bilinear', align_corners=False)
            batch['mask'] = F.interpolate(batch['mask'], size=cur_size, mode='nearest')

        if self.training and self.const_area_crop_kwargs is not None:
            batch = make_constant_area_crop_batch(batch, **self.const_area_crop_kwargs)

        img = batch['image']
        mask = batch['mask']

        masked_img = img * (1 - mask)

        if self.add_noise_kwargs is not None:
            noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
            masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        batch['predicted_image'] = self.generator(masked_img)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

        if self.fake_fakes_proba > 1e-3:
            if self.training and torch.rand(1).item() < self.fake_fakes_proba:
                batch['fake_fakes'], batch['fake_fakes_masks'] = self.fake_fakes_gen(img, mask)
                batch['use_fake_fakes'] = True
            else:
                batch['fake_fakes'] = torch.zeros_like(img)
                batch['fake_fakes_masks'] = torch.zeros_like(mask)
                batch['use_fake_fakes'] = False

        batch['mask_for_losses'] = self.refine_mask_for_losses(img, batch['predicted_image'], mask) \
            if self.refine_mask_for_losses is not None and self.training \
            else mask

        return batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses.l1.weight_known,
                                  self.config.losses.l1.weight_missing)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # vgg-based perceptual loss
        if self.config.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, img, mask=supervised_mask).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = supervised_mask if self.distance_weighted_mask_for_discr else original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(img)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.config.losses.feature_matching.weight > 0:
            need_mask_in_fm = OmegaConf.to_container(self.config.losses.feature_matching).get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value

        return total_loss, metrics

    def discriminator_loss(self, batch):
        total_loss = 0
        metrics = {}

        predicted_img = batch[self.image_to_discriminator].detach()
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(real_batch=batch['image'],
                                                                               fake_batch=predicted_img,
                                                                               discr_real_pred=discr_real_pred,
                                                                               discr_fake_pred=discr_fake_pred,
                                                                               mask=batch['mask'])
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))


        if batch.get('use_fake_fakes', False):
            fake_fakes = batch['fake_fakes']
            self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=fake_fakes,
                                                         generator=self.generator, discriminator=self.discriminator)
            discr_fake_fakes_pred, _ = self.discriminator(fake_fakes)
            fake_fakes_adv_discr_loss, fake_fakes_adv_metrics = self.adversarial_loss.discriminator_loss(
                real_batch=batch['image'],
                fake_batch=fake_fakes,
                discr_real_pred=discr_real_pred,
                discr_fake_pred=discr_fake_fakes_pred,
                mask=batch['mask']
            )
            total_loss = total_loss + fake_fakes_adv_discr_loss
            metrics['discr_adv_fake_fakes'] = fake_fakes_adv_discr_loss
            metrics.update(add_prefix_to_keys(fake_fakes_adv_metrics, 'adv_'))

        return total_loss, metrics


class RefineInpaintingTrainingModule(DefaultInpaintingTrainingModule):
    def __init__(self, *args, inpainter: nn.Module, modulo : int, n_iters : int, lr : float, min_side : int, 
    max_scales : int, px_budget : int,  **kwargs):
        # Copy all the attributes from DefaultInpaintingTrainingModule to RefineInpaintingTrainingModule
        for attr in inpainter.__dict__:
            setattr(self, attr, getattr(inpainter, attr))
        
        self.refine_modulo = modulo
        self.refine_n_iters = n_iters
        self.refine_lr = lr
        self.refine_min_side = min_side
        self.refine_max_scales = max_scales
        self.refine_px_budget = px_budget
        # separate the generator.model to two parts. 
        self.forward_front = inpainter.generator.model[0:5]
        self.forward_rear = inpainter.generator.model[5:]
    

    def forward(self, batch):

        ls_images, ls_masks = _get_image_mask_pyramid(
            batch, 
            self.refine_min_side, 
            self.refine_max_scales, 
            self.refine_px_budget
        )
        image_inpainted = None
       

        for ids, (image, mask) in enumerate(zip(ls_images, ls_masks)):
            orig_shape = image.shape[2:]
            image = pad_tensor_to_modulo(image, self.refine_modulo)
            mask = pad_tensor_to_modulo(mask, self.refine_modulo)
            mask[mask >= 1e-8] = 1.0
            mask[mask < 1e-8] = 0.0
            image, mask = move_to_device(image, self.device), move_to_device(mask, self.device)
            if image_inpainted is not None:
                image_inpainted = move_to_device(image_inpainted, self.device)
            image_inpainted = self._infer(image, mask, image_inpainted, orig_shape, ids, self.refine_n_iters, self.refine_lr)
            image_inpainted = image_inpainted[:,:,:orig_shape[0], :orig_shape[1]]
            # detach everything to save resources
            image = image.detach().cpu()
            mask = mask.detach().cpu()
        
        return image_inpainted

    def _infer(self, 
        image : torch.Tensor, mask : torch.Tensor, 
        ref_lower_res : torch.Tensor, orig_shape : tuple, 
        scale_ind : int, n_iters : int=15, lr : float=0.002):
        """Performs inference with refinement at a given scale.

        Parameters
        ----------
        image : torch.Tensor
            input image to be inpainted, of size (1,3,H,W)
        mask : torch.Tensor
            input inpainting mask, of size (1,1,H,W) 
        ref_lower_res : torch.Tensor
            the inpainting at previous scale, used as reference image
        orig_shape : tuple
            shape of the original input image before padding
        device : torch.device
            device used for inference.
        scale_ind : int
            the scale index
        n_iters : int, optional
            number of iterations of refinement, by default 15
        lr : float, optional
            learning rate, by default 0.002

        Returns
        -------
        torch.Tensor
            inpainted image
        """
        masked_image = image * (1 - mask)
        masked_image = torch.cat([masked_image, mask], dim=1)

        mask = mask.repeat(1,3,1,1)
        if ref_lower_res is not None:
            ref_lower_res = ref_lower_res.detach()
        with torch.no_grad():
            z1,z2 = self.forward_front(masked_image)
        # Inference
        mask = mask.to(self.device)
        ekernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)).astype(bool)).float()
        ekernel = ekernel.to(self.device)
        image = image.to(self.device)
        z1, z2 = z1.detach(), z2.detach()
        z1.requires_grad, z2.requires_grad = True, True

        optimizer = Adam([z1,z2], lr=lr)

        pbar = tqdm(range(n_iters), leave=False)
        for idi in pbar:
            optimizer.zero_grad()
            input_feat = (z1,z2)
            pred = self.forward_rear(input_feat)
            if ref_lower_res is None:
                break
            losses = {}
            ######################### multi-scale #############################
            # scaled loss with downsampler
            pred_downscaled = _pyrdown(pred[:,:,:orig_shape[0],:orig_shape[1]])
            mask_downscaled = _pyrdown_mask(mask[:,:1,:orig_shape[0],:orig_shape[1]], blur_mask=False, round_up=False)
            mask_downscaled = _erode_mask(mask_downscaled, ekernel=ekernel)
            mask_downscaled = mask_downscaled.repeat(1,3,1,1)
            losses["ms_l1"] = _l1_loss(pred, pred_downscaled, ref_lower_res, mask, mask_downscaled, image, on_pred=True)

            loss = sum(losses.values())
            pbar.set_description("Refining scale {} using scale {} ...current loss: {:.4f}".format(scale_ind+1, scale_ind, loss.item()))
            if idi < n_iters - 1:
                loss.backward()
                optimizer.step()
                del pred_downscaled
                del loss
                del pred
        # "pred" is the prediction after Plug-n-Play module
        inpainted = mask * pred + (1 - mask) * image
        inpainted = inpainted.detach().cpu()
        return inpainted