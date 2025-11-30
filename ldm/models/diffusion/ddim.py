"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from einops import repeat, rearrange

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


def get_noise_cond_gamma_schedule(timesteps, channels=3):
    gammas = np.linspace(
        5.0, 0.1, timesteps, dtype=np.float32
    )
    return torch.from_numpy(gammas / np.sqrt(channels))


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True, noise_cond=False):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
        
        self.noise_cond = noise_cond
        if noise_cond:
        #     cond_betas = get_noise_cond_beta_schedule(self.ddpm_num_timesteps)
        #     cond_betas = np.array(cond_betas, dtype=np.float64)
        #     self.cond_betas = cond_betas
        #     assert len(cond_betas.shape) == 1, "betas must be 1-D"
        #     assert (cond_betas > 0).all() and (cond_betas <= 1).all()

        #     cond_alphas = 1.0 - cond_betas
        #     self.gammas = np.cumprod(cond_alphas, axis=0)
            self.gammas = get_noise_cond_gamma_schedule(self.ddpm_num_timesteps, self.model.channels)
            
    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               change_background=False,
               label_B_ds=None,
               label_A_ds=None,
               content_correlation_scale=0.6,
               with_adain=False,
               x0_style=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose, noise_cond=kwargs["noise_cond"])
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    change_background=change_background,
                                                    label_A_ds=label_A_ds,
                                                    label_B_ds=label_B_ds,
                                                    content_correlation_scale=content_correlation_scale,
                                                    with_adain=with_adain,
                                                    x0_style=x0_style
                                                    )
        return samples, intermediates
    
    # for changeanywhere2
    @torch.no_grad()
    def ddim_sampling(self, cond, shape, label_A_ds=None, label_B_ds=None, content_correlation_scale=0.6, with_adain=False, x0_style=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, change_background=False):      
        
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='ChangeAnywhere2 Sampler based on the DDIM', total=total_steps, disable=True)

        if with_adain:
            assert (x0 is not None) and (x0_style is not None) , "Error: 'x0_style' must be provided when 'with_adain' is True."
            x0 = self.adain(x0, x0_style)    

        i_change = int(total_steps * content_correlation_scale) # TODO: test this
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)            
            
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                if with_adain:
                    img_style = self.model.q_sample(x0_style, ts)
                    img_orig = self.adain(img_orig, img_style)    
                
                # using inversion to adjust the std and mean of img_orig (Consistent with q_sample but time-consuming)
                # if with_adain and i == 0:
                #     img_style, _ = self.encode_ddim(x0_style, conditioning=cond, num_steps=total_steps)
                #     img_orig = self.adain(img_orig, img_style)  
                                
                if change_background and (i >= i_change):
                    p = torch.FloatTensor([1.]).to(device) # 0: copy from x0; 1: random generated
                    background_random_prob = torch.full(shape, float(p)).float().to(device) 
                else:
                    p = torch.FloatTensor([0.]).to(device)
                    background_random_prob = torch.full(shape, float(p)).float().to(device)  
                img = img_orig * mask * (1. - background_random_prob) + img * mask * background_random_prob + (1. - mask) * img
                  
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
        if with_adain:
            img = self.adain(img, x0_style)
            
        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
 
    @torch.no_grad()
    def adain(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        style_mean, style_std = self._calc_mean_std(style_feat)
        content_mean, content_std = self._calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean) / content_std
        return normalized_feat * style_std + style_mean

    @torch.no_grad()
    def cluster_wise_adain(self, content_feat, style_feat, content_cluster_mask, style_cluster_mask):
        assert (content_feat.size()[:2] == style_feat.size()[:2])

        normalized_feat = content_feat.clone()
        num_clusters = torch.unique(content_cluster_mask).long()
        
        for cluster in num_clusters:
            content_mask = (content_cluster_mask == cluster).float()
            style_mask = (style_cluster_mask == cluster).float()
            
            content_mean, content_std = self._calc_mean_std(content_feat, content_mask, is_content=True)
            style_mean, style_std = self._calc_mean_std(style_feat, style_mask)
            
            cluster_normalized_feat = (content_feat - content_mean) / content_std
            cluster_normalized_feat = cluster_normalized_feat * style_std + style_mean
            
            normalized_feat[content_mask.long()] = cluster_normalized_feat[content_mask.long()]
        return normalized_feat

    @torch.no_grad()
    def change_adain(self, content_feat, style_feat, change_mask):
        assert (content_feat.size()[:2] == style_feat.size()[:2])

        normalized_feat = content_feat.clone()
     
        content_mean, content_std = self._calc_mean_std(content_feat, change_mask, is_content=True)
        style_mean, style_std = self._calc_mean_std(style_feat, change_mask, is_content=True)
        
        non_change_normalized_feat = (content_feat - content_mean) / content_std
        non_change_normalized_feat = normalized_feat * style_std + style_mean
        
        normalized_feat[change_mask.long()] = non_change_normalized_feat[change_mask.long()]
        return normalized_feat
    
    @torch.no_grad()
    def _calc_mean_std(self, feat, mask=None, eps=1e-5, is_content=False):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        if mask is not None and (not is_content):
            mask = mask.view(1, -1)
            feat = rearrange(feat, 'b c h w -> c (b h w)')
            mask_sum = mask.sum(dim=1, keepdim=True)
            
            cluster_feat = feat * mask
            feat_mean = (cluster_feat.sum(dim=1, keepdim=True) / (mask_sum + eps))
            feat_var = ((cluster_feat - feat_mean) ** 2 * mask).sum(dim=1, keepdim=True) / (mask_sum + eps)
            feat_std = repeat(feat_var.sqrt(), 'c h -> b c h w', b=N, w=1)
            feat_mean = repeat(feat_mean, 'c h -> b c h w', b=N, w=1)
        elif mask is not None and is_content:
            mask = mask.view(N, 1, -1)
            feat = feat.view(N, C, -1) * mask
            feat_sum = feat.sum(dim=2)
            mask_sum = mask.sum(dim=2)
            feat_mean = (feat_sum / (mask_sum + eps)).view(N, C, 1, 1)
            feat_var = ((feat - feat_mean.view(N, C, -1)) ** 2 * mask).sum(dim=2) / (mask_sum + eps)      
            feat_std = feat_var.sqrt().view(N, C, 1, 1)      
        else:
            feat = feat.view(N, C, -1)
            feat_mean = feat.mean(dim=2).view(N, C, 1, 1)
            feat_var = feat.var(dim=2) + eps
            feat_std = feat_var.sqrt().view(N, C, 1, 1)
        return feat_mean, feat_std

    def _add_cond_noise(self, cond, timesteps):
        noise = torch.randn_like(cond)
        gamma = self.gammas[timesteps].to(cond.device)
        while len(gamma.shape) < len(noise.shape):
            gamma = gamma.unsqueeze(-1)        
        noise = gamma * noise
        noise_cond = cond + noise 
        return noise_cond 

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape, device=None):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        device = device if device is not None else timesteps.device
        res = torch.from_numpy(arr).to(device=device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    @torch.no_grad()
    def ddim_inversion():
        pass
    
    @torch.no_grad()
    def encode_ddim(self, img, num_steps, conditioning=None, unconditional_conditioning=None ,unconditional_guidance_scale=1., \
                    end_step=999, callback_ddim_timesteps=None, img_callback=None):
        
        print(f"Running DDIM inversion with {num_steps} timesteps")
        if num_steps == 999:
            T = 999
            c = T // num_steps
            iterator = tqdm(range(0,T ,c), desc='DDIM Inversion', total= num_steps)
            steps = list(range(0,T + c,c))
        else:
            T = self.ddpm_num_timesteps
            c = T // num_steps
            time_steps= range(1, T, c)
            iterator = tqdm(time_steps, desc='DDIM Inversion',total=num_steps)
            steps = list(range(1,T + c,c))
            steps[-1] = 999

        callback_ddim_timesteps_list = np.flip(make_ddim_timesteps("uniform", callback_ddim_timesteps, self.ddpm_num_timesteps))\
            if callback_ddim_timesteps is not None else np.flip(self.ddim_timesteps)
        for i, t in enumerate(iterator):
            if i > end_step:
                break
            img, pred_x0 = self.reverse_ddim(img, t, t_next=steps[i+1] ,c=conditioning, unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)
            if t in callback_ddim_timesteps_list:
                if img_callback: img_callback(pred_x0, img, t)

        return img, pred_x0

    @torch.no_grad()
    def reverse_ddim(self, x, t,t_next, c=None, quantize_denoised=False, unconditional_guidance_scale=1.,
                     unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
        if c is None:
            e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t_tensor, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_tensor] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond) # TODO:

        alphas = self.model.alphas_cumprod #.flip(0)
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod #.flip(0)
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[t], device=device)
        a_next = torch.full((b, 1, 1, 1), alphas[t_next], device=device) #a_next = torch.full((b, 1, 1, 1), alphas[t + 1], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[t], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_next).sqrt() * e_t
        x_next = a_next.sqrt() * pred_x0 + dir_xt
        return x_next, pred_x0   