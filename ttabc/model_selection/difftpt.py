"""
    Diverse Data Augmentation with Diffusions for Effective Test-time Prompt Tuning,
    https://arxiv.org/abs/2308.06038,
    https://github.com/chunmeifeng/DiffTPT
    NOTE: This implementation computes the diffusion augmentations on-the-fly, which is extremely slow and requires a large amount of CUDA memory. We'll release a more efficient, offline-augmentation version in the future.
"""
import time

from PIL import Image

import torch
import torch.optim

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, select_confident_samples, marginal_entropy, select_confident_samples_cosine
from ttabc.model_selection.base_method import BaseMethod
import torchvision.transforms as T

from diffusers import StableDiffusionImageVariationPipeline
from accelerate import Accelerator
# besides: install transformers and xformers

# def _unnormalize_clip(t: torch.Tensor):
#     mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=t.device).view(1,3,1,1)
#     std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=t.device).view(1,3,1,1)
#     return (t * std) + mean

# def _tensor_to_pil_clip(t: torch.Tensor):
#     # t shape: (1,3,H,W) or (3,H,W); assumes normalized by CLIP stats
#     if t.dim() == 4: t = t[0]
#     img = _unnormalize_clip(t.unsqueeze(0))[0].clamp(0,1)
#     img = (img * 255).byte().cpu().permute(1,2,0).numpy()
#     from PIL import Image
#     return Image.fromarray(img)

class DiffTPT(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        assert args.tpt is True, "DiffTPT only works with augmentations"
        self.args = args
        self.temperature_value = {'ViT': 1.16, 'RN': 1.15} #for temperature scaling experiments 
        
        # load diffusion pipeline
        self.diff_pipe = None
        self.diff_pil_transform = T.Compose([
            # pipeline 输出已是 PIL，不需要 resize（保持与输入一致），后续与主流程一致处理
            T.Resize(args.resolution, interpolation=BICUBIC),
            T.CenterCrop(args.resolution),
            T.ToTensor(),
            T.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711])
        ])
        self._init_diffusion_pipe()
        
    def _init_diffusion_pipe(self):
        if self.diff_pipe is not None:
            return

        acc = Accelerator()
        # dtype = torch.float16 if self.args.diff_pipe_fp16 else torch.float32
        # dtype = torch.float32
        dtype = torch.float16
        self.diff_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
            torch_dtype=dtype
        )
        self.diff_pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        self.diff_pipe.enable_attention_slicing()
        self.diff_pipe.enable_xformers_memory_efficient_attention()
        # self.diff_pipe.enable_sequential_cpu_offload()
        self.diff_pipe = self.diff_pipe.to(acc.device)
        self.diff_pipe.set_progress_bar_config(disable=True) # comment out this line to show the progress
        self.diff_accelerator = acc
        print("Diffusion pipe initialized.")
        
    def generate_diff_views(self, ori_img):
        """
        base_view_tensor: shape [1,3,H,W] 
        return: list[torch.Tensor], with each element shape [1,3,H,W]
        """
        # self._init_diffusion_pipe()

        num_views = self.args.diff_aug_size
        guidance = self.args.diff_guidance_scale
        num_inference_steps = self.args.diff_times

        # 利用 num_images_per_prompt 生成多视图 (若不支持则循环)
        with torch.no_grad():
            images = self.diff_pipe(ori_img, guidance_scale=guidance, num_images_per_prompt=num_views, num_inference_steps=num_inference_steps).images
            # except TypeError:
            #     images = []
            #     for _ in range(num_views):
            #         images.extend(self.diff_pipe([pil_img], guidance_scale=guidance).images)

            diff_tensors = []
            for im in images[:num_views]:
                t = self.diff_pil_transform(im).unsqueeze(0)  # (1,3,H,W)
                diff_tensors.append(t)
            torch.cuda.empty_cache()
                
        return diff_tensors

    def test_time_tuning(self, inputs):
        if self.args.prompt_type == 'cocoop':
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            self.optimizer = torch.optim.AdamW([pgen_ctx], self.args.lr)
        
        selected_idx = None
        batch_entropy = None
        for _ in range(self.args.tta_steps):
            with torch.amp.autocast(device_type='cuda'):
                if self.args.prompt_type == 'cocoop':
                    output = self.model((image_feature, pgen_ctx))
                else:
                    output = self.model(inputs) 

                if selected_idx is not None:
                    logits_cos = output[selected_idx[0]]
                    logits = torch.cat((output[0, :].unsqueeze(0), logits_cos), dim=0)
                    output = logits[selected_idx[1]]
                else:
                    output, selected_idx, batch_entropy = select_confident_samples_cosine(output, self.args.selection_cosine, self.args.selection_selfentro)

                loss = marginal_entropy(output)
                
            self.optimizer.zero_grad()
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.step(self.optimizer)
            self.scaler.update()
        if self.args.prompt_type == 'cocoop':
            return pgen_ctx

        return

    def test_time_adapt_eval(self, val_loader, result_dict=None):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, top1, top5],
            prefix='Test: ')

        # reset model and switch to evaluate mode
        self.model.eval()
        if self.args.prompt_type != 'cocoop': # no need to reset cocoop because it's fixed
            with torch.no_grad():
                self.model.reset()
        end = time.time()

        #define a softmax layer
        softmax = torch.nn.Softmax(dim=1)
    
        for i, (images, target) in enumerate(val_loader):
            assert self.args.gpu is not None
            # if isinstance(images, list): # There must be a list
            for k in range(len(images)):
                images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
            image = images[0]
            target = target.cuda(self.args.gpu, non_blocking=True)
            
            # KEY: diffusion augmentations
            diff_images = self.generate_diff_views(image)
            diff_images = [t.cuda(self.args.gpu, non_blocking=True) for t in diff_images]
            
            # Whole images (expected shape: [128,3,H,W])
            images = torch.cat(images + diff_images, dim=0)
            del diff_images
            torch.cuda.empty_cache()

            # reset the tunable prompt to its initial state
            if self.args.prompt_type != 'cocoop': # no need to reset cocoop because it's fixed
                if self.args.tta_steps > 0:
                    with torch.no_grad():
                        self.model.reset()
                self.optimizer.load_state_dict(self.optim_state)

                self.test_time_tuning(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        image_feature, pgen_ctx = self.model.gen_ctx(images, self.args.tpt)
                self.optimizer = None

                pgen_ctx = self.test_time_tuning((image_feature, pgen_ctx))

            # The actual inference goes here
            if self.args.prompt_type == 'cocoop':
                image_feature = image_feature[0].unsqueeze(0)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    if self.args.prompt_type == 'cocoop':
                        output = self.model((image_feature, pgen_ctx))
                    else:
                        output = self.model(image)


            if result_dict is not None:
                softmax_output = softmax(output) 

                #maximum confidence of the softmax_output and its index
                max_confidence, max_index = torch.max(softmax_output, 1)

                #save the max confidence, prediction, and label to the result_dict
                result_dict['max_confidence'].append(max_confidence.item())
                result_dict['prediction'].append(max_index.item())
                result_dict['label'].append(target.item())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % self.args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

        return [top1.avg, top5.avg]