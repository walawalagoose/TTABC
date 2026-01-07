"""
    O-TPT: Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models,
    https://arxiv.org/abs/2503.12096,
    https://github.com/ashshaksharifdeen/O-TPT

"""

import time

from PIL import Image

import torch
import torch.optim

from ttabc.utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, select_confident_samples, marginal_entropy
from ttabc.model_selection.base_method import BaseMethod

class OTPT(BaseMethod):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.temperature_value = {'ViT': 1.16, 'RN': 1.15} #for temperature scaling experiments 

    def test_time_tuning(self, inputs):
        output = None
        output2 = None
        single_output = None
        if self.args.prompt_type == 'cocoop':
            image_feature, pgen_ctx = inputs
            pgen_ctx.requires_grad = True
            self.optimizer = torch.optim.AdamW([pgen_ctx], self.args.lr)
        
        selected_idx = None
        for j in range(self.args.tta_steps):
            with torch.amp.autocast(device_type='cuda'):
                if self.args.prompt_type == 'maple':
                    output = self.model(inputs)['logits']
                elif self.args.prompt_type == 'cocoop':
                    output = self.model((image_feature, pgen_ctx))
                else:
                    output = self.model(inputs) 

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, self.args.selection_p)

                loss = marginal_entropy(output)

            if self.args.two_step:
                self.optimizer.zero_grad()
                # compute gradient and do SGD step
                self.scaler.scale(loss).backward(retain_graph=True)
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.step(self.optimizer)
                self.scaler.update()
                loss = 0

                with torch.amp.autocast(device_type='cuda'):
                    if self.args.prompt_type == 'cocoop':
                        output2 = self.model((image_feature, pgen_ctx))
                    else:
                        output2 = self.model(inputs) 
            if output == None and output2 == None:
                single_output = self.model(self.args.image)

            lambda_ = self.args.lambda_term

            number_of_class = output.shape[1]            
            #------------------------------------------------- Householder Transform--------
            text_feature = self.model.get_text_features()
            #print("text feature shape model:",text_feature.shape)
            #computing orthogonal constrained  SVD
            Wwt  =  torch.matmul(text_feature,text_feature.T)
            wwt_norm_col_HT = torch.linalg.norm(Wwt,dim=-1)
            Wwt_val_HT = wwt_norm_col_HT.mean()
            #wtW  =  torch.matmul(text_feature.T,text_feature)
            e = torch.eye(Wwt.shape[1], device=self.args.gpu)
            M_norm = torch.linalg.norm(Wwt, dim=0,keepdim=True)
            scaled_e = e * M_norm
            # Subtract the scaled identity matrix from Wwt
            u = Wwt - scaled_e
            u_norm = torch.linalg.norm(u, dim=-1,keepdim=True)
            #u_norm = u_norm ** 2
            # We need to expand u_norm to shape (47, 47, 1) for broadcasting
            #u_norm_exp = u_norm.unsqueeze(2)  # Shape: (1, 47, 1)
            
            #Transposing the u for batch element column and coresponding column transpose matrix multiplication
          
            v = u/u_norm
            normalized_matrix_exp = v.unsqueeze(2)  # Shape: (47, 47, 1)
            normalized_matrix_T_exp = v.unsqueeze(1)  # Shape: (47, 1, 47)
            
            # This will create a batch of 3 matrices, each of shape (47, 47)
            outer_products = normalized_matrix_exp @ normalized_matrix_T_exp  # Shape: (47, 47, 47)
            
            # Perform element-wise division of each outer product by the corresponding u_norm value
            divided_matrix = outer_products #/ u_norm_exp  # Shape: (47, 47, 47)

            # Multiply the result by 2
            scaled_matrix = 2 * divided_matrix  # Shape: (47, 47, 47)
            # Subtract the scaled result from the corresponding identity matrix for each batch
            identity_matrix_dim = e.unsqueeze(0).expand(Wwt.shape[1], -1, -1)  # Shape: (47, 47, 47)
            # Subtract from identity matrix
            transformed_matrix = identity_matrix_dim - scaled_matrix  # Shape: (47, 47, 47)
            # Reshape M so that its columns are aligned for batch multiplication
            Wwt_exp = Wwt.unsqueeze(2)  # Shape: (47, 47, 1)

            # Perform batched matrix multiplication between transformed matrix and M_exp
            Hx = torch.bmm(transformed_matrix, Wwt_exp)  # Shape: (47, 47, 1)
            
            # Reshape back the result to (3, 3) by removing the last singleton dimension
            Hx = Hx.squeeze(2)  # Shape: (47, 47)
            #print("shape of Hx:",Hx.shape)
            #print("Hx:",Hx)
            #normalizing Column wise
            #Hx_norm =torch.linalg.norm(Hx, dim=0,keepdim=True)
            #Hx = Hx/Hx_norm
            Ht_ortho = Hx - e  
            Ht_ortho_norm = torch.linalg.norm(Ht_ortho, dim=-1)
            Ht_ortho_norm_val = Ht_ortho_norm.mean()
            
            
            #-------------------------------------House holder end-----------------------------------------------          
      
            loss += (+(lambda_ * Ht_ortho_norm_val))

            self.optimizer.zero_grad()
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.args.prompt_type == 'cocoop':
            return pgen_ctx

        return None

    def test_time_adapt_eval(self, val_loader, result_dict):
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
            if isinstance(images, list):
                for k in range(len(images)):
                    images[k] = images[k].cuda(self.args.gpu, non_blocking=True)
                image = images[0]
            else:
                if len(images.size()) > 4:
                    # when using ImageNet Sampler as the dataset
                    assert images.size()[0] == 1
                    images = images.squeeze(0)
                images = images.cuda(self.args.gpu, non_blocking=True)
                image = images
            target = target.cuda(self.args.gpu, non_blocking=True)
            if self.args.tpt:
                images = torch.cat(images, dim=0)

            self.args.image = image # for ctpt

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
            if self.args.tpt:
                if self.args.prompt_type == 'cocoop':
                    image_feature = image_feature[0].unsqueeze(0)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    if self.args.prompt_type == 'maple':
                        output = self.model(image)['logits']
                    if self.args.prompt_type == 'cocoop':
                        output = self.model((image_feature, pgen_ctx))
                    else:
                        output = self.model(image)


            if result_dict is not None:
                softmax_output = softmax(output) 

                #maximum confidence of the softmax_output and its index
                max_confidence, max_index = torch.max(softmax_output, 1)

                #save the max confidence, prediction, and label to the result_dict
                if max_confidence.numel() == 1:
                        result_dict['max_confidence'].append(max_confidence.item())
                        result_dict['prediction'].append(max_index.item())
                        result_dict['label'].append(target.item())
                else:
                    for j in range(max_confidence.size(0)):
                        result_dict['max_confidence'].append(max_confidence[j].item())
                        result_dict['prediction'].append(max_index[j].item())
                        result_dict['label'].append(target[j].item())

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