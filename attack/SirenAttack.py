
from attack.Attack import Attack
import numpy as np
from adaptive_attack.EOT import EOT
from attack.utils import resolve_loss
import torch
from attack.utils import resolve_prediction

class SirenAttack(Attack):

    def __init__(self, model, threshold=None, 
                task='CSI', targeted=False, confidence=0.,
                epsilon=0.002, max_epoch=300, max_iter=30,
                c1=1.4961, c2=1.4961, n_particles=25, w_init=0.9, w_end=0.1,
                batch_size=1, EOT_size=1, EOT_batch_size=1, verbose=1, abort_early=True, abort_early_iter=10, abort_early_epoch=10):
        
        self.model = model
        self.threshold = threshold
        self.task = task
        self.targeted = targeted
        self.confidence = confidence
        self.epsilon = epsilon
        self.max_epoch = max_epoch
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.n_particles = n_particles
        self.w_init = w_init
        self.w_end = w_end
        self.batch_size = batch_size
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose

        self.abort_early = abort_early
        self.abort_early_iter = abort_early_iter
        self.abort_early_epoch = abort_early_epoch
    

    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):

        with torch.no_grad():

            v_upper = torch.abs(lower - upper)
            v_lower = -v_upper

            x_batch_clone = x_batch.clone() # for return
            n_audios, n_channels, N = x_batch.shape
            consider_index = list(range(n_audios))
            # pbest_locations = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
            #                     high=upper.unsqueeze(1).cpu().numpy(), size=(n_audios, self.n_particles, n_channels, N))
            # pbest_locations = torch.tensor(pbest_locations, device=x_batch.device, dtype=torch.float)

            gbest_location = torch.zeros(n_audios, n_channels, N, dtype=torch.float, device=x_batch.device)
            gbests = torch.ones(n_audios, device=x_batch.device, dtype=torch.float) * np.infty
            gbest_predict = np.array([None] * n_audios)
            prev_gbest = gbests.clone()
            prev_gbest_epoch = gbests.clone()

            continue_flag = True
            for epoch in range(self.max_epoch):
                
                if not continue_flag:
                    break

                if epoch == 0:
                    pbest_locations = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
                                high=upper.unsqueeze(1).cpu().numpy(), size=(n_audios, self.n_particles, n_channels, N))
                    pbest_locations = torch.tensor(pbest_locations, device=x_batch.device, dtype=torch.float)
                    pbests = torch.ones(n_audios, self.n_particles, device=x_batch.device, dtype=torch.float) * np.infty
                else:
                    best_index = torch.argmin(pbests, dim=1) # (len(consider_index), )
                    best_location = pbest_locations[np.arange(len(consider_index)), best_index] # (len(consider_index), n_channels, N)
                    pbest_locations = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
                                high=upper.unsqueeze(1).cpu().numpy(), size=(len(consider_index), self.n_particles-1, n_channels, N))
                    pbest_locations = torch.tensor(pbest_locations, device=x_batch.device, dtype=torch.float)
                    pbest_locations = torch.cat((best_location.unsqueeze(1), pbest_locations), dim=1)
                    pbests_new = torch.ones(len(consider_index), self.n_particles-1, device=x_batch.device, dtype=torch.float) * np.infty
                    pbests = torch.cat((pbests[np.arange(len(consider_index)), best_index].unsqueeze(1), pbests_new), dim=1)

                locations = pbest_locations.clone()
                # volicities = np.random.uniform(low=lower.unsqueeze(1).cpu().numpy(),
                #                 high=upper.unsqueeze(1).cpu().numpy(), size=(len(consider_index), self.n_particles, n_channels, N))
                volicities = np.random.uniform(low=v_lower.unsqueeze(1).cpu().numpy(),
                                high=v_upper.unsqueeze(1).cpu().numpy(), size=(len(consider_index), self.n_particles, n_channels, N))
                volicities = torch.tensor(volicities, device=x_batch.device, dtype=torch.float)

                ### ????
                # pbests = torch.ones(len(consider_index), self.n_particles, device=x_batch.device, dtype=torch.float) * np.infty

                continue_flag_inner = True

                # for iter in range(self.max_iter):
                for iter in range(self.max_iter+1):

                    if not continue_flag_inner:
                        break

                    eval_x = locations + x_batch.unsqueeze(1) # (n_audios, self.n_particles, n_channels, N)
                    eval_x = eval_x.view(-1, n_channels, N)
                    eval_y = None
                    for jj, y_ in enumerate(y_batch):
                        tmp = torch.tensor([y_] * self.n_particles, dtype=torch.long, device=x_batch.device)
                        if jj == 0:
                            eval_y = tmp
                        else:
                            eval_y = torch.cat((eval_y, tmp))
                    # print(eval_x.shape, eval_y.shape)
                    _, loss, _, decisions = self.EOT_wrapper(eval_x, eval_y)
                    EOT_num_batches = int(self.EOT_wrapper.EOT_size // self.EOT_wrapper.EOT_batch_size)
                    loss.data /= EOT_num_batches # (n_audios*n_p,)
                    loss = loss.view(len(consider_index), -1) # (n_audios, n_p)
                    predict = resolve_prediction(decisions).reshape(len(consider_index), -1)

                    update_index = torch.where(loss < pbests)
                    update_ii = update_index[0].cpu().numpy().tolist()
                    update_jj = update_index[1].cpu().numpy().tolist()
                    if len(update_ii) > 0:
                        for ii, jj in zip(update_ii, update_jj):
                            pbests[ii, jj] = loss[ii, jj]
                            pbest_locations[ii, jj, ...] = locations[ii, jj, ...]
                    
                    # if self.abort_early and (iter+1) % self.abort_early_iter == 0:
                    #     prev_gbest.data = gbests
                    
                    gbest_index = torch.argmin(pbests, 1)
                    for kk in range(gbest_index.shape[0]):
                        index = consider_index[kk]
                        if pbests[kk, gbest_index[kk]] < gbests[index]:
                            gbests[index] = pbests[kk, gbest_index[kk]]
                            gbest_location[index] = pbest_locations[kk, gbest_index[kk]]
                            gbest_predict[index] = predict[kk, gbest_index[kk]]
                    
                    if self.verbose:
                        print('batch: {}, epoch: {}, iter: {}, y: {}, y_pred: {}, gbest: {}'.format(batch_id,
                            epoch, iter, y_batch.cpu().numpy().tolist(), gbest_predict[consider_index], gbests[consider_index].cpu().numpy().tolist()))
                    
                    if self.abort_early and (iter+1) % self.abort_early_iter == 0:
                        if torch.mean(gbests) > 0.9999 * torch.mean(prev_gbest):
                            print('Converge, Break Inner Loop')
                            continue_flag_inner = False
                            # break
                        # prev_gbest.data = gbests
                        prev_gbest = gbests.clone()

                    # stop early
                    # x_batch, y_batch, lower, upper
                    # pbest_locations, locations, v, pbests
                    # consider_index
                    # delete alrady found examples
                    x_batch, y_batch, lower, upper, \
                    pbest_locations, locations, volicities, pbests, \
                    consider_index = self.delete_found(gbests[consider_index], x_batch, y_batch, lower, upper, 
                                                    pbest_locations, locations, volicities, pbests, 
                                                    consider_index)
                    if len(consider_index) == 0:
                        continue_flag = False # used to break the outer loop
                        break
                    else:
                        v_upper = torch.abs(lower - upper)
                        v_lower = -v_upper

                    if iter < self.max_iter:
                        w = (self.w_init - self.w_end) * (self.max_iter - iter - 1) / self.max_iter + self.w_end
                        # r1 = np.random.rand() + 0.00001
                        # r2 = np.random.rand() + 0.00001
                        r1 = np.random.rand(len(consider_index), self.n_particles, n_channels, N) + 0.00001
                        r2 = np.random.rand(len(consider_index), self.n_particles, n_channels, N) + 0.00001
                        r1 = torch.tensor(r1, device=x_batch.device, dtype=torch.float)
                        r2 = torch.tensor(r2, device=x_batch.device, dtype=torch.float)
                        volicities = (w * volicities + self.c1 * r1 * (pbest_locations - locations) +
                                self.c2 * r2 * (gbest_location[consider_index, ...].unsqueeze(1) - locations))
                        locations = locations + volicities
                        locations = torch.min(torch.max(locations, lower.unsqueeze(1)), upper.unsqueeze(1))
                
                if self.abort_early and (epoch+1) % self.abort_early_epoch == 0:
                    if torch.mean(gbests) > 0.9999 * torch.mean(prev_gbest_epoch):
                        print('Converge, Break Outer Loop')
                        continue_flag = False
                        # break
                    prev_gbest_epoch = gbests.clone()
            
            success = [False] * n_audios
            for kk, best_l in enumerate(gbests):
                if best_l < 0:
                    success[kk] = True

            return gbest_location + x_batch_clone, success

        
    def delete_found(self, gbests, x_batch, y_batch, lower, upper, 
                    pbest_locations, locations, volicities, pbests, 
                    consider_index):
        
        x_batch_u = None
        y_batch_u = None
        lower_u = None
        upper_u = None 
        pbest_locations_u = None
        locations_u = None
        volicities_u = None
        pbests_u = None
        consider_index_u = []

        for ii, g in enumerate(gbests):
            if g < 0:
                continue
            else:
                if x_batch_u is None:
                    x_batch_u = x_batch[ii:ii+1]
                    y_batch_u = y_batch[ii:ii+1]
                    lower_u = lower[ii:ii+1]
                    upper_u = upper[ii:ii+1]
                    pbest_locations_u = pbest_locations[ii:ii+1]
                    locations_u = locations[ii:ii+1]
                    volicities_u = volicities[ii:ii+1]
                    pbests_u = pbests[ii:ii+1]
                else:
                    x_batch_u = torch.cat((x_batch_u, x_batch[ii:ii+1]), 0)
                    y_batch_u = torch.cat((y_batch_u, y_batch[ii:ii+1]))
                    lower_u = torch.cat((lower_u, lower[ii:ii+1]), 0)
                    upper_u = torch.cat((upper_u, upper[ii:ii+1]), 0)
                    pbest_locations_u = torch.cat((pbest_locations_u, pbest_locations[ii:ii+1]), 0)
                    locations_u = torch.cat((locations_u, locations[ii:ii+1]), 0)
                    volicities_u = torch.cat((volicities_u, volicities[ii:ii+1]), 0)
                    pbests_u = torch.cat((pbests_u, pbests[ii:ii+1]), 0)
                index = consider_index[ii]
                consider_index_u.append(index)
        
        return x_batch_u, y_batch_u, lower_u, upper_u, \
                pbest_locations_u, locations_u, volicities_u, pbests_u, \
                consider_index_u
    

    def attack(self, x, y):

        if self.task in ['SV', 'OSI'] and self.threshold is None:
            raise NotImplementedError('You are running black box attack for {} task, \
                        but the threshold not specified. Consider Estimating the threshold by FAKEBOB!')
        self.loss, self.grad_sign = resolve_loss('Margin', self.targeted, self.confidence, self.task, self.threshold, False)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, False)

        lower = -1
        upper = 1
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain' 
        n_audios, n_channels, _ = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal' 
        # upper = torch.clamp(x+self.epsilon, max=upper)
        # lower = torch.clamp(x-self.epsilon, min=lower)
        lower = torch.clamp(-1-x, min=-self.epsilon) # for distortion, not adver audio
        upper = torch.clamp(1-x, max=self.epsilon) # for distortion, not adver audio

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))
        for batch_id in range(n_batches):
            x_batch = x[batch_id*batch_size:(batch_id+1)*batch_size] # (batch_size, 1, max_len)
            y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
            lower_batch = lower[batch_id*batch_size:(batch_id+1)*batch_size]
            upper_batch = upper[batch_id*batch_size:(batch_id+1)*batch_size]
            adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, batch_id)
            if batch_id == 0:
                adver_x = adver_x_batch
                success = success_batch
            else:
                adver_x = torch.cat((adver_x, adver_x_batch), 0)
                success += success_batch

        return adver_x, success
