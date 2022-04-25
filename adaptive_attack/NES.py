
from attack.utils import resolve_prediction
import torch
import torch.nn as nn

class NES(nn.Module):

    def __init__(self, samples_per_draw, samples_per_draw_batch, sigma, EOT_wrapper):
        super().__init__()
        self.samples_per_draw = samples_per_draw
        self.samples_per_draw_batch_size = samples_per_draw_batch
        self.sigma = sigma
        self.EOT_wrapper = EOT_wrapper # EOT wraps the model
    
    def forward(self, x, y):
        n_audios, n_channels, N = x.shape
        num_batches = self.samples_per_draw // self.samples_per_draw_batch_size
        for i in range(num_batches):
            noise = torch.randn([n_audios, self.samples_per_draw_batch_size // 2, n_channels, N], device=x.device)
            # noise = (torch.rand([n_audios, self.samples_per_draw_batch_size // 2, n_channels, N], device=x.device) * 2 - 1).sign()
            noise = torch.cat((noise, -noise), 1)
            if i == 0:
                noise = torch.cat((torch.zeros_like(x, device=x.device).unsqueeze(1), noise), 1)
            eval_input = noise * self.sigma + x.unsqueeze(1)
            eval_input = eval_input.view(-1, n_channels, N) # (n_audios*samples_per_draw_batch_size, n_channels, N)
            eval_y = None
            for jj, y_ in enumerate(y):
                tmp = torch.tensor([y_] * (self.samples_per_draw_batch_size+1 if i == 0 else self.samples_per_draw_batch_size), dtype=torch.long, device=x.device)
                if jj == 0:
                    eval_y = tmp
                else:
                    eval_y = torch.cat((eval_y, tmp))
            # scores, loss, _ = EOT_wrapper(eval_input, eval_y, EOT_num_batches, self.EOT_batch_size, use_grad=False)
            scores, loss, _, decisions = self.EOT_wrapper(eval_input, eval_y)
            EOT_num_batches = int(self.EOT_wrapper.EOT_size // self.EOT_wrapper.EOT_batch_size)
            loss.data /= EOT_num_batches # (n_audios*samples_per_draw_batch_size,)
            scores.data /= EOT_num_batches

            loss = loss.view(n_audios, -1)
            scores = scores.view(n_audios, -1, scores.shape[1])

            if i == 0:
                adver_loss = loss[..., 0] # (n_audios, )
                loss = loss[..., 1:] # (n_audios, samples_batch)
                adver_score = scores[:, 0, :] # (n_audios, n_spks)
                noise = noise[:, 1:, :, :] # (n_audios, samples_batch, n_channels, N)
                grad = torch.mean(loss.unsqueeze(2).unsqueeze(3) * noise, 1)
                mean_loss = loss.mean(1)
                predicts = resolve_prediction(decisions).reshape(n_audios, -1) # (n_audios, samples_batch)
                predict = predicts[:, 0]
            else:
                grad += torch.mean(loss.unsqueeze(2).unsqueeze(3) * noise, 1)
                mean_loss += loss.mean(1)
        grad = grad / self.sigma / num_batches
        mean_loss = mean_loss / num_batches
        return mean_loss, grad, adver_loss, adver_score, predict