

from attack.PGD import PGD

class CWinf(PGD):
    
    def __init__(self, model, task='CSI', epsilon=0.002, step_size=0.0004, max_iter=10, num_random_init=0, 
                loss='Margin', targeted=False,
                batch_size=1, EOT_size=1, EOT_batch_size=1, 
                verbose=1):

        loss = 'Margin' # hard coding: using Margin Loss
        super().__init__(model, task=task, epsilon=epsilon, step_size=step_size, max_iter=max_iter, num_random_init=num_random_init, 
                        loss=loss, targeted=targeted,
                        batch_size=batch_size, EOT_size=EOT_size, EOT_batch_size=EOT_batch_size, 
                        verbose=verbose)