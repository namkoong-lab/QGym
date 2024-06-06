import torch

class MaxPressurePolicy:
    def __init__(self, queue_event_options):
        self.queue_event_options = queue_event_options

    def test_forward(self, step, batch_queue, batch_time, repeated_queue, repeated_network, repeated_mu, repeated_h):
        q = repeated_network.shape[-1]
        A = self.queue_event_options[q:]
        return repeated_mu * (torch.sum(-A.unsqueeze(0).repeat(repeated_queue.shape[0], 1, 1) * repeated_queue[:, 0].unsqueeze(2).repeat(1, 1, q) * repeated_h[:, 0].unsqueeze(2).repeat(1, 1, q), 2)).unsqueeze(1).repeat(1, repeated_network.shape[1], 1)