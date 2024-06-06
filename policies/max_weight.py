class MaxWeightCMuQPolicy:
    def __init__(self):
        pass

    def test_forward(self, step, batch_queue, batch_time, repeated_queue, repeated_network, repeated_mu, repeated_h):
        return repeated_queue * repeated_h * repeated_network * repeated_mu

