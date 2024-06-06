class MaxWeightCMuPolicy:
    def __init__(self):
        pass

    def test_forward(self, step, batch_queue, batch_time, repeated_queue, repeated_network, repeated_mu, repeated_h):
        return repeated_h * repeated_mu

