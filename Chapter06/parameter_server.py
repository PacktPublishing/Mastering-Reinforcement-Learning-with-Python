import ray
from models import get_Q_network


@ray.remote
class ParameterServer:
    def __init__(self, config):
        self.weights = None
        self.eval_weights = None
        self.Q = get_Q_network(config)

    def update_weights(self, new_parameters):
        self.weights = new_parameters
        return True

    def get_weights(self):
        return self.weights

    def get_eval_weights(self):
        return self.eval_weights

    def set_eval_weights(self):
        self.eval_weights = self.weights
        return True

    def save_eval_weights(self,
                          filename=
                          'checkpoints/model_checkpoint'):
        self.Q.set_weights(self.eval_weights)
        self.Q.save_weights(filename)
        print("Saved.")
