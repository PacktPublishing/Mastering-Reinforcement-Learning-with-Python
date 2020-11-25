from gym.spaces import Box
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()


class ParametricActionsModel(DistributionalQTFModel):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        true_obs_shape=(2,),
        **kw
    ):
        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw
        )
        self.action_value_model = FullyConnectedNetwork(
            Box(-1, 1, shape=true_obs_shape),
            action_space,
            num_outputs,
            model_config,
            name + "_action_values",
        )
        self.register_variables(self.action_value_model.variables())

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        action_values, _ = self.action_value_model(
            {"obs": input_dict["obs"]["actual_obs"]}
        )
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_values + inf_mask, state

