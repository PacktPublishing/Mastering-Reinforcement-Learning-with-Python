from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def masked_loss(args):
    y_true, y_pred, mask = args
    masked_pred = K.sum(mask * y_pred, axis=1, keepdims=True)
    loss = K.square(y_true - masked_pred)
    return K.mean(loss, axis=-1)


def get_Q_network(config):
    obs_input = Input(shape=config["obs_shape"],
                      name='Q_input')

    x = Flatten()(obs_input)
    for i, n_units in enumerate(config["fcnet_hiddens"]):
        layer_name = 'Q_' + str(i + 1)
        x = Dense(n_units,
                  activation=config["fcnet_activation"],
                  name=layer_name)(x)
    q_estimate_output = Dense(config["n_actions"],
                              activation='linear',
                              name='Q_output')(x)
    # Q Model
    Q_model = Model(inputs=obs_input,
                    outputs=q_estimate_output)
    Q_model.summary()
    Q_model.compile(optimizer=Adam(), loss='mse')
    return Q_model


def get_trainable_model(config):
    Q_model = get_Q_network(config)
    obs_input = Q_model.get_layer("Q_input").output
    q_estimate_output = Q_model.get_layer("Q_output").output
    mask_input = Input(shape=(config["n_actions"],),
                       name='Q_mask')
    sampled_bellman_input = Input(shape=(1,),
                                  name='Q_sampled')

    # Trainable model
    loss_output = Lambda(masked_loss,
                         output_shape=(1,),
                         name='Q_masked_out')\
                        ([sampled_bellman_input,
                          q_estimate_output,
                          mask_input])
    trainable_model = Model(inputs=[obs_input,
                                    mask_input,
                                    sampled_bellman_input],
                            outputs=loss_output)
    trainable_model.summary()
    trainable_model.compile(optimizer=
                            Adam(lr=config["lr"],
                            clipvalue=config["grad_clip"]),
                            loss=[lambda y_true,
                                         y_pred: y_pred])
    return Q_model, trainable_model

