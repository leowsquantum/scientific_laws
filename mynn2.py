import tensorflow as tf
from data_gen_maf_1d import data_gen_maf_1d
from time import time

class MyNN2(tf.keras.Model):
    '''
    Model architecture:
    (n_points, in_dim)
    -- w_in -->
    (n_points, hidden_dim)
    -- act1 -->
    (n_points, hidden_dim)
    -- w_hidden -->
    (n_points, hidden_dim)
    -- act2 -->
    (n_points, hidden_dim)
    -- w_out -->
    (n_points, out_dim)
    '''
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        # This layer is now deactivated
        self.w_in = tf.Variable(tf.eye(in_dim), dtype=tf.float32, trainable=False)
        # log activationi function, taking absolute value to handle negative numbers
        self.act1 = tf.keras.layers.Lambda(lambda x: tf.math.log(tf.abs(x + 1e-6)))
        self.w_hidden = tf.Variable(tf.random.uniform([in_dim, hidden_dim], -1.0, 1.0),
                                    dtype=tf.float32, trainable=True)
        self.act2 = tf.keras.layers.Lambda(lambda x: tf.exp(x))
        self.w_out = tf.Variable(tf.random.uniform([hidden_dim, out_dim], -1.0, 1.0),
                                 dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, x):
        x = tf.matmul(x, self.w_in)
        x = self.act1(x)
        x = tf.matmul(x, self.w_hidden)
        x = self.act2(x)
        x = tf.matmul(x, self.w_out)
        return x

class MyNorm(tf.keras.metrics.Metric):
    def __init__(self, model: MyNN2, **kwargs):
        super().__init__()
        self.model = model
        self.norm = 0.0
        self.p = 2.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        # norm_w_in = tf.norm(self.model.w_in, ord=self.p)
        norm_w_hidden = tf.norm(self.model.w_hidden, ord=self.p)
        norm_w_out = tf.norm(self.model.w_out, ord=self.p)
        self.norm = (norm_w_hidden / 4.0 + 1.0 / norm_w_hidden + norm_w_out / 4.0 + 1.0 / norm_w_out) / 2.0 - 1.0

    def result(self):
        return self.norm

class IntProximity(tf.keras.metrics.Metric):
    def __init__(self, model: MyNN2, **kwargs):
        super().__init__()
        self.model = model
        self.int_proximity = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.int_proximity = (tf.reduce_mean(tf.square(self.model.w_hidden - tf.round(self.model.w_hidden)))
                         + tf.reduce_mean(tf.square(self.model.w_out - tf.round(self.model.w_out)))) / 3.0

    def result(self):
        return self.int_proximity

class MyLoss2(tf.keras.losses.Loss):
    def __init__(self, model: MyNN2):
        super().__init__()
        self.model = model
        self.p = 2.0
        self.w_mse = 1.0
        self.w_norm = 10.0
        self.w_int_proximity = 30.0
        self.my_norm = MyNorm(model)
        self.int_proximity = IntProximity(model)

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        self.my_norm.update_state(y_true, y_pred)
        self.int_proximity.update_state(y_true, y_pred)
        return (mse * self.w_mse + self.my_norm.result() * self.w_norm
                + self.int_proximity.result() * self.w_int_proximity)

if __name__ == "__main__":
    model = MyNN2(3, 1, 3)
    loss = MyLoss2(model)
    my_norm = MyNorm(model)
    int_proximity = IntProximity(model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=loss,
                  metrics=[tf.keras.metrics.MeanSquaredError(), my_norm, int_proximity])
    m, a, f = data_gen_maf_1d(n_points=int(1e6), m_lim=(-1.0, 1.0), a_lim=(-1.0, 1.0), noise=0.0) # (n_points)
    x = tf.transpose([m, a, f]) # (n_points, 3)
    y = tf.expand_dims(m * a - f, 1) # (n_points, 1)
    print('=' * 40 + 'Initial weights' + '=' * 40)
    print('w_in:\n', model.w_in)
    print('w_hidden:\n', model.w_hidden)
    print('w_out:\n', model.w_out)
    t0 = time()
    model.fit(x, y, epochs=100, batch_size=int(1e4), steps_per_epoch=100)
    print("Time:", time() - t0)
    print('=' * 40 + 'trained weights' + '=' * 40)
    print('w_in:\n', model.w_in)
    print('w_hidden:\n', model.w_hidden)
    print('w_out:\n', model.w_out)
    print('=' * 40 + 'Rounded trained weights' + '=' * 40)
    print('w_in:\n', tf.round(model.w_in))
    print('w_hidden:\n', tf.round(model.w_hidden))
    print('w_out:\n', tf.round(model.w_out))



