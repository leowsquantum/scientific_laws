import tensorflow as tf
from data_gen_maf_1d import data_gen_maf_1d

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
        self.w_in = tf.Variable(tf.random.uniform([in_dim, hidden_dim], 0.1, 2.0), dtype=tf.float32, trainable=True) # (in_dim, hidden_dim)
        self.act1 = tf.keras.layers.Lambda(lambda x: tf.math.log(x))
        self.w_hidden = tf.Variable(tf.random.uniform([hidden_dim, hidden_dim], 0.1, 2.0), dtype=tf.float32, trainable=True)
        self.act2 = tf.keras.layers.Lambda(lambda x: tf.exp(x))
        self.w_out = tf.Variable(tf.random.uniform([hidden_dim, out_dim], 0.1, 2.0), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, x):
        x = tf.matmul(x, self.w_in)
        x = self.act1(x)
        x = tf.matmul(x, self.w_hidden)
        x = self.act2(x)
        x = tf.matmul(x, self.w_out)
        return x
class MyLoss2(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

if __name__ == "__main__":
    model = MyNN2(3, 1, 3)
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=MyLoss2())
    m, a, f = data_gen_maf_1d(n_points=1000, m_lim=(0.1, 1.0), a_lim=(0.1, 1.0), noise=0.0) # (n_points)
    x = tf.transpose([m, a, f]) # (n_points, 3)
    y = tf.expand_dims(m * a - f, 1) # (n_points, 1)
    print('w_in:\n', model.w_in)
    print('w_hidden:\n', model.w_hidden)
    print('w_out:\n', model.w_out)
    model.fit(x, y, epochs=1000, batch_size=1000)
    print('w_in:\n', model.w_in)
    print('w_hidden:\n', model.w_hidden)
    print('w_out:\n', model.w_out)


