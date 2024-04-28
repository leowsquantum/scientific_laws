import tensorflow as tf
from data_gen_maf_1d import data_gen_maf_1d

class MyNN1(tf.keras.Model):
    # w: list of weight matrices in each layer, shape (n_layers, n_dim_(i-1), n_dim_i) for i = 1,...,n_layers
    # x: input tensor, shape (n_points, n_dim)
    def __init__(self, in_dim: int, out_dim: int, libfunc_width=None, layers=6, name=None):
        super().__init__(name=name)
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.lib_func_add = [
            tf.keras.layers.Lambda(lambda x: 1),
            tf.keras.layers.Lambda(lambda x: x),
            tf.keras.layers.Lambda(lambda x: tf.sin(x)),
            tf.keras.layers.Lambda(lambda x: tf.exp(x)),
            tf.keras.layers.Lambda(lambda x: tf.math.log(x)),
        ]
        self.lib_func_mul = [
            tf.keras.layers.Lambda(lambda x: x),
            tf.keras.layers.Lambda(lambda x: tf.sin(x)),
            tf.keras.layers.Lambda(lambda x: tf.exp(x)),
            tf.keras.layers.Lambda(lambda x: tf.math.log(x)),
        ]
        self.libfunc_width = []
        if libfunc_width is None:
            self.libfunc_width = [3] * (self.lib_func_add.__len__() + self.lib_func_mul.__len__())
        else:
            assert libfunc_width.__len__() == self.lib_func_add.__len__() + self.lib_func_mul.__len__()
            self.libfunc_width = libfunc_width
        self.width = 0
        for i in range(self.libfunc_width.__len__()):
            self.width += self.libfunc_width[i]
        self.act_add = []
        self.act_mul = []
        for i in range(self.libfunc_width.__len__()):
            for j in range(self.libfunc_width[i]):
                if i < self.lib_func_add.__len__():
                    self.act_add.append(self.lib_func_add[i])
                else:
                    self.act_mul.append(self.lib_func_mul[i - self.lib_func_add.__len__()])
        self.act = self.act_add + self.act_mul
        self.w = []
        self.w.append(tf.Variable(tf.random.uniform([in_dim, self.width]), dtype=tf.float32, trainable=True))
        for i in range(layers - 2):
            self.w.append(tf.Variable(tf.random.normal([self.width, self.width]), dtype=tf.float32, trainable=True))
        self.w.append(tf.Variable(tf.random.normal([self.width, out_dim]), dtype=tf.float32, trainable=True))

    def set_weights_to_zeros(self):
        for i in range(self.w.__len__()):
            self.w[i].assign(tf.Variable(tf.zeros(self.w[i].shape), dtype=tf.float32))

    @tf.function
    def activation(self, x: tf.Tensor):
        x = tf.identity(x)  # shape (n_points, n_dim_(i-1))
        for i in range(self.lib_func_add.__len__()):
            x[:, i] = self.lib_func_add[i](x[:, i])
        for i in range(self.lib_func_mul.__len__()):
            x[:, i + self.lib_func_add.__len__()] = self.lib_func_mul[i](x[:, i + self.lib_func_add.__len__()])
        return x  # shape (n_points, n_dim_(i))

    # w shape (n_dim_(i-1), n_dim_i)
    # x shape (n_points, n_dim_(i-1))
    @tf.function
    def power_weighting(self, w: tf.Tensor, x: tf.Tensor):
        w = tf.expand_dims(w, 0)  # shape (1, n_dim_(i-1), n_dim_i)
        x = tf.expand_dims(x, -1)  # shape (n_points, n_dim_(i-1), 1)
        return tf.reduce_prod(tf.pow(x, w), axis=1)  # shape (n_points, n_dim_i)

    # x shape (n_points, in_dim)
    def call(self, inputs, training=False):
        assert inputs.shape[1] == self.in_dim
        x = tf.matmul(inputs, self.w[0])  # shape (n_points, n_dim_1)
        for i in range(1, self.w.__len__() - 1):
            x = tf.concat([tf.matmul(x, self.w[i][:, 0:self.act_add.__len__()]),
                           self.power_combination(x, self.w[i][:, self.act_add.__len__():])], axis=1)
            x = self.activation(x)  # shape (n_points, n_dim_i)
        x = tf.matmul(x, self.w[-1])  # shape (n_points, out_dim)
        return x

if __name__ == '__main__':
    model = MyNN1(in_dim=3, out_dim=1, layers=2, libfunc_width=[0, 1, 0, 0, 0, 1, 0, 0, 0])
    model.set_weights_to_zeros()
    model.w[0].assign(tf.Variable([[0, 1], [0, 1], [1, 0]], dtype=tf.float32))
    model.w[1].assign(tf.Variable([[1], [-1]], dtype=tf.float32))
    print()

