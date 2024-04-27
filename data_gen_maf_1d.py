import tensorflow as tf
import matplotlib.pyplot as plt

def data_gen_maf_1d(n_points=1000, m_lim=(-1.0, 1.0), a_lim=(-1.0, 1.0), noise=0.03, seed=None):
    m0 = tf.random.uniform([n_points], m_lim[0], m_lim[1], seed=seed, dtype=tf.float32)
    a0 = tf.random.uniform([n_points], a_lim[0], a_lim[1], seed=seed, dtype=tf.float32)
    f0 = m0 * a0
    m = m0 + tf.random.normal(m0.shape, 0, tf.abs(m0 * noise), seed=seed, dtype=tf.float32)
    a = a0 + tf.random.normal(a0.shape, 0, tf.abs(a0 * noise), seed=seed, dtype=tf.float32)
    f = f0 + tf.random.normal(f0.shape, 0, tf.abs(f0 * noise), seed=seed, dtype=tf.float32)
    return tf.Variable([m, a, f])

if __name__ == "__main__":
    print(tf.__version__)
    data = data_gen_maf_1d(n_points=300)

    plt.scatter(data[0, :] * data[1, :], data[2, :])
    plt.show()
