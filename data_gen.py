import tensorflow as tf
import matplotlib.pyplot as plt
import time

def gravity_gen(n_points=1000, n_dim=2, x_lim=(-1.0, 1.0), t_lim=(0.0, 1.0), m_lim=(0.0, 1.0), t_step=1000, v_lim=(-10, 10.0), a_lim=(-10, 10.0), noise=0.03, seed=None):
    x0 = tf.random.uniform([n_points, n_dim], x_lim[0], x_lim[1], seed=seed, dtype=tf.float32)
    v0 = tf.random.uniform([n_points, n_dim], v_lim[0], v_lim[1], seed=seed, dtype=tf.float32)
    a0 = tf.random.uniform([n_points, n_dim], a_lim[0], a_lim[1], seed=seed, dtype=tf.float32)
    m0 = tf.random.uniform([n_points], m_lim[0], m_lim[1], seed=seed, dtype=tf.float32)
    t = tf.linspace(t_lim[0], t_lim[1], t_step)
    x0 = tf.broadcast_to(tf.expand_dims(x0, 1), [n_points, t_step, n_dim])
    v0 = tf.broadcast_to(tf.expand_dims(v0, 1), [n_points, t_step, n_dim])
    a0 = tf.broadcast_to(tf.expand_dims(a0, 1), [n_points, t_step, n_dim])
    t = tf.broadcast_to(tf.expand_dims(tf.expand_dims(t, 0), -1), [n_points, t_step, n_dim])
    m0 = tf.broadcast_to(tf.expand_dims(tf.expand_dims(m0, -1), -1), [n_points, t_step, n_dim])
    f0 = m0 * a0
    x = x0 + v0 * t + 0.5 * a0 * t * t
    x += tf.random.normal(x.shape, 0, tf.abs(x0 * noise), seed=seed, dtype=tf.float32)
    m = m0 + tf.random.normal(m0.shape, 0, tf.abs(m0 * noise), seed=seed, dtype=tf.float32)
    f = m * a0 + tf.random.normal(f0.shape, 0, tf.abs(f0 * noise), seed=seed, dtype=tf.float32)
    return tf.Variable([x, t, m, f])

if __name__ == "__main__":
    print(tf.__version__)
    t0 = time.time()
    data = gravity_gen(n_points=30, n_dim=2)
    print("Time:", time.time() - t0)

    for i in range(data.shape[1]):
        plt.plot(data[0, i, :, 0], data[0, i, :, 1])
    plt.show()

