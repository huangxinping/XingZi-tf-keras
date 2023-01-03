import tensorflow as tf

class FeatureExtractionLoss(tf.keras.losses.Loss):

    def __init__(self, char_size=80, reduction=tf.keras.losses.Reduction.AUTO, name='rewrite_loss'):
        super(FeatureExtractionLoss, self).__init__(reduction=reduction, name=name)
        self.char_size = char_size

    def total_variation_loss(self, y_pred, side):
        """
        Total variation loss for regularization of image smoothness
        """
        loss = tf.nn.l2_loss(y_pred[:, 1:, :, :] - y_pred[:, :side - 1, :, :]) / side + \
            tf.nn.l2_loss(y_pred[:, :, 1:, :] - y_pred[:, :, :side - 1, :]) / side
        return loss

    def call(self, y_true, y_pred):
        true = tf.cast(y_true, dtype=tf.float32)
        pred = tf.cast(y_pred, dtype=tf.float32)
        pixel_abs_loss = tf.reduce_mean(tf.abs(true - pred))
        tv_loss = 0.0002 * self.total_variation_loss(pred, self.char_size)
        combined_loss = pixel_abs_loss + tv_loss
        return combined_loss