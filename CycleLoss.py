class CircleLoss(kls.Loss):
    
    def __init__(self,gamma: int = 64,margin: float = 0.25,batch_size: int = None,reduction='auto',name=None):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.margin = margin
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin
        if batch_size:
            self.batch_size = batch_size
            self.batch_idxs = tf.expand_dims(tf.range(0, batch_size, dtype=tf.int32), 1)  # shape [batch,1]

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(y_pred))
        alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - self.O_n)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) 
                  + (1 - y_true) * (alpha_n * (y_pred - self.Delta_n))
                 ) * self.gamma
        return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)