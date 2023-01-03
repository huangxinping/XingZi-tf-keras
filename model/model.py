import tensorflow as tf


class Encoder(tf.keras.models.Model):
    
    def __init__(self):
        input = tf.keras.layers.Input(shape=(80, 80, 1))
        super(Encoder, self).__init__(inputs=input, outputs=tf.keras.layers.Flatten()(input))
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(80, 80, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        ])
        super(Encoder, self).__init__(inputs=self.model.input, outputs=self.model.output)
    
    def call(self, x, training=None):
        return self.model(x)
    
    
class Decoder(tf.keras.models.Model):
    
    def __init__(self):
        input = tf.keras.layers.Input(shape=(80, 80, 1))
        super(Decoder, self).__init__(inputs=input, outputs=tf.keras.layers.Flatten()(input))
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(10, 10, 256)),
            # tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
            # tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
            # tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
            
            tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid')
        ])
        super(Decoder, self).__init__(inputs=self.model.input, outputs=self.model.output)
    
    def call(self, x, training=None):
        return self.model(x)
    
    
class FeatureExtraction(tf.keras.models.Model):    
    
    def __init__(self):
        input = tf.keras.layers.Input(shape=(80, 80, 1))
        super(FeatureExtraction, self).__init__(inputs=input, outputs=tf.keras.layers.Flatten()(input))
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.model = tf.keras.models.Sequential([
            self.encoder,
            self.decoder
        ])
        super(FeatureExtraction, self).__init__(inputs=self.model.input, outputs=self.model.output)
        
    def call(self, x, training=None):
        return self.model(x)


class FeatureTransfer(tf.keras.models.Model):
    
    def __init__(self):
        # placehold
        input = tf.keras.layers.Input(shape=(10, 10, 256))
        super(FeatureTransfer, self).__init__(inputs=input, outputs=tf.keras.layers.Flatten()(input))
        
        # real
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(10, 10, 256)),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')
        ])
        super(FeatureTransfer, self).__init__(inputs=self.model.input, outputs=self.model.output)
    
    
    def call(self, x, training=None):
        return self.model(x)
        

if __name__ == '__main__':
    # encoder = Encoder()
    # encoder.summary()
    
    # decoder = Decoder()
    # decoder.summary()
    
    # fe = FeatureExtraction()
    # fe.summary()
    
    ft = FeatureTransfer()
    ft.summary()
    
    