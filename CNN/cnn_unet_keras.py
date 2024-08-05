import tensorflow as tf
import keras
import keras.api.layers as layers

from tensorflow.keras.layers import concatenate

def encoder_block(inputs, num_filters): 
  
    # Convolution with 3x3 filter followed by ReLU activation 
    x = tf.keras.layers.Conv2D(num_filters,  
                               3,  
                               padding = 'same')(inputs) 
    x = tf.keras.layers.Activation('relu')(x) 
      
    # Convolution with 3x3 filter followed by ReLU activation 
    x = tf.keras.layers.Conv2D(num_filters,  
                               3,  
                               padding = 'same')(x) 
    _skip = tf.keras.layers.Activation('relu')(x) 
  
    # Max Pooling with 2x2 filter 
    x = tf.keras.layers.MaxPool2D(pool_size = (2, 2), 
                                  strides = 2)(_skip) 
      
    return x, _skip

def decoder_block(inputs, skip_features, num_filters): 
  
    # Upsampling with 2x2 filter 
    x = tf.keras.layers.Conv2DTranspose(num_filters, 
                                        (3, 3),  
                                        strides = 2,  
                                        padding = 'same')(inputs) 
      
    # Copy and crop the skip features  
    # to match the shape of the upsampled input 
    '''
    skip_features = tf.image.resize(skip_features, 
                                    size = (x.shape[1], 
                                            x.shape[2])) 
    '''
    
    #x = tf.keras.layers.Concatenate()([x, skip_features], axis=3) 
    x = concatenate([x, skip_features], axis=3)
    
    # Convolution with 3x3 filter followed by ReLU activation 
    x = tf.keras.layers.Conv2D(num_filters, 
                               3,  
                               padding = 'same')(x) 
    x = tf.keras.layers.Activation('relu')(x) 
  
    # Convolution with 3x3 filter followed by ReLU activation 
    x = tf.keras.layers.Conv2D(num_filters, 3, padding = 'same')(x) 
    x = tf.keras.layers.Activation('relu')(x) 
      
    return x

def unet_model(input_shape = (256, 256, 3), num_classes = 1): 
    inputs = tf.keras.layers.Input(input_shape) 
      
    # Contracting Path 
    s1, sk1 = encoder_block(inputs, 64) 
    s2, sk2 = encoder_block(s1, 128) 
    s3, sk3 = encoder_block(s2, 256) 
    s4, sk4 = encoder_block(s3, 512) 
      
    # Bottleneck 
    b1 = tf.keras.layers.Conv2D(1024, 3, padding = 'same')(s4) 
    b1 = tf.keras.layers.Activation('relu')(b1) 
    b1 = tf.keras.layers.Conv2D(1024, 3, padding = 'same')(b1) 
    b1 = tf.keras.layers.Activation('relu')(b1) 
      
    # Expansive Path 
    print(s4.shape, sk4.shape, b1.shape)
    
    s5 = decoder_block(b1, sk4, 512) 
    s6 = decoder_block(s5, sk3, 256) 
    s7 = decoder_block(s6, sk2, 128) 
    s8 = decoder_block(s7, sk1, 64) 
      
    # Output 
    outputs = tf.keras.layers.Conv2D(num_classes,  
                                     1,  
                                     padding = 'valid',  
                                     activation = 'sigmoid')(s8) 
      
    model = tf.keras.models.Model(inputs = inputs,  
                                  outputs = outputs,  
                                  name = 'U-Net') 
    return model 
  
if __name__ == '__main__': 
    model = unet_model(input_shape=(576, 576, 3), num_classes=2) 
    model.summary()
    