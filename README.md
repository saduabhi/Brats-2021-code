# Brats-2021-code


def Hausdorff_distance(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
        
    print(type(y_pred_f))
    # first method
    d2_matrix = pairwise_distances(y_true_f,y_pred_f, metric ='euclidean')
    result = tf.math.reduce_mean(tf.math.minimum(d2_matrix, axis=0)) + tf.math.reduce_mean(tf.math.minimum(d2_matrix, axis=1))
    return result



https://www.kaggle.com/code/rastislav/3d-mri-brain-tumor-segmentation-u-net#Problem-definiton


def Unet_3d(input_img, n_filters = 32, batch_norm = True):

    #Contraction path
    c1 = Conv3D(n_filters, (5, 5, 5), activation='relu', kernel_initializer=kernel_initializer, padding='same')(input_img)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(n_filters, (5, 5, 5), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(n_filters*2, (5, 5, 5), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(n_filters*2, (5, 5, 5), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(n_filters*4, (5, 5, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(n_filters*4, (5, 5, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 1))(c3)
     
    c4 = Conv3D(n_filters*8, (5, 5, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.3)(c4)
    c4 = Conv3D(n_filters*8, (5, 5, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    
    #Expansive path 
    u5 = Conv3DTranspose(n_filters*4, (2, 2, 1), strides=(2, 2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv3D(n_filters*4, (2, 2, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u5)
    c5 = Dropout(0.2)(c6)
    c5 = Conv3D(n_filters*4, (2, 2, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    u6 = Conv3DTranspose(n_filters*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv3D(n_filters*2, (2, 2, 2), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv3D(n_filters*2, (2, 2, 2), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
    
    u7 = Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv3D(n_filters, (2, 2, 2), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv3D(n_filters, (2, 2, 2), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    outputs = Conv3D(4, (1, 1, 1), activation='softmax')(c7)
     
    model = Model(inputs=input_img, outputs=outputs)
    #concat =  concatenate([x.output, y.output])
    return model

